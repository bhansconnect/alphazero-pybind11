import glob
import io
import os
from collections import namedtuple
import math
import random
import time
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import threading
import tqdm
import queue
import numpy as np
import gc
import zstandard as zstd
import alphazero
from neural_net import get_device
from config import load_config
from tracy_utils import tracy_zone, tracy_thread, TracyZone, tracy_frame

ZSTD_LEVEL = 3


def save_compressed(tensor, path):
    """Save tensor as bfloat16 + zstd compressed (.ptz)."""
    buffer = io.BytesIO()
    torch.save(tensor.bfloat16(), buffer)
    with open(path, 'wb') as f:
        f.write(zstd.ZstdCompressor(level=ZSTD_LEVEL, threads=-1).compress(buffer.getvalue()))


def load_compressed(path):
    """Load a .ptz file, returning float32 tensor."""
    with open(path, 'rb') as f:
        data = zstd.ZstdDecompressor().decompress(f.read())
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True).float()


def _load_hist_tensor(path):
    """Load a history tensor from .ptz (compressed) or .pt (legacy) file."""
    if path.endswith(".ptz"):
        return load_compressed(path)
    return torch.load(path, map_location="cpu", mmap=True)


def _glob_hist_files(location, pattern):
    """Glob for both .ptz and .pt files, preferring .ptz."""
    files = sorted(glob.glob(os.path.join(location, pattern + ".ptz")))
    if not files:
        files = sorted(glob.glob(os.path.join(location, pattern + ".pt")))
    return files


GRArgs = namedtuple(
    "GRArgs",
    [
        "title",
        "game",
        "max_batch_size",
        "iteration",
        "data_save_size",
        "data_folder",
        "concurrent_batches",
        "batch_workers",
        "nn_workers",
        "result_workers",
        "mcts_workers",
    ],
    defaults=(0, 30_000, None, 0, 0, 1, 1, os.cpu_count() - 1),
)


class GameRunner:
    def __init__(self, players, pm, args):
        self.players = players
        self.pm = pm
        self.args = args
        self.device = get_device()
        self.num_players = self.args.game.NUM_PLAYERS()
        self.batch_workers = self.args.batch_workers
        if self.batch_workers == 0:
            self.batch_workers = self.num_players
        self.concurrent_batches = self.args.concurrent_batches
        if self.concurrent_batches == 0:
            self.concurrent_batches = self.num_players
        if self.batch_workers % self.num_players != 0:
            raise Exception(
                "batch workers should be a multiple of the number of players"
            )
        if self.concurrent_batches % self.batch_workers != 0:
            raise Exception(
                "concurrent batches should be a multiple of the number of batch workers"
            )
        if len(self.players) != self.num_players:
            raise Exception("There must be a player for each player")
        self.ready_queues = []
        for i in range(self.batch_workers):
            self.ready_queues.append(queue.SimpleQueue())
        self.batch_queue = queue.SimpleQueue()
        self.result_queue = queue.SimpleQueue()
        self.monitor_queue = queue.SimpleQueue()
        self.saved_samples = 0
        cs = self.args.game.CANONICAL_SHAPE()
        self.hist_canonical = torch.zeros(self.args.data_save_size, cs[0], cs[1], cs[2])
        self.hist_v = torch.zeros(self.args.data_save_size, self.num_players + 1)
        self.hist_pi = torch.zeros(self.args.data_save_size, self.args.game.NUM_MOVES())
        shape = (self.args.max_batch_size, cs[0], cs[1], cs[2])
        self.batches = []
        self.v = []
        self.pi = []
        for i in range(self.concurrent_batches):
            self.batches.append(torch.zeros(shape))
            self.v.append(torch.zeros((self.num_players + 1)))
            self.pi.append(torch.zeros((self.args.game.NUM_MOVES())))
            if str(self.device) == "cuda":
                self.batches[i].pin_memory()
                self.v[i].pin_memory()
                self.pi[i].pin_memory()
            self.ready_queues[i % self.num_players].put(i)

    @tracy_zone
    def run(self):
        nn_players = set()
        for i in range(self.num_players):
            if not isinstance(self.players[i], (PlayoutPlayer, RandPlayer)):
                nn_players.add(i)

        batch_workers = []
        for i in range(self.batch_workers):
            player = i % self.num_players
            if player not in nn_players:
                continue
            batch_workers.append(
                threading.Thread(
                    target=self.batch_builder, args=(player,)
                )
            )
            batch_workers[-1].start()
        result_workers = []
        for i in range(self.args.result_workers):
            result_workers.append(threading.Thread(target=self.result_processor))
            result_workers[i].start()
        player_workers = []
        for i in range(self.num_players):
            player_workers.append(threading.Thread(target=self.player_executor))
            player_workers[i].start()
        mcts_workers = []
        for i in range(self.args.mcts_workers):
            mcts_workers.append(threading.Thread(target=self.pm.play))
            mcts_workers[i].start()

        monitor = threading.Thread(target=self.monitor)
        monitor.start()
        if self.pm.params().history_enabled:
            hist_saver = threading.Thread(target=self.hist_saver)
            hist_saver.start()

        for bw in batch_workers:
            bw.join()
        for rw in result_workers:
            rw.join()
        for pw in player_workers:
            pw.join()
        for mw in mcts_workers:
            mw.join()
        monitor.join()
        if self.pm.params().history_enabled:
            hist_saver.join()

    @tracy_zone
    def monitor(self):
        tracy_thread("monitor")
        last_completed = 0
        last_update = time.time()
        n = self.pm.params().games_to_play
        pbar = tqdm.tqdm(total=n, unit="games", desc=self.args.title, leave=False)
        while self.pm.remaining_games() > 0:
            try:
                self.monitor_queue.get(timeout=1)
            except queue.Empty:
                continue
            if time.time() - last_update > 1:
                hr = 0
                hits = self.pm.cache_hits()
                total = hits + self.pm.cache_misses()
                if total > 0:
                    hr = hits / total
                completed = self.pm.games_completed()
                scores = self.pm.scores()
                win_rates = [0] * len(scores)
                if completed > 0:
                    for i in range(len(scores)):
                        win_rates[i] = scores[i] / completed
                win_rates = list(map(lambda x: f"{x:0.3f}", win_rates))
                pbar.set_postfix({"win rates": win_rates, "cache rate": hr})
                pbar.update(completed - last_completed)
                last_completed = completed
                last_update = time.time()
        hr = 0
        hits = self.pm.cache_hits()
        total = hits + self.pm.cache_misses()
        if total > 0:
            hr = hits / total
        completed = self.pm.games_completed()
        scores = self.pm.scores()
        win_rates = [0] * len(scores)
        for i in range(len(scores)):
            win_rates[i] = scores[i] / completed
        win_rates = list(map(lambda x: f"{x:0.3f}", win_rates))
        pbar.set_postfix({"win rates": win_rates, "cache hit": hr})
        pbar.update(n - last_completed)
        pbar.close()

    @tracy_zone
    def batch_builder(self, player):
        tracy_thread(f"batch_builder_{player}")
        while self.pm.remaining_games() > 0:
            try:
                batch_index = self.ready_queues[player].get(timeout=1)
            except queue.Empty:
                continue
            batch = self.batches[batch_index]
            game_indices = self.pm.build_batch(
                batch_index % self.num_players, batch, self.batch_workers
            )
            if len(game_indices) == 0:
                self.ready_queues[player].put(batch_index)
                continue
            out = batch[: len(game_indices)]
            out = out.contiguous().to(self.device, non_blocking=True)
            self.batch_queue.put((out, batch_index, game_indices))

    @tracy_zone
    def player_executor(self):
        tracy_thread("player_executor")
        while self.pm.remaining_games() > 0:
            try:
                batch, batch_index, game_indices = self.batch_queue.get(timeout=1)
            except queue.Empty:
                continue
            self.v[batch_index], self.pi[batch_index] = self.players[
                batch_index % self.num_players
            ].process(batch)
            self.result_queue.put((batch_index, game_indices))

    @tracy_zone
    def result_processor(self):
        tracy_thread("result_processor")
        while self.pm.remaining_games() > 0:
            try:
                batch_index, game_indices = self.result_queue.get(timeout=1)
            except queue.Empty:
                continue
            v = self.v[batch_index].cpu().numpy()
            pi = self.pi[batch_index].cpu().numpy()
            if v.size == 0 or pi.size == 0:
                continue
            self.pm.update_inferences(
                batch_index % self.num_players, game_indices, v, pi
            )
            self.ready_queues[batch_index % self.num_players].put(batch_index)
            self.monitor_queue.put(v.shape[0])

    @tracy_zone
    def hist_saver(self):
        tracy_thread("hist_saver")
        batch = 0
        data_folder = self.args.data_folder
        os.makedirs(data_folder, exist_ok=True)
        while self.pm.remaining_games() > 0 or self.pm.hist_count() > 0:
            size = self.pm.build_history_batch(
                self.hist_canonical, self.hist_v, self.hist_pi
            )
            if size == 0:
                continue
            torch.save(
                self.hist_canonical[:size],
                os.path.join(
                    data_folder,
                    f"{self.args.iteration:04d}-{batch:04d}-canonical-{size}.pt",
                ),
            )
            torch.save(
                self.hist_v[:size],
                os.path.join(
                    data_folder,
                    f"{self.args.iteration:04d}-{batch:04d}-v-{size}.pt",
                ),
            )
            torch.save(
                self.hist_pi[:size],
                os.path.join(
                    data_folder,
                    f"{self.args.iteration:04d}-{batch:04d}-pi-{size}.pt",
                ),
            )
            self.saved_samples += size
            batch += 1


class RandPlayer:
    """Marker for random evaluation. Actual eval runs inline in C++ MCTS threads."""
    pass


class PlayoutPlayer:
    """Marker for rollout-based evaluation. Actual eval runs inline in C++ MCTS threads."""
    pass


def set_eval_types(params, players):
    """Set eval_type on params based on player types."""
    eval_types = []
    for p in players:
        if isinstance(p, PlayoutPlayer):
            eval_types.append(alphazero.EvalType.PLAYOUT)
        elif isinstance(p, RandPlayer):
            eval_types.append(alphazero.EvalType.RANDOM)
        else:
            eval_types.append(alphazero.EvalType.NN)
    params.eval_type = eval_types


def base_params(config, start_temp, bs, cb):
    """Create PlayParams from config."""
    params = alphazero.PlayParams()
    params.cache_shards = config.resolved_cache_shards
    params.max_cache_size = config.max_cache_size
    params.cpuct = config.cpuct
    params.start_temp = start_temp
    params.temp_decay_half_life = config.temp_decay_half_life
    params.final_temp = config.final_temp
    params.max_batch_size = bs
    params.concurrent_games = bs * cb
    params.fpu_reduction = config.fpu_reduction
    return params


_ELO_ALPHA = math.log(10) / 400.0


def elo_prob(r1, r2):
    x = _ELO_ALPHA * (r2 - r1)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


@tracy_zone
def get_elo(past_elo, win_rates, new_agent):
    if new_agent != 0:
        past_elo[new_agent] = past_elo[new_agent - 1]
    iters = 5000
    for _ in tqdm.trange(iters, leave=False):
        mean_update = 0
        for j in range(past_elo.shape[0]):
            if not math.isnan(win_rates[new_agent, j]):
                rate = win_rates[new_agent, j]
                rate = max(0.001, rate)
                rate = min(0.999, rate)
                mean_update += rate - elo_prob(past_elo[j], past_elo[new_agent])
        past_elo[new_agent] += mean_update * 32
    return past_elo


def calc_hist_size(config, i):
    return int(
        config.window_size_scalar
        * (
            1
            + config.window_size_beta
            * (((i + 1) / config.window_size_scalar) ** config.window_size_alpha - 1)
            / config.window_size_alpha
        )
    )


def maybe_save(
    config,
    c,
    v,
    p,
    size,
    batch,
    iteration,
    location,
    name="",
    force=False,
    use_compression=False,
):
    if size == config.hist_size or (force and size > 0):
        save_fn = save_compressed if use_compression else torch.save
        ext = ".ptz" if use_compression else ".pt"
        save_fn(
            c[:size],
            os.path.join(
                location, f"{iteration:04d}-{batch:04d}{name}-canonical-{size}{ext}"
            ),
        )
        save_fn(
            v[:size],
            os.path.join(
                location, f"{iteration:04d}-{batch:04d}{name}-v-{size}{ext}"
            ),
        )
        save_fn(
            p[:size],
            os.path.join(
                location, f"{iteration:04d}-{batch:04d}{name}-pi-{size}{ext}"
            ),
        )
        return True
    return False


@tracy_zone
def exploit_symmetries(config, paths, iteration):
    Game = config.Game
    if Game.NUM_SYMMETRIES() <= 1:
        return

    tmp_hist = paths["tmp_history"]
    c_names = sorted(
        glob.glob(os.path.join(tmp_hist, f"{iteration:04d}-*-canonical-*.pt"))
    )
    v_names = sorted(
        glob.glob(os.path.join(tmp_hist, f"{iteration:04d}-*-v-*.pt"))
    )
    p_names = sorted(
        glob.glob(os.path.join(tmp_hist, f"{iteration:04d}-*-pi-*.pt"))
    )

    datasets = []
    for j in range(len(c_names)):
        c_tensor = torch.load(c_names[j], map_location="cpu", mmap=True)
        v_tensor = torch.load(v_names[j], map_location="cpu", mmap=True)
        p_tensor = torch.load(p_names[j], map_location="cpu", mmap=True)
        datasets.append(TensorDataset(c_tensor, v_tensor, p_tensor))
        del c_tensor, v_tensor, p_tensor

    dataset = ConcatDataset(datasets)
    sample_count = len(dataset)

    i_out = 0
    batch_out = 0
    cs = Game.CANONICAL_SHAPE()
    c_out = torch.zeros(config.hist_size, cs[0], cs[1], cs[2])
    v_out = torch.zeros(config.hist_size, Game.NUM_PLAYERS() + 1)
    p_out = torch.zeros(config.hist_size, Game.NUM_MOVES())

    for i in tqdm.trange(
        sample_count, desc="Creating Symmetric Samples", leave=False
    ):
        c, v, pi = dataset[i]
        ph = alphazero.PlayHistory(c, v, pi)
        syms = Game().symmetries(ph)
        for sym in syms:
            c_out[i_out] = torch.from_numpy(np.array(sym.canonical()))
            v_out[i_out] = torch.from_numpy(np.array(sym.v()))
            p_out[i_out] = torch.from_numpy(np.array(sym.pi()))
            i_out += 1
            if maybe_save(
                config,
                c_out,
                v_out,
                p_out,
                i_out,
                batch_out,
                iteration,
                location=tmp_hist,
                name="syms",
            ):
                i_out = 0
                batch_out += 1
        del c, v, pi, ph, syms
    maybe_save(
        config,
        c_out,
        v_out,
        p_out,
        i_out,
        batch_out,
        iteration,
        location=tmp_hist,
        name="syms",
        force=True,
    )

    del datasets, dataset
    del c_out, v_out, p_out

    gc.collect()
    for fn in c_names + v_names + p_names:
        os.remove(fn)


@tracy_zone
def resample_by_surprise(config, paths, experiment_name, iteration):
    import neural_net

    Game = config.Game
    tmp_hist = paths["tmp_history"]
    hist_location = paths["history"]

    c_names = sorted(
        glob.glob(os.path.join(tmp_hist, f"{iteration:04d}-*-canonical-*.pt"))
    )
    v_names = sorted(
        glob.glob(os.path.join(tmp_hist, f"{iteration:04d}-*-v-*.pt"))
    )
    p_names = sorted(
        glob.glob(os.path.join(tmp_hist, f"{iteration:04d}-*-pi-*.pt"))
    )

    datasets = []
    for j in range(len(c_names)):
        c_tensor = torch.load(c_names[j], map_location="cpu", mmap=True)
        v_tensor = torch.load(v_names[j], map_location="cpu", mmap=True)
        p_tensor = torch.load(p_names[j], map_location="cpu", mmap=True)
        datasets.append(TensorDataset(c_tensor, v_tensor, p_tensor))
        del c_tensor, v_tensor, p_tensor

    dataset = ConcatDataset(datasets)
    sample_count = len(dataset)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False)

    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    loss = nn.sample_loss(dataloader, sample_count)
    total_loss = np.sum(loss)

    i_out = 0
    batch_out = 0
    cs = Game.CANONICAL_SHAPE()
    c_out = torch.zeros(config.hist_size, cs[0], cs[1], cs[2])
    v_out = torch.zeros(config.hist_size, Game.NUM_PLAYERS() + 1)
    p_out = torch.zeros(config.hist_size, Game.NUM_MOVES())
    os.makedirs(hist_location, exist_ok=True)

    # Clear old history for iteration before saving new history.
    gc.collect()
    for fn in glob.glob(os.path.join(hist_location, f"{iteration:04d}-*.pt*")):
        os.remove(fn)

    for i in tqdm.trange(sample_count, desc="Resampling Data", leave=False):
        sample_weight = 0.5 + (loss[i] / total_loss) * 0.5 * sample_count
        for _ in range(math.floor(sample_weight)):
            c, v, pi = dataset[i]
            c_out[i_out] = c
            v_out[i_out] = v
            p_out[i_out] = pi
            i_out += 1
            if maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration, location=hist_location, use_compression=True):
                i_out = 0
                batch_out += 1
            del c, v, pi
        if random.random() < sample_weight - math.floor(sample_weight):
            c, v, pi = dataset[i]
            c_out[i_out] = c
            v_out[i_out] = v
            p_out[i_out] = pi
            i_out += 1
            if maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration, location=hist_location, use_compression=True):
                i_out = 0
                batch_out += 1
            del c, v, pi

    maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration, location=hist_location, use_compression=True, force=True)

    del datasets, dataset, dataloader, nn
    del c_out, v_out, p_out

    gc.collect()
    for fn in glob.glob(os.path.join(tmp_hist, "*")):
        os.remove(fn)


@tracy_zone
def iteration_loss(config, paths, experiment_name, iteration):
    import neural_net

    Game = config.Game
    hist_location = paths["history"]
    datasets = []
    c = _glob_hist_files(hist_location, f"{iteration:04d}-*-canonical-*")
    v = _glob_hist_files(hist_location, f"{iteration:04d}-*-v-*")
    p = _glob_hist_files(hist_location, f"{iteration:04d}-*-pi-*")
    for j in range(len(c)):
        c_tensor = _load_hist_tensor(c[j])
        v_tensor = _load_hist_tensor(v[j])
        p_tensor = _load_hist_tensor(p[j])
        datasets.append(TensorDataset(c_tensor, v_tensor, p_tensor))
        del c_tensor, v_tensor, p_tensor

    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    v_loss, pi_loss = nn.losses(dataloader)

    del datasets, dataset, dataloader, nn

    return v_loss, pi_loss


@tracy_zone
def train(config, paths, experiment_name, iteration, hist_size, run, total_train_steps):
    import neural_net

    Game = config.Game
    hist_location = paths["history"]
    total_size = 0
    datasets = []
    for i in range(max(0, iteration - hist_size), iteration + 1):
        c = _glob_hist_files(hist_location, f"{i:04d}-*-canonical-*")
        v = _glob_hist_files(hist_location, f"{i:04d}-*-v-*")
        p = _glob_hist_files(hist_location, f"{i:04d}-*-pi-*")
        for j in range(len(c)):
            size = int(c[j].split("-")[-1].split(".")[0])
            total_size += size
            c_tensor = _load_hist_tensor(c[j])
            v_tensor = _load_hist_tensor(v[j])
            p_tensor = _load_hist_tensor(p[j])
            datasets.append(TensorDataset(c_tensor, v_tensor, p_tensor))
            del c_tensor, v_tensor, p_tensor

    bs = config.train_batch_size
    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    average_generation = total_size / min(hist_size, iteration + 1)
    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    steps_to_train = int(math.ceil(average_generation / bs * config.train_sample_rate))
    v_loss, pi_loss = nn.train(
        dataloader, steps_to_train, run, iteration, total_train_steps
    )
    total_train_steps += steps_to_train
    nn.save_checkpoint(paths["checkpoint"], f"{iteration + 1:04d}-{experiment_name}.pt")
    del datasets, dataset, dataloader, nn
    return v_loss, pi_loss, total_train_steps


def update_reservoir(config, paths, iteration, hist_size):
    """Merge evicted window data into the reservoir and delete old files."""
    hist_location = paths["history"]
    reservoir_location = paths["reservoir"]

    oldest_in_window = max(0, iteration - hist_size)
    prev_oldest = max(0, (iteration - 1) - calc_hist_size(config, iteration - 1))
    evicted_iters = list(range(prev_oldest, oldest_in_window))
    if not evicted_iters:
        return

    new_c, new_v, new_pi = [], [], []
    new_iters = []
    for it in evicted_iters:
        c_files = _glob_hist_files(hist_location, f"{it:04d}-*-canonical-*")
        v_files = _glob_hist_files(hist_location, f"{it:04d}-*-v-*")
        p_files = _glob_hist_files(hist_location, f"{it:04d}-*-pi-*")
        for j in range(len(c_files)):
            ct = _load_hist_tensor(c_files[j])
            new_c.append(ct)
            new_v.append(_load_hist_tensor(v_files[j]))
            new_pi.append(_load_hist_tensor(p_files[j]))
            new_iters.append(torch.full((ct.shape[0],), it, dtype=torch.int16))

    if not new_c:
        return

    new_c = torch.cat(new_c)
    new_v = torch.cat(new_v)
    new_pi = torch.cat(new_pi)
    new_iters = torch.cat(new_iters)

    os.makedirs(reservoir_location, exist_ok=True)
    res_c_path = os.path.join(reservoir_location, "canonical.ptz")
    if os.path.exists(res_c_path):
        old_c = load_compressed(res_c_path)
        old_v = load_compressed(os.path.join(reservoir_location, "v.ptz"))
        old_pi = load_compressed(os.path.join(reservoir_location, "pi.ptz"))
        old_iters = load_compressed(os.path.join(reservoir_location, "meta.ptz")).to(torch.int16)
        all_c = torch.cat([old_c, new_c])
        all_v = torch.cat([old_v, new_v])
        all_pi = torch.cat([old_pi, new_pi])
        all_iters = torch.cat([old_iters, new_iters])
        del old_c, old_v, old_pi, old_iters
    else:
        all_c, all_v, all_pi, all_iters = new_c, new_v, new_pi, new_iters

    # Compute capacity = total samples in current window
    capacity = 0
    for i in range(max(0, iteration - hist_size), iteration + 1):
        for fn in glob.glob(os.path.join(hist_location, f"{i:04d}-*-canonical-*.pt*")):
            capacity += int(fn.split("-")[-1].split(".")[0])

    if len(all_c) > capacity > 0:
        ages = (iteration - all_iters.float()).clamp(min=0)
        weights = config.reservoir_recency_decay ** ages
        indices = torch.multinomial(weights, capacity, replacement=False)
        all_c = all_c[indices]
        all_v = all_v[indices]
        all_pi = all_pi[indices]
        all_iters = all_iters[indices]

    save_compressed(all_c, res_c_path)
    save_compressed(all_v, os.path.join(reservoir_location, "v.ptz"))
    save_compressed(all_pi, os.path.join(reservoir_location, "pi.ptz"))
    save_compressed(all_iters.float(), os.path.join(reservoir_location, "meta.ptz"))

    del new_c, new_v, new_pi, new_iters, all_c, all_v, all_pi, all_iters
    gc.collect()
    for it in evicted_iters:
        for fn in glob.glob(os.path.join(hist_location, f"{it:04d}-*.pt*")):
            os.remove(fn)


def load_reservoir(paths):
    """Load reservoir as a TensorDataset, or None if no reservoir exists."""
    res_c_path = os.path.join(paths["reservoir"], "canonical.ptz")
    if not os.path.exists(res_c_path):
        return None
    c = load_compressed(res_c_path)
    v = load_compressed(os.path.join(paths["reservoir"], "v.ptz"))
    pi = load_compressed(os.path.join(paths["reservoir"], "pi.ptz"))
    return TensorDataset(c, v, pi)


def load_available_window(paths, start, end):
    """Load whatever window history files exist across an iteration range."""
    hist_location = paths["history"]
    datasets = []
    for i in range(start, end):
        c = _glob_hist_files(hist_location, f"{i:04d}-*-canonical-*")
        v = _glob_hist_files(hist_location, f"{i:04d}-*-v-*")
        p = _glob_hist_files(hist_location, f"{i:04d}-*-pi-*")
        for j in range(len(c)):
            c_tensor = _load_hist_tensor(c[j])
            v_tensor = _load_hist_tensor(v[j])
            p_tensor = _load_hist_tensor(p[j])
            datasets.append(TensorDataset(c_tensor, v_tensor, p_tensor))
            del c_tensor, v_tensor, p_tensor
    return datasets


@tracy_zone
def self_play(config, paths, experiment_name, best, iteration, depth, fast_depth):
    import neural_net

    Game = config.Game
    bs = config.self_play_batch_size
    cb = Game.NUM_PLAYERS() * config.self_play_concurrent_batch_mult
    n = bs * cb * config.self_play_chunks
    params = base_params(config, config.self_play_temp, bs, cb)
    params.games_to_play = n
    params.mcts_depth = [depth] * Game.NUM_PLAYERS()
    params.self_play = True
    params.history_enabled = True
    params.add_noise = True
    params.playout_cap_randomization = True
    params.playout_cap_depth = fast_depth
    params.playout_cap_percent = 0.75
    params.resign_percent = config.resign_percent
    params.resign_playthrough_percent = config.resign_playthrough_percent

    use_rand = best == 0

    if use_rand:
        nn = RandPlayer()
        params.max_cache_size = 0
    else:
        nn = neural_net.NNWrapper.load_checkpoint(
            Game, paths["checkpoint"], f"{best:04d}-{experiment_name}.pt"
        )

    players = [nn] * Game.NUM_PLAYERS()
    set_eval_types(params, players)

    pm = alphazero.PlayManager(Game(), params)
    grargs = GRArgs(
        title="Self Play",
        game=Game,
        iteration=iteration,
        max_batch_size=bs,
        concurrent_batches=cb,
        result_workers=config.result_workers,
        data_folder=paths["tmp_history"],
    )

    gr = GameRunner(players, pm, grargs)
    gr.run()
    scores = pm.scores()
    win_rates = [0] * len(scores)
    sn = sum(scores)
    for i in range(len(scores)):
        win_rates[i] = scores[i] / sn
    resign_scores = pm.resign_scores()
    resign_win_rates = [0] * len(resign_scores)
    rn = sum(resign_scores)
    for i in range(len(resign_scores)):
        resign_win_rates[i] = resign_scores[i] / rn
    resign_rate = rn / sn
    hits = pm.cache_hits()
    total = hits + pm.cache_misses()
    hr = 0
    if total > 0:
        hr = hits / total
    agl = pm.avg_game_length()
    avg_depth = pm.avg_leaf_depth()
    avg_entropy = pm.avg_search_entropy()
    fast_avg_depth = pm.fast_avg_leaf_depth()
    fast_avg_entropy = pm.fast_avg_search_entropy()
    avg_mpt = pm.avg_moves_per_turn()
    avg_vm = pm.avg_valid_moves()
    del pm, nn
    return win_rates, hr, agl, resign_win_rates, resign_rate, avg_depth, avg_entropy, fast_avg_depth, fast_avg_entropy, avg_mpt, avg_vm


@tracy_zone
def play_past(config, paths, experiment_name, depth, iteration, past_iter, batch_size=64):
    import neural_net

    Game = config.Game
    nn_rate = 0
    draw_rate = 0
    hr = 0
    agl = 0
    avg_depth = 0
    avg_entropy = 0
    avg_mpt = 0
    avg_vm = 0
    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    if past_iter == 0:
        nn_past = RandPlayer()
    else:
        nn_past = neural_net.NNWrapper.load_checkpoint(
            Game, paths["checkpoint"], f"{past_iter:04d}-{experiment_name}.pt"
        )
    cb = Game.NUM_PLAYERS()
    if Game.NUM_PLAYERS() > 2:
        bs = batch_size
        n = bs * cb
        for i in tqdm.trange(
            Game.NUM_PLAYERS(),
            leave=False,
            desc=f"Bench 1 new vs {Game.NUM_PLAYERS() - 1} old",
        ):
            params = base_params(config, config.eval_temp, bs, cb)
            params.games_to_play = n
            params.mcts_depth = [depth] * Game.NUM_PLAYERS()
            players = [nn_past] * Game.NUM_PLAYERS()
            players[i] = nn
            set_eval_types(params, players)
            pm = alphazero.PlayManager(Game(), params)

            grargs = GRArgs(
                title=f"Bench {iteration} v {past_iter} as p{i + 1}",
                game=Game,
                iteration=iteration,
                max_batch_size=bs,
                concurrent_batches=cb,
                result_workers=config.result_workers,
            )
            gr = GameRunner(players, pm, grargs)
            gr.run()
            scores = pm.scores()
            nn_rate += scores[i] / n
            draw_rate += scores[-1] / n
            hits = pm.cache_hits()
            total = hits + pm.cache_misses()
            if total > 0:
                hr += hits / total
            agl += pm.avg_game_length()
            avg_depth += pm.avg_leaf_depth()
            avg_entropy += pm.avg_search_entropy()
            avg_mpt += pm.avg_moves_per_turn()
            avg_vm += pm.avg_valid_moves()
            del pm
            gc.collect()
        for i in tqdm.trange(
            Game.NUM_PLAYERS(),
            leave=False,
            desc=f"Bench {Game.NUM_PLAYERS() - 1} new vs 1 old",
        ):
            params = base_params(config, config.eval_temp, bs, cb)
            params.games_to_play = n
            params.mcts_depth = [depth] * Game.NUM_PLAYERS()
            players = [nn] * Game.NUM_PLAYERS()
            players[i] = nn_past
            set_eval_types(params, players)
            pm = alphazero.PlayManager(Game(), params)

            grargs = GRArgs(
                title=f"Bench {iteration} v {past_iter} as p{i + 1}",
                game=Game,
                iteration=iteration,
                max_batch_size=bs,
                concurrent_batches=cb,
                result_workers=config.result_workers,
            )
            gr = GameRunner(players, pm, grargs)
            gr.run()
            scores = pm.scores()
            for j in range(1, Game.NUM_PLAYERS()):
                nn_rate += scores[(i + j) % Game.NUM_PLAYERS()] / n
            draw_rate += scores[-1] / n
            hits = pm.cache_hits()
            total = hits + pm.cache_misses()
            if total > 0:
                hr += hits / total
            agl += pm.avg_game_length()
            avg_depth += pm.avg_leaf_depth()
            avg_entropy += pm.avg_search_entropy()
            avg_mpt += pm.avg_moves_per_turn()
            avg_vm += pm.avg_valid_moves()
            del pm
            gc.collect()
        nn_rate /= 2 * Game.NUM_PLAYERS()
        draw_rate /= 2 * Game.NUM_PLAYERS()
        hr /= 2 * Game.NUM_PLAYERS()
        agl /= 2 * Game.NUM_PLAYERS()
        avg_depth /= 2 * Game.NUM_PLAYERS()
        avg_entropy /= 2 * Game.NUM_PLAYERS()
        avg_mpt /= 2 * Game.NUM_PLAYERS()
        avg_vm /= 2 * Game.NUM_PLAYERS()
    else:
        bs = batch_size
        n = bs * cb
        for i in tqdm.trange(
            Game.NUM_PLAYERS(), leave=False, desc="Bench new vs old"
        ):
            params = base_params(config, config.eval_temp, bs, cb)
            params.games_to_play = n
            params.mcts_depth = [depth] * Game.NUM_PLAYERS()
            players = [nn_past] * Game.NUM_PLAYERS()
            players[i] = nn
            set_eval_types(params, players)
            pm = alphazero.PlayManager(Game(), params)

            grargs = GRArgs(
                title=f"Bench {iteration} v {past_iter} as p{i + 1}",
                game=Game,
                iteration=iteration,
                max_batch_size=bs,
                concurrent_batches=cb,
                result_workers=config.result_workers,
            )
            gr = GameRunner(players, pm, grargs)
            gr.run()
            scores = pm.scores()
            nn_rate += scores[i] / n
            draw_rate += scores[-1] / n
            hits = pm.cache_hits()
            total = hits + pm.cache_misses()
            if total > 0:
                hr += hits / total
            agl += pm.avg_game_length()
            avg_depth += pm.avg_leaf_depth()
            avg_entropy += pm.avg_search_entropy()
            avg_mpt += pm.avg_moves_per_turn()
            avg_vm += pm.avg_valid_moves()
            del pm
            gc.collect()
        nn_rate /= Game.NUM_PLAYERS()
        draw_rate /= Game.NUM_PLAYERS()
        hr /= Game.NUM_PLAYERS()
        agl /= Game.NUM_PLAYERS()
        avg_depth /= Game.NUM_PLAYERS()
        avg_entropy /= Game.NUM_PLAYERS()
        avg_mpt /= Game.NUM_PLAYERS()
        avg_vm /= Game.NUM_PLAYERS()

    del nn, nn_past
    return nn_rate, draw_rate, hr, agl, avg_depth, avg_entropy, avg_mpt, avg_vm


def main(config, experiment_dir, aim_repo=None):
    """Main training loop.

    Args:
        config: TrainConfig instance with all training parameters.
        experiment_dir: Path to experiment directory (e.g. data/connect4/densenet-4d-12c-5k-100sims/).
        aim_repo: Path for aim logging directory. Default None uses project root.
    """
    import neural_net

    Game = config.Game
    paths = config.resolve_paths(experiment_dir)
    experiment_name = config.auto_experiment_name

    # Ensure all directories exist
    for key in ("checkpoint", "history", "tmp_history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    # Save config at start
    config.save(os.path.join(experiment_dir, "config.yaml"))

    @tracy_zone
    def create_init_net():
        nnargs = neural_net.NNArgs(
            num_channels=config.channels,
            depth=config.depth,
            lr_milestone=config.lr_milestone,
            dense_net=config.dense_net,
            kernel_size=config.kernel_size,
            star_gambit_spatial=config.star_gambit_spatial,
            lr=config.lr,
            cv=config.cv,
        )
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.save_checkpoint(paths["checkpoint"], f"0000-{experiment_name}.pt")

    try:
        import aim
        run = aim.Run(experiment=experiment_name, repo=aim_repo)
        run.name = config.game
        run["hparams"] = {
            "network": config.network_name,
            "panel_size": config.gating_panel_size,
            "panel_win_rate": config.gating_panel_win_rate,
            "best_win_rate": config.gating_best_win_rate,
            "expected_opening_length": config.expected_opening_length,
            "cpuct": config.cpuct,
            "fpu_reduction": config.fpu_reduction,
            "self_play_temp": config.self_play_temp,
            "eval_temp": config.eval_temp,
            "final_temp": config.final_temp,
            "training_sample_rate": config.train_sample_rate,
            "depth": config.depth,
            "channels": config.channels,
            "kernel_size": config.kernel_size,
            "lr_milestone": config.lr_milestone,
            "full_mcts_depth": config.selfplay_mcts_depth,
            "fast_mcts_depth": config.fast_mcts_depth,
        }
    except ImportError:
        print("aim is used for nice web logging with graphs. I would advise `pip install aim`.")
        print("Using a dummy logger for now that does nothing.\n")

        class DummyRun:
            def track(*args, **kwargs):
                pass

        run = DummyRun()

    total_agents = config.iterations + 1  # + base

    start = config.start

    if start == 0:
        create_init_net()
        wr = np.empty((total_agents, total_agents))
        wr[:] = np.nan
        elo = np.zeros(total_agents)
        current_best = 0
        total_train_steps = 0

        # Handle bootstrap from existing experiment
        if config.bootstrap_from:
            source_dir = config.bootstrap_from
            source_config_path = os.path.join(source_dir, "config.yaml")

            # Determine source final iteration from elo.csv
            source_elo_path = os.path.join(source_dir, "elo.csv")
            if os.path.exists(source_elo_path):
                source_elo = np.genfromtxt(source_elo_path, delimiter=",")
                source_n = len(source_elo)
            else:
                raise RuntimeError(f"No elo.csv in bootstrap source: {source_dir}")

            source_paths = config.resolve_paths(source_dir)

            # Copy reservoir
            import shutil
            if os.path.exists(os.path.join(source_paths["reservoir"], "canonical.ptz")):
                os.makedirs(paths["reservoir"], exist_ok=True)
                for fname in ("canonical.ptz", "v.ptz", "pi.ptz", "meta.ptz"):
                    src = os.path.join(source_paths["reservoir"], fname)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(paths["reservoir"], fname))

            # Copy elo/wr from source
            source_wr_path = os.path.join(source_dir, "win_rate.csv")
            if os.path.exists(source_wr_path):
                source_wr = np.genfromtxt(source_wr_path, delimiter=",")
                copy_size = min(source_n, total_agents)
                wr[:copy_size, :copy_size] = source_wr[:copy_size, :copy_size]
                elo[:copy_size] = source_elo[:copy_size]

            # Check if same architecture
            same_arch = True
            if os.path.exists(source_config_path):
                source_cfg = load_config(source_config_path, {})
                same_arch = (
                    source_cfg.depth == config.depth
                    and source_cfg.channels == config.channels
                    and source_cfg.kernel_size == config.kernel_size
                    and source_cfg.dense_net == config.dense_net
                    and source_cfg.star_gambit_spatial == config.star_gambit_spatial
                )

            if same_arch:
                # Copy best checkpoint
                source_checkpoints = sorted(glob.glob(os.path.join(source_paths["checkpoint"], "*.pt")))
                if source_checkpoints:
                    best_cp = source_checkpoints[-1]
                    dest_name = f"{source_n:04d}-{experiment_name}.pt"
                    shutil.copy2(best_cp, os.path.join(paths["checkpoint"], dest_name))
            else:
                # Retrain on source data
                reservoir_ds = load_reservoir(source_paths)
                window_datasets = load_available_window(source_paths, 0, source_n)
                nn = neural_net.NNWrapper.load_checkpoint(
                    Game, paths["checkpoint"], f"0000-{experiment_name}.pt"
                )
                all_datasets = ([reservoir_ds] if reservoir_ds else []) + window_datasets
                if all_datasets:
                    combined = ConcatDataset(all_datasets)
                    dataloader = DataLoader(combined, batch_size=config.train_batch_size, shuffle=True)
                    steps_p1 = int(math.ceil(len(combined) / config.train_batch_size)) * config.bootstrap_full_passes
                    nn.train(dataloader, steps_p1, run, source_n, total_train_steps)
                    total_train_steps += steps_p1

                    if window_datasets:
                        window_only = ConcatDataset(window_datasets)
                        dataloader2 = DataLoader(window_only, batch_size=config.train_batch_size, shuffle=True)
                        steps_p2 = int(math.ceil(len(window_only) / config.train_batch_size)) * config.bootstrap_window_passes
                        nn.train(dataloader2, steps_p2, run, source_n, total_train_steps)
                        total_train_steps += steps_p2

                nn.save_checkpoint(paths["checkpoint"], f"{source_n:04d}-{experiment_name}.pt")

            current_best = source_n
            start = source_n

            # Calibrate bootstrap network ELO against source networks
            compare_start = max(0, source_n - config.compare_past)
            for past_iter in range(compare_start, source_n):
                past_cp = os.path.join(source_paths["checkpoint"], f"{past_iter:04d}-{experiment_name}.pt")
                if not os.path.exists(past_cp):
                    # Try to find a checkpoint with a different experiment name
                    cps = glob.glob(os.path.join(source_paths["checkpoint"], f"{past_iter:04d}-*.pt"))
                    if cps:
                        # Copy it with our experiment name
                        import shutil
                        shutil.copy2(cps[0], os.path.join(paths["checkpoint"], f"{past_iter:04d}-{experiment_name}.pt"))
                    else:
                        continue

                nn_rate, draw_rate, _, _, _, _, _, _ = play_past(
                    config, paths, experiment_name, config.compare_mcts_depth,
                    source_n, past_iter, config.past_compare_batch_size
                )
                wr[source_n, past_iter] = nn_rate + draw_rate / Game.NUM_PLAYERS()
                wr[past_iter, source_n] = 1 - (nn_rate + draw_rate / Game.NUM_PLAYERS())
                gc.collect()

            elo = get_elo(elo, wr, source_n)

        np.savetxt(os.path.join(experiment_dir, "elo.csv"), elo, delimiter=",")
        np.savetxt(os.path.join(experiment_dir, "win_rate.csv"), wr, delimiter=",")
        np.savetxt(
            os.path.join(experiment_dir, "total_train_steps.txt"),
            [total_train_steps],
            delimiter=",",
        )
    else:
        tmp_wr = np.genfromtxt(os.path.join(experiment_dir, "win_rate.csv"), delimiter=",")
        wr = np.empty((total_agents, total_agents))
        wr[:] = np.nan
        old_size = min(tmp_wr.shape[0], total_agents)
        wr[:old_size, :old_size] = tmp_wr[:old_size, :old_size]
        tmp_elo = np.genfromtxt(os.path.join(experiment_dir, "elo.csv"), delimiter=",")
        elo = np.zeros(total_agents)
        old_size = min(tmp_elo.shape[0], total_agents)
        elo[:old_size] = tmp_elo[:old_size]
        current_best = np.argmax(elo[: start + 1])
        total_train_steps = int(
            np.genfromtxt(os.path.join(experiment_dir, "total_train_steps.txt"))
        )

    postfix = {"best": current_best}
    panel = [current_best]

    with tqdm.trange(start, config.iterations, desc="Build Amazing Network") as pbar:
        for i in pbar:
            stage_times = {}
            iteration_start = time.time()

            run.track(
                current_best, name="best_network", epoch=i, step=total_train_steps
            )

            stage_start = time.time()
            with TracyZone("stage_history"):
                past_iter = max(0, i - config.compare_past)
                if past_iter != i and math.isnan(wr[i, past_iter]):
                    nn_rate, draw_rate, _, game_length, bench_depth, bench_entropy, bench_mpt, bench_vm = play_past(
                        config, paths, experiment_name,
                        config.compare_mcts_depth, i, past_iter, config.past_compare_batch_size
                    )
                    wr[i, past_iter] = nn_rate + draw_rate / Game.NUM_PLAYERS()
                    wr[past_iter, i] = 1 - (nn_rate + draw_rate / Game.NUM_PLAYERS())
                    run.track(nn_rate, name="win_rate", epoch=i, step=total_train_steps, context={"vs": f"-{config.compare_past}", "from": "all_games"})
                    run.track(draw_rate, name="draw_rate", epoch=i, step=total_train_steps, context={"vs": f"-{config.compare_past}", "from": "all_games"})
                    run.track(game_length, name="average_game_length", epoch=i, step=total_train_steps, context={"vs": f"-{config.compare_past}"})
                    run.track(bench_depth, name="avg_leaf_depth", epoch=i, step=total_train_steps, context={"vs": f"-{config.compare_past}", "search": "full"})
                    run.track(bench_entropy, name="search_entropy", epoch=i, step=total_train_steps, context={"vs": f"-{config.compare_past}", "search": "full"})
                    run.track(bench_mpt, name="moves_per_turn", epoch=i, step=total_train_steps, context={"vs": f"-{config.compare_past}"})
                    run.track(bench_vm, name="avg_valid_moves", epoch=i, step=total_train_steps, context={"vs": f"-{config.compare_past}"})
                    postfix[f"vs -{config.compare_past}"] = (nn_rate + draw_rate / Game.NUM_PLAYERS())
                    gc.collect()
            stage_times["history"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_elo"):
                elo = get_elo(elo, wr, i)
                run.track(elo[i], name="elo", epoch=i, step=total_train_steps, context={"type": "current"})
                run.track(elo[current_best], name="elo", epoch=i, step=total_train_steps, context={"type": "best"})
                postfix["elo"] = int(elo[i])
                pbar.set_postfix(postfix)
                np.savetxt(os.path.join(experiment_dir, "elo.csv"), elo, delimiter=",")
            stage_times["elo"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_selfplay"):
                win_rates, hit_rate, game_length, resign_win_rates, resignation_rate, selfplay_depth, selfplay_entropy, fast_selfplay_depth, fast_selfplay_entropy, selfplay_mpt, selfplay_vm = (
                    self_play(
                        config, paths, experiment_name,
                        current_best, i,
                        config.selfplay_mcts_depth,
                        config.fast_mcts_depth,
                    )
                )
                for j in range(len(win_rates) - 1):
                    run.track(win_rates[j], name="win_rate", epoch=i, step=total_train_steps, context={"vs": "self", "player": j + 1, "from": "all_games"})
                for j in range(len(resign_win_rates) - 1):
                    run.track(resign_win_rates[j], name="win_rate", epoch=i, step=total_train_steps, context={"vs": "self", "player": j + 1, "from": "resignation"})
                run.track(resignation_rate, name="resignation_rate", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(win_rates[-1], name="draw_rate", epoch=i, step=total_train_steps, context={"vs": "self", "from": "all_games"})
                run.track(resign_win_rates[-1], name="draw_rate", epoch=i, step=total_train_steps, context={"vs": "self", "from": "resignation"})
                run.track(float(hit_rate), name="cache_hit_rate", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(game_length, name="average_game_length", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(selfplay_depth, name="avg_leaf_depth", epoch=i, step=total_train_steps, context={"vs": "self", "search": "full"})
                run.track(selfplay_entropy, name="search_entropy", epoch=i, step=total_train_steps, context={"vs": "self", "search": "full"})
                if fast_selfplay_depth > 0:
                    run.track(fast_selfplay_depth, name="avg_leaf_depth", epoch=i, step=total_train_steps, context={"vs": "self", "search": "fast"})
                if fast_selfplay_entropy > 0:
                    run.track(fast_selfplay_entropy, name="search_entropy", epoch=i, step=total_train_steps, context={"vs": "self", "search": "fast"})
                run.track(selfplay_mpt, name="moves_per_turn", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(selfplay_vm, name="avg_valid_moves", epoch=i, step=total_train_steps, context={"vs": "self"})
                postfix["win_rates"] = list(map(lambda x: f"{x:0.3f}", win_rates))
                pbar.set_postfix(postfix)
                gc.collect()
            stage_times["selfplay"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_symmetries"):
                exploit_symmetries(config, paths, i)
                gc.collect()
            stage_times["symmetries"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_resampling"):
                resample_by_surprise(config, paths, experiment_name, i)
                gc.collect()
            stage_times["resampling"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_training"):
                hist_size = calc_hist_size(config, i)
                oldest_iteration = max(0, i - hist_size)
                run.track(hist_size, name="history", epoch=i, step=total_train_steps, context={"type": "window_size"})
                run.track(oldest_iteration, name="history", epoch=i, step=total_train_steps, context={"type": "oldest_iteration"})
                v_loss, pi_loss, total_train_steps = train(
                    config, paths, experiment_name, i, hist_size, run, total_train_steps
                )
                np.savetxt(
                    os.path.join(experiment_dir, "total_train_steps.txt"),
                    [total_train_steps],
                    delimiter=",",
                )
                postfix["vloss"] = v_loss
                postfix["ploss"] = pi_loss
                pbar.set_postfix(postfix)
                gc.collect()
            stage_times["training"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_reservoir"):
                update_reservoir(config, paths, i, hist_size)
                gc.collect()
            stage_times["reservoir"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_gating"):
                next_net = i + 1
                panel_nn_rate = 0
                panel_draw_rate = 0
                panel_game_length = 0
                panel_depth = 0
                panel_entropy = 0
                panel_mpt = 0
                panel_vm = 0
                best_win_rate = 0
                for gate_net in tqdm.tqdm(
                    panel, desc=f"Pitting against Panel {panel}", leave=False
                ):
                    nn_rate, draw_rate, _, game_length, gate_depth, gate_entropy, gate_mpt, gate_vm = play_past(
                        config, paths, experiment_name,
                        config.compare_mcts_depth, next_net, gate_net, config.gate_compare_batch_size
                    )
                    panel_nn_rate += nn_rate
                    panel_draw_rate += draw_rate
                    panel_game_length += game_length
                    panel_depth += gate_depth
                    panel_entropy += gate_entropy
                    panel_mpt += gate_mpt
                    panel_vm += gate_vm
                    wr[next_net, gate_net] = nn_rate + draw_rate / Game.NUM_PLAYERS()
                    wr[gate_net, next_net] = 1 - (nn_rate + draw_rate / Game.NUM_PLAYERS())
                    gc.collect()
                    if gate_net == current_best:
                        run.track(nn_rate, name="win_rate", epoch=next_net, step=total_train_steps, context={"vs": "best", "from": "all_games"})
                        run.track(draw_rate, name="draw_rate", epoch=next_net, step=total_train_steps, context={"vs": "best", "from": "all_games"})
                        run.track(game_length, name="average_game_length", epoch=next_net, context={"vs": "best"})
                        run.track(gate_depth, name="avg_leaf_depth", epoch=next_net, step=total_train_steps, context={"vs": "best", "search": "full"})
                        run.track(gate_entropy, name="search_entropy", epoch=next_net, step=total_train_steps, context={"vs": "best", "search": "full"})
                        run.track(gate_mpt, name="moves_per_turn", epoch=next_net, step=total_train_steps, context={"vs": "best"})
                        run.track(gate_vm, name="avg_valid_moves", epoch=next_net, step=total_train_steps, context={"vs": "best"})
                        best_win_rate = nn_rate + draw_rate / Game.NUM_PLAYERS()
                        postfix["vs best"] = best_win_rate
                        pbar.set_postfix(postfix)
                panel_nn_rate /= len(panel)
                panel_draw_rate /= len(panel)
                panel_game_length /= len(panel)
                panel_depth /= len(panel)
                panel_entropy /= len(panel)
                panel_mpt /= len(panel)
                panel_vm /= len(panel)
                run.track(panel_nn_rate, name="win_rate", epoch=next_net, step=total_train_steps, context={"vs": "panel", "from": "all_games"})
                run.track(panel_draw_rate, name="draw_rate", epoch=next_net, step=total_train_steps, context={"vs": "panel", "from": "all_games"})
                run.track(panel_game_length, name="average_game_length", epoch=next_net, context={"vs": "panel"})
                run.track(panel_depth, name="avg_leaf_depth", epoch=next_net, step=total_train_steps, context={"vs": "panel"})
                run.track(panel_entropy, name="search_entropy", epoch=next_net, step=total_train_steps, context={"vs": "panel"})
                run.track(panel_mpt, name="moves_per_turn", epoch=next_net, step=total_train_steps, context={"vs": "panel"})
                run.track(panel_vm, name="avg_valid_moves", epoch=next_net, step=total_train_steps, context={"vs": "panel"})
                panel_win_rate = panel_nn_rate + panel_draw_rate / Game.NUM_PLAYERS()
                postfix["vs panel"] = panel_win_rate
                panel_ratio = len(panel) / config.gating_panel_size
                wanted_panel_win_rate = (config.gating_panel_win_rate * panel_ratio) + (
                    config.gating_best_win_rate * (1.0 - panel_ratio)
                )
                if (
                    panel_win_rate > wanted_panel_win_rate
                    and best_win_rate > config.gating_best_win_rate
                ):
                    current_best = next_net
                    postfix["best"] = current_best
                    pbar.set_postfix(postfix)
                    panel.append(current_best)
                    while len(panel) > config.gating_panel_size:
                        panel = panel[1:]
                np.savetxt(os.path.join(experiment_dir, "win_rate.csv"), wr, delimiter=",")
            stage_times["gating"] = time.time() - stage_start

            total_time = time.time() - iteration_start
            for stage_name, stage_time in stage_times.items():
                percentage = (stage_time / total_time) * 100.0
                run.track(percentage, name="time_percent", epoch=i, step=total_train_steps, context={"stage": stage_name})
                run.track(stage_time, name="time_seconds", epoch=i, step=total_train_steps, context={"stage": stage_name})
            run.track(total_time, name="time_seconds", epoch=i, step=total_train_steps, context={"stage": "total"})

            tracy_frame()
