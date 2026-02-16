import glob
import io
import json
import os
import signal
import shutil
import tempfile
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import random
import time
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, IterableDataset
import threading
import tqdm
import queue
import numpy as np
import gc
import zstandard as zstd
import alphazero
from neural_net import get_device, get_storage_dtype, to_half_safe
from config import load_config, find_latest_checkpoint
from tracy_utils import tracy_zone, tracy_thread, TracyZone, tracy_frame

def save_compressed(tensor, path, half_storage=True, zstd_level=1):
    """Save tensor with zstd compression, optionally as half-precision."""
    buffer = io.BytesIO()
    if half_storage:
        tensor = to_half_safe(tensor, get_storage_dtype())
    torch.save(tensor, buffer)
    with open(path, 'wb') as f:
        f.write(zstd.ZstdCompressor(level=zstd_level, threads=-1).compress(buffer.getvalue()))


def load_compressed(path):
    """Load a .ptz file, returning tensor in storage dtype (typically float16)."""
    with open(path, 'rb') as f:
        data = zstd.ZstdDecompressor().decompress(f.read())
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _atomic_save_compressed(tensor, path, half_storage=True, zstd_level=1):
    """Save tensor to path atomically via temp file."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    os.close(fd)
    try:
        save_compressed(tensor, tmp_path, half_storage, zstd_level=zstd_level)
        os.replace(tmp_path, path)  # atomic on POSIX
    except:
        os.unlink(tmp_path)
        raise


def _load_hist_tensor(path):
    """Load a history tensor from .ptz (compressed) or .pt (legacy) file."""
    if path.endswith(".ptz"):
        return load_compressed(path)
    return torch.load(path, map_location="cpu", mmap=True)


def prepare_inference_model(nn, config):
    """Apply inference optimizations from config."""
    if hasattr(nn, 'enable_inference_optimizations'):
        nn.enable_inference_optimizations(
            amp=config.amp_inference,
            compile=config.torch_compile,
        )


def _glob_hist_files(location, pattern):
    """Glob for both .ptz and .pt files, preferring .ptz."""
    files = sorted(glob.glob(os.path.join(location, pattern + ".ptz")))
    if not files:
        files = sorted(glob.glob(os.path.join(location, pattern + ".pt")))
    return files


def glob_file_triples(directory, pattern="*-canonical-*.ptz"):
    """Glob for file triples (canonical, v, pi) and return (c_path, v_path, pi_path, size) tuples."""
    c_files = sorted(glob.glob(os.path.join(directory, pattern)))
    triples = []
    for c_path in c_files:
        v_path = c_path.replace("-canonical-", "-v-")
        pi_path = c_path.replace("-canonical-", "-pi-")
        size = int(c_path.rsplit("-", 1)[-1].split(".")[0])
        triples.append((c_path, v_path, pi_path, size))
    return triples


def _load_triple(args):
    """Load a single file triple. Runs in thread pool (GIL released by I/O + zstd)."""
    c_path, v_path, pi_path = args
    return (_load_hist_tensor(c_path), _load_hist_tensor(v_path), _load_hist_tensor(pi_path))


def _load_triple_float(triple):
    """Load a file triple and convert to float32. Used for prefetching in symmetry expansion."""
    c_path, v_path, pi_path = triple[:3]
    return (load_compressed(c_path).float(), load_compressed(v_path).float(), load_compressed(pi_path).float())


def _parallel_load_triples(triples, num_workers, desc="Loading Data"):
    """Load file triples in parallel, preserving input order.

    Args:
        triples: list of (c_path, v_path, pi_path[, ...]) tuples
        num_workers: thread count (<=1 falls back to sequential)
        desc: tqdm progress bar label
    Returns:
        list of (c_tensor, v_tensor, pi_tensor) in input order
    """
    work = [(t[0], t[1], t[2]) for t in triples]
    if num_workers <= 1:
        return [_load_triple(w) for w in tqdm.tqdm(work, desc=desc, leave=False)]

    results = [None] * len(work)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(_load_triple, w): i for i, w in enumerate(work)}
        for future in tqdm.tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx), desc=desc, leave=False,
        ):
            results[future_to_idx[future]] = future.result()
    return results


def _load_and_select(args):
    """Load a file triple and select subset by indices. Keeps only selected rows in memory."""
    c_path, v_path, pi_path, indices = args
    c = _load_hist_tensor(c_path)
    v = _load_hist_tensor(v_path)
    pi = _load_hist_tensor(pi_path)
    idx = torch.tensor(indices, dtype=torch.long)
    result = (c[idx], v[idx], pi[idx])
    del c, v, pi
    return result


def _thin_triple(args):
    """Thin a single file triple: load, subsample, save, cleanup. Runs in thread pool."""
    c_path, v_path, pi_path, file_size, keep, zstd_level = args
    c = _load_hist_tensor(c_path)
    v = _load_hist_tensor(v_path)
    pi = _load_hist_tensor(pi_path)
    perm = torch.randperm(file_size)[:keep]
    c, v, pi = c[perm], v[perm], pi[perm]

    new_c = _replace_size_in_path(c_path, keep)
    new_v = _replace_size_in_path(v_path, keep)
    new_pi = _replace_size_in_path(pi_path, keep)
    # Migrate legacy .pt -> .ptz
    if not new_c.endswith(".ptz"):
        new_c = os.path.splitext(new_c)[0] + ".ptz"
        new_v = os.path.splitext(new_v)[0] + ".ptz"
        new_pi = os.path.splitext(new_pi)[0] + ".ptz"

    _atomic_save_compressed(c, new_c, zstd_level=zstd_level)
    _atomic_save_compressed(v, new_v, zstd_level=zstd_level)
    _atomic_save_compressed(pi, new_pi, zstd_level=zstd_level)

    for old, new in ((c_path, new_c), (v_path, new_v), (pi_path, new_pi)):
        if old != new and os.path.exists(old):
            os.remove(old)
    del c, v, pi


def _replace_size_in_path(path, new_size):
    """Replace the sample count in a history/reservoir filename."""
    base, ext = os.path.splitext(path)
    prefix = base.rsplit("-", 1)[0]
    return f"{prefix}-{new_size}{ext}"


GRArgs = namedtuple(
    "GRArgs",
    [
        "title",
        "game",
        "max_batch_size",
        "iteration",
        "data_save_size",
        "mcts_workers",
        "record_batch_metrics",
        "data_folder",
    ],
    defaults=(0, 30_000, os.cpu_count() - 1, False, None),
)


SelfPlayResult = namedtuple(
    "SelfPlayResult",
    [
        "win_rates",
        "hit_rate",
        "game_length",
        "resign_win_rates",
        "resign_rate",
        "avg_depth",
        "avg_entropy",
        "fast_avg_depth",
        "fast_avg_entropy",
        "avg_mpt",
        "avg_vm",
        "saturation",
        "churn_rate",
        "avg_batch_size",
        "median_batch_size",
        "min_batch_size",
        "max_batch_size",
        "avg_inference_ms",
        "median_inference_ms",
        "min_inference_ms",
        "max_inference_ms",
        "theoretical_hr",
    ],
)


class GameRunner:
    def __init__(self, players, pm, args):
        self.players = players
        self.pm = pm
        self.args = args
        self.device = get_device()
        self.num_players = self.args.game.NUM_PLAYERS()
        if len(self.players) != self.num_players:
            raise Exception("There must be a player for each player")

        self.num_model_groups = self.pm.num_model_groups()

        # Build model_group → model mapping for GPU inference
        model_groups = list(self.pm.params().model_groups)
        self._group_models = [None] * self.num_model_groups
        for i, p in enumerate(self.players):
            if not isinstance(p, (RandPlayer, PlayoutPlayer)):
                g = model_groups[i]
                if self._group_models[g] is None:
                    self._group_models[g] = p

        # NN model groups (groups that actually need GPU inference)
        self._nn_groups = [g for g in range(self.num_model_groups)
                           if self._group_models[g] is not None]

        # Pipeline queues
        self.gpu_ready_queue = queue.SimpleQueue()
        self.gpu_result_queue = queue.SimpleQueue()
        self.buffer_pool = queue.SimpleQueue()
        self.monitor_queue = queue.SimpleQueue()
        self.saved_samples = 0
        self._batch_lock = threading.Lock()
        self._batch_sizes = []
        self._inference_times = []

        # Buffer pool: 1 per batcher thread + 1 GPU + 1 result worker
        cs = self.args.game.CANONICAL_SHAPE()
        num_buffers = len(self._nn_groups) + 2
        for _ in range(num_buffers):
            buf = torch.zeros(self.args.max_batch_size, cs[0], cs[1], cs[2])
            if str(self.device) == "cuda":
                buf = buf.pin_memory()
            self.buffer_pool.put(buf)

        # History buffers
        self.hist_canonical = torch.zeros(self.args.data_save_size, cs[0], cs[1], cs[2])
        self.hist_v = torch.zeros(self.args.data_save_size, self.num_players + 1)
        self.hist_pi = torch.zeros(self.args.data_save_size, self.args.game.NUM_MOVES())

    @tracy_zone
    def run(self):
        interrupted = False
        prev_handler = signal.getsignal(signal.SIGINT)
        def _sigint_handler(sig, frame):
            nonlocal interrupted
            interrupted = True
            self.pm.stop()
        signal.signal(signal.SIGINT, _sigint_handler)
        try:
            self._run_inner()
        finally:
            signal.signal(signal.SIGINT, prev_handler)
            if interrupted:
                raise KeyboardInterrupt

    def _run_inner(self):
        self._batch_sizes.clear()
        self._inference_times.clear()
        # Check if any model group has an NN model
        has_nn = any(m is not None for m in self._group_models)

        if has_nn:
            # Pre-warm CUDA graphs before starting threads to avoid
            # capture race conditions (graph capture is device-wide).
            for g in self._nn_groups:
                model = self._group_models[g]
                if model is not None:
                    model.warmup_graphs(self.args.max_batch_size)

            batcher_threads = []
            for g in self._nn_groups:
                t = threading.Thread(target=self.batcher, args=(g,))
                t.start()
                batcher_threads.append(t)
            gpu_thread = threading.Thread(target=self.gpu_loop)
            gpu_thread.start()
            result_thread = threading.Thread(target=self.result_worker)
            result_thread.start()

        mcts_workers = []
        for i in range(self.args.mcts_workers):
            mcts_workers.append(threading.Thread(target=self.pm.play))
            mcts_workers[i].start()

        monitor = threading.Thread(target=self.monitor)
        monitor.start()
        if self.pm.params().history_enabled:
            hist_saver = threading.Thread(target=self.hist_saver)
            hist_saver.start()

        for mw in mcts_workers:
            mw.join()
        if has_nn:
            for bt in batcher_threads:
                bt.join()
            gpu_thread.join()
            result_thread.join()
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

        def _build_postfix():
            hr = 0
            hits = self.pm.cache_hits()
            misses = self.pm.cache_misses()
            total = hits + misses
            if total > 0:
                hr = hits / total
            max_cache = self.pm.cache_max_size()
            if max_cache > 0:
                cache_sz = self.pm.cache_size()
                sat = cache_sz / max_cache
                evictions = self.pm.cache_evictions()
                churn = evictions / hits if hits > 0 else 0
                reinserts = self.pm.cache_reinserts()
                thr = (hits + reinserts) / total if total > 0 else 0
                postfix = {"cache hr": f"{thr:.3f}"}
                postfix["sat"] = f"{sat:.2f}"
                postfix["churn"] = f"{churn:.2f}"
            else:
                postfix = {"cache hr": f"{hr:.3f}"}
            if self.args.record_batch_metrics:
                with self._batch_lock:
                    sizes = list(self._batch_sizes)
                    inf_times = list(self._inference_times)
                if sizes:
                    sizes.sort()
                    n_bs = len(sizes)
                    postfix["bs"] = f"{sizes[0]}/{sizes[n_bs//2]}/{sizes[-1]}"
                if inf_times:
                    resolved = []
                    for t in inf_times:
                        if isinstance(t, tuple):
                            t[1].synchronize()
                            resolved.append(t[0].elapsed_time(t[1]) / 1000.0)
                        else:
                            resolved.append(t)
                    inf_times = resolved
                    inf_times.sort()
                    n_it = len(inf_times)
                    postfix["nn_ms"] = (
                        f"{inf_times[0]*1000:.1f}/"
                        f"{inf_times[n_it//2]*1000:.1f}/"
                        f"{inf_times[-1]*1000:.1f}"
                    )
            completed = self.pm.games_completed()
            num_perms = self.pm.num_seat_perms()
            if completed > 0 and num_perms > 1:
                seat_perms = self.pm.params().seat_perms
                num_groups = self.pm.num_model_groups()
                group_wins = [0.0] * (num_groups + 1)  # last = draws
                total_games = 0
                for perm_idx in range(num_perms):
                    ps = self.pm.perm_scores(perm_idx)
                    pg = self.pm.perm_games_completed(perm_idx)
                    if pg == 0:
                        continue
                    total_games += pg
                    perm = seat_perms[perm_idx]
                    for seat in range(self.num_players):
                        group_wins[perm[seat]] += ps[seat]
                    group_wins[-1] += ps[self.num_players]
                if total_games > 0:
                    win_rates = [gw / total_games for gw in group_wins]
                else:
                    win_rates = [0] * (num_groups + 1)
            else:
                scores = self.pm.scores()
                win_rates = [0] * len(scores)
                if completed > 0:
                    for i in range(len(scores)):
                        win_rates[i] = scores[i] / completed
            postfix["wr"] = list(map(lambda x: f"{x:0.3f}", win_rates))
            return postfix, completed

        while self.pm.remaining_games() > 0:
            try:
                self.monitor_queue.get(timeout=1)
            except queue.Empty:
                pass
            if time.time() - last_update > 1:
                postfix, completed = _build_postfix()
                pbar.set_postfix(postfix)
                pbar.update(completed - last_completed)
                last_completed = completed
                last_update = time.time()

        postfix, completed = _build_postfix()
        pbar.set_postfix(postfix)
        pbar.update(n - last_completed)
        pbar.close()

    @tracy_zone
    def batcher(self, group):
        tracy_thread(f"batcher_{group}")
        while self.pm.remaining_games() > 0:
            try:
                batch_tensor = self.buffer_pool.get(timeout=1)
            except queue.Empty:
                continue

            game_indices = self.pm.build_batch(group, batch_tensor)
            if not game_indices:
                self.buffer_pool.put(batch_tensor)
                continue

            if self.args.record_batch_metrics:
                with self._batch_lock:
                    self._batch_sizes.append(len(game_indices))

            gpu_tensor = batch_tensor[:len(game_indices)].to(
                self.device, non_blocking=True
            )
            self.gpu_ready_queue.put((gpu_tensor, batch_tensor, game_indices, group))

    @tracy_zone
    def gpu_loop(self):
        tracy_thread("gpu_loop")
        while self.pm.remaining_games() > 0:
            # Try non-blocking first — if batch already queued, grab immediately
            try:
                gpu_tensor, batch_tensor, game_indices, group = (
                    self.gpu_ready_queue.get_nowait()
                )
            except queue.Empty:
                # No batch ready — signal batcher to hand off partial batch
                self.pm.set_eager(True)
                try:
                    gpu_tensor, batch_tensor, game_indices, group = (
                        self.gpu_ready_queue.get(timeout=1)
                    )
                except queue.Empty:
                    self.pm.set_eager(False)
                    continue
                self.pm.set_eager(False)

            model = self._group_models[group]
            if self.args.record_batch_metrics and self.device.type == 'cuda':
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
            elif self.args.record_batch_metrics:
                t0 = time.perf_counter()
            v, pi = model.process(gpu_tensor)
            if self.args.record_batch_metrics:
                if self.device.type == 'cuda':
                    end_ev.record()
                    with self._batch_lock:
                        self._inference_times.append((start_ev, end_ev))
                else:
                    with self._batch_lock:
                        self._inference_times.append(time.perf_counter() - t0)
            self.gpu_result_queue.put((batch_tensor, game_indices, v, pi, group))

    @tracy_zone
    def result_worker(self):
        tracy_thread("result_worker")
        while self.pm.remaining_games() > 0:
            try:
                batch_tensor, game_indices, v, pi, group = (
                    self.gpu_result_queue.get(timeout=1)
                )
            except queue.Empty:
                continue
            v_np = v.cpu().numpy()
            pi_np = pi.cpu().numpy()
            self.pm.update_inferences(group, game_indices, v_np, pi_np)
            self.buffer_pool.put(batch_tensor)
            self.monitor_queue.put(len(game_indices))

    @tracy_zone
    def hist_saver(self):
        tracy_thread("hist_saver")
        batch = 0
        data_folder = self.args.data_folder
        if data_folder is None:
            raise ValueError("GRArgs.data_folder must be set when history saving is needed")
        os.makedirs(data_folder, exist_ok=True)
        while self.pm.remaining_games() > 0 or self.pm.hist_count() > 0:
            size = self.pm.build_history_batch(
                self.hist_canonical, self.hist_v, self.hist_pi
            )
            if size == 0:
                continue
            prefix = os.path.join(data_folder, f"{self.args.iteration:04d}-{batch:04d}")
            save_compressed(self.hist_canonical[:size], f"{prefix}-canonical-{size}.ptz")
            save_compressed(self.hist_v[:size], f"{prefix}-v-{size}.ptz")
            save_compressed(self.hist_pi[:size], f"{prefix}-pi-{size}.ptz")
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


def set_model_groups(params, players):
    """Set model_groups on params based on model object identity.

    Players sharing the same model object get the same group.
    Non-NN groups get an unused inference queue (eval_type routes them
    inline in C++, so the queue is never consumed).
    """
    model_ids = {}
    groups = []
    for p in players:
        pid = id(p)
        if pid not in model_ids:
            model_ids[pid] = len(model_ids)
        groups.append(model_ids[pid])
    params.model_groups = groups


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
    mask = ~np.isnan(win_rates[new_agent])
    for _ in tqdm.trange(iters, leave=False):
        rates = np.clip(win_rates[new_agent, mask], 0.001, 0.999)
        x = _ELO_ALPHA * (past_elo[new_agent] - past_elo[mask])
        # Clamp to avoid overflow in exp (sigmoid saturates beyond ~700)
        x_safe = np.clip(x, -500, 500)
        probs = np.where(x_safe >= 0, 1.0 / (1.0 + np.exp(-x_safe)),
                         np.exp(x_safe) / (1.0 + np.exp(x_safe)))
        past_elo[new_agent] += np.sum(rates - probs) * 32
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
    half_storage=True,
):
    if size == config.hist_size or (force and size > 0):
        save_fn = (lambda t, path: save_compressed(t, path, half_storage, zstd_level=config.zstd_level)) if use_compression else torch.save
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
    # Find raw files (exclude any already-symmetrized files)
    all_triples = glob_file_triples(tmp_hist, f"{iteration:04d}-*-canonical-*.ptz")
    raw_triples = [t for t in all_triples if "-syms-" not in t[0]]
    if not raw_triples:
        return

    # Delete stale syms files from a previous interrupted run
    stale_syms = glob_file_triples(tmp_hist, f"{iteration:04d}-*-syms-canonical-*.ptz")
    for c_path, v_path, pi_path, _ in stale_syms:
        for fp in (c_path, v_path, pi_path):
            if os.path.exists(fp):
                os.remove(fp)

    i_out = 0
    batch_out = 0
    cs = Game.CANONICAL_SHAPE()
    c_out = torch.zeros(config.hist_size, cs[0], cs[1], cs[2])
    v_out = torch.zeros(config.hist_size, Game.NUM_PLAYERS() + 1)
    p_out = torch.zeros(config.hist_size, Game.NUM_MOVES())

    max_workers = min(os.cpu_count() or 4, 8)
    _thread_local = threading.local()

    def _compute_symmetries(args):
        """Compute symmetries for a single sample. Runs in thread pool (GIL released by C++)."""
        c, v, pi = args
        if not hasattr(_thread_local, 'game'):
            _thread_local.game = Game()
        ph = alphazero.PlayHistory(c, v, pi)
        syms = _thread_local.game.symmetries(ph)
        result = []
        for sym in syms:
            result.append((
                torch.from_numpy(np.array(sym.canonical())),
                torch.from_numpy(np.array(sym.v())),
                torch.from_numpy(np.array(sym.pi())),
            ))
        return result

    total_samples = sum(size for _, _, _, size in raw_triples)

    with ThreadPoolExecutor(max_workers=1) as io_pool, \
         ThreadPoolExecutor(max_workers=max_workers) as executor:
        pbar = tqdm.tqdm(total=total_samples, desc="Creating Symmetric Samples", leave=False)
        next_future = io_pool.submit(_load_triple_float, raw_triples[0])
        for idx, (c_path, v_path, pi_path, size) in enumerate(raw_triples):
            c_data, v_data, pi_data = next_future.result()
            if idx + 1 < len(raw_triples):
                next_future = io_pool.submit(_load_triple_float, raw_triples[idx + 1])

            def _samples():
                for j in range(size):
                    yield (c_data[j], v_data[j], pi_data[j])

            futures = executor.map(
                _compute_symmetries,
                _samples(),
                chunksize=max(1, size // (max_workers * 4)),
            )
            for sym_list in futures:
                for c_sym, v_sym, p_sym in sym_list:
                    c_out[i_out] = c_sym
                    v_out[i_out] = v_sym
                    p_out[i_out] = p_sym
                    i_out += 1
                    if i_out == config.hist_size:
                        if maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration,
                                      location=tmp_hist, name="-syms", use_compression=True,
                                      half_storage=config.half_storage):
                            i_out = 0
                            batch_out += 1
                pbar.update(1)

            del c_data, v_data, pi_data
        pbar.close()

    # Save remaining samples
    if i_out > 0:
        maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration,
                   location=tmp_hist, name="-syms", use_compression=True, force=True,
                   half_storage=config.half_storage)

    del c_out, v_out, p_out

    # Delete the raw files now that symmetrized versions exist
    for c_path, v_path, pi_path, _ in raw_triples:
        for fp in (c_path, v_path, pi_path):
            if os.path.exists(fp):
                os.remove(fp)


@tracy_zone
def resample_by_surprise(config, paths, experiment_name, iteration):
    import neural_net

    Game = config.Game
    hist_location = paths["history"]
    tmp_hist = paths["tmp_history"]

    # Find files in tmp_history
    file_triples = glob_file_triples(tmp_hist, f"{iteration:04d}-*-canonical-*.ptz")
    if not file_triples:
        return

    # Pass 1: compute per-sample loss by loading one file at a time
    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    loss_arrays = []
    with ThreadPoolExecutor(max_workers=1) as io_pool:
        next_future = io_pool.submit(_load_triple, (file_triples[0][0], file_triples[0][1], file_triples[0][2]))
        for idx, (c_path, v_path, pi_path, size) in enumerate(tqdm.tqdm(file_triples, desc="Computing Loss", leave=False)):
            c_t, v_t, pi_t = next_future.result()
            if idx + 1 < len(file_triples):
                nxt = file_triples[idx + 1]
                next_future = io_pool.submit(_load_triple, (nxt[0], nxt[1], nxt[2]))
            ds = TensorDataset(c_t, v_t, pi_t)
            dl = DataLoader(ds, batch_size=config.train_batch_size, shuffle=False,
                            pin_memory=torch.cuda.is_available())
            file_loss = nn.sample_loss(dl, size)
            loss_arrays.append(file_loss)
            del c_t, v_t, pi_t, ds, dl
    del nn

    loss = np.concatenate(loss_arrays)
    del loss_arrays
    sample_count = len(loss)
    total_loss = np.sum(loss)

    # Compute copy counts vectorized
    if total_loss <= 0:
        copies = np.ones(sample_count, dtype=np.int64)
    else:
        weights = 0.5 + (loss / total_loss) * 0.5 * sample_count
        int_w = np.floor(weights).astype(np.int64)
        frac_w = weights - int_w
        frac_mask = np.random.random(sample_count) < frac_w
        copies = int_w + frac_mask.astype(np.int64)
    del loss

    # Clear old history for iteration before saving new history
    os.makedirs(hist_location, exist_ok=True)
    for fn in glob.glob(os.path.join(hist_location, f"{iteration:04d}-*.pt*")):
        os.remove(fn)

    # Pass 2: resample and write, loading one file at a time
    i_out = 0
    batch_out = 0
    cs = Game.CANONICAL_SHAPE()
    c_out = torch.zeros(config.hist_size, cs[0], cs[1], cs[2])
    v_out = torch.zeros(config.hist_size, Game.NUM_PLAYERS() + 1)
    p_out = torch.zeros(config.hist_size, Game.NUM_MOVES())

    sample_idx = 0
    with ThreadPoolExecutor(max_workers=1) as io_pool:
        next_future = io_pool.submit(_load_triple, (file_triples[0][0], file_triples[0][1], file_triples[0][2]))
        for idx, (c_path, v_path, pi_path, size) in enumerate(tqdm.tqdm(file_triples, desc="Resampling Data", leave=False)):
            c_t, v_t, pi_t = next_future.result()
            if idx + 1 < len(file_triples):
                nxt = file_triples[idx + 1]
                next_future = io_pool.submit(_load_triple, (nxt[0], nxt[1], nxt[2]))

            ds_copies = copies[sample_idx:sample_idx + size]
            expanded_idx = np.repeat(np.arange(size), ds_copies)
            if len(expanded_idx) == 0:
                del c_t, v_t, pi_t
                sample_idx += size
                continue
            c_exp = c_t[expanded_idx]
            v_exp = v_t[expanded_idx]
            p_exp = pi_t[expanded_idx]
            del c_t, v_t, pi_t

            pos = 0
            while pos < len(expanded_idx):
                space = config.hist_size - i_out
                chunk = min(space, len(expanded_idx) - pos)
                c_out[i_out:i_out + chunk] = c_exp[pos:pos + chunk]
                v_out[i_out:i_out + chunk] = v_exp[pos:pos + chunk]
                p_out[i_out:i_out + chunk] = p_exp[pos:pos + chunk]
                i_out += chunk
                pos += chunk
                if maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration, location=hist_location, use_compression=True, half_storage=config.half_storage):
                    i_out = 0
                    batch_out += 1
            del c_exp, v_exp, p_exp, expanded_idx
            sample_idx += size

    maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration, location=hist_location, use_compression=True, force=True, half_storage=config.half_storage)

    del c_out, v_out, p_out

    # Clean up tmp_history files for this iteration
    for c_path, v_path, pi_path, _ in file_triples:
        for fp in (c_path, v_path, pi_path):
            if os.path.exists(fp):
                os.remove(fp)


@tracy_zone
def iteration_loss(config, paths, experiment_name, iteration):
    import neural_net

    Game = config.Game
    hist_location = paths["history"]
    c = _glob_hist_files(hist_location, f"{iteration:04d}-*-canonical-*")
    v = _glob_hist_files(hist_location, f"{iteration:04d}-*-v-*")
    p = _glob_hist_files(hist_location, f"{iteration:04d}-*-pi-*")
    loaded = _parallel_load_triples(
        list(zip(c, v, p)), config.resolved_loader_threads, desc="Loading Iteration Data"
    )
    datasets = [TensorDataset(ct, vt, pt) for ct, vt, pt in loaded]
    del loaded

    dataset = ConcatDataset(datasets)
    _dl_workers = config.resolved_loader_threads
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, pin_memory=torch.cuda.is_available(),
                            num_workers=_dl_workers)

    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    v_loss, pi_loss = nn.losses(dataloader)

    del datasets, dataset, dataloader, nn

    return v_loss, pi_loss


@tracy_zone
def train(config, paths, experiment_name, iteration, hist_size, run, total_train_steps, lr=None):
    """Index-then-extract training: pre-select samples, load one file at a time.

    Memory bounded at ~1.5 GB regardless of window size.
    """
    import neural_net

    Game = config.Game
    hist_location = paths["history"]
    bs = config.train_batch_size

    # Phase 1: Discover files and sizes from filenames (no loading)
    file_triples = []
    iter_range = range(max(0, iteration - hist_size), iteration + 1)
    for i in tqdm.tqdm(iter_range, desc="Discovering Training Files", leave=False):
        triples = glob_file_triples(hist_location, f"{i:04d}-*-canonical-*.ptz")
        if not triples:
            triples = glob_file_triples(hist_location, f"{i:04d}-*-canonical-*.pt")
        file_triples.extend(triples)

    sizes = [size for _, _, _, size in file_triples]
    total_size = sum(sizes)

    num_iters_in_window = min(hist_size, iteration + 1)
    average_generation = total_size / num_iters_in_window
    steps_to_train = int(math.ceil(average_generation / bs * config.train_sample_rate))
    samples_needed = steps_to_train * bs

    # Phase 2: Pre-select sample indices
    if samples_needed >= total_size:
        # Use all samples (early iterations)
        selected_per_file = [list(range(s)) for s in sizes]
    else:
        # Generate sorted random indices, map to (file_idx, local_offset)
        all_indices = sorted(random.sample(range(total_size), samples_needed))
        cum_sizes = []
        cum = 0
        for s in sizes:
            cum_sizes.append(cum)
            cum += s
        selected_per_file = [[] for _ in range(len(file_triples))]
        fi = 0
        for idx in all_indices:
            while fi < len(cum_sizes) - 1 and idx >= cum_sizes[fi + 1]:
                fi += 1
            selected_per_file[fi].append(idx - cum_sizes[fi])

    # Phase 3: Load files and extract selected indices (parallel)
    work_items = [
        (c_path, v_path, pi_path, selected_per_file[i])
        for i, (c_path, v_path, pi_path, _) in enumerate(file_triples)
        if selected_per_file[i]
    ]

    num_workers = config.resolved_loader_threads
    acc_c, acc_v, acc_pi = [], [], []
    if num_workers <= 1:
        for item in tqdm.tqdm(work_items, desc="Extracting Training Samples", leave=False):
            c, v, pi = _load_and_select(item)
            acc_c.append(c); acc_v.append(v); acc_pi.append(pi)
    else:
        results = [None] * len(work_items)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(_load_and_select, item): i for i, item in enumerate(work_items)}
            for future in tqdm.tqdm(
                as_completed(future_to_idx), total=len(future_to_idx),
                desc="Extracting Training Samples", leave=False,
            ):
                results[future_to_idx[future]] = future.result()
        for c, v, pi in results:
            acc_c.append(c); acc_v.append(v); acc_pi.append(pi)

    c_data = torch.cat(acc_c)
    v_data = torch.cat(acc_v)
    pi_data = torch.cat(acc_pi)
    del acc_c, acc_v, acc_pi

    # Phase 4: Train
    dataset = TensorDataset(c_data, v_data, pi_data)
    _dl_workers = config.resolved_loader_threads
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, pin_memory=torch.cuda.is_available(),
                            num_workers=_dl_workers)
    del c_data, v_data, pi_data

    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    if lr is not None:
        nn.set_lr(lr)
    v_loss, pi_loss = nn.train(
        dataloader, steps_to_train, run, iteration, total_train_steps,
        ema_averaging=config.ema_averaging,
    )
    total_train_steps += steps_to_train
    nn.save_checkpoint(paths["checkpoint"], f"{iteration + 1:04d}-{experiment_name}.pt",
                       zstd_level=config.zstd_level)
    del dataset, dataloader, nn
    return v_loss, pi_loss, total_train_steps


def update_reservoir(config, paths, iteration, hist_size):
    """Move evicted window data to reservoir and thin if over capacity.

    Uses per-iteration .ptz files (same naming as history). Eviction is O(1)
    via os.rename. Thinning uses age-weighted removal biased toward oldest data.
    """
    hist_location = paths["history"]
    reservoir_location = paths["reservoir"]

    oldest_in_window = max(0, iteration - hist_size)
    prev_oldest = max(0, (iteration - 1) - calc_hist_size(config, iteration - 1))
    evicted_iters = list(range(prev_oldest, oldest_in_window))
    if not evicted_iters:
        return

    # Move evicted files to reservoir (zero memory, O(1) per file)
    os.makedirs(reservoir_location, exist_ok=True)
    for it in tqdm.tqdm(evicted_iters, desc="Moving Evicted History", leave=False):
        for pattern in (f"{it:04d}-*.ptz", f"{it:04d}-*.pt"):
            for src in glob.glob(os.path.join(hist_location, pattern)):
                dst = os.path.join(reservoir_location, os.path.basename(src))
                os.rename(src, dst)

    # Only thin every N iterations to batch expensive I/O
    if iteration % config.reservoir_thin_interval != 0:
        return

    # Check if thinning is needed
    reservoir_triples = glob_file_triples(reservoir_location)
    # Also check for .pt files in reservoir (legacy)
    reservoir_triples += glob_file_triples(reservoir_location, "*-canonical-*.pt")
    reservoir_total = sum(s for _, _, _, s in reservoir_triples)

    capacity = 0
    for i in range(oldest_in_window, iteration + 1):
        for fn in glob.glob(os.path.join(hist_location, f"{i:04d}-*-canonical-*.pt*")):
            capacity += int(fn.rsplit("-", 1)[-1].split(".")[0])

    excess = reservoir_total - capacity
    if excess <= 0:
        return

    # Build per-iteration info from reservoir filenames
    iter_info = {}   # iter_num -> total_samples
    iter_files = {}  # iter_num -> [(c, v, pi, size), ...]
    for c_path, v_path, pi_path, size in reservoir_triples:
        iter_num = int(os.path.basename(c_path).split("-")[0])
        iter_info[iter_num] = iter_info.get(iter_num, 0) + size
        iter_files.setdefault(iter_num, []).append((c_path, v_path, pi_path, size))

    # Compute age-weighted removal targets (older → more removals per sample)
    total_weight = 0.0
    iter_weights = {}
    for iter_num, iter_samples in iter_info.items():
        age = iteration - iter_num
        weight = ((1.0 / config.reservoir_recency_decay) ** age) * iter_samples
        iter_weights[iter_num] = weight
        total_weight += weight

    removals = {}
    for iter_num, weight in iter_weights.items():
        removals[iter_num] = round(excess * weight / total_weight)

    # Phase A: pre-compute work items (sequential, fast integer math)
    delete_files = []  # file paths to remove outright
    thin_work = []     # args tuples for _thin_triple

    for iter_num in sorted(removals.keys()):
        budget = removals[iter_num]
        if budget <= 0:
            continue
        iter_samples = iter_info[iter_num]
        if budget >= iter_samples:
            for c_path, v_path, pi_path, _ in iter_files[iter_num]:
                delete_files.extend([c_path, v_path, pi_path])
            continue
        remaining_budget = budget
        for c_path, v_path, pi_path, file_size in iter_files[iter_num]:
            if remaining_budget <= 0:
                break
            sub_budget = min(
                round(budget * file_size / iter_samples),
                remaining_budget, file_size
            )
            if sub_budget <= 0:
                continue
            keep = file_size - sub_budget
            if keep <= 0:
                delete_files.extend([c_path, v_path, pi_path])
                remaining_budget -= file_size
            else:
                thin_work.append((c_path, v_path, pi_path, file_size, keep,
                                  config.zstd_level))
                remaining_budget -= sub_budget

    # Phase B: execute deletions then thin in parallel
    for fp in delete_files:
        if os.path.exists(fp):
            os.remove(fp)

    num_workers = config.resolved_loader_threads
    if thin_work:
        if num_workers <= 1:
            for item in tqdm.tqdm(thin_work, desc="Thinning Reservoir", leave=False):
                _thin_triple(item)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_thin_triple, item) for item in thin_work]
                for fut in tqdm.tqdm(as_completed(futures), total=len(futures),
                                     desc="Thinning Reservoir", leave=False):
                    fut.result()  # propagate exceptions

    gc.collect()


def load_reservoir(paths, num_workers=1):
    """Load reservoir as a TensorDataset, or None if no reservoir exists.

    Supports both per-iteration files (new format) and monolithic files (legacy).
    """
    # Try new per-iteration format first
    triples = glob_file_triples(paths["reservoir"])
    if triples:
        loaded = _parallel_load_triples(triples, num_workers, desc="Loading Reservoir")
        cs = [ct for ct, _, _ in loaded]
        vs = [vt for _, vt, _ in loaded]
        pis = [pt for _, _, pt in loaded]
        return TensorDataset(torch.cat(cs), torch.cat(vs), torch.cat(pis))

    # Fall back to old monolithic format
    res_c_path = os.path.join(paths["reservoir"], "canonical.ptz")
    if not os.path.exists(res_c_path):
        return None
    c = load_compressed(res_c_path)
    v = load_compressed(os.path.join(paths["reservoir"], "v.ptz"))
    pi = load_compressed(os.path.join(paths["reservoir"], "pi.ptz"))
    return TensorDataset(c, v, pi)


def load_available_window(paths, start, end, num_workers=1):
    """Load whatever window history files exist across an iteration range."""
    hist_location = paths["history"]
    all_triples = []
    for i in range(start, end):
        c = _glob_hist_files(hist_location, f"{i:04d}-*-canonical-*")
        v = _glob_hist_files(hist_location, f"{i:04d}-*-v-*")
        p = _glob_hist_files(hist_location, f"{i:04d}-*-pi-*")
        all_triples.extend(zip(c, v, p))
    loaded = _parallel_load_triples(all_triples, num_workers, desc="Loading Window")
    return [TensorDataset(ct, vt, pt) for ct, vt, pt in loaded]


class StreamingCompressedDataset(IterableDataset):
    """Memory-bounded streaming dataset that loads one file at a time.

    Yields pre-formed batches of (canonical, v, pi) tuples. Compatible with
    nn.train() which iterates `for batch in batches` and unpacks the tuple.
    Use with DataLoader(batch_size=None, num_workers=0).
    """

    def __init__(self, file_triples, batch_size, passes=1):
        self.file_triples = file_triples  # [(c_path, v_path, pi_path, size), ...]
        self.batch_size = batch_size
        self.passes = passes

    def __iter__(self):
        for _ in range(self.passes):
            file_order = list(range(len(self.file_triples)))
            random.shuffle(file_order)
            for fi in file_order:
                c_path, v_path, pi_path, size = self.file_triples[fi]
                c = _load_hist_tensor(c_path)
                v = _load_hist_tensor(v_path)
                pi = _load_hist_tensor(pi_path)
                perm = torch.randperm(size)
                c = c[perm]
                v = v[perm]
                pi = pi[perm]
                for start in range(0, size, self.batch_size):
                    end = min(start + self.batch_size, size)
                    yield c[start:end], v[start:end], pi[start:end]
                del c, v, pi


@tracy_zone
def self_play(config, paths, experiment_name, best, iteration, depth, fast_depth):
    import neural_net

    Game = config.Game
    bs = config.self_play_batch_size
    cb = Game.NUM_PLAYERS() * config.self_play_concurrent_batch_mult
    n = bs * cb * config.self_play_chunks
    params = base_params(config, config.self_play_temp, bs, cb)
    params.games_to_play = n
    params.mcts_visits = [depth] * Game.NUM_PLAYERS()
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
        prepare_inference_model(nn, config)

    players = [nn] * Game.NUM_PLAYERS()
    set_model_groups(params, players)
    set_eval_types(params, players)

    # Clean tmp_history before self-play
    shutil.rmtree(paths["tmp_history"], ignore_errors=True)

    pm = alphazero.PlayManager(Game(), params)
    grargs = GRArgs(
        title="Self Play",
        game=Game,
        iteration=iteration,
        max_batch_size=bs,
        record_batch_metrics=True,
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
    misses = pm.cache_misses()
    total = hits + misses
    hr = 0
    if total > 0:
        hr = hits / total
    reinserts = pm.cache_reinserts()
    theoretical_hr = (hits + reinserts) / total if total > 0 else 0
    agl = pm.avg_game_length()
    avg_depth = pm.avg_leaf_depth()
    avg_entropy = pm.avg_search_entropy()
    fast_avg_depth = pm.fast_avg_leaf_depth()
    fast_avg_entropy = pm.fast_avg_search_entropy()
    avg_mpt = pm.avg_moves_per_turn()
    avg_vm = pm.avg_valid_moves()
    # Cache pressure metrics
    max_cache = pm.cache_max_size()
    cache_sz = pm.cache_size()
    saturation = cache_sz / max_cache if max_cache > 0 else 0
    evictions = pm.cache_evictions()
    churn_rate = evictions / hits if hits > 0 else 0
    # Batch size metrics
    with gr._batch_lock:
        sizes = gr._batch_sizes
    if sizes:
        sizes_sorted = sorted(sizes)
        avg_batch_size = sum(sizes) / len(sizes)
        min_batch_size = sizes_sorted[0]
        max_batch_size = sizes_sorted[-1]
        n_bs = len(sizes_sorted)
        median_batch_size = (sizes_sorted[n_bs // 2] + sizes_sorted[(n_bs - 1) // 2]) / 2
    else:
        avg_batch_size = min_batch_size = max_batch_size = median_batch_size = 0
    # Inference time metrics
    with gr._batch_lock:
        inf_times = gr._inference_times
    if inf_times:
        resolved = []
        for t in inf_times:
            if isinstance(t, tuple):
                t[1].synchronize()
                resolved.append(t[0].elapsed_time(t[1]) / 1000.0)
            else:
                resolved.append(t)
        inf_sorted = sorted(resolved)
        avg_inference_ms = (sum(inf_sorted) / len(inf_sorted)) * 1000
        min_inference_ms = inf_sorted[0] * 1000
        max_inference_ms = inf_sorted[-1] * 1000
        n_it = len(inf_sorted)
        median_inference_ms = ((inf_sorted[n_it // 2] + inf_sorted[(n_it - 1) // 2]) / 2) * 1000
    else:
        avg_inference_ms = min_inference_ms = max_inference_ms = median_inference_ms = 0
    del gr, pm, nn
    gc.collect()
    return SelfPlayResult(
        win_rates=win_rates, hit_rate=hr, game_length=agl,
        resign_win_rates=resign_win_rates, resign_rate=resign_rate,
        avg_depth=avg_depth, avg_entropy=avg_entropy,
        fast_avg_depth=fast_avg_depth, fast_avg_entropy=fast_avg_entropy,
        avg_mpt=avg_mpt, avg_vm=avg_vm,
        saturation=saturation, churn_rate=churn_rate,
        avg_batch_size=avg_batch_size, median_batch_size=median_batch_size,
        min_batch_size=min_batch_size, max_batch_size=max_batch_size,
        avg_inference_ms=avg_inference_ms, median_inference_ms=median_inference_ms,
        min_inference_ms=min_inference_ms, max_inference_ms=max_inference_ms,
        theoretical_hr=theoretical_hr,
    )


@tracy_zone
def play_past(config, paths, experiment_name, depth, iteration, past_iter, batch_size=64):
    import neural_net

    Game = config.Game
    num_players = Game.NUM_PLAYERS()
    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    prepare_inference_model(nn, config)
    if past_iter == 0:
        nn_past = RandPlayer()
    else:
        nn_past = neural_net.NNWrapper.load_checkpoint(
            Game, paths["checkpoint"], f"{past_iter:04d}-{experiment_name}.pt"
        )
        prepare_inference_model(nn_past, config)

    bs = batch_size
    cb = num_players

    if num_players == 2:
        # 2 players, 2 unique models
        # players list must be length num_players
        players = [nn, nn_past]
        model_groups = [0, 1]
        seat_perms = [[0, 1], [1, 0]]
        n_perms = 2
    else:
        # >2 players: nn (group 0) and nn_past (group 1)
        # players/model_groups must be length num_players
        # Default mapping: seat 0 = nn, rest = nn_past
        players = [nn_past] * num_players
        players[0] = nn
        model_groups = [1] * num_players
        model_groups[0] = 0
        # Generate all seat permutations:
        # First N: nn in each seat, rest nn_past
        seat_perms = []
        for i in range(num_players):
            perm = [1] * num_players
            perm[i] = 0
            seat_perms.append(perm)
        # Next N: nn_past in each seat, rest nn
        for i in range(num_players):
            perm = [0] * num_players
            perm[i] = 1
            seat_perms.append(perm)
        n_perms = len(seat_perms)

    n = bs * cb * n_perms  # total games across all permutations
    params = base_params(config, config.eval_temp, bs, cb)
    params.games_to_play = n
    params.mcts_visits = [depth] * num_players
    params.model_groups = model_groups
    params.seat_perms = seat_perms
    set_eval_types(params, players)

    pm = alphazero.PlayManager(Game(), params)
    grargs = GRArgs(
        title=f"Bench {iteration} v {past_iter}",
        game=Game,
        iteration=iteration,
        max_batch_size=bs,
    )
    gr = GameRunner(players, pm, grargs)
    gr.run()

    # Extract per-perm results to compute nn win rate and draw rate
    nn_rate = 0
    draw_rate = 0
    hr = 0
    agl = pm.avg_game_length()
    avg_depth = pm.avg_leaf_depth()
    avg_entropy = pm.avg_search_entropy()
    avg_mpt = pm.avg_moves_per_turn()
    avg_vm = pm.avg_valid_moves()

    hits = pm.cache_hits()
    total = hits + pm.cache_misses()
    if total > 0:
        hr = hits / total

    for perm_idx in range(pm.num_seat_perms()):
        perm_scores = pm.perm_scores(perm_idx)
        perm_games = pm.perm_games_completed(perm_idx)
        if perm_games == 0:
            continue
        perm = seat_perms[perm_idx]
        # Find which seats have nn (group 0) and accumulate their wins
        for seat in range(num_players):
            if perm[seat] == 0:  # nn is in this seat
                nn_rate += perm_scores[seat] / perm_games
        draw_rate += perm_scores[num_players] / perm_games  # draw slot

    nn_rate /= n_perms
    draw_rate /= n_perms

    del gr, nn, nn_past, pm
    gc.collect()
    return nn_rate, draw_rate, hr, agl, avg_depth, avg_entropy, avg_mpt, avg_vm


def get_lr(config, iteration, lr_state):
    """Compute learning rate for the current iteration."""
    if config.lr_schedule == "constant":
        lr = config.lr
    elif config.lr_schedule == "step":
        lr = config.lr_steps[0][1] if config.lr_steps else config.lr
        for step_iter, step_lr in config.lr_steps:
            if iteration >= step_iter:
                lr = step_lr
    elif config.lr_schedule == "adaptive":
        lr = lr_state['current_lr']
        can_drop = (
            iteration >= config.lr_min_iter
            and iteration - lr_state['last_drop_iter'] >= config.lr_min_between_drops
            and (config.lr_max_drops == 0 or lr_state['num_drops'] < config.lr_max_drops)
            and iteration - lr_state['last_best_iter'] >= config.lr_patience
        )
        if can_drop:
            lr *= config.lr_drop_factor
            lr_state['num_drops'] += 1
            lr_state['last_drop_iter'] = iteration
            lr_state['current_lr'] = lr
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")

    # Apply window-fill warmup
    if config.lr_warmup_target > 0 and iteration < config.lr_warmup_target:
        warmup_factor = config.lr_warmup_floor + (1.0 - config.lr_warmup_floor) * (iteration + 1) / config.lr_warmup_target
        lr *= warmup_factor

    return lr


def _default_lr_state(config):
    return {
        'current_lr': config.lr,
        'num_drops': 0,
        'last_drop_iter': -config.lr_min_between_drops,
        'last_best_iter': 0,
    }


def main(config, experiment_dir, start=0, aim_repo=None, bootstrap_from=""):
    """Main training loop.

    Args:
        config: TrainConfig instance with all training parameters.
        experiment_dir: Path to experiment directory (e.g. data/connect4/densenet-4d-12c-5k-100sims/).
        start: Iteration to resume from (0 = fresh start).
        aim_repo: Path for aim logging directory. Default None uses project root.
        bootstrap_from: Path to source experiment dir to bootstrap from (empty = no bootstrap).
    """
    import neural_net

    Game = config.Game
    paths = config.resolve_paths(experiment_dir)
    experiment_name = os.path.basename(experiment_dir)

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
            kernel_size=config.kernel_size,
            dense_net=config.dense_net,
            lr=config.lr,
            cv=config.cv,
            star_gambit_spatial=config.star_gambit_spatial,
        )
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.save_checkpoint(paths["checkpoint"], f"0000-{experiment_name}.pt",
                           zstd_level=config.zstd_level)

    try:
        import aim
        aim_hash_path = os.path.join(experiment_dir, ".aim_run_hash")
        if start > 0 and os.path.exists(aim_hash_path):
            run = aim.Run(run_hash=open(aim_hash_path).read().strip(), repo=aim_repo)
        else:
            run = aim.Run(experiment=experiment_name, repo=aim_repo)
            run.name = config.game
            run["hparams"] = {
                "network": config.network_name,
                "panel_size": config.gating_panel_size,
                "panel_win_rate": config.gating_panel_win_rate,
                "best_win_rate": config.gating_best_win_rate,
                "cpuct": config.cpuct,
                "fpu_reduction": config.fpu_reduction,
                "self_play_temp": config.self_play_temp,
                "eval_temp": config.eval_temp,
                "final_temp": config.final_temp,
                "training_sample_rate": config.train_sample_rate,
                "depth": config.depth,
                "channels": config.channels,
                "kernel_size": config.kernel_size,
                "lr_schedule": config.lr_schedule,
                "full_mcts_visits": config.selfplay_mcts_visits,
                "fast_mcts_visits": config.fast_mcts_visits,
            }
        with open(aim_hash_path, "w") as f:
            f.write(run.hash)
    except ImportError:
        print("aim is used for nice web logging with graphs. I would advise `pip install aim`.")
        print("Using a dummy logger for now that does nothing.\n")

        class DummyRun:
            def track(*args, **kwargs):
                pass

        run = DummyRun()

    import atexit
    atexit.register(lambda: hasattr(run, 'close') and run.close())

    # LR state for adaptive schedule
    lr_state_path = os.path.join(experiment_dir, "lr_state.json")

    if start == 0:
        create_init_net()

        # Read bootstrap source size from checkpoints (ground truth)
        source_n = 0
        if bootstrap_from:
            source_checkpoint_dir = os.path.join(bootstrap_from, "checkpoint")
            source_n = find_latest_checkpoint(source_checkpoint_dir)
            if source_n == 0:
                raise RuntimeError(f"No checkpoints found in bootstrap source: {bootstrap_from}")

        total_agents = max(config.iterations, source_n) + 1
        wr = np.empty((total_agents, total_agents))
        wr[:] = np.nan
        elo = np.zeros(total_agents)
        current_best = 0
        total_train_steps = 0
        lr_state = _default_lr_state(config)

        # Handle bootstrap from existing experiment
        if bootstrap_from:
            source_dir = bootstrap_from
            source_config_path = os.path.join(source_dir, "config.yaml")

            source_paths = config.resolve_paths(source_dir)

            # Copy reservoir (per-iteration .ptz files)
            source_reservoir = source_paths["reservoir"]
            source_res_triples = glob_file_triples(source_reservoir)
            if source_res_triples:
                os.makedirs(paths["reservoir"], exist_ok=True)
                all_res_files = []
                for c, v, p, _ in source_res_triples:
                    all_res_files.extend([c, v, p])
                for src in tqdm.tqdm(all_res_files, desc="Copying Reservoir", leave=False):
                    shutil.copy2(src, os.path.join(paths["reservoir"], os.path.basename(src)))

            # Copy elo/wr from source (if available)
            source_elo_path = os.path.join(source_dir, "elo.csv")
            source_wr_path = os.path.join(source_dir, "win_rate.csv")
            if os.path.exists(source_elo_path) and os.path.exists(source_wr_path):
                source_elo = np.genfromtxt(source_elo_path, delimiter=",")
                source_wr = np.genfromtxt(source_wr_path, delimiter=",")
                copy_size = min(len(source_elo), total_agents)
                wr[:copy_size, :copy_size] = source_wr[:copy_size, :copy_size]
                elo[:copy_size] = source_elo[:copy_size]

            # Check if same architecture
            same_arch = True
            if os.path.exists(source_config_path):
                source_cfg = load_config(source_config_path, {}, warn=False)
                same_arch = (
                    source_cfg.depth == config.depth
                    and source_cfg.channels == config.channels
                    and source_cfg.kernel_size == config.kernel_size
                    and source_cfg.dense_net == config.dense_net
                    and source_cfg.star_gambit_spatial == config.star_gambit_spatial
                )

            if same_arch:
                # Copy best-gated checkpoint from source
                best_source = source_n  # default to latest
                if os.path.exists(source_elo_path):
                    best_source = int(np.argmax(source_elo[:source_n + 1]))
                cps = sorted(glob.glob(os.path.join(source_paths["checkpoint"], f"{best_source:04d}-*.pt")))
                if cps:
                    dest_name = f"{source_n:04d}-{experiment_name}.pt"
                    shutil.copy2(cps[-1], os.path.join(paths["checkpoint"], dest_name))
            else:
                # Retrain on source data using streaming (bounded memory)
                reservoir_files = glob_file_triples(source_paths["reservoir"])
                window_files = []
                for wi in range(0, source_n):
                    triples = glob_file_triples(source_paths["history"], f"{wi:04d}-*-canonical-*.ptz")
                    if not triples:
                        triples = glob_file_triples(source_paths["history"], f"{wi:04d}-*-canonical-*.pt")
                    window_files.extend(triples)
                all_files = reservoir_files + window_files

                nn = neural_net.NNWrapper.load_checkpoint(
                    Game, paths["checkpoint"], f"0000-{experiment_name}.pt"
                )

                if all_files:
                    total_samples = sum(s for _, _, _, s in all_files)
                    bbs = config.train_batch_size
                    streaming_ds = StreamingCompressedDataset(all_files, bbs, passes=config.bootstrap_full_passes)
                    dl = DataLoader(streaming_ds, batch_size=None, num_workers=0)
                    steps_p1 = int(math.ceil(total_samples / bbs)) * config.bootstrap_full_passes
                    nn.train(dl, steps_p1, run, source_n, total_train_steps)
                    total_train_steps += steps_p1

                    if window_files:
                        window_total = sum(s for _, _, _, s in window_files)
                        streaming_ds2 = StreamingCompressedDataset(window_files, bbs, passes=config.bootstrap_window_passes)
                        dl2 = DataLoader(streaming_ds2, batch_size=None, num_workers=0)
                        steps_p2 = int(math.ceil(window_total / bbs)) * config.bootstrap_window_passes
                        nn.train(dl2, steps_p2, run, source_n, total_train_steps)
                        total_train_steps += steps_p2

                nn.save_checkpoint(paths["checkpoint"], f"{source_n:04d}-{experiment_name}.pt",
                                   zstd_level=config.zstd_level)

            current_best = source_n
            start = source_n

            # Calibrate bootstrap network ELO against source networks
            compare_start = max(0, source_n - config.bootstrap_compare_past)
            for past_iter in tqdm.tqdm(range(compare_start, source_n), desc="Calibrating ELO", leave=False):
                dest_cp = os.path.join(paths["checkpoint"], f"{past_iter:04d}-{experiment_name}.pt")
                if not os.path.exists(dest_cp):
                    # Copy from source (try exact name first, then glob)
                    source_cp = os.path.join(source_paths["checkpoint"], f"{past_iter:04d}-{experiment_name}.pt")
                    if not os.path.exists(source_cp):
                        cps = glob.glob(os.path.join(source_paths["checkpoint"], f"{past_iter:04d}-*.pt"))
                        source_cp = cps[0] if cps else None
                    if source_cp:
                        shutil.copy2(source_cp, dest_cp)
                    else:
                        continue

                nn_rate, draw_rate, _, _, _, _, _, _ = play_past(
                    config, paths, experiment_name, config.compare_mcts_visits,
                    source_n, past_iter, config.past_compare_batch_size
                )
                wr[source_n, past_iter] = nn_rate + draw_rate / Game.NUM_PLAYERS()
                wr[past_iter, source_n] = 1 - (nn_rate + draw_rate / Game.NUM_PLAYERS())
                gc.collect()

            elo = get_elo(elo, wr, source_n)

            if os.path.exists(source_elo_path):
                best_source_iter = int(np.argmax(source_elo[:source_n + 1]))
                if elo[source_n] < elo[best_source_iter]:
                    print(f"Warning: bootstrapped network (ELO {elo[source_n]:.0f}) is weaker than "
                          f"source best at iter {best_source_iter} (ELO {elo[best_source_iter]:.0f}). "
                          f"Training will continue and should improve.")

        np.savetxt(os.path.join(experiment_dir, "elo.csv"), elo, delimiter=",")
        np.savetxt(os.path.join(experiment_dir, "win_rate.csv"), wr, delimiter=",")
        np.savetxt(
            os.path.join(experiment_dir, "total_train_steps.txt"),
            [total_train_steps],
            delimiter=",",
        )
    else:
        total_agents = config.iterations + 1
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
        if os.path.exists(lr_state_path):
            with open(lr_state_path) as f:
                lr_state = json.load(f)
        else:
            lr_state = _default_lr_state(config)

    postfix = {"best": current_best}
    panel = [current_best]

    with tqdm.tqdm(range(start, config.iterations), initial=start, total=config.iterations, desc="Build Amazing Network") as pbar:
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
                        config.compare_mcts_visits, i, past_iter, config.past_compare_batch_size
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
                sp = self_play(
                    config, paths, experiment_name,
                    current_best, i,
                    config.selfplay_mcts_visits,
                    config.fast_mcts_visits,
                )
                for j in range(len(sp.win_rates) - 1):
                    run.track(sp.win_rates[j], name="win_rate", epoch=i, step=total_train_steps, context={"vs": "self", "player": j + 1, "from": "all_games"})
                for j in range(len(sp.resign_win_rates) - 1):
                    run.track(sp.resign_win_rates[j], name="win_rate", epoch=i, step=total_train_steps, context={"vs": "self", "player": j + 1, "from": "resignation"})
                run.track(sp.resign_rate, name="resignation_rate", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(sp.win_rates[-1], name="draw_rate", epoch=i, step=total_train_steps, context={"vs": "self", "from": "all_games"})
                run.track(sp.resign_win_rates[-1], name="draw_rate", epoch=i, step=total_train_steps, context={"vs": "self", "from": "resignation"})
                run.track(float(sp.hit_rate), name="cache_rate", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "hit_rate"})
                run.track(float(sp.theoretical_hr), name="cache_rate", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "theoretical_hit_rate"})
                run.track(sp.game_length, name="average_game_length", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(sp.avg_depth, name="avg_leaf_depth", epoch=i, step=total_train_steps, context={"vs": "self", "search": "full"})
                run.track(sp.avg_entropy, name="search_entropy", epoch=i, step=total_train_steps, context={"vs": "self", "search": "full"})
                if sp.fast_avg_depth > 0:
                    run.track(sp.fast_avg_depth, name="avg_leaf_depth", epoch=i, step=total_train_steps, context={"vs": "self", "search": "fast"})
                if sp.fast_avg_entropy > 0:
                    run.track(sp.fast_avg_entropy, name="search_entropy", epoch=i, step=total_train_steps, context={"vs": "self", "search": "fast"})
                run.track(sp.avg_mpt, name="moves_per_turn", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(sp.avg_vm, name="avg_valid_moves", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(float(sp.saturation), name="cache_rate", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "saturation"})
                run.track(float(sp.churn_rate), name="cache_rate", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "churn"})
                run.track(float(sp.avg_batch_size), name="batch_size", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "avg"})
                run.track(float(sp.median_batch_size), name="batch_size", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "median"})
                run.track(float(sp.min_batch_size), name="batch_size", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "min"})
                run.track(float(sp.max_batch_size), name="batch_size", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "max"})
                run.track(float(sp.avg_inference_ms), name="inference_ms", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "avg"})
                run.track(float(sp.median_inference_ms), name="inference_ms", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "median"})
                run.track(float(sp.min_inference_ms), name="inference_ms", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "min"})
                run.track(float(sp.max_inference_ms), name="inference_ms", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "max"})
                postfix["win_rates"] = list(map(lambda x: f"{x:0.3f}", sp.win_rates))
                pbar.set_postfix(postfix)
            stage_times["selfplay"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_symmetries"):
                exploit_symmetries(config, paths, i)
            stage_times["symmetries"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_resampling"):
                resample_by_surprise(config, paths, experiment_name, i)
            stage_times["resampling"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_training"):
                hist_size = calc_hist_size(config, i)
                oldest_iteration = max(0, i - hist_size)
                run.track(hist_size, name="history", epoch=i, step=total_train_steps, context={"type": "window_size"})
                run.track(oldest_iteration, name="history", epoch=i, step=total_train_steps, context={"type": "oldest_iteration"})
                lr = get_lr(config, i, lr_state)
                run.track(lr, name="learning_rate", epoch=i, step=total_train_steps)
                v_loss, pi_loss, total_train_steps = train(
                    config, paths, experiment_name, i, hist_size, run, total_train_steps, lr=lr
                )
                np.savetxt(
                    os.path.join(experiment_dir, "total_train_steps.txt"),
                    [total_train_steps],
                    delimiter=",",
                )
                with open(lr_state_path, 'w') as f:
                    json.dump(lr_state, f)
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
                        config.compare_mcts_visits, next_net, gate_net, config.gate_compare_batch_size
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
                    lr_state['last_best_iter'] = next_net
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

    if hasattr(run, 'close'):
        run.close()
