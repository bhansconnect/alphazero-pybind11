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
import frozen_eval

def _fmt_pct(x):
    pct = x * 100
    return f"{int(pct)}%" if pct == int(pct) else f"{pct:.1f}%"


# ============================================================================
# Unified Star Gambit variant mixing helpers
# ============================================================================

UNIFIED_VARIANT_NAMES = ["skirmish", "showdown", "clash", "battle"]
_UNIFIED_GAME_NAMES = frozenset({
    "star_gambit_unified",
    "star_gambit_unified_skirmish",
    "star_gambit_unified_showdown",
    "star_gambit_unified_clash",
    "star_gambit_unified_battle",
})


def _is_unified_game(config):
    return config.game in _UNIFIED_GAME_NAMES


def _compute_unified_probs(config, prev_sample_counts=None):
    """Variant selection probabilities [skirmish, showdown, clash, battle].

    game_based: variant_fractions used directly as game probabilities.
    sample_based: adjust game probabilities each iteration so actual sample
                  fractions approach the target (uses previous iteration's counts).
    """
    if config.variant_fractions:
        target = [config.variant_fractions.get(v, 0.0) for v in UNIFIED_VARIANT_NAMES]
    else:
        target = [0.25] * 4
    total = sum(target) or 1.0
    target = [t / total for t in target]

    if config.variant_mixing_mode != "sample_based" or prev_sample_counts is None:
        probs = target
    else:
        total_samples = sum(prev_sample_counts)
        if total_samples == 0:
            probs = target
        else:
            actual = [c / total_samples for c in prev_sample_counts]
            # Scale each variant's game prob by target/actual so sample fractions converge.
            adjusted = [target[v] / actual[v] if actual[v] > 1e-6 else target[v] * 4.0
                        for v in range(4)]
            s = sum(adjusted) or 1.0
            probs = [a / s for a in adjusted]

    # Guarantee a minimum probability so no variant is starved of games.
    MIN_PROB = 0.02
    probs = [max(p, MIN_PROB) for p in probs]
    s = sum(probs)
    return [p / s for p in probs]


def _compute_gating_probs(config):
    """Variant selection probabilities to use during gating evaluation."""
    weights = config.gating_variant_weights or config.variant_fractions or {}
    if not weights:
        return [0.25] * 4
    probs = [weights.get(v, 0.0) for v in UNIFIED_VARIANT_NAMES]
    total = sum(probs) or 1.0
    return [p / total for p in probs]


def _make_game_instance(config, probs=None):
    """Create a game instance; for star_gambit_unified injects variant probs."""
    Game = config.Game
    if config.game == "star_gambit_unified" and probs is not None:
        return Game(probs=probs)
    return Game()


_VARIANT_METRIC_TRACK_NAMES = {
    "game_length": ("average_game_length", None),
    "avg_depth": ("avg_leaf_depth", {"search": "full"}),
    "avg_entropy": ("search_entropy", {"search": "full"}),
    "fast_avg_depth": ("avg_leaf_depth", {"search": "fast"}),
    "fast_avg_entropy": ("search_entropy", {"search": "fast"}),
    "avg_mpt": ("moves_per_turn", None),
    "avg_vm": ("avg_valid_moves", None),
}


def _track_variant_play_metrics(run, variant_metrics, *, epoch, step, vs_context):
    """Track per-variant play metrics on the aim Run with consistent metric names."""
    if not variant_metrics:
        return
    for vid, metrics in variant_metrics.items():
        vname = UNIFIED_VARIANT_NAMES[vid]
        for key, value in metrics.items():
            name_extra = _VARIANT_METRIC_TRACK_NAMES.get(key)
            if name_extra is None:
                continue
            metric_name, extra = name_extra
            if key.startswith("fast_") and not value:
                continue
            ctx = {"vs": vs_context, "variant": vname}
            if extra:
                ctx.update(extra)
            run.track(value, name=metric_name, epoch=epoch, step=step, context=ctx)


def _collect_variant_metrics(pm, include_fast=False):
    """Return {vid: {metric: value}} for variants with completed games.

    Metrics: game_length, avg_depth, avg_entropy, avg_mpt, avg_vm
    (plus fast_avg_depth/fast_avg_entropy when include_fast=True).
    Empty dict for non-unified games.
    """
    out = {}
    for vid in range(pm.num_tracked_variants()):
        if pm.variant_games_completed(vid) == 0:
            continue
        m = {
            "game_length": pm.variant_avg_game_length(vid),
            "avg_depth": pm.variant_avg_leaf_depth(vid),
            "avg_entropy": pm.variant_avg_search_entropy(vid),
            "avg_mpt": pm.variant_avg_moves_per_turn(vid),
            "avg_vm": pm.variant_avg_valid_moves(vid),
        }
        if include_fast:
            m["fast_avg_depth"] = pm.variant_fast_avg_leaf_depth(vid)
            m["fast_avg_entropy"] = pm.variant_fast_avg_search_entropy(vid)
        out[vid] = m
    return out


def _count_variant_samples(paths, iteration):
    """Count samples per variant from an iteration's history files.

    Returns [skirmish, showdown, clash, battle] counts or None if unavailable.
    Reads channel 32+v at grid position (6,6) — board center, always a valid hex.
    """
    c_files = _glob_hist_files(paths["history"], f"{iteration:04d}-*-canonical-*")
    if not c_files:
        return None
    counts = [0, 0, 0, 0]
    for c_path in c_files:
        try:
            c = load_compressed(c_path).float()
            if c.ndim != 4 or c.shape[1] < 36:
                continue
            for v in range(4):
                counts[v] += int((c[:, 32 + v, 6, 6] > 0.5).sum().item())
        except Exception:
            pass
    return counts if sum(counts) > 0 else None


def save_compressed(tensor, path, half_storage=True, zstd_level=1):
    """Save tensor with zstd compression, optionally as half-precision."""
    buffer = io.BytesIO()
    if half_storage:
        tensor = to_half_safe(tensor, get_storage_dtype())
    # Clone views/slices to avoid serializing the entire backing storage
    if tensor.storage_offset() or tensor.untyped_storage().size() != tensor.nelement() * tensor.element_size():
        tensor = tensor.clone()
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


def _sample_recent_history_batch(paths, batch_size):
    """Sample a batch of canonical positions from the most recent history file.

    Used by per-checkpoint diagnostics (effective rank). Returns a CPU float
    tensor of shape [<=batch_size, C, H, W], or None if no history exists.
    """
    hist_dir = paths["history"]
    files = sorted(glob.glob(os.path.join(hist_dir, "*-canonical-*.ptz")), reverse=True)
    if not files:
        files = sorted(glob.glob(os.path.join(hist_dir, "*-canonical-*.pt")), reverse=True)
    if not files:
        return None
    tensor = _load_hist_tensor(files[0]).float()
    n = tensor.size(0)
    if n == 0:
        return None
    take = min(batch_size, n)
    if take == n:
        return tensor
    idx = torch.randperm(n)[:take]
    return tensor[idx]


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


def _load_and_shuffle(c_path, v_path, pi_path, size):
    """Load a file triple and randomly permute samples. Used for streaming dataset prefetch."""
    c = _load_hist_tensor(c_path)
    v = _load_hist_tensor(v_path)
    pi = _load_hist_tensor(pi_path)
    perm = torch.randperm(size)
    return c[perm], v[perm], pi[perm]


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
        "variant_game_counts",   # dict {vid: count} or {} for non-unified games
        "variant_win_rates",     # dict {vid: [rate_per_player...]} or {} for non-unified
        "variant_metrics",       # dict {vid: {metric_name: value}} or {} for non-unified
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
        # Two-stage Ctrl+C:
        #   1st press: ack visibly + call pm.stop() so workers drain gracefully
        #              (preserves current iteration's data).
        #   2nd press: restore the default handler and re-raise immediately,
        #              so an unresponsive drain (long MCTS step, queued
        #              history saving, etc.) doesn't leave the user stuck
        #              mashing Ctrl+C with no feedback.
        interrupted = False
        prev_handler = signal.getsignal(signal.SIGINT)
        def _sigint_handler(sig, frame):
            nonlocal interrupted
            if interrupted:
                signal.signal(signal.SIGINT, prev_handler)
                tqdm.tqdm.write("\nSecond Ctrl+C — forcing exit.")
                raise KeyboardInterrupt
            interrupted = True
            tqdm.tqdm.write(
                "\nCtrl+C received — stopping selfplay; press again to force exit."
            )
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
                postfix = {"hr": _fmt_pct(hr), "thr": _fmt_pct(thr)}
                postfix["churn"] = f"{churn:.2f}"
            else:
                postfix = {"hr": "N/A"}
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
                player_wins = [0.0] * (self.num_players + 1)  # last = draws
                total_games = 0
                for perm_idx in range(num_perms):
                    ps = self.pm.perm_scores(perm_idx)
                    pg = self.pm.perm_games_completed(perm_idx)
                    if pg == 0:
                        continue
                    total_games += pg
                    for seat in range(self.num_players):
                        original_player = (seat + perm_idx) % self.num_players
                        player_wins[original_player] += ps[seat]
                    player_wins[-1] += ps[self.num_players]
                if total_games > 0:
                    win_rates = [pw / total_games for pw in player_wins]
                else:
                    win_rates = [0] * (self.num_players + 1)
            else:
                scores = self.pm.scores()
                win_rates = [0] * len(scores)
                if completed > 0:
                    for i in range(len(scores)):
                        win_rates[i] = scores[i] / completed
            postfix["wr"] = "/".join(_fmt_pct(x) for x in win_rates)
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
        return np.zeros(0)

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
    loss_for_logging = loss
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

    return loss_for_logging


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


def _load_reservoir_meta(reservoir_dir):
    """Load reservoir metadata, or None if not found."""
    meta_path = os.path.join(reservoir_dir, "reservoir_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def _save_reservoir_meta(reservoir_dir, meta):
    """Save reservoir metadata atomically."""
    meta_path = os.path.join(reservoir_dir, "reservoir_meta.json")
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_path, meta_path)


def _load_chunk(reservoir_dir, chunk_idx):
    """Load a reservoir chunk, returning (c, v, pi, iters) tensors."""
    prefix = os.path.join(reservoir_dir, f"chunk_{chunk_idx:02d}")
    c = load_compressed(prefix + "_canonical.ptz")
    v = load_compressed(prefix + "_v.ptz")
    pi = load_compressed(prefix + "_pi.ptz")
    iters = load_compressed(prefix + "_iters.ptz")
    return c, v, pi, iters


def _save_chunk(reservoir_dir, chunk_idx, c, v, pi, iters, zstd_level=1):
    """Save a reservoir chunk atomically."""
    prefix = os.path.join(reservoir_dir, f"chunk_{chunk_idx:02d}")
    _atomic_save_compressed(c, prefix + "_canonical.ptz", zstd_level=zstd_level)
    _atomic_save_compressed(v, prefix + "_v.ptz", zstd_level=zstd_level)
    _atomic_save_compressed(pi, prefix + "_pi.ptz", zstd_level=zstd_level)
    _atomic_save_compressed(iters, prefix + "_iters.ptz", half_storage=False, zstd_level=zstd_level)


def glob_reservoir_chunks(reservoir_dir):
    """Glob for reservoir chunk files, returning (c_path, v_path, pi_path, size) tuples.

    Uses metadata chunk_size when available, falls back to loading canonical tensor.
    """
    c_files = sorted(glob.glob(os.path.join(reservoir_dir, "chunk_*_canonical.ptz")))
    if not c_files:
        return []

    # Try to get per-chunk sizes from metadata (avoids loading tensors)
    meta = _load_reservoir_meta(reservoir_dir)
    chunk_sizes = meta.get("chunk_sizes") if meta else None

    triples = []
    for i, c_path in enumerate(c_files):
        v_path = c_path.replace("_canonical.ptz", "_v.ptz")
        pi_path = c_path.replace("_canonical.ptz", "_pi.ptz")
        if os.path.exists(v_path) and os.path.exists(pi_path):
            if chunk_sizes and i < len(chunk_sizes):
                size = chunk_sizes[i]
            else:
                # Backward compat: no chunk_sizes in metadata, load tensor
                c_tensor = load_compressed(c_path)
                size = c_tensor.shape[0]
                del c_tensor
            triples.append((c_path, v_path, pi_path, size))
    return triples


def _index_staging(staging_dir):
    """Lightweight listing of staging files.

    Returns a list of (c_path, v_path, pi_path, size, iter_num) — no data
    loaded. The streaming reservoir code uses this so it never has to hold
    all of staging in memory at once.
    """
    staging_triples = glob_file_triples(staging_dir)
    out = []
    for c_path, v_path, pi_path, size in staging_triples:
        iter_num = int(os.path.basename(c_path).split("-")[0])
        out.append((c_path, v_path, pi_path, size, iter_num))
    return out


def _sample_staging_rows(file_index, n_samples, offset=0):
    """Random-sample rows from staging[offset:] across the indexed files.

    Loads only the files that contain selected rows; uses ``torch.randperm``
    on the *remaining* row count so RNG consumption matches the legacy
    ``staging_x[perm]`` path exactly. Returns (c, v, pi, iters) tensors in
    perm-order (i.e. the same row order ``staging[perm]`` would produce),
    or ``None`` if there are no remaining rows.
    """
    total = sum(sz for _, _, _, sz, _ in file_index)
    remaining = total - offset
    if remaining <= 0:
        return None

    if n_samples >= remaining:
        # All-rows path: matches legacy "use staging slice directly" (no RNG).
        perm = torch.arange(remaining, dtype=torch.long)
    else:
        perm = torch.randperm(remaining)[:n_samples]

    abs_perm = perm + offset
    sorted_abs, sort_back = abs_perm.sort()
    n_sp = sorted_abs.shape[0]

    cs, vs, pis, iters_list = [], [], [], []
    cum = 0
    sp_idx = 0
    for c_path, v_path, pi_path, sz, iter_num in file_index:
        end = sp_idx
        while end < n_sp and int(sorted_abs[end]) < cum + sz:
            end += 1
        if end > sp_idx:
            local = (sorted_abs[sp_idx:end] - cum).long()
            c = load_compressed(c_path)
            v = load_compressed(v_path)
            pi = load_compressed(pi_path)
            cs.append(c[local].clone())
            vs.append(v[local].clone())
            pis.append(pi[local].clone())
            iters_list.append(torch.full((local.numel(),), iter_num, dtype=torch.int16))
            del c, v, pi
        sp_idx = end
        cum += sz

    c_sorted = torch.cat(cs)
    v_sorted = torch.cat(vs)
    pi_sorted = torch.cat(pis)
    iters_sorted = torch.cat(iters_list)

    inv = sort_back.argsort()
    return c_sorted[inv], v_sorted[inv], pi_sorted[inv], iters_sorted[inv]


def _migrate_legacy_reservoir(config, reservoir_dir):
    """One-time migration from per-iteration reservoir files to chunked format."""
    legacy_triples = glob_file_triples(reservoir_dir)
    legacy_triples += glob_file_triples(reservoir_dir, "*-canonical-*.pt")
    if not legacy_triples:
        return

    tqdm.tqdm.write(f"Migrating {len(legacy_triples)} legacy reservoir files to chunked format...")
    chunk_size = config.reservoir_chunk_size
    chunks_filled = 0
    chunk_sizes_list = []
    buf_c, buf_v, buf_pi, buf_iters = [], [], [], []
    buf_len = 0

    for c_path, v_path, pi_path, size in tqdm.tqdm(legacy_triples, desc="Migration", leave=False):
        c = _load_hist_tensor(c_path)
        v = _load_hist_tensor(v_path)
        pi = _load_hist_tensor(pi_path)
        iter_num = int(os.path.basename(c_path).split("-")[0])
        buf_c.append(c)
        buf_v.append(v)
        buf_pi.append(pi)
        buf_iters.append(torch.full((size,), iter_num, dtype=torch.int16))
        buf_len += size

        while buf_len >= chunk_size:
            all_c = torch.cat(buf_c)
            all_v = torch.cat(buf_v)
            all_pi = torch.cat(buf_pi)
            all_iters = torch.cat(buf_iters)
            _save_chunk(reservoir_dir, chunks_filled,
                        all_c[:chunk_size], all_v[:chunk_size],
                        all_pi[:chunk_size], all_iters[:chunk_size],
                        zstd_level=config.zstd_level)
            # Keep remainder
            remainder = buf_len - chunk_size
            if remainder > 0:
                buf_c = [all_c[chunk_size:]]
                buf_v = [all_v[chunk_size:]]
                buf_pi = [all_pi[chunk_size:]]
                buf_iters = [all_iters[chunk_size:]]
            else:
                buf_c, buf_v, buf_pi, buf_iters = [], [], [], []
            buf_len = remainder
            chunk_sizes_list.append(chunk_size)
            chunks_filled += 1
            del all_c, all_v, all_pi, all_iters

    # Save final partial chunk
    if buf_len > 0:
        all_c = torch.cat(buf_c)
        all_v = torch.cat(buf_v)
        all_pi = torch.cat(buf_pi)
        all_iters = torch.cat(buf_iters)
        _save_chunk(reservoir_dir, chunks_filled,
                    all_c, all_v, all_pi, all_iters,
                    zstd_level=config.zstd_level)
        chunk_sizes_list.append(buf_len)
        chunks_filled += 1

    # Determine last_updated from iteration numbers in filenames
    last_updated = []
    for i in range(chunks_filled):
        _, _, _, iters = _load_chunk(reservoir_dir, i)
        last_updated.append(int(iters.max().item()))

    meta = {
        "version": 2,
        "n_chunks": config.reservoir_n_chunks,
        "chunk_size": chunk_size,
        "chunk_sizes": chunk_sizes_list,
        "chunks_filled": chunks_filled,
        "last_updated": last_updated,
    }
    _save_reservoir_meta(reservoir_dir, meta)

    # Delete legacy files
    for c_path, v_path, pi_path, _ in legacy_triples:
        for p in (c_path, v_path, pi_path):
            if os.path.exists(p):
                os.remove(p)

    tqdm.tqdm.write(f"Migration complete: {chunks_filled} chunks created")


def update_reservoir(config, paths, iteration, hist_size):
    """Move evicted window data to reservoir staging, then periodically merge into chunks.

    Streams staging file-by-file so peak memory is bounded by one chunk's
    worth of samples regardless of how many iterations of staging
    accumulated. RNG consumption matches the previous concat-then-sample
    implementation; see test_reservoir_streaming.py for the equivalence
    proof.
    """
    hist_location = paths["history"]
    reservoir_location = paths["reservoir"]
    staging_dir = os.path.join(reservoir_location, "staging")

    oldest_in_window = max(0, iteration - hist_size)
    prev_oldest = max(0, (iteration - 1) - calc_hist_size(config, iteration - 1))
    evicted_iters = list(range(prev_oldest, oldest_in_window))

    if evicted_iters:
        os.makedirs(staging_dir, exist_ok=True)
        for it in tqdm.tqdm(evicted_iters, desc="Moving Evicted History", leave=False):
            for pattern in (f"{it:04d}-*.ptz", f"{it:04d}-*.pt"):
                for src in glob.glob(os.path.join(hist_location, pattern)):
                    dst = os.path.join(staging_dir, os.path.basename(src))
                    os.rename(src, dst)

    if iteration % config.reservoir_update_interval != 0:
        return

    os.makedirs(reservoir_location, exist_ok=True)

    meta = _load_reservoir_meta(reservoir_location)
    if meta is None:
        legacy_triples = glob_file_triples(reservoir_location)
        legacy_triples += glob_file_triples(reservoir_location, "*-canonical-*.pt")
        if legacy_triples:
            _migrate_legacy_reservoir(config, reservoir_location)
            meta = _load_reservoir_meta(reservoir_location)

    file_index = _index_staging(staging_dir) if os.path.isdir(staging_dir) else []
    total_staging = sum(sz for _, _, _, sz, _ in file_index)
    if total_staging == 0:
        return

    if meta is None:
        meta = {
            "version": 2,
            "n_chunks": config.reservoir_n_chunks,
            "chunk_size": config.reservoir_chunk_size,
            "chunk_sizes": [],
            "chunks_filled": 0,
            "last_updated": [],
        }

    n_chunks = config.reservoir_n_chunks
    chunk_size = config.reservoir_chunk_size
    chunks_filled = meta["chunks_filled"]

    if "chunk_sizes" not in meta:
        meta["chunk_sizes"] = [chunk_size] * chunks_filled

    # FILLING PHASE — stream files sequentially into chunk-sized batches.
    # Peak memory: one staging file + one chunk in flight.
    consumed = 0
    if chunks_filled < n_chunks:
        buf_c, buf_v, buf_pi, buf_iters = [], [], [], []
        buf_n = 0
        file_iter = iter(file_index)
        cur = None  # [c, v, pi, iters, pos, sz]

        while chunks_filled < n_chunks and consumed < total_staging:
            # Refill buf up to chunk_size or until staging is exhausted.
            while buf_n < chunk_size and consumed < total_staging:
                if cur is None:
                    try:
                        c_path, v_path, pi_path, sz, iter_num = next(file_iter)
                    except StopIteration:
                        break
                    c = load_compressed(c_path)
                    v = load_compressed(v_path)
                    pi = load_compressed(pi_path)
                    iters = torch.full((sz,), iter_num, dtype=torch.int16)
                    cur = [c, v, pi, iters, 0, sz]
                c, v, pi, iters, pos, sz = cur
                take = min(chunk_size - buf_n, sz - pos)
                buf_c.append(c[pos:pos + take])
                buf_v.append(v[pos:pos + take])
                buf_pi.append(pi[pos:pos + take])
                buf_iters.append(iters[pos:pos + take])
                buf_n += take
                consumed += take
                cur[4] = pos + take
                if cur[4] >= sz:
                    cur = None

            if buf_n == 0:
                break

            out_c = torch.cat(buf_c)
            out_v = torch.cat(buf_v)
            out_pi = torch.cat(buf_pi)
            out_iters = torch.cat(buf_iters)
            _save_chunk(reservoir_location, chunks_filled,
                        out_c, out_v, out_pi, out_iters,
                        zstd_level=config.zstd_level)
            meta["last_updated"].append(iteration)
            meta["chunk_sizes"].append(buf_n)
            chunks_filled += 1
            buf_c, buf_v, buf_pi, buf_iters = [], [], [], []
            buf_n = 0
            del out_c, out_v, out_pi, out_iters

        meta["chunks_filled"] = chunks_filled
        del buf_c, buf_v, buf_pi, buf_iters, cur

    # MERGE PHASE — sample new_per_chunk rows per chunk via the streaming
    # sampler, materializing only the rows needed for this chunk.
    if chunks_filled >= n_chunks and consumed < total_staging:
        remaining = total_staging - consumed
        K = min(config.reservoir_chunks_per_update, chunks_filled)
        last_updated = meta["last_updated"]
        target_indices = sorted(range(chunks_filled),
                                key=lambda i: last_updated[i])[:K]

        C = (n_chunks / K) * config.reservoir_update_interval
        decay = config.reservoir_recency_decay
        target_rate = 1 - decay ** C
        w_old_est = decay ** (C / 2)
        new_per_chunk = int(target_rate * chunk_size * w_old_est / (1 - target_rate))
        new_per_chunk = max(1, min(new_per_chunk, remaining))

        for chunk_idx in tqdm.tqdm(target_indices, desc="Merging Reservoir Chunks", leave=False):
            sampled = _sample_staging_rows(file_index, new_per_chunk, offset=consumed)
            if sampled is None:
                break
            new_c, new_v, new_pi, new_iters = sampled
            del sampled

            old_c, old_v, old_pi, old_iters = _load_chunk(reservoir_location, chunk_idx)
            pool_c = torch.cat([old_c, new_c])
            pool_v = torch.cat([old_v, new_v])
            pool_pi = torch.cat([old_pi, new_pi])
            pool_iters = torch.cat([old_iters, new_iters])
            del old_c, old_v, old_pi, old_iters, new_c, new_v, new_pi, new_iters

            ages = (iteration - pool_iters.float()).clamp(min=0)
            weights = decay ** ages
            select_size = min(chunk_size, pool_c.shape[0])
            if select_size < pool_c.shape[0]:
                selected = torch.multinomial(weights, select_size, replacement=False)
            else:
                selected = torch.arange(pool_c.shape[0])

            _save_chunk(reservoir_location, chunk_idx,
                        pool_c[selected], pool_v[selected],
                        pool_pi[selected], pool_iters[selected],
                        zstd_level=config.zstd_level)
            meta["chunk_sizes"][chunk_idx] = select_size
            del pool_c, pool_v, pool_pi, pool_iters, weights, selected, ages
            last_updated[chunk_idx] = iteration
            gc.collect()

        meta["last_updated"] = last_updated

    _save_reservoir_meta(reservoir_location, meta)

    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)

    gc.collect()


def load_reservoir(paths, num_workers=1):
    """Load reservoir as a TensorDataset, or None if no reservoir exists.

    Supports chunked format (primary), per-iteration files (legacy), and
    monolithic files (oldest legacy).
    """
    reservoir_dir = paths["reservoir"]

    # Try chunked format first
    meta = _load_reservoir_meta(reservoir_dir)
    if meta is not None and meta.get("chunks_filled", 0) > 0:
        chunk_triples = glob_reservoir_chunks(reservoir_dir)
        if chunk_triples:
            loaded = _parallel_load_triples(chunk_triples, num_workers, desc="Loading Reservoir")
            cs = [ct for ct, _, _ in loaded]
            vs = [vt for _, vt, _ in loaded]
            pis = [pt for _, _, pt in loaded]
            return TensorDataset(torch.cat(cs), torch.cat(vs), torch.cat(pis))

    # Try per-iteration format
    triples = glob_file_triples(reservoir_dir)
    if triples:
        loaded = _parallel_load_triples(triples, num_workers, desc="Loading Reservoir")
        cs = [ct for ct, _, _ in loaded]
        vs = [vt for _, vt, _ in loaded]
        pis = [pt for _, _, pt in loaded]
        return TensorDataset(torch.cat(cs), torch.cat(vs), torch.cat(pis))

    # Fall back to old monolithic format
    res_c_path = os.path.join(reservoir_dir, "canonical.ptz")
    if not os.path.exists(res_c_path):
        return None
    c = load_compressed(res_c_path)
    v = load_compressed(os.path.join(reservoir_dir, "v.ptz"))
    pi = load_compressed(os.path.join(reservoir_dir, "pi.ptz"))
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
    """Memory-bounded streaming dataset with cross-file batch mixing.

    Holds up to ``active_files`` loaded triples at once and builds each batch
    by round-robin slicing across them, so samples in a single batch come from
    multiple iters' history. When a slot drains, the prefetched next file
    takes its place. Yields (canonical, v, pi) batches; use with
    DataLoader(batch_size=None, num_workers=0).
    """

    def __init__(self, file_triples, batch_size, passes=1, active_files=8):
        self.file_triples = file_triples  # [(c_path, v_path, pi_path, size), ...]
        self.batch_size = batch_size
        self.passes = passes
        self.active_files = max(1, active_files)

    def __iter__(self):
        for _ in range(self.passes):
            yield from self._one_pass()

    def _one_pass(self):
        order = list(range(len(self.file_triples)))
        random.shuffle(order)
        if not order:
            return

        target = min(self.active_files, len(order))
        next_idx = 0

        def _load(i):
            return list(_load_and_shuffle(*self.file_triples[order[i]])) + [0]
            # slot = [c, v, pi, cursor]

        with ThreadPoolExecutor(max_workers=1) as executor:
            active = []
            while next_idx < len(order) and len(active) < target:
                active.append(_load(next_idx))
                next_idx += 1

            future = executor.submit(_load, next_idx) if next_idx < len(order) else None
            if future is not None:
                next_idx += 1

            while active:
                n = len(active)
                per = self.batch_size // n
                rem = self.batch_size - per * n

                cs, vs, pis = [], [], []
                for fi in range(n):
                    slot = active[fi]
                    want = per + (1 if fi < rem else 0)
                    avail = slot[0].size(0) - slot[3]
                    take = min(want, avail)
                    if take > 0:
                        end = slot[3] + take
                        cs.append(slot[0][slot[3]:end])
                        vs.append(slot[1][slot[3]:end])
                        pis.append(slot[2][slot[3]:end])
                        slot[3] = end

                if cs:
                    yield (
                        torch.cat(cs) if len(cs) > 1 else cs[0],
                        torch.cat(vs) if len(vs) > 1 else vs[0],
                        torch.cat(pis) if len(pis) > 1 else pis[0],
                    )

                # Evict drained slots, refill from prefetch
                drained = [i for i, s in enumerate(active) if s[3] >= s[0].size(0)]
                for i in reversed(drained):
                    active.pop(i)
                while len(active) < target and future is not None:
                    active.append(future.result())
                    future = executor.submit(_load, next_idx) if next_idx < len(order) else None
                    if future is not None:
                        next_idx += 1


@tracy_zone
def self_play(config, paths, experiment_name, best, iteration, depth, fast_depth,
              variant_probs=None, fallback_checkpoint_dir=None):
    import neural_net

    Game = config.Game
    bs = config.self_play_batch_size
    cb = Game.NUM_PLAYERS() * config.self_play_concurrent_batch_mult
    n = bs * cb * config.self_play_chunks
    params = base_params(config, config.self_play_temp, bs, cb)
    params.games_to_play = n
    params.mcts_visits = [depth] * Game.NUM_PLAYERS()
    params.history_enabled = True
    params.epsilon = 0.25
    params.playout_cap_randomization = True
    params.playout_cap_depth = fast_depth
    params.playout_cap_percent = 0.75
    params.resign_percent = config.resign_percent
    params.resign_playthrough_percent = config.resign_playthrough_percent
    params.mcts_root_temp = config.mcts_root_temp
    params.root_fpu_zero = config.root_fpu_zero
    params.shaped_dirichlet = config.shaped_dirichlet
    params.policy_target_pruning = config.policy_target_pruning

    use_rand = best == 0

    if use_rand:
        nn = RandPlayer()
        params.max_cache_size = 0
    else:
        nn = _resolve_checkpoint(Game, paths["checkpoint"], experiment_name,
                                 best, fallback_checkpoint_dir)
        prepare_inference_model(nn, config)

    players = [nn] * Game.NUM_PLAYERS()
    set_model_groups(params, players)
    set_eval_types(params, players)

    # Clean tmp_history before self-play
    shutil.rmtree(paths["tmp_history"], ignore_errors=True)

    pm = alphazero.PlayManager(_make_game_instance(config, variant_probs), params)
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
    variant_game_counts = {}
    variant_win_rates = {}
    for vid in range(pm.num_tracked_variants()):
        vgames = pm.variant_games_completed(vid)
        variant_game_counts[vid] = vgames
        if vgames > 0:
            vscores = pm.variant_scores(vid)
            vn = sum(vscores) or 1.0
            variant_win_rates[vid] = [float(vscores[j]) / vn for j in range(len(vscores))]
    variant_metrics = _collect_variant_metrics(pm, include_fast=True)
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
        variant_game_counts=variant_game_counts,
        variant_win_rates=variant_win_rates,
        variant_metrics=variant_metrics,
    )


def _resolve_checkpoint(Game, primary_dir, experiment_name, iteration, fallback_dir=None):
    """Load a checkpoint, falling back to a secondary directory if not found locally."""
    import neural_net
    primary_path = os.path.join(primary_dir, f"{iteration:04d}-{experiment_name}.pt")
    if os.path.exists(primary_path):
        return neural_net.NNWrapper.load_checkpoint(Game, primary_dir, f"{iteration:04d}-{experiment_name}.pt")
    if fallback_dir:
        fallback_path = os.path.join(fallback_dir, f"{iteration:04d}-{experiment_name}.pt")
        if os.path.exists(fallback_path):
            return neural_net.NNWrapper.load_checkpoint(Game, fallback_dir, f"{iteration:04d}-{experiment_name}.pt")
        cps = glob.glob(os.path.join(fallback_dir, f"{iteration:04d}-*.pt"))
        if cps:
            return neural_net.NNWrapper.load_checkpoint(Game, "", cps[0])
    raise FileNotFoundError(f"Checkpoint {iteration:04d} not found in {primary_dir}" +
                           (f" or {fallback_dir}" if fallback_dir else ""))


@tracy_zone
def play_past(config, paths, experiment_name, depth, iteration, past_iter, batch_size=64,
              fallback_checkpoint_dir=None, variant_probs=None):
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
        nn_past = _resolve_checkpoint(Game, paths["checkpoint"], experiment_name,
                                      past_iter, fallback_checkpoint_dir)
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

    pm = alphazero.PlayManager(_make_game_instance(config, variant_probs), params)
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

    # Per-variant win/draw rates (only populated for unified games with num_variants > 0).
    variant_nn_rates = {}
    variant_draw_rates = {}
    for vid in range(pm.num_tracked_variants()):
        vgames = pm.variant_games_completed(vid)
        if vgames == 0:
            continue
        vnn = 0.0
        # Mirror the perm-aggregation logic used for nn_rate: use per-perm variant
        # scores so nn-seat wins and nn_past-seat wins don't cancel each other out.
        for perm_idx in range(pm.num_seat_perms()):
            vperm_scores = pm.variant_perm_scores(vid, perm_idx)
            vperm_games = pm.variant_perm_games_completed(vid, perm_idx)
            if vperm_games == 0:
                continue
            perm = seat_perms[perm_idx]
            for seat in range(num_players):
                if perm[seat] == 0:  # nn is in this seat
                    vnn += vperm_scores[seat] / vperm_games
        variant_nn_rates[vid] = vnn / n_perms
        # Draw rate doesn't depend on seat assignment, use global aggregate.
        vscores = pm.variant_scores(vid)
        variant_draw_rates[vid] = vscores[num_players] / vgames

    variant_metrics = _collect_variant_metrics(pm)
    del gr, nn, nn_past, pm
    gc.collect()
    return nn_rate, draw_rate, hr, agl, avg_depth, avg_entropy, avg_mpt, avg_vm, variant_nn_rates, variant_draw_rates, variant_metrics


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


def _bootstrap_train_phase(nn, files, config, run, source_n, total_train_steps, phase_name):
    """Run one epoch of bootstrap training with PyTorch-style ReduceLROnPlateau.

    Returns (steps_trained, early_stopped).
    """
    from lr_scheduler import PlateauLRScheduler, ema_update

    bbs = config.train_batch_size
    epochs = max(1, config.bootstrap_epochs)
    total_steps = sum(int(math.ceil(s / bbs)) for _, _, _, s in files) * epochs
    eval_interval = min(config.bootstrap_eval_interval, max(total_steps, 1))

    scheduler = PlateauLRScheduler(
        set_lr=nn.set_lr,
        initial_lr=config.bootstrap_lr,
        drop_factor=config.bootstrap_lr_drop_factor,
        max_drops=config.bootstrap_lr_max_drops,
        patience=eval_interval * config.bootstrap_lr_patience,
        conv_patience=eval_interval * config.bootstrap_convergence_patience,
        cooldown=eval_interval * config.bootstrap_lr_cooldown,
        threshold=config.bootstrap_convergence_threshold,
    )
    ema_loss = None

    streaming_ds = StreamingCompressedDataset(
        files, bbs, passes=epochs, active_files=config.streaming_active_files,
    )
    dl = DataLoader(streaming_ds, batch_size=None, num_workers=0)

    current_step = 0
    early_stopped = False

    nn.nnet.train()
    pbar = tqdm.tqdm(total=total_steps, desc=f"  {phase_name}", leave=False, unit="step")

    for batch in dl:
        canonical, target_vs, target_pis = batch
        canonical = canonical.float().contiguous().to(nn.device, non_blocking=nn._non_blocking)
        target_vs = target_vs.float().contiguous().to(nn.device, non_blocking=nn._non_blocking)
        target_pis = target_pis.float().contiguous().to(nn.device, non_blocking=nn._non_blocking)

        nn.optimizer.zero_grad()
        out_v, out_pi = nn.nnet(canonical)
        l_v = nn.loss_v(target_vs, out_v)
        l_pi = nn.loss_pi(target_pis, out_pi)
        total_loss = l_v + l_pi
        total_loss.backward()
        nn.optimizer.step()

        v_val = l_v.item()
        pi_val = l_pi.item()
        current_step += 1
        step_loss = v_val + pi_val
        ema_loss = ema_update(ema_loss, step_loss)

        run.track(v_val, name="loss", epoch=source_n,
                  step=total_train_steps + current_step, context={"type": "value"})
        run.track(pi_val, name="loss", epoch=source_n,
                  step=total_train_steps + current_step, context={"type": "policy"})
        run.track(step_loss, name="loss", epoch=source_n,
                  step=total_train_steps + current_step, context={"type": "total"})

        event, lr = scheduler.step(ema_loss)

        pbar.update()
        pbar.set_postfix(
            ema=f"{ema_loss:.4f}",
            best=f"{scheduler.best:.4f}",
            bad=f"{scheduler.num_bad}",
            lr=f"{lr:.6f}",
        )

        if event == "drop":
            pbar.write(f"  LR drop #{scheduler.drops} at step {current_step}: "
                       f"lr={lr:.6f} (best={scheduler.best:.4f}, cur={ema_loss:.4f})")
        elif event == "stop":
            pbar.write(f"  {phase_name} converged at step {current_step}/{total_steps} "
                       f"(ema={ema_loss:.4f}, best={scheduler.best:.4f}, lr={lr:.6f})")
            early_stopped = True
            break

    pbar.close()
    if not early_stopped:
        tqdm.tqdm.write(f"  {phase_name} completed {current_step} steps")

    return current_step, early_stopped


def _bootstrap_retrain(nn, reservoir_files, window_files, config, run, source_n, total_train_steps):
    """Bootstrap retrain: 1 epoch on reservoir, then 1 epoch on window.

    Each phase uses step-level LR patience with early stopping.
    LR resets to bootstrap_lr at the start of each phase.

    Returns updated total_train_steps.
    """
    phases = []
    if reservoir_files:
        phases.append(("Retrain: Reservoir", reservoir_files))
    if window_files:
        phases.append(("Retrain: Window", window_files))

    for phase_name, files in phases:
        steps, _ = _bootstrap_train_phase(
            nn, files, config, run, source_n, total_train_steps, phase_name,
        )
        total_train_steps += steps

    return total_train_steps


def _analyze_iteration_variants(config, paths, experiment_name, iteration):
    """Compute per-variant pi-loss, v-loss, and target entropy after training.

    Loads up to 4096 samples from the current iteration's history files, runs
    inference with the newly-trained checkpoint, and returns per-variant mean losses.

    Returns dict: {variant_name: {"pi_loss": float, "v_loss": float, "entropy": float}}
    Returns {} for non-unified games or if history files are missing.
    """
    import neural_net

    Game = config.Game
    hist_location = paths["history"]

    file_triples = glob_file_triples(hist_location, f"{iteration:04d}-*-canonical-*.ptz")
    if not file_triples:
        return {}

    MAX_SAMPLES = 32_000
    all_c, all_v, all_pi = [], [], []
    total = 0
    per_file = max(1, MAX_SAMPLES // len(file_triples))
    for c_path, v_path, pi_path, size in file_triples:
        take = min(per_file, size, MAX_SAMPLES - total)
        if take <= 0:
            continue
        c = load_compressed(c_path).float()
        v = load_compressed(v_path).float()
        pi = load_compressed(pi_path).float()
        if take < size:
            idx = torch.randperm(size)[:take]
            all_c.append(c[idx]); all_v.append(v[idx]); all_pi.append(pi[idx])
        else:
            all_c.append(c); all_v.append(v); all_pi.append(pi)
        total += take

    if not all_c:
        return {}

    c_data = torch.cat(all_c)
    v_data = torch.cat(all_v)
    pi_data = torch.cat(all_pi)
    del all_c, all_v, all_pi

    # Variant membership: channels 32-35 one-hot, sampled at center of 13×13 grid
    variant_ids = c_data[:, 32:36, 6, 6].argmax(dim=1).numpy()

    try:
        nn = neural_net.NNWrapper.load_checkpoint(
            Game, paths["checkpoint"], f"{iteration + 1:04d}-{experiment_name}.pt"
        )
    except Exception:
        del c_data, v_data, pi_data
        return {}

    device = nn.device
    nn.nnet.eval()

    all_pi_loss, all_v_loss, all_entropy, all_top1, all_v_pred, all_v_actual = [], [], [], [], [], []
    bs = config.train_batch_size
    n_samples = len(c_data)
    with torch.no_grad():
        for start in tqdm.tqdm(range(0, n_samples, bs), desc="Variant analysis", leave=False):
            end = min(start + bs, n_samples)
            cb = c_data[start:end].to(device, non_blocking=True)
            vb = v_data[start:end].to(device, non_blocking=True)
            pib = pi_data[start:end].to(device, non_blocking=True)
            out_v, out_pi = nn.nnet(cb)
            all_pi_loss.append(nn.sample_loss_pi(pib, out_pi).cpu().numpy())
            all_v_loss.append(nn.sample_loss_v(vb, out_v).cpu().numpy())
            all_entropy.append((-(pib * (pib + 1e-9).log()).sum(dim=1)).cpu().numpy())
            all_top1.append(pib.max(dim=1).values.cpu().numpy())
            # Value calibration: predicted vs actual current-player win prob (index 0)
            all_v_pred.append(torch.exp(out_v)[:, 0].cpu().numpy())
            all_v_actual.append(vb[:, 0].cpu().numpy())

    pi_losses  = np.concatenate(all_pi_loss)
    v_losses   = np.concatenate(all_v_loss)
    entropies  = np.concatenate(all_entropy)
    top1       = np.concatenate(all_top1)
    v_pred     = np.concatenate(all_v_pred)
    v_actual   = np.concatenate(all_v_actual)
    del nn, c_data, v_data, pi_data
    gc.collect()

    result = {}
    for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
        mask = variant_ids == vid
        if mask.sum() == 0:
            continue
        result[vname] = {
            "pi_loss":  pi_losses[mask],
            "v_loss":   v_losses[mask],
            "entropy":  entropies[mask],
            "top1":     top1[mask],
            "v_pred":   v_pred[mask],
            "v_actual": v_actual[mask],
        }
    return result


def _log_win_rate_matrix(paths, iteration, run, total_train_steps):
    """Log win rate matrix heatmap to aim. Works for any game."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import aim as aim_lib
    except ImportError:
        return
    try:
        wr_path = os.path.join(paths["experiment"], "win_rate.csv")
        if not os.path.exists(wr_path):
            return
        wr_mat = np.genfromtxt(wr_path, delimiter=",")
        n = min(iteration + 2, wr_mat.shape[0])
        # Mask unplayed matchups so they render distinctly from genuine 0.5
        # draws (previously they were painted yellow-green like a fair draw).
        wr_display = np.ma.masked_invalid(wr_mat[:n, :n])
        sz = min(max(4, n // 3), 12)
        fig, ax = plt.subplots(figsize=(sz, sz))
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#d0d0d0")
        # origin="lower" flips the y axis so the latest agent (highest index)
        # sits at the top — biggest-vs-biggest lands in the upper-right corner.
        im = ax.imshow(wr_display, cmap=cmap, vmin=0, vmax=1,
                       interpolation="nearest", origin="lower")
        ax.set_title(f"Win Rate Matrix (iteration {iteration})", fontsize=11)
        ax.set_xlabel("Opponent iteration"); ax.set_ylabel("Agent iteration")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Overlay numeric labels for small matrices.
        if n <= 12:
            for i in range(n):
                for j in range(n):
                    v = wr_mat[i, j]
                    if np.isnan(v):
                        continue
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="black")
        plt.tight_layout()
        run.track(aim_lib.Image(fig), name="win_rate_matrix",
                  epoch=iteration, step=total_train_steps)
    except Exception:
        pass
    finally:
        plt.close("all")


def _generate_visualizations(config, paths, iteration, run, total_train_steps, variant_stats=None):
    """Generate and log phase-aware action heatmap and win rate matrix as aim images.

    Only called for unified games. All errors are caught silently so a missing
    dependency or bad data never interrupts training.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import aim as aim_lib
    except ImportError:
        return

    try:
        hist_location = paths["history"]
        file_triples = glob_file_triples(hist_location, f"{iteration:04d}-*-canonical-*.ptz")
        if not file_triples:
            return

        MAX_SAMPLES = 32_000
        all_c, all_pi = [], []
        total = 0
        per_file = max(1, MAX_SAMPLES // len(file_triples))
        for c_path, v_path, pi_path, size in file_triples:
            take = min(per_file, size, MAX_SAMPLES - total)
            if take <= 0:
                continue
            c = load_compressed(c_path).float()
            pi = load_compressed(pi_path).float()
            if take < size:
                idx = torch.randperm(size)[:take]
                all_c.append(c[idx]); all_pi.append(pi[idx])
            else:
                all_c.append(c); all_pi.append(pi)
            total += take

        if not all_c:
            return

        c_data = torch.cat(all_c)
        pi_np = torch.cat(all_pi).numpy()
        del all_c, all_pi

        # Variant id per sample: channels 32-35 one-hot at grid center (6,6)
        variant_ids = c_data[:, 32:36, 6, 6].numpy().argmax(axis=1)

        # Unit presence (channels 1-8 summed over board)
        unit_presence_sum = c_data[:, 1:9, :, :].sum(dim=(1, 2, 3)).numpy()

        # Four monotonic phases using two independently-decreasing signals:
        #   reserves (ch 24-29): only decrease as units deploy
        #   portal HP (ch 30-31): only decrease as damage lands
        #   unit count: only decreases after deploy is done (no new units enter)
        #
        # Phase 1 — Deploy:      reserves above per-variant median
        # Phase 2 — Standstill:  fleet committed, portals intact, unit count high
        # Phase 3 — Attrition:   unit count declining, portals still intact
        #                        (game often decided here before portal is touched)
        # Phase 4 — Portal:      any portal damage (prioritized over phases 2-3)
        #
        # Portal damage is checked first so it always "wins" regardless of unit count.
        # Within portal-intact states, per-variant median unit count splits standstill/attrition.
        reserves = c_data[:, 24:30, 6, 6].sum(dim=1).numpy()
        portal_total = c_data[:, 30:32, 6, 6].sum(dim=1).numpy()

        deploy_mask  = np.zeros(len(variant_ids), dtype=bool)
        for _vid in range(4):
            _vm = variant_ids == _vid
            if _vm.sum() > 0:
                deploy_mask[_vm] = reserves[_vm] > float(np.median(reserves[_vm]))

        post_deploy    = ~deploy_mask
        portal_mask    = post_deploy & (portal_total <= 1.99)
        intact_mask    = post_deploy & (portal_total >  1.99)
        standstill_mask = np.zeros(len(variant_ids), dtype=bool)
        attrition_mask  = np.zeros(len(variant_ids), dtype=bool)
        for _vid in range(4):
            _vm = intact_mask & (variant_ids == _vid)
            if _vm.sum() > 1:
                _med = float(np.median(unit_presence_sum[_vm]))
                standstill_mask |= _vm & (unit_presence_sum >  _med)
                attrition_mask  |= _vm & (unit_presence_sum <= _med)
            elif _vm.sum() == 1:
                standstill_mask |= _vm

        # Per-unit-type presence at each hex: channels 1=Fighter, 2=Cruiser, 3=Dreadnought
        unit_presence = [c_data[:, ch, :, :].numpy() > 0.5 for ch in [1, 2, 3]]

        # Valid spatial action slots per unit type (10 slots, 0..9).
        # Fighter: move f/fl/fr (0,1,2), fire forward (5)
        # Cruiser: move f/fl/fr/l/r (0..4), fire f/fl/fr (5,6,7)
        # Dreadnought: move fl/fr/l/r (1..4), fire fl/fr/rl/rr (6..9)
        VALID_SLOTS_BY_UNIT = np.zeros((3, 10), dtype=bool)
        VALID_SLOTS_BY_UNIT[0, [0, 1, 2, 5]]             = True
        VALID_SLOTS_BY_UNIT[1, [0, 1, 2, 3, 4, 5, 6, 7]] = True
        VALID_SLOTS_BY_UNIT[2, [1, 2, 3, 4, 6, 7, 8, 9]] = True

        # Valid deploy facings in canonical (current-player == P0) perspective.
        VALID_DEPLOY_FACINGS = np.zeros((3, 6), dtype=bool)
        VALID_DEPLOY_FACINGS[0, [1, 2, 3]]    = True   # Fighter:     NE, NW, W
        VALID_DEPLOY_FACINGS[1, [1, 2, 3]]    = True   # Cruiser:     NE, NW, W
        VALID_DEPLOY_FACINGS[2, [0, 1, 2, 3]] = True   # Dreadnought: E, NE, NW, W

        # Which variants each unit type can exist in (Skirmish has no Dread,
        # Showdown has no Cruiser). Used by fig4/5/6 to grey out N/A cells.
        UNIT_VALID_IN_VARIANT = np.array([
            [True,  True,  True,  True ],   # Fighter
            [True,  False, True,  True ],   # Cruiser
            [False, True,  True,  True ],   # Dreadnought
        ], dtype=bool)

        # Inner board_side per variant id (variants 0-2 use 5; Battle uses 6).
        # Drives deploy-hex annotations on per-variant heatmaps.
        VARIANT_BOARD_SIDE = [5, 5, 5, 6]

        # Minimum sample count below which a renormalized distribution is too
        # noisy to plot; we render an "insufficient data" placeholder instead.
        MIN_SAMPLES_FOR_PLOT = 50

        # Per-sample unit counts by type (my + opponent).
        # Fighter=1 hex, Cruiser=2 hexes, Dreadnought=3 hexes — divide channel sum by hex size.
        _hex_sizes = [1, 2, 3]
        unit_counts_my  = [c_data[:, ch,   :, :].sum(dim=(1, 2)).numpy() / sz
                           for ch, sz in zip([1, 2, 3], _hex_sizes)]
        unit_counts_opp = [c_data[:, ch+4, :, :].sum(dim=(1, 2)).numpy() / sz
                           for ch, sz in zip([1, 2, 3], _hex_sizes)]

        del c_data

        # Precompute spatial pi tensor once: (N, 13, 13, 10)
        pi_spatial = pi_np[:, :1690].reshape(-1, 13, 13, 10)

        # Hex valid-cell mask for the 13×13 unified board (board_side=6)
        hex_mask = np.zeros((13, 13), dtype=bool)
        for r in range(13):
            for c in range(13):
                q = r - 6; rh = c - 6; s = -q - rh
                if abs(q) <= 6 and abs(rh) <= 6 and abs(s) <= 6:
                    hex_mask[r, c] = True

        # Slots 0-4 are movement (MvFwd/MvFL/MvFR/RotL/RotR), 5-9 are firing.
        MOVE_KIND_MASK = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        FIRE_KIND_MASK = ~MOVE_KIND_MASK
        ALL_KIND_MASK  = np.ones(10, dtype=bool)
        _KIND_MASKS = {"all": ALL_KIND_MASK, "move": MOVE_KIND_MASK, "fire": FIRE_KIND_MASK}

        def _spatial_heatmap(sample_mask, unit_idx=None, action_kind="all"):
            """(13,13) heatmap of π mass per unit-instance at each source hex.

            Mass is restricted to (hex, slot) pairs whose slot is type-legal
            for a unit type present at that hex. action_kind in
            {"all","move","fire"} further restricts which slots contribute.

            The grid is normalized by total unit-instance count of the
            relevant type(s) across the samples — so attrition (fewer units
            alive in later phases) doesn't artificially shrink the heatmap;
            you see "what each remaining unit does," not "how many units
            there are doing it."

            Returns (grid, mass_per_unit):
              grid[r,c]      = avg π mass at hex (r,c) per unit-instance,
                               NaN outside the hex hull. Cross-panel
                               comparable.
              mass_per_unit  = total π mass per unit-instance for this
                               unit/kind selection. Equal to nansum(grid).
            """
            n = int(sample_mask.sum())
            if n == 0:
                return np.full((13, 13), np.nan), 0.0
            sp = pi_spatial[sample_mask]                              # (n, 13, 13, 10)
            grid = np.zeros((13, 13))
            units = [unit_idx] if unit_idx is not None else [0, 1, 2]
            kind_mask = _KIND_MASKS[action_kind]
            denom = 0.0
            for u in units:
                up = unit_presence[u][sample_mask]                    # (n, 13, 13)
                slot_mask = VALID_SLOTS_BY_UNIT[u] & kind_mask        # (10,)
                grid += (sp * up[:, :, :, np.newaxis] * slot_mask
                         ).sum(axis=(0, 3))                           # (13, 13)
                denom += float(unit_counts_my[u][sample_mask].sum())
            if denom <= 0:
                return np.full((13, 13), np.nan), 0.0
            grid = grid / denom
            mass_per_unit = float(grid.sum())
            return np.where(hex_mask, grid, np.nan), mass_per_unit

        def _action_dist(sample_mask, unit_idx):
            """Per-slot policy mass for a unit type, with per-unit magnitude.

            Returns (dist10, mass_per_unit):
              dist10        sums to 1 across the 10 slots (renormalized
                            over type-legal slots) — "of this unit type's
                            action mass in these samples, what share goes
                            to each slot".
              mass_per_unit = Σπ on this unit type's type-legal slots
                            divided by total unit-instance count of this
                            type across the samples — cross-cell magnitude
                            that is NOT confounded by unit attrition.
            """
            n = int(sample_mask.sum())
            if n == 0:
                return np.zeros(10), 0.0
            sp = pi_spatial[sample_mask]                              # (n, 13, 13, 10)
            up = unit_presence[unit_idx][sample_mask]                 # (n, 13, 13)
            slot_mask = VALID_SLOTS_BY_UNIT[unit_idx]                 # (10,)
            dist = (sp * up[:, :, :, np.newaxis] * slot_mask
                    ).sum(axis=(0, 1, 2))                             # (10,)
            total = float(dist.sum())
            denom = float(unit_counts_my[unit_idx][sample_mask].sum())
            mass_per_unit = (total / denom) if denom > 0 else 0.0
            return (dist / total if total > 0 else dist), mass_per_unit

        def _draw_na(ax, label="N/A"):
            """Render a grey 'N/A' / insufficient-data placeholder panel."""
            ax.set_facecolor("#dddddd")
            ax.text(0.5, 0.5, label, transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#888888")
            ax.set_xticks([])
            ax.set_yticks([])

        # ---- Hex-canvas rendering ---------------------------------------------------
        # The 13×13 grid is indexed [q+6, r+6]. The game's actual hex layout
        # (matching StarGambitGS::dump and play.py) places P0 deploy at the
        # bottom-right corner and P1 deploy at the top-left. We render heatmaps
        # using axial→pixel coords so peaks land at the corners a player expects.
        from matplotlib.collections import PolyCollection

        _HEX_DY = math.sqrt(3) / 2
        _hex_unit_verts = np.array([
            (math.cos(math.radians(60 * k + 30)),
             math.sin(math.radians(60 * k + 30)))
            for k in range(6)
        ])  # pointy-top hexagon, circumradius=1

        def _axial_to_xy(q, r):
            """Axial (q, r) → screen (x, y) for pointy-top hexes, +y points down."""
            x = q + r * 0.5
            y = r * _HEX_DY
            return x, y

        def _hex_in_bounds(q, r, side=6):
            return abs(q) <= side and abs(r) <= side and abs(-q - r) <= side

        def _draw_hex_heatmap(ax, grid, cmap="YlOrRd", vmin=0, vmax=None,
                              board_side_hint=None, show_colorbar=True,
                              show_compass=True):
            """Render a 13×13 grid (indexed [q+6, r+6]) as a hex polygon collection.

            board_side_hint: int marks the P0 deploy hex for that inner board
                size AND clips the rendered hexes to that variant's board
                (so the outer empty ring is hidden for board_side=5 variants).
                None renders all hexes within the unified board_side=6 hull
                and marks both 5- and 6-side spawn hexes.
            """
            side = board_side_hint if board_side_hint is not None else 6
            polys = []
            values = []
            for q in range(-side, side + 1):
                for r in range(-side, side + 1):
                    if not _hex_in_bounds(q, r, side):
                        continue
                    val = grid[q + 6, r + 6]
                    if np.isnan(val):
                        continue
                    cx, cy = _axial_to_xy(q, r)
                    polys.append(_hex_unit_verts * 0.95 + (cx, cy))
                    values.append(val)
            if not polys:
                _draw_na(ax)
                return None
            if vmax is None:
                vmax = max(values) if values else 1.0
            if vmax <= 0:
                vmax = 1.0
            coll = PolyCollection(polys, array=np.asarray(values),
                                  cmap=cmap, edgecolors="white", linewidths=0.3)
            coll.set_clim(vmin, vmax)
            ax.add_collection(coll)

            # Spawn markers.
            spawn_sides = [board_side_hint] if board_side_hint is not None else [5, 6]
            for s in spawn_sides:
                cx, cy = _axial_to_xy(0, s)
                ax.plot(cx, cy, marker="x", markersize=8, mew=1.5,
                        color="#222222", linestyle="none")
            # Axis cosmetics: equal aspect, +y down, no ticks; small compass.
            x_min = -side - 0.5 * side - 1
            x_max =  side + 0.5 * side + 1
            y_min = -side * _HEX_DY - 1
            y_max =  side * _HEX_DY + 1
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  # invert y so +y is down (matches dump())
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if show_compass:
                ax.text(x_max - 0.4, 0, "E", ha="right", va="center",
                        fontsize=7, color="#888888")
                ax.text(x_min + 0.4, 0, "W", ha="left",  va="center",
                        fontsize=7, color="#888888")
                ax.text(0, y_min + 0.4, "↑P1", ha="center", va="top",
                        fontsize=7, color="#888888")
                ax.text(0, y_max - 0.4, "P0↓", ha="center", va="bottom",
                        fontsize=7, color="#888888")
            if show_colorbar:
                import matplotlib.pyplot as _plt
                _plt.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
            return coll

        unit_names  = ["Fighter", "Cruiser", "Dreadnought"]
        unit_colors = ["#4e79a7", "#f28e2b", "#59a14f"]
        facing_labels = ["E", "NE", "NW", "W", "SW", "SE"]
        action_labels = ["MvFwd", "MvFL", "MvFR", "RotL", "RotR",
                         "FirFwd", "FirFL", "FirFR", "FirRL", "FirRR"]
        # Blue=move, orange=rotate, red=fire
        action_colors = ["#4e79a7","#5fa2ce","#a0cbe8",
                         "#f28e2b","#ffbe7d",
                         "#e15759","#ff9d9a","#b07aa1","#9c755f","#d37295"]

        # --- Figure 1: aggregate action frequency by phase ---
        fig1, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig1.suptitle(f"Iteration {iteration} — Action Frequency by Phase", fontsize=12)

        from matplotlib.patches import Patch as _Patch

        # --- Figure 1 (per-variant): each row is a variant. Columns are
        # Deploy bars + (Move, Fire) heatmaps for each of 3 phases = 7 cols.
        # Variants 0-2 use board_side=5 so the outer ring of hexes is hidden.
        # Heatmap values are π mass per unit-instance; vmax is shared across
        # all 6 phase heatmaps within a variant row so cross-phase magnitude
        # is visible at a glance (rare-action panels look dim, common-action
        # panels look bright). Variant rows have independent scales.
        fig1, axes = plt.subplots(4, 7, figsize=(34, 20))
        fig1.suptitle(
            f"Iteration {iteration} — Action Frequency by Variant × Phase "
            "(π mass per unit-instance; Move slots 0-4 vs Fire slots 5-9; "
            "vmax shared per variant row)",
            fontsize=13)
        deploy_valid_mask = np.concatenate(
            [VALID_DEPLOY_FACINGS[ui] for ui in range(3)]).astype(bool)  # (18,)
        phase_specs = [
            ("Standstill / Maneuver",       standstill_mask),
            ("Attrition (no portal dmg)",   attrition_mask),
            ("Portal Damage",               portal_mask),
        ]

        # Pass 1: pre-compute every spatial panel so we can share vmax per row.
        spatial_cache = {}  # (vid, phase_idx, kind) -> (grid_or_None, mass, n)
        for vid in range(len(UNIFIED_VARIANT_NAMES)):
            vmask_v = variant_ids == vid
            for phase_idx, (_, pmask) in enumerate(phase_specs):
                combo = pmask & vmask_v
                n_combo = int(combo.sum())
                for kind in ("move", "fire"):
                    if n_combo < MIN_SAMPLES_FOR_PLOT:
                        spatial_cache[(vid, phase_idx, kind)] = (None, 0.0, n_combo)
                    else:
                        grid, mass = _spatial_heatmap(combo, action_kind=kind)
                        spatial_cache[(vid, phase_idx, kind)] = (grid, mass, n_combo)

        # Per-variant-row vmax (None falls back to per-panel auto in renderer)
        row_vmax = {}
        for vid in range(len(UNIFIED_VARIANT_NAMES)):
            peaks = []
            for phase_idx in range(len(phase_specs)):
                for kind in ("move", "fire"):
                    grid, _, _ = spatial_cache[(vid, phase_idx, kind)]
                    if grid is None or np.all(np.isnan(grid)):
                        continue
                    m = np.nanmax(grid)
                    if np.isfinite(m):
                        peaks.append(m)
            row_vmax[vid] = max(peaks) if peaks else None

        for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
            vmask = variant_ids == vid
            side = VARIANT_BOARD_SIDE[vid]

            # ---- Deploy bars for this variant (col 0) ----
            ax = axes[vid, 0]
            d_mask = deploy_mask & vmask
            d_n = int(d_mask.sum())
            if d_n >= MIN_SAMPLES_FOR_PLOT:
                dep_pi = pi_np[d_mask, 1690:1708].sum(axis=0)
                dep_pi = dep_pi * deploy_valid_mask
                dep_total = dep_pi.sum()
                if dep_total > 0:
                    dep_pi = dep_pi / dep_total
            else:
                dep_pi = np.zeros(18)
            x = np.arange(6); bar_w = 0.25
            legend_handles = []
            for ui, (uname, col) in enumerate(zip(unit_names, unit_colors)):
                # Grey the bar if either the facing or the unit is invalid here.
                unit_ok = UNIT_VALID_IN_VARIANT[ui, vid]
                bar_colors = [col if (unit_ok and VALID_DEPLOY_FACINGS[ui, f])
                              else "#dddddd" for f in range(6)]
                ax.bar(x + ui * bar_w, dep_pi[ui * 6: ui * 6 + 6], bar_w,
                       color=bar_colors, alpha=0.85)
                legend_handles.append(_Patch(facecolor=col, alpha=0.85, label=uname))
            ax.set_xticks(x + bar_w); ax.set_xticklabels(facing_labels, fontsize=8)
            if d_n < MIN_SAMPLES_FOR_PLOT:
                _draw_na(ax, label=f"n={d_n}\n(insufficient)")
            ax.set_title(f"{vname.capitalize()} — Deploy\n(n={d_n})", fontsize=10)
            ax.set_ylabel("Fraction of deploy pi mass\n(valid only)", fontsize=8)
            ax.legend(handles=legend_handles, fontsize=7)

            # ---- Spatial-phase heatmaps: (Move, Fire) per phase ----
            vmax = row_vmax[vid]
            for phase_idx, (title, _) in enumerate(phase_specs):
                for kind_idx, kind in enumerate(("move", "fire")):
                    col_idx = 1 + phase_idx * 2 + kind_idx
                    ax = axes[vid, col_idx]
                    grid, mass, n = spatial_cache[(vid, phase_idx, kind)]
                    label = "Move" if kind == "move" else "Fire"
                    if n < MIN_SAMPLES_FOR_PLOT or grid is None:
                        _draw_na(ax, label=f"n={n}\n(insufficient)")
                        ax.set_title(f"{vname.capitalize()} — {title}\n{label} (n={n})",
                                     fontsize=9)
                        continue
                    _draw_hex_heatmap(ax, grid, board_side_hint=side, vmax=vmax)
                    ax.set_title(
                        f"{vname.capitalize()} — {title}\n"
                        f"{label} (n={n}, Σπ/unit={mass:.3f})",
                        fontsize=9)
        plt.tight_layout()

        fig2 = None

        # --- Figure 3: per-unit action breakdown (3 units × 4 phases).
        # Bars sum to 1 within each cell (slot share). Σπ/unit in the title
        # is the cross-cell-comparable magnitude: total π mass for this
        # unit type per unit-instance, so attrition doesn't bias it.
        fig3, axes3 = plt.subplots(3, 4, figsize=(20, 12))
        fig3.suptitle(f"Iteration {iteration} — Action Breakdown by Unit Type "
                      "(type-legal slots only; Σπ/unit gives per-instance "
                      "magnitude)", fontsize=13)
        _phases = [(deploy_mask, "Deploy"), (standstill_mask, "Standstill"),
                   (attrition_mask, "Attrition"), (portal_mask, "Portal")]
        for ui, uname in enumerate(unit_names):
            valid_slots = VALID_SLOTS_BY_UNIT[ui]
            n_valid = int(valid_slots.sum())
            for pi_col, (mask, phase) in enumerate(_phases):
                ax = axes3[ui, pi_col]
                n = int(mask.sum())
                bar_colors = [action_colors[s] if valid_slots[s] else "#dddddd"
                              for s in range(10)]
                if n < MIN_SAMPLES_FOR_PLOT:
                    _draw_na(ax, label=f"n={n}\n(insufficient)")
                    ax.set_title(f"{uname} — {phase}\n(n={n})", fontsize=9)
                    continue
                dist, mass_per_unit = _action_dist(mask, ui)
                ax.bar(range(10), dist, color=bar_colors, alpha=0.85)
                ax.set_xticks(range(10))
                ax.set_xticklabels(action_labels, fontsize=6, rotation=45, ha="right")
                ax.set_title(
                    f"{uname} — {phase}\n"
                    f"(n={n}, valid:{n_valid}, Σπ/unit={mass_per_unit:.3f})",
                    fontsize=9)
                ax.set_ylabel("Fraction", fontsize=8)
        plt.tight_layout()

        # --- Figure 4: per-unit × per-variant spatial heatmaps (all positions).
        # 3 rows (units) × 8 cols ((variant, kind) pairs). vmax is shared
        # within each (unit, variant) pair so move/fire are directly
        # comparable; pairs across variants/units are NOT scale-coupled.
        fig4, axes4 = plt.subplots(3, 8, figsize=(28, 12))
        fig4.suptitle(
            f"Iteration {iteration} — Unit × Variant Action Heatmaps "
            "(All Positions; π mass per unit-instance; "
            "vmax shared per (unit, variant) pair)",
            fontsize=13)
        for ui, uname in enumerate(unit_names):
            for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                vmask = variant_ids == vid
                n = int(vmask.sum())
                # Compute both kinds first so the (move, fire) pair can share vmax.
                kind_data = {}
                for kind in ("move", "fire"):
                    if not UNIT_VALID_IN_VARIANT[ui, vid] or n < MIN_SAMPLES_FOR_PLOT:
                        kind_data[kind] = (None, 0.0)
                    else:
                        grid, mass = _spatial_heatmap(
                            vmask, unit_idx=ui, action_kind=kind)
                        kind_data[kind] = (grid, mass)
                peaks = [np.nanmax(g) for g, _ in kind_data.values()
                         if g is not None and not np.all(np.isnan(g))]
                peaks = [p for p in peaks if np.isfinite(p)]
                pair_vmax = max(peaks) if peaks else None
                for kind_idx, kind in enumerate(("move", "fire")):
                    col_idx = vid * 2 + kind_idx
                    ax = axes4[ui, col_idx]
                    label = "Move" if kind == "move" else "Fire"
                    grid, mass = kind_data[kind]
                    if not UNIT_VALID_IN_VARIANT[ui, vid]:
                        _draw_na(ax)
                        ax.set_title(f"{uname} — {label}\n{vname.capitalize()} (N/A)",
                                     fontsize=8)
                    elif n < MIN_SAMPLES_FOR_PLOT or grid is None:
                        _draw_na(ax, label=f"n={n}\n(insufficient)")
                        ax.set_title(f"{uname} — {label}\n{vname.capitalize()} (n={n})",
                                     fontsize=8)
                    else:
                        _draw_hex_heatmap(
                            ax, grid,
                            board_side_hint=VARIANT_BOARD_SIDE[vid],
                            vmax=pair_vmax)
                        ax.set_title(
                            f"{uname} — {label}\n"
                            f"{vname.capitalize()} (n={n}, Σπ/unit={mass:.3f})",
                            fontsize=8)
        plt.tight_layout()

        # --- Figure 5: unit count by type × phase × variant ---
        _phase_list = [
            (deploy_mask,     "Deploy"),
            (standstill_mask, "Standstill"),
            (attrition_mask,  "Attrition"),
            (portal_mask,     "Portal"),
        ]
        fig5, axes5u = plt.subplots(4, 4, figsize=(16, 12))
        fig5.suptitle(f"Iteration {iteration} — Unit Count by Type × Phase × Variant "
                      "(y-axis shared per variant column)", fontsize=12)
        _bar_x = np.arange(3); _bar_w = 0.35

        # Precompute averages so we can lock the y-axis per variant column.
        # Shape: [phase][variant][side][unit] where side 0=mine, 1=opp.
        avgs = np.full((len(_phase_list), len(UNIFIED_VARIANT_NAMES), 2, 3), np.nan)
        ns   = np.zeros((len(_phase_list), len(UNIFIED_VARIANT_NAMES)), dtype=int)
        for pi_row, (pmask, _) in enumerate(_phase_list):
            for vid in range(len(UNIFIED_VARIANT_NAMES)):
                combo = pmask & (variant_ids == vid)
                ns[pi_row, vid] = int(combo.sum())
                if ns[pi_row, vid] < MIN_SAMPLES_FOR_PLOT:
                    continue
                for ui in range(3):
                    if not UNIT_VALID_IN_VARIANT[ui, vid]:
                        continue
                    avgs[pi_row, vid, 0, ui] = unit_counts_my[ui][combo].mean()
                    avgs[pi_row, vid, 1, ui] = unit_counts_opp[ui][combo].mean()

        # Per-variant y-max: largest avg across all phases / sides / units, +10% headroom.
        ymax_per_variant = np.nanmax(avgs, axis=(0, 2, 3))
        ymax_per_variant = np.where(np.isnan(ymax_per_variant), 1.0, ymax_per_variant)
        ymax_per_variant = np.maximum(ymax_per_variant * 1.1, 0.5)

        for pi_row, (pmask, pname) in enumerate(_phase_list):
            for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                ax = axes5u[pi_row, vid]
                n = int(ns[pi_row, vid])
                ax.set_title(f"{vname.capitalize()} — {pname}\n(n={n})", fontsize=7)
                if n < MIN_SAMPLES_FOR_PLOT:
                    _draw_na(ax, label=f"n={n}\n(insufficient)" if n > 0 else "N/A")
                    continue
                my_avg  = avgs[pi_row, vid, 0]
                opp_avg = avgs[pi_row, vid, 1]
                # Per-bar alpha — fade invalid (unit, variant) bars to grey.
                my_colors  = [unit_colors[ui] if UNIT_VALID_IN_VARIANT[ui, vid]
                              else "#dddddd" for ui in range(3)]
                opp_colors = [unit_colors[ui] if UNIT_VALID_IN_VARIANT[ui, vid]
                              else "#dddddd" for ui in range(3)]
                ax.bar(_bar_x - _bar_w/2, np.nan_to_num(my_avg),  _bar_w,
                       color=my_colors,  alpha=0.85)
                ax.bar(_bar_x + _bar_w/2, np.nan_to_num(opp_avg), _bar_w,
                       color=opp_colors, alpha=0.45)
                ax.set_xticks(_bar_x); ax.set_xticklabels(["F", "C", "D"], fontsize=8)
                ax.set_ylim(0, ymax_per_variant[vid])
                ax.set_ylabel("Avg units", fontsize=7)
                if pi_row == 0 and vid == 0:
                    legend_p = [_Patch(facecolor="#888888", alpha=0.85, label="Mine"),
                                _Patch(facecolor="#888888", alpha=0.45, label="Opp")]
                    ax.legend(handles=legend_p, fontsize=6)
        plt.tight_layout()

        # --- Figure 6: per-phase unit × variant spatial heatmaps.
        # Same 3×8 layout as Figure 4, one figure per phase. vmax is shared
        # within each (unit, variant) pair for move/fire comparison; cross-
        # phase magnitude comparison goes through Fig 1 (or by reading the
        # Σπ/unit numbers in titles here).
        fig6_list = []
        for pmask, pname in _phase_list:
            f, axs = plt.subplots(3, 8, figsize=(28, 12))
            f.suptitle(
                f"Iteration {iteration} — Unit × Variant Heatmap: {pname} "
                "(π mass per unit-instance; vmax shared per (unit, variant) pair)",
                fontsize=13)
            for ui, uname in enumerate(unit_names):
                for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                    vmask = pmask & (variant_ids == vid)
                    n = int(vmask.sum())
                    kind_data = {}
                    for kind in ("move", "fire"):
                        if not UNIT_VALID_IN_VARIANT[ui, vid] or n < MIN_SAMPLES_FOR_PLOT:
                            kind_data[kind] = (None, 0.0)
                        else:
                            grid, mass = _spatial_heatmap(
                                vmask, unit_idx=ui, action_kind=kind)
                            kind_data[kind] = (grid, mass)
                    peaks = [np.nanmax(g) for g, _ in kind_data.values()
                             if g is not None and not np.all(np.isnan(g))]
                    peaks = [p for p in peaks if np.isfinite(p)]
                    pair_vmax = max(peaks) if peaks else None
                    for kind_idx, kind in enumerate(("move", "fire")):
                        col_idx = vid * 2 + kind_idx
                        ax = axs[ui, col_idx]
                        label = "Move" if kind == "move" else "Fire"
                        grid, mass = kind_data[kind]
                        if not UNIT_VALID_IN_VARIANT[ui, vid]:
                            _draw_na(ax)
                            ax.set_title(
                                f"{uname} — {label}\n{vname.capitalize()} (N/A)",
                                fontsize=8)
                        elif n < MIN_SAMPLES_FOR_PLOT or grid is None:
                            _draw_na(ax, label=f"n={n}\n(insufficient)")
                            ax.set_title(
                                f"{uname} — {label}\n{vname.capitalize()} (n={n})",
                                fontsize=8)
                        else:
                            _draw_hex_heatmap(
                                ax, grid,
                                board_side_hint=VARIANT_BOARD_SIDE[vid],
                                vmax=pair_vmax)
                            ax.set_title(
                                f"{uname} — {label}\n"
                                f"{vname.capitalize()} (n={n}, Σπ/unit={mass:.3f})",
                                fontsize=8)
            plt.tight_layout()
            fig6_list.append((f, pname))

        # --- Figure 7: value calibration curves per variant ---
        fig7 = None
        if variant_stats:
            n_variants = len(variant_stats)
            fig7, axes7 = plt.subplots(1, n_variants, figsize=(4 * n_variants, 4))
            if n_variants == 1:
                axes7 = [axes7]
            fig7.suptitle(f"Iteration {iteration} — Value Calibration", fontsize=12)
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            MIN_BIN_COUNT = 10
            for ax, (vname, stats) in zip(axes7, variant_stats.items()):
                vp = stats["v_pred"]
                va = stats["v_actual"]
                bin_idx = np.digitize(vp, bins) - 1
                bin_idx = np.clip(bin_idx, 0, 9)
                counts = np.array([(bin_idx == b).sum() for b in range(10)])
                # Only plot bins with enough support to be informative.
                actual_means = np.array([
                    va[bin_idx == b].mean() if counts[b] >= MIN_BIN_COUNT else np.nan
                    for b in range(10)
                ])
                ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
                valid = ~np.isnan(actual_means)
                sc = ax.scatter(bin_centers[valid], actual_means[valid],
                                c=counts[valid], cmap="YlOrRd", s=60, zorder=3)
                ax.plot(bin_centers[valid], actual_means[valid], alpha=0.7)
                # Annotate per-bin n below the curve.
                for b in range(10):
                    if counts[b] > 0:
                        ax.text(bin_centers[b], 0.02, f"{counts[b]}",
                                ha="center", va="bottom", fontsize=6,
                                color="#666666",
                                alpha=1.0 if counts[b] >= MIN_BIN_COUNT else 0.4)
                plt.colorbar(sc, ax=ax, label="n samples")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xlabel("Predicted win prob"); ax.set_ylabel("Actual win rate")
                ax.set_title(f"{vname.capitalize()}\n(n={len(vp)}, "
                             f"bins≥{MIN_BIN_COUNT}: {int(valid.sum())})",
                             fontsize=9)
                ax.legend(fontsize=7)
            plt.tight_layout()

        run.track(aim_lib.Image(fig1), name="action_heatmap",
                  epoch=iteration, step=total_train_steps)
        run.track(aim_lib.Image(fig3), name="unit_action_breakdown",
                  epoch=iteration, step=total_train_steps)
        run.track(aim_lib.Image(fig4), name="unit_variant_heatmap",
                  epoch=iteration, step=total_train_steps)
        run.track(aim_lib.Image(fig5), name="unit_count_by_phase",
                  epoch=iteration, step=total_train_steps)
        for f6, pname in fig6_list:
            run.track(aim_lib.Image(f6), name="unit_variant_heatmap_by_phase",
                      epoch=iteration, step=total_train_steps, context={"phase": pname})
        if fig7 is not None:
            run.track(aim_lib.Image(fig7), name="value_calibration",
                      epoch=iteration, step=total_train_steps)
    except Exception:
        pass
    finally:
        plt.close("all")


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
        nnargs = neural_net.nnargs_from_config(config)
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

    fallback_checkpoint_dir = None

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
        if _is_unified_game(config):
            wr_v = [np.full((total_agents, total_agents), np.nan) for _ in range(4)]
            elo_v = [np.zeros(total_agents) for _ in range(4)]
        current_best = 0
        total_train_steps = 0
        lr_state = _default_lr_state(config)
        if bootstrap_from:
            lr_state['last_best_iter'] = source_n

        # Handle bootstrap from existing experiment
        if bootstrap_from:
            source_dir = bootstrap_from
            source_config_path = os.path.join(source_dir, "config.yaml")
            source_paths = config.resolve_paths(source_dir)
            source_elo_path = os.path.join(source_dir, "elo.csv")
            source_wr_path = os.path.join(source_dir, "win_rate.csv")

            # Copy elo/wr from source (if available)
            if os.path.exists(source_elo_path) and os.path.exists(source_wr_path):
                source_elo = np.genfromtxt(source_elo_path, delimiter=",")
                source_wr = np.genfromtxt(source_wr_path, delimiter=",")
                copy_size = min(len(source_elo), total_agents)
                wr[:copy_size, :copy_size] = source_wr[:copy_size, :copy_size]
                elo[:copy_size] = source_elo[:copy_size]

            # Detect architecture match before phase loop
            same_arch = True
            if os.path.exists(source_config_path):
                source_cfg = load_config(source_config_path, {}, warn=False)
                same_arch = (
                    source_cfg.depth == config.depth
                    and source_cfg.channels == config.channels
                    and source_cfg.kernel_size == config.kernel_size
                    and source_cfg.dense_net == config.dense_net
                    and source_cfg.star_gambit_spatial == config.star_gambit_spatial
                    and source_cfg.head_channels == config.head_channels
                    and source_cfg.head_pool == config.head_pool
                )

            # Build phase list
            bootstrap_phases = [] if config.bootstrap_window_only else ["Copy Reservoir"]
            bootstrap_phases.append("Copy Window")
            if same_arch:
                bootstrap_phases.append("Copy Checkpoint")
            else:
                bootstrap_phases.append("Retrain")
            bootstrap_phases.append("Calibrate ELO")

            phase_bar = tqdm.tqdm(bootstrap_phases, desc="Bootstrap", leave=True)
            for phase in phase_bar:
                phase_bar.set_description(f"Bootstrap: {phase}")

                if phase == "Copy Reservoir":
                    os.makedirs(paths["reservoir"], exist_ok=True)
                    # Copy all reservoir files (chunks, metadata, and any legacy files)
                    source_res = source_paths["reservoir"]
                    res_files = (
                        glob.glob(os.path.join(source_res, "chunk_*.ptz"))
                        + glob.glob(os.path.join(source_res, "reservoir_meta.json"))
                        + [p for c, v, p2, _ in glob_file_triples(source_res)
                           for p in (c, v, p2)]
                    )
                    if res_files:
                        for src in tqdm.tqdm(res_files, desc="  Files", leave=False):
                            shutil.copy2(src, os.path.join(paths["reservoir"], os.path.basename(src)))

                elif phase == "Copy Window":
                    source_window_triples = []
                    for wi in range(0, source_n):
                        triples = glob_file_triples(source_paths["history"], f"{wi:04d}-*-canonical-*.ptz")
                        if not triples:
                            triples = glob_file_triples(source_paths["history"], f"{wi:04d}-*-canonical-*.pt")
                        source_window_triples.extend(triples)
                    if source_window_triples:
                        os.makedirs(paths["history"], exist_ok=True)
                        all_win_files = []
                        for c, v, p, _ in source_window_triples:
                            all_win_files.extend([c, v, p])
                        for src in tqdm.tqdm(all_win_files, desc="  Files", leave=False):
                            shutil.copy2(src, os.path.join(paths["history"], os.path.basename(src)))

                elif phase == "Copy Checkpoint":
                    best_source = source_n
                    if os.path.exists(source_elo_path):
                        best_source = int(np.argmax(source_elo[:source_n + 1]))
                    cps = sorted(glob.glob(os.path.join(source_paths["checkpoint"], f"{best_source:04d}-*.pt")))
                    if cps:
                        dest_name = f"{source_n:04d}-{experiment_name}.pt"
                        shutil.copy2(cps[-1], os.path.join(paths["checkpoint"], dest_name))

                elif phase == "Retrain":
                    # Try chunked reservoir first, fall back to per-iteration
                    reservoir_files = glob_reservoir_chunks(paths["reservoir"])
                    if not reservoir_files:
                        reservoir_files = glob_file_triples(paths["reservoir"])
                    window_files = glob_file_triples(paths["history"])

                    nn = neural_net.NNWrapper.load_checkpoint(
                        Game, paths["checkpoint"], f"0000-{experiment_name}.pt"
                    )

                    if reservoir_files or window_files:
                        total_train_steps = _bootstrap_retrain(
                            nn, reservoir_files, window_files, config, run, source_n, total_train_steps
                        )

                    nn.save_checkpoint(paths["checkpoint"], f"{source_n:04d}-{experiment_name}.pt",
                                       zstd_level=config.zstd_level)

                elif phase == "Calibrate ELO":
                    compare_start = max(0, source_n - config.bootstrap_compare_past)
                    for past_iter in tqdm.tqdm(range(compare_start, source_n), desc="  ELO", leave=False):
                        nn_rate, draw_rate, _, _, _, _, _, _, _, _, _ = play_past(
                            config, paths, experiment_name, config.compare_mcts_visits,
                            source_n, past_iter, config.past_compare_batch_size,
                            fallback_checkpoint_dir=source_paths["checkpoint"],
                        )
                        wr[source_n, past_iter] = nn_rate + draw_rate / Game.NUM_PLAYERS()
                        wr[past_iter, source_n] = 1 - (nn_rate + draw_rate / Game.NUM_PLAYERS())
                        gc.collect()

                    elo = get_elo(elo, wr, source_n)

            phase_bar.close()

            current_best = source_n
            start = source_n
            fallback_checkpoint_dir = source_paths["checkpoint"]

            # Save bootstrap metadata for resume
            bootstrap_meta_path = os.path.join(experiment_dir, "bootstrap_meta.json")
            with open(bootstrap_meta_path, "w") as f:
                json.dump({"source_checkpoint_dir": source_paths["checkpoint"]}, f)

            if os.path.exists(source_elo_path):
                best_source_iter = int(np.argmax(source_elo[:source_n + 1]))
                if elo[source_n] < elo[best_source_iter]:
                    tqdm.tqdm.write(f"Warning: bootstrapped network (ELO {elo[source_n]:.0f}) is weaker than "
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
        if _is_unified_game(config):
            wr_v = []
            elo_v = []
            for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                vwr_path = os.path.join(experiment_dir, f"win_rate_{vname}.csv")
                velo_path = os.path.join(experiment_dir, f"elo_{vname}.csv")
                m = np.full((total_agents, total_agents), np.nan)
                if os.path.exists(vwr_path):
                    tmp = np.genfromtxt(vwr_path, delimiter=",")
                    sz = min(tmp.shape[0], total_agents)
                    m[:sz, :sz] = tmp[:sz, :sz]
                wr_v.append(m)
                ev = np.zeros(total_agents)
                if os.path.exists(velo_path):
                    tmp = np.genfromtxt(velo_path, delimiter=",")
                    sz = min(len(tmp), total_agents)
                    ev[:sz] = tmp[:sz]
                elo_v.append(ev)
        total_train_steps = int(
            np.genfromtxt(os.path.join(experiment_dir, "total_train_steps.txt"))
        )
        if os.path.exists(lr_state_path):
            with open(lr_state_path) as f:
                lr_state = json.load(f)
        else:
            lr_state = _default_lr_state(config)

        # Recover bootstrap fallback checkpoint dir on resume
        bootstrap_meta_path = os.path.join(experiment_dir, "bootstrap_meta.json")
        if os.path.exists(bootstrap_meta_path):
            with open(bootstrap_meta_path) as f:
                fallback_checkpoint_dir = json.load(f).get("source_checkpoint_dir")

        # Use the gating-derived best (matches what was tracked to aim during
        # the prior run). argmax(elo) can point at a slot whose checkpoint
        # never existed locally — e.g. after bootstrap, the source's elo array
        # is copied verbatim but the source's best checkpoint is renamed into
        # slot `source_n` in the new dir.
        current_best = int(lr_state.get('last_best_iter', 0))
        if current_best > start or not (
            os.path.exists(os.path.join(paths["checkpoint"], f"{current_best:04d}-{experiment_name}.pt"))
            or (fallback_checkpoint_dir and glob.glob(
                os.path.join(fallback_checkpoint_dir, f"{current_best:04d}-*.pt")))
        ):
            # Legacy run with no lr_state, or stale value — fall back to
            # argmax restricted to slots whose checkpoint actually exists.
            candidates = [
                i for i in range(start + 1)
                if os.path.exists(os.path.join(paths["checkpoint"], f"{i:04d}-{experiment_name}.pt"))
            ]
            current_best = max(candidates, key=lambda i: elo[i]) if candidates else 0

    postfix = {"best": current_best}
    panel = [current_best]

    # Unified variant mixing state (only used for star_gambit_unified).
    unified_probs = None
    variant_sample_count_history = []  # rolling 5-iter window to smooth prob corrections
    if _is_unified_game(config) and start > 0:
        for past in range(max(0, start - 5), start):
            counts = _count_variant_samples(paths, past)
            if counts is not None:
                variant_sample_count_history.append(counts)
    if not _is_unified_game(config):
        wr_v = None
        elo_v = None

    # Rolling per-anchor history of (iter, mean_kl) for frozen-eval trajectory
    # analysis. Built up across iters as data arrives. Persisted to disk via
    # frozen_eval.append_kl_history, so resumes / restarts pick up the prior
    # trajectory instead of needing 3+ fresh iters to print a slope.
    _FROZEN_KL_HIST_LEN = 30
    frozen_kl_history = {a: [] for a in config.frozen_eval_anchor_iters}
    if config.frozen_eval_anchor_iters and start > 0:
        for anchor in config.frozen_eval_anchor_iters:
            prior = frozen_eval.load_kl_history(paths, anchor, max_iter=start)
            if prior:
                frozen_kl_history[anchor] = prior[-_FROZEN_KL_HIST_LEN:]

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
                    nn_rate, draw_rate, _, game_length, bench_depth, bench_entropy, bench_mpt, bench_vm, variant_nn_rates, variant_draw_rates, variant_play_metrics = play_past(
                        config, paths, experiment_name,
                        config.compare_mcts_visits, i, past_iter, config.past_compare_batch_size,
                        fallback_checkpoint_dir=fallback_checkpoint_dir,
                        variant_probs=unified_probs,
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
                    for vid, vrate in variant_nn_rates.items():
                        run.track(vrate, name="win_rate", epoch=i, step=total_train_steps,
                                  context={"vs": f"-{config.compare_past}", "variant": UNIFIED_VARIANT_NAMES[vid]})
                        if wr_v is not None:
                            vadj = vrate + variant_draw_rates.get(vid, 0.0) / Game.NUM_PLAYERS()
                            wr_v[vid][i, past_iter] = vadj
                            wr_v[vid][past_iter, i] = 1.0 - vadj
                    for vid, drate in variant_draw_rates.items():
                        run.track(drate, name="draw_rate", epoch=i, step=total_train_steps,
                                  context={"vs": f"-{config.compare_past}", "variant": UNIFIED_VARIANT_NAMES[vid]})
                    _track_variant_play_metrics(run, variant_play_metrics, epoch=i,
                                                step=total_train_steps,
                                                vs_context=f"-{config.compare_past}")
                    postfix[f"vs -{config.compare_past}"] = _fmt_pct(nn_rate + draw_rate / Game.NUM_PLAYERS())
                    gc.collect()
            stage_times["history"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_elo"):
                elo = get_elo(elo, wr, i)
                run.track(elo[i], name="elo", epoch=i, step=total_train_steps, context={"type": "current"})
                run.track(elo[current_best], name="elo", epoch=i, step=total_train_steps, context={"type": "best"})
                if elo_v is not None:
                    for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                        if not np.all(np.isnan(wr_v[vid][i])):
                            elo_v[vid] = get_elo(elo_v[vid], wr_v[vid], i)
                        run.track(elo_v[vid][i], name="variant_elo", epoch=i, step=total_train_steps,
                                  context={"variant": vname})
                postfix["elo"] = int(elo[i])
                pbar.set_postfix(postfix)
                np.savetxt(os.path.join(experiment_dir, "elo.csv"), elo, delimiter=",")
                # Persist wr matrices here too. Without this, a crash between
                # training (which writes checkpoint i+1) and gating's save would
                # cause iter i to be skipped on resume, leaving wr[i, i-compare_past]
                # permanently NaN and dragging variant elo down by 100s of points.
                np.savetxt(os.path.join(experiment_dir, "win_rate.csv"), wr, delimiter=",")
                if wr_v is not None:
                    for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                        np.savetxt(os.path.join(experiment_dir, f"win_rate_{vname}.csv"), wr_v[vid], delimiter=",")
                        np.savetxt(os.path.join(experiment_dir, f"elo_{vname}.csv"), elo_v[vid], delimiter=",")
            stage_times["elo"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_selfplay"):
                if _is_unified_game(config):
                    counts = _count_variant_samples(paths, i - 1) if i > 0 else None
                    if counts is not None:
                        variant_sample_count_history.append(counts)
                        if len(variant_sample_count_history) > 5:
                            variant_sample_count_history.pop(0)
                    rolled_counts = (
                        [sum(h[v] for h in variant_sample_count_history) for v in range(4)]
                        if variant_sample_count_history else None
                    )
                    unified_probs = _compute_unified_probs(config, rolled_counts)
                    for vi, vname in enumerate(UNIFIED_VARIANT_NAMES):
                        run.track(unified_probs[vi], name="variant_prob", epoch=i,
                                  step=total_train_steps, context={"variant": vname})
                sp = self_play(
                    config, paths, experiment_name,
                    current_best, i,
                    config.selfplay_mcts_visits,
                    config.fast_mcts_visits,
                    variant_probs=unified_probs,
                    fallback_checkpoint_dir=fallback_checkpoint_dir,
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
                postfix["wr"] = "/".join(_fmt_pct(x) for x in sp.win_rates)
                pbar.set_postfix(postfix)
                if sp.variant_game_counts:
                    total_vg = sum(sp.variant_game_counts.values()) or 1
                    for vid, cnt in sp.variant_game_counts.items():
                        run.track(cnt / total_vg, name="variant_game_frac", epoch=i,
                                  step=total_train_steps,
                                  context={"variant": UNIFIED_VARIANT_NAMES[vid], "vs": "self"})
                if sp.variant_win_rates:
                    for vid, vrates in sp.variant_win_rates.items():
                        vname = UNIFIED_VARIANT_NAMES[vid]
                        for j in range(len(vrates) - 1):
                            run.track(vrates[j], name="win_rate", epoch=i,
                                      step=total_train_steps,
                                      context={"vs": "self", "player": j + 1,
                                               "variant": vname, "from": "all_games"})
                        run.track(vrates[-1], name="draw_rate", epoch=i,
                                  step=total_train_steps,
                                  context={"vs": "self", "variant": vname, "from": "all_games"})
                _track_variant_play_metrics(run, sp.variant_metrics, epoch=i,
                                            step=total_train_steps, vs_context="self")
            stage_times["selfplay"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_symmetries"):
                exploit_symmetries(config, paths, i)
            stage_times["symmetries"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_resampling"):
                surprise_loss_arr = resample_by_surprise(config, paths, experiment_name, i)
            stage_times["resampling"] = time.time() - stage_start
            if len(surprise_loss_arr) > 0:
                try:
                    import aim as _aim
                    run.track(_aim.Distribution(surprise_loss_arr), name="surprise_loss",
                              epoch=i, step=total_train_steps)
                except Exception:
                    pass

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
            with TracyZone("stage_variant_analysis"):
                variant_stats = {}
                if _is_unified_game(config):
                    variant_stats = _analyze_iteration_variants(config, paths, experiment_name, i)
                    for vname, stats in variant_stats.items():
                        # Mean scalars for time-series trend lines
                        run.track(float(stats["pi_loss"].mean()), name="loss", epoch=i,
                                  step=total_train_steps,
                                  context={"type": "policy", "variant": vname})
                        run.track(float(stats["v_loss"].mean()), name="loss", epoch=i,
                                  step=total_train_steps,
                                  context={"type": "value", "variant": vname})
                        run.track(float(stats["entropy"].mean()), name="loss", epoch=i,
                                  step=total_train_steps,
                                  context={"type": "target_entropy", "variant": vname})
                        run.track(float(stats["top1"].mean()), name="policy_top1_mass", epoch=i,
                                  step=total_train_steps, context={"variant": vname})
                        # Distributions for spread/skew visibility
                        try:
                            import aim as _aim
                            run.track(_aim.Distribution(stats["pi_loss"]), name="pi_loss_dist",
                                      epoch=i, step=total_train_steps, context={"variant": vname})
                            run.track(_aim.Distribution(stats["v_loss"]), name="v_loss_dist",
                                      epoch=i, step=total_train_steps, context={"variant": vname})
                            run.track(_aim.Distribution(stats["entropy"]), name="entropy_dist",
                                      epoch=i, step=total_train_steps, context={"variant": vname})
                            run.track(_aim.Distribution(stats["top1"]), name="pi_top1_dist",
                                      epoch=i, step=total_train_steps, context={"variant": vname})
                        except Exception:
                            pass
            stage_times["variant_analysis"] = time.time() - stage_start

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
                gating_probs = _compute_gating_probs(config) if _is_unified_game(config) else None
                for gate_net in tqdm.tqdm(
                    panel, desc=f"Pitting against Panel {panel}", leave=False
                ):
                    nn_rate, draw_rate, _, game_length, gate_depth, gate_entropy, gate_mpt, gate_vm, variant_nn_rates, variant_draw_rates, variant_play_metrics = play_past(
                        config, paths, experiment_name,
                        config.compare_mcts_visits, next_net, gate_net, config.gate_compare_batch_size,
                        fallback_checkpoint_dir=fallback_checkpoint_dir,
                        variant_probs=gating_probs,
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
                        for vid, vrate in variant_nn_rates.items():
                            run.track(vrate, name="win_rate", epoch=next_net, step=total_train_steps,
                                      context={"vs": "best", "variant": UNIFIED_VARIANT_NAMES[vid]})
                            if wr_v is not None:
                                vadj = vrate + variant_draw_rates.get(vid, 0.0) / Game.NUM_PLAYERS()
                                wr_v[vid][next_net, gate_net] = vadj
                                wr_v[vid][gate_net, next_net] = 1.0 - vadj
                        for vid, drate in variant_draw_rates.items():
                            run.track(drate, name="draw_rate", epoch=next_net, step=total_train_steps,
                                      context={"vs": "best", "variant": UNIFIED_VARIANT_NAMES[vid]})
                        _track_variant_play_metrics(run, variant_play_metrics, epoch=next_net,
                                                    step=total_train_steps, vs_context="best")
                        best_win_rate = nn_rate + draw_rate / Game.NUM_PLAYERS()
                        postfix["vs best"] = _fmt_pct(best_win_rate)
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
                if len(panel) > 1:
                    postfix["vs panel"] = _fmt_pct(panel_win_rate)
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
                if wr_v is not None:
                    for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                        np.savetxt(os.path.join(experiment_dir, f"win_rate_{vname}.csv"), wr_v[vid], delimiter=",")
                        np.savetxt(os.path.join(experiment_dir, f"elo_{vname}.csv"), elo_v[vid], delimiter=",")
            stage_times["gating"] = time.time() - stage_start

            # Per-checkpoint diagnostics (Features 1, 3).
            # Load the new checkpoint once and run all enabled diagnostics.
            wants_frozen = bool(config.frozen_eval_anchor_iters)
            wants_rank = config.effective_rank_enabled
            if wants_frozen or wants_rank:
                stage_start = time.time()
                with TracyZone("stage_diagnostics"):
                    diag_nn = None
                    try:
                        import neural_net
                        diag_nn = neural_net.NNWrapper.load_checkpoint(
                            Game, paths["checkpoint"],
                            f"{next_net:04d}-{experiment_name}.pt",
                        )
                    except Exception as exc:
                        tqdm.tqdm.write(f"  Diagnostics warning [iter {next_net}]: failed to "
                                        f"load checkpoint: {exc}")

                    if diag_nn is not None and wants_rank:
                        try:
                            sample = _sample_recent_history_batch(
                                paths, config.effective_rank_batch_size,
                            )
                            if sample is not None:
                                pr = diag_nn.effective_rank(sample)
                                run.track(pr, name="effective_rank",
                                          epoch=next_net, step=total_train_steps,
                                          context={})
                        except Exception as exc:
                            tqdm.tqdm.write(f"  Effective rank warning [iter {next_net}]: {exc}")

                    if diag_nn is not None and wants_frozen:
                        for anchor in config.frozen_eval_anchor_iters:
                            snap_ok = frozen_eval.ensure_snapshot(
                                config, paths, experiment_name, anchor,
                                fallback_checkpoint_dir=fallback_checkpoint_dir,
                            )
                            if not (snap_ok and (i % max(1, config.frozen_eval_interval) == 0)):
                                continue
                            try:
                                metrics = frozen_eval.evaluate_checkpoint(
                                    diag_nn, config, paths, anchor,
                                )
                                frozen_eval.log_metrics_to_aim(
                                    run, metrics, anchor_iter=anchor,
                                    epoch=next_net, step=total_train_steps,
                                )
                                kls = [vm["kl_mcts_net"] for vm in metrics.values()
                                       if "kl_mcts_net" in vm]
                                if kls:
                                    mean_kl = sum(kls) / len(kls)
                                    frozen_eval.append_kl_history(
                                        paths, anchor, next_net, mean_kl,
                                    )
                                    hist = frozen_kl_history.setdefault(anchor, [])
                                    hist.append((next_net, mean_kl))
                                    if len(hist) > _FROZEN_KL_HIST_LEN:
                                        frozen_kl_history[anchor] = hist[-_FROZEN_KL_HIST_LEN:]
                                    frozen_eval.print_kl_health(
                                        run, frozen_kl_history[anchor],
                                        epoch=next_net, step=total_train_steps,
                                        anchor_iter=anchor,
                                    )
                            except Exception as exc:
                                tqdm.tqdm.write(f"  Frozen eval warning [iter {next_net}, "
                                                f"anchor {anchor}]: skipped: {exc}")

                    if diag_nn is not None:
                        del diag_nn
                    gc.collect()
                stage_times["diagnostics"] = time.time() - stage_start

            # Log variant_sample_frac at end of iteration from this iteration's actual data.
            if _is_unified_game(config):
                current_counts = _count_variant_samples(paths, i)
                if current_counts:
                    total_sc = sum(current_counts) or 1
                    for vi, vname in enumerate(UNIFIED_VARIANT_NAMES):
                        run.track(current_counts[vi] / total_sc, name="variant_sample_frac",
                                  epoch=i, step=total_train_steps, context={"variant": vname})

            stage_start = time.time()
            _log_win_rate_matrix(paths, i, run, total_train_steps)
            if _is_unified_game(config):
                _generate_visualizations(config, paths, i, run, total_train_steps, variant_stats)
            stage_times["visualizations"] = time.time() - stage_start

            total_time = time.time() - iteration_start
            for stage_name, stage_time in stage_times.items():
                percentage = (stage_time / total_time) * 100.0
                run.track(percentage, name="time_percent", epoch=i, step=total_train_steps, context={"stage": stage_name})
                run.track(stage_time, name="time_seconds", epoch=i, step=total_train_steps, context={"stage": stage_name})
            run.track(total_time, name="time_seconds", epoch=i, step=total_train_steps, context={"stage": "total"})

            tracy_frame()

    if hasattr(run, 'close'):
        run.close()
