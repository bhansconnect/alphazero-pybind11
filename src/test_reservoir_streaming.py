"""Behavioral equivalence tests for the streaming `update_reservoir` refactor.

The legacy version of `update_reservoir` (concat-then-sample) is preserved
verbatim in this file as `_legacy_update_reservoir`. The production code
in `game_runner.py` has been refactored to stream staging file-by-file
to bound peak memory. With a fixed RNG seed the two paths must produce
bitwise-identical reservoir chunks. These tests prove that.
"""

import glob
import os
import random
import shutil
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import game_runner
from game_runner import (
    calc_hist_size,
    glob_file_triples,
    load_compressed,
    save_compressed,
    update_reservoir,
    _load_chunk,
    _load_reservoir_meta,
    _migrate_legacy_reservoir,
    _save_chunk,
    _save_reservoir_meta,
)


# ---------------------------------------------------------------------------
# Legacy implementation (pre-refactor) — preserved verbatim from game_runner.
# Production code no longer contains this; tests use it as the ground truth.
# ---------------------------------------------------------------------------

def _legacy_load_staging_data(staging_dir):
    """Load all staging files, returning (c, v, pi, iters) concatenated tensors, or None if empty."""
    staging_triples = glob_file_triples(staging_dir)
    if not staging_triples:
        return None
    cs, vs, pis, iters_list = [], [], [], []
    for c_path, v_path, pi_path, size in staging_triples:
        cs.append(load_compressed(c_path))
        vs.append(load_compressed(v_path))
        pis.append(load_compressed(pi_path))
        iter_num = int(os.path.basename(c_path).split("-")[0])
        iters_list.append(torch.full((size,), iter_num, dtype=torch.int16))
    return torch.cat(cs), torch.cat(vs), torch.cat(pis), torch.cat(iters_list)


def _legacy_update_reservoir(config, paths, iteration, hist_size):
    """Pre-refactor monolithic-staging version. For equivalence testing only."""
    import gc
    import tqdm

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

    staging_data = _legacy_load_staging_data(staging_dir) if os.path.isdir(staging_dir) else None
    if staging_data is None:
        return

    staging_c, staging_v, staging_pi, staging_iters = staging_data
    total_staging = staging_c.shape[0]

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

    if chunks_filled < n_chunks:
        offset = 0
        while offset < total_staging and chunks_filled < n_chunks:
            end = min(offset + chunk_size, total_staging)
            chunk_c = staging_c[offset:end]
            chunk_v = staging_v[offset:end]
            chunk_pi = staging_pi[offset:end]
            chunk_iters = staging_iters[offset:end]
            _save_chunk(reservoir_location, chunks_filled,
                        chunk_c, chunk_v, chunk_pi, chunk_iters,
                        zstd_level=config.zstd_level)
            meta["last_updated"].append(iteration)
            meta["chunk_sizes"].append(end - offset)
            chunks_filled += 1
            offset = end
        meta["chunks_filled"] = chunks_filled

        if offset < total_staging:
            staging_c = staging_c[offset:]
            staging_v = staging_v[offset:]
            staging_pi = staging_pi[offset:]
            staging_iters = staging_iters[offset:]
            total_staging = staging_c.shape[0]
        else:
            total_staging = 0

    if chunks_filled >= n_chunks and total_staging > 0:
        K = min(config.reservoir_chunks_per_update, chunks_filled)
        last_updated = meta["last_updated"]
        target_indices = sorted(range(chunks_filled),
                                key=lambda i: last_updated[i])[:K]

        C = (n_chunks / K) * config.reservoir_update_interval
        decay = config.reservoir_recency_decay
        target_rate = 1 - decay ** C
        w_old_est = decay ** (C / 2)
        new_per_chunk = int(target_rate * chunk_size * w_old_est / (1 - target_rate))
        new_per_chunk = max(1, min(new_per_chunk, total_staging))

        for chunk_idx in tqdm.tqdm(target_indices, desc="Merging Reservoir Chunks", leave=False):
            if total_staging <= new_per_chunk:
                new_c, new_v, new_pi, new_iters = staging_c, staging_v, staging_pi, staging_iters
            else:
                perm = torch.randperm(total_staging)[:new_per_chunk]
                new_c = staging_c[perm]
                new_v = staging_v[perm]
                new_pi = staging_pi[perm]
                new_iters = staging_iters[perm]

            old_c, old_v, old_pi, old_iters = _load_chunk(reservoir_location, chunk_idx)

            pool_c = torch.cat([old_c, new_c])
            pool_v = torch.cat([old_v, new_v])
            pool_pi = torch.cat([old_pi, new_pi])
            pool_iters = torch.cat([old_iters, new_iters])
            del old_c, old_v, old_pi, old_iters

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
            del pool_c, pool_v, pool_pi, pool_iters, weights
            last_updated[chunk_idx] = iteration

        meta["last_updated"] = last_updated

    _save_reservoir_meta(reservoir_location, meta)

    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)

    del staging_c, staging_v, staging_pi, staging_iters
    gc.collect()


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _Config:
    """Minimal config carrying only the fields update_reservoir touches."""
    def __init__(self, **kwargs):
        # Defaults match TrainConfig for the relevant fields.
        self.reservoir_n_chunks = 5
        self.reservoir_chunk_size = 20
        self.reservoir_chunks_per_update = 3
        self.reservoir_update_interval = 1
        self.reservoir_recency_decay = 0.99
        self.zstd_level = 1
        self.half_storage = False
        # Window-size knobs for calc_hist_size. The merge path doesn't depend
        # on these unless evicted_iters is non-empty; tests pre-populate
        # staging directly so we set safe defaults.
        self.window_size_scalar = 6.0
        self.window_size_alpha = 0.5
        self.window_size_beta = 0.7
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_staging(staging_dir, num_files, samples_per_file, iter_offset=0, seed=42):
    """Populate `staging_dir` with `num_files` history triples.

    Each sample carries a deterministic value so off-by-one mix-ups are
    obvious. Returns the list of iteration numbers used.
    """
    os.makedirs(staging_dir, exist_ok=True)
    g = torch.Generator()
    g.manual_seed(seed)
    iter_nums = []
    for fi in range(num_files):
        iter_num = iter_offset + fi
        iter_nums.append(iter_num)
        # Small deterministic tensors; shape doesn't have to match a real game.
        c = torch.arange(samples_per_file * 4 * 5 * 5,
                         dtype=torch.float32).reshape(samples_per_file, 4, 5, 5)
        c += iter_num * 1000.0  # tag with iter so we can identify origin
        v = torch.rand((samples_per_file, 3), generator=g)
        pi = torch.rand((samples_per_file, 8), generator=g)
        # Filename encodes sample count as the last token (parsed by
        # `glob_file_triples` to determine each file's size).
        prefix = os.path.join(staging_dir, f"{iter_num:04d}-0000")
        save_compressed(c, f"{prefix}-canonical-{samples_per_file}.ptz",
                        half_storage=False, zstd_level=1)
        save_compressed(v, f"{prefix}-v-{samples_per_file}.ptz",
                        half_storage=False, zstd_level=1)
        save_compressed(pi, f"{prefix}-pi-{samples_per_file}.ptz",
                        half_storage=False, zstd_level=1)
    return iter_nums


def _build_reservoir(reservoir_dir, n_chunks_filled, chunk_size, seed=99):
    """Populate reservoir with `n_chunks_filled` existing chunks."""
    os.makedirs(reservoir_dir, exist_ok=True)
    g = torch.Generator()
    g.manual_seed(seed)
    last_updated = []
    chunk_sizes = []
    for ci in range(n_chunks_filled):
        c = torch.rand((chunk_size, 4, 5, 5), generator=g) - 100.0  # negative tag
        v = torch.rand((chunk_size, 3), generator=g)
        pi = torch.rand((chunk_size, 8), generator=g)
        iters = torch.full((chunk_size,), -ci - 1, dtype=torch.int16)
        _save_chunk(reservoir_dir, ci, c, v, pi, iters, zstd_level=1)
        last_updated.append(-ci)  # ensure they all look "stale"
        chunk_sizes.append(chunk_size)
    meta = {
        "version": 2,
        "n_chunks": 5,  # caller's n_chunks
        "chunk_size": chunk_size,
        "chunk_sizes": chunk_sizes,
        "chunks_filled": n_chunks_filled,
        "last_updated": last_updated,
    }
    _save_reservoir_meta(reservoir_dir, meta)


def _setup_two_dirs(num_staging_files, samples_per_file, n_chunks_filled,
                    chunk_size, base_dir):
    """Build two identical (reservoir + staging) trees rooted under base_dir."""
    dir_a = os.path.join(base_dir, "A")
    dir_b = os.path.join(base_dir, "B")
    for d in (dir_a, dir_b):
        _build_reservoir(d, n_chunks_filled, chunk_size)
        _build_staging(os.path.join(d, "staging"),
                       num_staging_files, samples_per_file)
    return dir_a, dir_b


def _seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def _read_all_chunks(reservoir_dir):
    """Read all chunks back from disk for byte-level comparison."""
    meta = _load_reservoir_meta(reservoir_dir)
    if meta is None:
        return None, None
    chunks = []
    for ci in range(meta["chunks_filled"]):
        c, v, pi, iters = _load_chunk(reservoir_dir, ci)
        chunks.append((c, v, pi, iters))
    return meta, chunks


def _assert_chunks_equal(meta_a, chunks_a, meta_b, chunks_b):
    # Metadata
    assert meta_a == meta_b, f"meta mismatch:\nA={meta_a}\nB={meta_b}"
    # Chunk tensors
    assert len(chunks_a) == len(chunks_b)
    for ci, ((ca, va, pia, ia), (cb, vb, pib, ib)) in enumerate(zip(chunks_a, chunks_b)):
        assert torch.equal(ca, cb), f"chunk {ci} canonical differs"
        assert torch.equal(va, vb), f"chunk {ci} value differs"
        assert torch.equal(pia, pib), f"chunk {ci} policy differs"
        assert torch.equal(ia, ib), f"chunk {ci} iters differs"


def _paths_for(reservoir_dir):
    return {
        "history": os.path.join(reservoir_dir, "history"),
        "reservoir": reservoir_dir,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_update_reservoir_full_merge_matches_legacy(tmp_path):
    """End-to-end: filling+merge path with leftover staging."""
    cfg = _Config(reservoir_n_chunks=3, reservoir_chunk_size=20,
                  reservoir_chunks_per_update=2, reservoir_update_interval=1)
    # 5 staging files × 20 samples = 100 staging samples.
    # 1 chunk already filled → need 2 more chunks (40 samples) → 60 left for merge.
    dir_a, dir_b = _setup_two_dirs(num_staging_files=5, samples_per_file=20,
                                    n_chunks_filled=1, chunk_size=20,
                                    base_dir=str(tmp_path))

    _seed_all(0)
    _legacy_update_reservoir(cfg, _paths_for(dir_a), iteration=1, hist_size=0)
    _seed_all(0)
    update_reservoir(cfg, _paths_for(dir_b), iteration=1, hist_size=0)

    meta_a, chunks_a = _read_all_chunks(dir_a)
    meta_b, chunks_b = _read_all_chunks(dir_b)
    _assert_chunks_equal(meta_a, chunks_a, meta_b, chunks_b)


def test_filling_phase_matches_legacy(tmp_path):
    """Only the filling phase runs (empty reservoir, enough staging to fill all)."""
    cfg = _Config(reservoir_n_chunks=3, reservoir_chunk_size=20,
                  reservoir_chunks_per_update=2, reservoir_update_interval=1)
    # 3 chunks × 20 samples = exactly 60 staging samples (no leftover).
    dir_a, dir_b = _setup_two_dirs(num_staging_files=3, samples_per_file=20,
                                    n_chunks_filled=0, chunk_size=20,
                                    base_dir=str(tmp_path))

    _seed_all(0)
    _legacy_update_reservoir(cfg, _paths_for(dir_a), iteration=1, hist_size=0)
    _seed_all(0)
    update_reservoir(cfg, _paths_for(dir_b), iteration=1, hist_size=0)

    meta_a, chunks_a = _read_all_chunks(dir_a)
    meta_b, chunks_b = _read_all_chunks(dir_b)
    _assert_chunks_equal(meta_a, chunks_a, meta_b, chunks_b)


def test_merge_phase_matches_legacy(tmp_path):
    """Only the merge phase runs (reservoir already full)."""
    cfg = _Config(reservoir_n_chunks=3, reservoir_chunk_size=20,
                  reservoir_chunks_per_update=3, reservoir_update_interval=1)
    # All 3 chunks pre-filled. Staging: 4 files × 25 = 100 samples, all for merge.
    dir_a, dir_b = _setup_two_dirs(num_staging_files=4, samples_per_file=25,
                                    n_chunks_filled=3, chunk_size=20,
                                    base_dir=str(tmp_path))

    _seed_all(0)
    _legacy_update_reservoir(cfg, _paths_for(dir_a), iteration=1, hist_size=0)
    _seed_all(0)
    update_reservoir(cfg, _paths_for(dir_b), iteration=1, hist_size=0)

    meta_a, chunks_a = _read_all_chunks(dir_a)
    meta_b, chunks_b = _read_all_chunks(dir_b)
    _assert_chunks_equal(meta_a, chunks_a, meta_b, chunks_b)


def test_empty_staging_is_noop(tmp_path):
    """Empty staging dir: both paths leave reservoir unchanged."""
    cfg = _Config(reservoir_n_chunks=3, reservoir_chunk_size=20,
                  reservoir_update_interval=1)
    dir_a = str(tmp_path / "A")
    dir_b = str(tmp_path / "B")
    for d in (dir_a, dir_b):
        _build_reservoir(d, n_chunks_filled=2, chunk_size=20)
        # Create empty staging dir
        os.makedirs(os.path.join(d, "staging"), exist_ok=True)

    _seed_all(0)
    _legacy_update_reservoir(cfg, _paths_for(dir_a), iteration=1, hist_size=0)
    _seed_all(0)
    update_reservoir(cfg, _paths_for(dir_b), iteration=1, hist_size=0)

    meta_a, chunks_a = _read_all_chunks(dir_a)
    meta_b, chunks_b = _read_all_chunks(dir_b)
    _assert_chunks_equal(meta_a, chunks_a, meta_b, chunks_b)


def test_partial_chunk_fill_handoff(tmp_path):
    """Staging has fewer samples than one full chunk; last chunk is partial."""
    cfg = _Config(reservoir_n_chunks=5, reservoir_chunk_size=20,
                  reservoir_chunks_per_update=2, reservoir_update_interval=1)
    # 2 staging files × 7 = 14 samples; chunk_size = 20.
    # Filling: writes 1 partial chunk (size 14), no leftover, no merge.
    dir_a, dir_b = _setup_two_dirs(num_staging_files=2, samples_per_file=7,
                                    n_chunks_filled=0, chunk_size=20,
                                    base_dir=str(tmp_path))

    _seed_all(0)
    _legacy_update_reservoir(cfg, _paths_for(dir_a), iteration=1, hist_size=0)
    _seed_all(0)
    update_reservoir(cfg, _paths_for(dir_b), iteration=1, hist_size=0)

    meta_a, chunks_a = _read_all_chunks(dir_a)
    meta_b, chunks_b = _read_all_chunks(dir_b)
    _assert_chunks_equal(meta_a, chunks_a, meta_b, chunks_b)


def test_filling_with_chunk_size_not_aligned_to_files(tmp_path):
    """Staging file boundaries don't align to chunk boundaries — exercises the
    cross-file slicing in the streaming fill loop."""
    cfg = _Config(reservoir_n_chunks=4, reservoir_chunk_size=20,
                  reservoir_chunks_per_update=2, reservoir_update_interval=1)
    # 5 files × 13 = 65 samples; chunk_size = 20 → 3 full + leftover 5.
    dir_a, dir_b = _setup_two_dirs(num_staging_files=5, samples_per_file=13,
                                    n_chunks_filled=0, chunk_size=20,
                                    base_dir=str(tmp_path))

    _seed_all(0)
    _legacy_update_reservoir(cfg, _paths_for(dir_a), iteration=1, hist_size=0)
    _seed_all(0)
    update_reservoir(cfg, _paths_for(dir_b), iteration=1, hist_size=0)

    meta_a, chunks_a = _read_all_chunks(dir_a)
    meta_b, chunks_b = _read_all_chunks(dir_b)
    _assert_chunks_equal(meta_a, chunks_a, meta_b, chunks_b)
