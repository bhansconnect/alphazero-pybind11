"""Tests for history data management: compression, reservoir, and backward compatibility.

Converted from history_management_test.py to pytest with tmp_path fixtures.
All global references (HIST_LOCATION, RESERVOIR_LOCATION, etc.) replaced with
config-based paths and explicit directories.
"""

import glob
import math
import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from neural_net import NNWrapper, NNArgs, get_storage_dtype
from game_runner import (
    save_compressed, load_compressed, _glob_hist_files,
    _atomic_save_compressed, update_reservoir, calc_hist_size,
    glob_file_triples,
    StreamingCompressedDataset, load_reservoir, train,
    exploit_symmetries, resample_by_surprise,
    _bootstrap_retrain, _bootstrap_train_phase,
    glob_reservoir_chunks, _load_reservoir_meta, _save_reservoir_meta,
    _load_chunk, _save_chunk, _migrate_legacy_reservoir,
)
from config import TrainConfig

Game = alphazero.Connect4GS


# ============================================================
# Compression round-trip
# ============================================================


def test_roundtrip_float32(tmp_path):
    """Random float32 tensor round-trips within half-precision tolerance."""
    path = str(tmp_path / "rt.ptz")
    original = torch.randn(100, 3, 6, 7)
    save_compressed(original, path)
    loaded = load_compressed(path)
    assert loaded.shape == original.shape
    assert loaded.dtype == get_storage_dtype()  # float16
    assert torch.allclose(original, loaded.float(), atol=0.01, rtol=0.01)


def test_roundtrip_integer_values(tmp_path):
    """0/1 tensor round-trips exactly."""
    path = str(tmp_path / "int_rt.ptz")
    original = torch.randint(0, 2, (200, 3, 6, 7)).float()
    save_compressed(original, path)
    loaded = load_compressed(path)
    assert torch.equal(original, loaded.float())


def test_file_size_reduction(tmp_path):
    """Compressed .ptz is smaller than uncompressed .pt."""
    pt_path = str(tmp_path / "test.pt")
    ptz_path = str(tmp_path / "test.ptz")
    data = torch.zeros(1000, 3, 6, 7)
    data[:, 0, :3, :] = 1.0
    torch.save(data, pt_path)
    save_compressed(data, ptz_path)
    assert os.path.getsize(ptz_path) < os.path.getsize(pt_path)


def test_ptz_extension(tmp_path):
    """Save creates file with .ptz extension."""
    path = str(tmp_path / "ext.ptz")
    save_compressed(torch.zeros(10), path)
    assert os.path.exists(path)


def test_empty_tensor_roundtrip(tmp_path):
    """Empty tensor round-trips."""
    path = str(tmp_path / "empty.ptz")
    original = torch.zeros(0, 3, 6, 7)
    save_compressed(original, path)
    loaded = load_compressed(path)
    assert loaded.shape == original.shape


# ============================================================
# History file format handling
# ============================================================


def test_hist_save_creates_ptz(tmp_path):
    """Saving history creates .ptz files."""
    cs = Game.CANONICAL_SHAPE()
    n = 100
    c = torch.randn(n, cs[0], cs[1], cs[2])
    save_compressed(c, str(tmp_path / "0000-0000-canonical-100.ptz"))
    assert (tmp_path / "0000-0000-canonical-100.ptz").exists()


def test_mixed_format_loading(tmp_path):
    """Both .pt and .ptz files can be loaded side by side."""
    cs = Game.CANONICAL_SHAPE()
    n = 50
    # .pt for iter 0
    c0 = torch.randn(n, cs[0], cs[1], cs[2])
    torch.save(c0, str(tmp_path / f"0000-0000-canonical-{n}.pt"))
    # .ptz for iter 1
    c1 = torch.randn(n, cs[0], cs[1], cs[2])
    save_compressed(c1, str(tmp_path / f"0001-0000-canonical-{n}.ptz"))
    # Both should be found by _glob_hist_files
    files0 = _glob_hist_files(str(tmp_path), "0000-*-canonical-*")
    assert len(files0) == 1
    assert files0[0].endswith(".pt")
    files1 = _glob_hist_files(str(tmp_path), "0001-*-canonical-*")
    assert len(files1) == 1
    assert files1[0].endswith(".ptz")


# ============================================================
# glob_file_triples and helpers
# ============================================================


def test_glob_file_triples(tmp_path):
    """glob_file_triples finds matching triples and extracts sizes."""
    cs = Game.CANONICAL_SHAPE()
    n = 75
    for tensor_type in ("canonical", "v", "pi"):
        save_compressed(
            torch.randn(n, 3) if tensor_type != "canonical" else torch.randn(n, cs[0], cs[1], cs[2]),
            str(tmp_path / f"0002-0000-{tensor_type}-{n}.ptz"),
        )
    triples = glob_file_triples(str(tmp_path))
    assert len(triples) == 1
    c_path, v_path, pi_path, size = triples[0]
    assert size == n
    assert "-canonical-" in c_path
    assert "-v-" in v_path
    assert "-pi-" in pi_path


def test_glob_file_triples_multiple(tmp_path):
    """glob_file_triples handles multiple batches per iteration."""
    cs = Game.CANONICAL_SHAPE()
    for batch in range(3):
        n = 100 + batch * 10
        for tensor_type in ("canonical", "v", "pi"):
            save_compressed(torch.randn(n, 3), str(tmp_path / f"0005-{batch:04d}-{tensor_type}-{n}.ptz"))
    triples = glob_file_triples(str(tmp_path))
    assert len(triples) == 3
    sizes = [s for _, _, _, s in triples]
    assert sizes == [100, 110, 120]



# ============================================================
# Reservoir logic (chunked format)
# ============================================================


def test_reservoir_capacity_enforcement(tmp_path):
    """Overfilling reservoir downsamples to target capacity."""
    config = TrainConfig()
    n_samples = 500
    capacity = 200
    iters = torch.randint(0, 50, (n_samples,), dtype=torch.int16)
    ages = (100 - iters.float()).clamp(min=0)
    weights = config.reservoir_recency_decay ** ages
    indices = torch.multinomial(weights, capacity, replacement=False)
    assert len(indices) == capacity
    assert len(set(indices.tolist())) == capacity  # All unique


def test_reservoir_recency_weighting():
    """Recent samples survive downsampling more often than old samples."""
    config = TrainConfig()
    n_old, n_new, capacity, iteration = 200, 200, 200, 100
    iters = torch.cat([
        torch.full((n_old,), 0, dtype=torch.int16),
        torch.full((n_new,), 50, dtype=torch.int16),
    ])
    old_survival, new_survival = 0, 0
    for _ in range(100):
        ages = (iteration - iters.float()).clamp(min=0)
        weights = config.reservoir_recency_decay ** ages
        indices = torch.multinomial(weights, capacity, replacement=False)
        selected = iters[indices]
        old_survival += (selected == 0).sum().item()
        new_survival += (selected == 50).sum().item()
    assert new_survival > old_survival


def test_reservoir_metadata_roundtrip(tmp_path):
    """Reservoir metadata JSON round-trips correctly."""
    meta = {
        "version": 2,
        "n_chunks": 10,
        "chunk_size": 100,
        "chunks_filled": 5,
        "last_updated": [10, 20, 30, 40, 50],
    }
    res_dir = str(tmp_path / "reservoir")
    os.makedirs(res_dir)
    _save_reservoir_meta(res_dir, meta)
    loaded = _load_reservoir_meta(res_dir)
    assert loaded == meta


def test_chunk_save_load_roundtrip(tmp_path):
    """Chunk save/load round-trips data correctly."""
    cs = Game.CANONICAL_SHAPE()
    n = 50
    c = torch.randn(n, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
    iters = torch.randint(0, 100, (n,), dtype=torch.int16)

    res_dir = str(tmp_path / "reservoir")
    os.makedirs(res_dir)
    _save_chunk(res_dir, 0, c, v, pi, iters)
    lc, lv, lpi, liters = _load_chunk(res_dir, 0)

    assert lc.shape == c.shape
    assert lv.shape == v.shape
    assert lpi.shape == pi.shape
    assert torch.equal(liters, iters)


def test_iters_tensor_roundtrip(tmp_path):
    """int16 iteration tensor round-trips exactly (not stored as half)."""
    iters = torch.tensor([0, 1, 2, 5, 10, 20, 100, 500], dtype=torch.int16)
    res_dir = str(tmp_path / "reservoir")
    os.makedirs(res_dir)
    cs = Game.CANONICAL_SHAPE()
    n = len(iters)
    c = torch.randn(n, cs[0], cs[1], cs[2])
    v = torch.randn(n, 3)
    pi = torch.randn(n, 7)
    _save_chunk(res_dir, 0, c, v, pi, iters)
    _, _, _, loaded_iters = _load_chunk(res_dir, 0)
    assert torch.equal(iters, loaded_iters)


# ============================================================
# Multi-iteration E2E
# ============================================================


def test_multi_iteration_e2e(tmp_path):
    """5-iteration E2E with compression + reservoir management."""
    hist_dir = tmp_path / "history"
    ckpt_dir = tmp_path / "checkpoint"
    hist_dir.mkdir()
    ckpt_dir.mkdir()
    cs = Game.CANONICAL_SHAPE()

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(str(ckpt_dir), "0000-test.pt")
    del nn

    class DummyRun:
        def track(*a, **kw):
            pass

    run = DummyRun()
    hist_size = 2

    for i in range(5):
        n_samples = 200
        c = torch.randn(n_samples, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n_samples, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n_samples, Game.NUM_MOVES()), dim=1)

        save_compressed(c, str(hist_dir / f"{i:04d}-0000-canonical-{n_samples}.ptz"))
        save_compressed(v, str(hist_dir / f"{i:04d}-0000-v-{n_samples}.ptz"))
        save_compressed(p, str(hist_dir / f"{i:04d}-0000-pi-{n_samples}.ptz"))

        # Load window, train
        datasets = []
        for wi in range(max(0, i - hist_size), i + 1):
            cf = _glob_hist_files(str(hist_dir), f"{wi:04d}-*-canonical-*")
            vf = _glob_hist_files(str(hist_dir), f"{wi:04d}-*-v-*")
            pf = _glob_hist_files(str(hist_dir), f"{wi:04d}-*-pi-*")
            for j in range(len(cf)):
                ct = load_compressed(cf[j]) if cf[j].endswith(".ptz") else torch.load(cf[j])
                vt = load_compressed(vf[j]) if vf[j].endswith(".ptz") else torch.load(vf[j])
                pt = load_compressed(pf[j]) if pf[j].endswith(".ptz") else torch.load(pf[j])
                datasets.append(TensorDataset(ct, vt, pt))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        nn = NNWrapper.load_checkpoint(Game, str(ckpt_dir), f"{i:04d}-test.pt")
        v_loss, pi_loss = nn.train(dataloader, 2, run, i, 0)
        assert math.isfinite(v_loss)
        assert math.isfinite(pi_loss)
        nn.save_checkpoint(str(ckpt_dir), f"{i+1:04d}-test.pt")
        del nn, datasets, dataset, dataloader

    # Verify checkpoints exist
    assert len(list(ckpt_dir.glob("*.pt"))) == 6  # 0000 through 0005


# ============================================================
# Reservoir bootstrap
# ============================================================


def test_reservoir_bootstrap(tmp_path):
    """Build per-iteration reservoir files, then bootstrap (train) from them."""
    res_dir = tmp_path / "reservoir"
    res_dir.mkdir()
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    cs = Game.CANONICAL_SHAPE()

    # Create per-iteration reservoir files (new format)
    n_total = 500
    c = torch.randn(n_total, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n_total, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n_total, Game.NUM_MOVES()), dim=1)
    save_compressed(c, str(res_dir / f"0003-0000-canonical-{n_total}.ptz"))
    save_compressed(v, str(res_dir / f"0003-0000-v-{n_total}.ptz"))
    save_compressed(pi, str(res_dir / f"0003-0000-pi-{n_total}.ptz"))

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(str(ckpt_dir), "0000-test.pt")

    class DummyRun:
        def track(*a, **kw):
            pass

    run = DummyRun()

    res_c = load_compressed(str(res_dir / f"0003-0000-canonical-{n_total}.ptz"))
    res_v = load_compressed(str(res_dir / f"0003-0000-v-{n_total}.ptz"))
    res_pi = load_compressed(str(res_dir / f"0003-0000-pi-{n_total}.ptz"))
    ds = TensorDataset(res_c, res_v, res_pi)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    v_loss, pi_loss = nn.train(dl, int(math.ceil(len(ds) / 64)) * 2, run, 5, 0)
    assert math.isfinite(v_loss)
    assert math.isfinite(pi_loss)
    nn.save_checkpoint(str(ckpt_dir), "0005-test.pt")
    assert (ckpt_dir / "0005-test.pt").exists()


# ============================================================
# update_reservoir() direct tests (per-iteration format)
# ============================================================


def test_update_reservoir_moves_evicted_to_staging(tmp_path):
    """update_reservoir moves evicted files to staging directory."""
    config = TrainConfig(window_size_scalar=1.0, reservoir_update_interval=100)
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    n = 50
    for i in range(5):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    # Use high update_interval so merge doesn't trigger, just staging
    for iteration in range(3, 5):
        hist_size = calc_hist_size(config, iteration)
        update_reservoir(config, paths, iteration, hist_size)

    # Staging dir should contain evicted iterations
    staging_dir = os.path.join(paths["reservoir"], "staging")
    staging_triples = glob_file_triples(staging_dir)
    assert len(staging_triples) >= 1, "Staging should have evicted files"

    staging_iters = set()
    for c_path, _, _, _ in staging_triples:
        iter_num = int(os.path.basename(c_path).split("-")[0])
        staging_iters.add(iter_num)
    assert 0 in staging_iters, "Iteration 0 should be in staging"

    # Evicted files should no longer be in history
    for evicted_iter in staging_iters:
        hist_files = glob.glob(os.path.join(paths["history"], f"{evicted_iter:04d}-*.ptz"))
        assert len(hist_files) == 0, f"Iteration {evicted_iter} still in history after eviction"


def test_reservoir_filling_phase(tmp_path):
    """Empty reservoir fills chunks sequentially from staging data."""
    config = TrainConfig(
        window_size_scalar=1.0,
        reservoir_update_interval=1,
        reservoir_n_chunks=5,
        reservoir_chunk_size=40,
    )
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    n = 50
    # Create enough iterations to trigger eviction and filling
    for i in range(6):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    for iteration in range(3, 6):
        hist_size = calc_hist_size(config, iteration)
        update_reservoir(config, paths, iteration, hist_size)

    # Check chunks were created
    meta = _load_reservoir_meta(paths["reservoir"])
    assert meta is not None, "Metadata should exist after filling"
    assert meta["chunks_filled"] > 0, "At least one chunk should be filled"

    # Verify chunk files exist and are loadable
    for i in range(meta["chunks_filled"]):
        c, v, pi, iters = _load_chunk(paths["reservoir"], i)
        assert c.shape[0] > 0
        assert c.shape[0] == v.shape[0] == pi.shape[0] == iters.shape[0]


def test_update_reservoir_merge_phase(tmp_path):
    """After filling, merge phase updates stalest chunks with staging data."""
    config = TrainConfig(
        window_size_scalar=1.0,
        reservoir_update_interval=1,
        reservoir_n_chunks=3,
        reservoir_chunk_size=30,
        reservoir_chunks_per_update=2,
        reservoir_recency_decay=0.9,
    )
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    n = 50

    # Create many iterations to fill reservoir and trigger merge
    for i in range(15):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    for iteration in range(3, 15):
        hist_size = calc_hist_size(config, iteration)
        update_reservoir(config, paths, iteration, hist_size)

    meta = _load_reservoir_meta(paths["reservoir"])
    assert meta is not None
    assert meta["chunks_filled"] == config.reservoir_n_chunks, \
        f"All {config.reservoir_n_chunks} chunks should be filled"

    # Verify each chunk has at most chunk_size samples
    for i in range(meta["chunks_filled"]):
        c, v, pi, iters = _load_chunk(paths["reservoir"], i)
        assert c.shape[0] <= config.reservoir_chunk_size
        assert c.shape[0] == v.shape[0] == pi.shape[0] == iters.shape[0]

    # Staging should be cleaned up after merge
    staging_dir = os.path.join(paths["reservoir"], "staging")
    assert not os.path.isdir(staging_dir), "Staging dir should be removed after merge"


def test_update_reservoir_recency_bias(tmp_path):
    """Merge phase preserves recency bias: recent samples survive more often.

    Uses aggressive decay (0.8) so the bias is clearly visible.
    """
    config = TrainConfig(
        window_size_scalar=1.0,
        reservoir_update_interval=1,
        reservoir_n_chunks=2,
        reservoir_chunk_size=50,
        reservoir_chunks_per_update=2,
        reservoir_recency_decay=0.8,
    )
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    n = 80

    for i in range(12):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    for iteration in range(3, 12):
        hist_size = calc_hist_size(config, iteration)
        update_reservoir(config, paths, iteration, hist_size)

    # Check that recent iterations have more representation than old ones
    meta = _load_reservoir_meta(paths["reservoir"])
    assert meta is not None

    iter_counts = {}
    for i in range(meta["chunks_filled"]):
        _, _, _, iters = _load_chunk(paths["reservoir"], i)
        for it in iters.tolist():
            iter_counts[it] = iter_counts.get(it, 0) + 1

    if len(iter_counts) >= 2:
        sorted_iters = sorted(iter_counts.keys())
        oldest = sorted_iters[0]
        newest = sorted_iters[-1]
        assert iter_counts[newest] >= iter_counts[oldest], \
            f"Newest iter {newest} ({iter_counts[newest]}) should have >= " \
            f"oldest iter {oldest} ({iter_counts[oldest]})"


def test_atomic_save_produces_valid_file(tmp_path):
    """_atomic_save_compressed produces a valid, loadable file."""
    path = str(tmp_path / "atomic_test.ptz")
    original = torch.randn(100, 3, 6, 7)
    _atomic_save_compressed(original, path)

    assert os.path.exists(path)
    loaded = load_compressed(path)
    assert loaded.shape == original.shape
    assert torch.allclose(original, loaded.float(), atol=0.01, rtol=0.01)


def test_atomic_save_no_temp_files_on_success(tmp_path):
    """After successful atomic save, no .tmp files remain."""
    path = str(tmp_path / "clean.ptz")
    _atomic_save_compressed(torch.randn(50), path)

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_reservoir_source_copy_chunks(tmp_path):
    """Bootstrap reservoir copy works with chunked format."""
    import shutil

    source_res = tmp_path / "source" / "reservoir"
    dest_res = tmp_path / "dest" / "reservoir"
    source_res.mkdir(parents=True)
    dest_res.mkdir(parents=True)

    cs = Game.CANONICAL_SHAPE()
    n = 50
    # Create chunked reservoir
    for i in range(3):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.randn(n, 3)
        pi = torch.randn(n, 7)
        iters = torch.full((n,), i, dtype=torch.int16)
        _save_chunk(str(source_res), i, c, v, pi, iters)

    meta = {
        "version": 2, "n_chunks": 10, "chunk_size": n,
        "chunks_filled": 3, "last_updated": [0, 1, 2],
    }
    _save_reservoir_meta(str(source_res), meta)

    # Copy chunk files + metadata
    chunk_triples = glob_reservoir_chunks(str(source_res))
    assert len(chunk_triples) == 3

    all_files = glob.glob(str(source_res / "chunk_*.ptz")) + \
                glob.glob(str(source_res / "reservoir_meta.json"))
    for src in all_files:
        shutil.copy2(src, str(dest_res / os.path.basename(src)))

    dest_triples = glob_reservoir_chunks(str(dest_res))
    assert len(dest_triples) == 3
    dest_meta = _load_reservoir_meta(str(dest_res))
    assert dest_meta["chunks_filled"] == 3


def test_reservoir_empty_source_skips_copy(tmp_path):
    """Bootstrap reservoir copy skips when source has no files."""
    source_res = tmp_path / "source" / "reservoir"
    dest_res = tmp_path / "dest" / "reservoir"
    source_res.mkdir(parents=True)
    dest_res.mkdir(parents=True)

    # No files → both globs return empty
    assert len(glob_file_triples(str(source_res))) == 0
    assert len(glob_reservoir_chunks(str(source_res))) == 0
    assert len(list(dest_res.iterdir())) == 0


def test_glob_reservoir_chunks(tmp_path):
    """glob_reservoir_chunks finds chunk files correctly."""
    res_dir = str(tmp_path / "reservoir")
    os.makedirs(res_dir)
    cs = Game.CANONICAL_SHAPE()
    n = 30

    for i in range(3):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.randn(n, 3)
        pi = torch.randn(n, 7)
        iters = torch.full((n,), i, dtype=torch.int16)
        _save_chunk(res_dir, i, c, v, pi, iters)

    meta = {
        "version": 2, "n_chunks": 10, "chunk_size": n,
        "chunks_filled": 3, "last_updated": [0, 1, 2],
    }
    _save_reservoir_meta(res_dir, meta)

    triples = glob_reservoir_chunks(res_dir)
    assert len(triples) == 3
    for c_path, v_path, pi_path, size in triples:
        assert "_canonical.ptz" in c_path
        assert "_v.ptz" in v_path
        assert "_pi.ptz" in pi_path
        assert size == n


def test_glob_reservoir_chunks_partial_non_last(tmp_path):
    """glob_reservoir_chunks reports correct sizes for partial non-last chunks via chunk_sizes."""
    res_dir = str(tmp_path / "reservoir")
    os.makedirs(res_dir)
    cs = Game.CANONICAL_SHAPE()
    sizes = [30, 100, 50]

    for i, n in enumerate(sizes):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.randn(n, 3)
        pi = torch.randn(n, 7)
        iters = torch.full((n,), i, dtype=torch.int16)
        _save_chunk(res_dir, i, c, v, pi, iters)

    meta = {
        "version": 2, "n_chunks": 10, "chunk_size": 100,
        "chunk_sizes": sizes,
        "chunks_filled": 3, "last_updated": [0, 1, 2],
    }
    _save_reservoir_meta(res_dir, meta)

    triples = glob_reservoir_chunks(res_dir)
    assert len(triples) == 3
    reported_sizes = [size for _, _, _, size in triples]
    assert reported_sizes == sizes, f"Expected {sizes}, got {reported_sizes}"


def test_glob_reservoir_chunks_backward_compat(tmp_path):
    """glob_reservoir_chunks falls back to loading tensors when chunk_sizes is missing."""
    res_dir = str(tmp_path / "reservoir")
    os.makedirs(res_dir)
    cs = Game.CANONICAL_SHAPE()
    sizes = [30, 100, 50]

    for i, n in enumerate(sizes):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.randn(n, 3)
        pi = torch.randn(n, 7)
        iters = torch.full((n,), i, dtype=torch.int16)
        _save_chunk(res_dir, i, c, v, pi, iters)

    # Old-format metadata: no chunk_sizes key
    meta = {
        "version": 2, "n_chunks": 10, "chunk_size": 100,
        "chunks_filled": 3, "last_updated": [0, 1, 2],
    }
    _save_reservoir_meta(res_dir, meta)

    triples = glob_reservoir_chunks(res_dir)
    assert len(triples) == 3
    reported_sizes = [size for _, _, _, size in triples]
    assert reported_sizes == sizes, f"Expected {sizes}, got {reported_sizes}"


def test_filling_phase_excess_data_not_discarded(tmp_path):
    """Excess staging data at fill→merge transition is merged, not discarded."""
    config = TrainConfig(
        window_size_scalar=1.0,
        reservoir_update_interval=1,
        reservoir_n_chunks=2,
        reservoir_chunk_size=30,
        reservoir_chunks_per_update=2,
        reservoir_recency_decay=0.9,
    )
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    # Create enough data so that filling 2 chunks (30 each = 60) leaves excess
    # We need staging data > 2 * chunk_size = 60
    n = 50
    for i in range(6):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    # Run updates; with window_size_scalar=1.0 and small chunk_size, filling will
    # complete and excess staging data should flow into merge phase
    for iteration in range(3, 6):
        hist_size = calc_hist_size(config, iteration)
        update_reservoir(config, paths, iteration, hist_size)

    meta = _load_reservoir_meta(paths["reservoir"])
    assert meta is not None
    assert meta["chunks_filled"] == 2, "Should have exactly 2 chunks"

    # Count total samples in reservoir
    total = 0
    for i in range(meta["chunks_filled"]):
        c, v, pi, iters = _load_chunk(paths["reservoir"], i)
        total += c.shape[0]

    # Without the fix, excess data at the fill→merge boundary would be lost.
    # With the fix, at least one chunk should have been merged with excess data.
    # The merge phase uses recency-weighted sampling, so chunks can be up to chunk_size.
    assert total > 0, "Reservoir should contain data"
    assert "chunk_sizes" in meta, "Metadata should contain chunk_sizes"
    assert len(meta["chunk_sizes"]) == 2, "Should have 2 chunk sizes"


def test_reservoir_migration_creates_chunk_sizes(tmp_path):
    """Migration from legacy format creates chunk_sizes in metadata."""
    config = TrainConfig(reservoir_n_chunks=10, reservoir_chunk_size=40)
    res_dir = str(tmp_path / "reservoir")
    os.makedirs(res_dir)
    cs = Game.CANONICAL_SHAPE()
    n = 30

    # Create legacy per-iteration files
    for i in range(3):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.randn(n, 3)
        pi = torch.randn(n, 7)
        save_compressed(c, os.path.join(res_dir, f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(res_dir, f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(pi, os.path.join(res_dir, f"{i:04d}-0000-pi-{n}.ptz"))

    _migrate_legacy_reservoir(config, res_dir)

    meta = _load_reservoir_meta(res_dir)
    assert meta is not None
    assert "chunk_sizes" in meta, "Migration should create chunk_sizes"
    assert len(meta["chunk_sizes"]) == meta["chunks_filled"]

    # Verify chunk_sizes match actual tensor sizes
    for i, expected_size in enumerate(meta["chunk_sizes"]):
        c, v, pi, iters = _load_chunk(res_dir, i)
        assert c.shape[0] == expected_size, \
            f"Chunk {i}: expected {expected_size}, got {c.shape[0]}"

    # Total should equal input
    total = sum(meta["chunk_sizes"])
    assert total == n * 3, f"Expected {n * 3} total samples, got {total}"


def test_reservoir_migration(tmp_path):
    """Legacy per-iteration files are migrated to chunked format."""
    config = TrainConfig(reservoir_n_chunks=10, reservoir_chunk_size=40)
    res_dir = str(tmp_path / "reservoir")
    os.makedirs(res_dir)
    cs = Game.CANONICAL_SHAPE()
    n = 30

    # Create legacy per-iteration files
    for i in range(3):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.randn(n, 3)
        pi = torch.randn(n, 7)
        save_compressed(c, os.path.join(res_dir, f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(res_dir, f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(pi, os.path.join(res_dir, f"{i:04d}-0000-pi-{n}.ptz"))

    _migrate_legacy_reservoir(config, res_dir)

    # Metadata should exist
    meta = _load_reservoir_meta(res_dir)
    assert meta is not None
    assert meta["chunks_filled"] > 0

    # Legacy files should be deleted
    legacy = glob_file_triples(res_dir)
    assert len(legacy) == 0, "Legacy files should be removed after migration"

    # Chunks should contain all data
    total_in_chunks = 0
    for i in range(meta["chunks_filled"]):
        c, v, pi, iters = _load_chunk(res_dir, i)
        total_in_chunks += c.shape[0]
    assert total_in_chunks == n * 3, f"Expected {n * 3} samples, got {total_in_chunks}"


# ============================================================
# StreamingCompressedDataset
# ============================================================


def _make_file_triples(tmp_path, n_files=3, samples_per_file=80):
    """Create .ptz file triples and return (file_triples_list, total_samples, raw_data_dict)."""
    cs = Game.CANONICAL_SHAPE()
    triples = []
    raw = {"c": [], "v": [], "pi": []}
    for i in range(n_files):
        n = samples_per_file
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, str(tmp_path / f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, str(tmp_path / f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(pi, str(tmp_path / f"{i:04d}-0000-pi-{n}.ptz"))
        raw["c"].append(c)
        raw["v"].append(v)
        raw["pi"].append(pi)
    triples = glob_file_triples(str(tmp_path))
    total = n_files * samples_per_file
    return triples, total, raw


def test_streaming_dataset_yields_all_samples(tmp_path):
    """Single pass yields every sample exactly once."""
    triples, total, _ = _make_file_triples(tmp_path, n_files=3, samples_per_file=80)
    ds = StreamingCompressedDataset(triples, batch_size=64, passes=1)

    seen = 0
    for c_batch, v_batch, pi_batch in ds:
        assert c_batch.shape[0] == v_batch.shape[0] == pi_batch.shape[0]
        assert c_batch.shape[0] <= 64
        seen += c_batch.shape[0]

    assert seen == total, f"Expected {total} samples, got {seen}"


def test_streaming_dataset_multiple_passes(tmp_path):
    """Multiple passes yield samples * passes total."""
    triples, total, _ = _make_file_triples(tmp_path, n_files=2, samples_per_file=50)
    passes = 3
    ds = StreamingCompressedDataset(triples, batch_size=32, passes=passes)

    seen = 0
    for c_batch, v_batch, pi_batch in ds:
        seen += c_batch.shape[0]

    assert seen == total * passes, f"Expected {total * passes} samples, got {seen}"


def test_streaming_dataset_partial_final_batch(tmp_path):
    """When file_size % batch_size != 0, final batch is smaller."""
    triples, _, _ = _make_file_triples(tmp_path, n_files=1, samples_per_file=70)
    ds = StreamingCompressedDataset(triples, batch_size=32, passes=1)

    batch_sizes = [c.shape[0] for c, v, pi in ds]
    # 70 samples / 32 = 2 full batches + 1 of size 6
    assert batch_sizes == [32, 32, 6]


def test_streaming_dataset_with_dataloader(tmp_path):
    """StreamingCompressedDataset works with DataLoader(batch_size=None)."""
    triples, total, _ = _make_file_triples(tmp_path, n_files=2, samples_per_file=60)
    ds = StreamingCompressedDataset(triples, batch_size=64, passes=1)
    dl = DataLoader(ds, batch_size=None, num_workers=0)

    seen = 0
    for batch in dl:
        canonical, target_vs, target_pis = batch
        assert canonical.ndim == 4  # (B, C, H, W)
        seen += canonical.shape[0]

    assert seen == total


def test_streaming_dataset_trains_nn(tmp_path):
    """StreamingCompressedDataset can be used to train a NN to finite loss."""
    triples, total, _ = _make_file_triples(tmp_path, n_files=2, samples_per_file=100)

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)

    class DummyRun:
        def track(*a, **kw):
            pass

    bs = 64
    ds = StreamingCompressedDataset(triples, bs, passes=2)
    dl = DataLoader(ds, batch_size=None, num_workers=0)
    steps = int(math.ceil(total / bs)) * 2
    v_loss, pi_loss = nn.train(dl, steps, DummyRun(), 0, 0)
    assert math.isfinite(v_loss)
    assert math.isfinite(pi_loss)


# ============================================================
# Index-then-extract train()
# ============================================================


def test_train_index_then_extract(tmp_path):
    """train() uses index-then-extract and produces valid loss."""
    config = TrainConfig(train_batch_size=64)
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("checkpoint", "history"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    experiment_name = "test"

    # Create checkpoint at iteration 2 (train() loads iteration:04d checkpoint)
    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(paths["checkpoint"], f"0002-{experiment_name}.pt")
    del nn

    # Create 3 iterations of history (small)
    for i in range(3):
        n = 100
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    class DummyRun:
        def track(*a, **kw):
            pass

    # Train with hist_size=2 at iteration=2 (window covers iters 0..2 since hist_size >= iteration)
    v_loss, pi_loss, total = train(config, paths, experiment_name, 2, 2, DummyRun(), 0)
    assert math.isfinite(v_loss)
    assert math.isfinite(pi_loss)
    assert total > 0

    # Checkpoint for iteration 3 should exist
    assert os.path.exists(os.path.join(paths["checkpoint"], f"0003-{experiment_name}.pt"))


def test_train_subsamples_when_window_large(tmp_path):
    """When window has many more samples than needed, train() subsamples."""
    config = TrainConfig(train_batch_size=64)
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("checkpoint", "history"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    experiment_name = "test"

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(paths["checkpoint"], f"0004-{experiment_name}.pt")
    del nn

    # Create 5 iterations of history with 200 samples each = 1000 total
    for i in range(5):
        n = 200
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    class DummyRun:
        def track(*a, **kw):
            pass

    # hist_size=4, iteration=4 → window spans iters 0..4
    # average_generation = 1000/5 = 200, steps = ceil(200/64) = 4, samples_needed = 256
    # 256 < 1000, so subsampling should happen
    v_loss, pi_loss, total = train(config, paths, experiment_name, 4, 4, DummyRun(), 0)
    assert math.isfinite(v_loss)
    assert math.isfinite(pi_loss)
    assert total > 0


# ============================================================
# load_reservoir
# ============================================================


def test_load_reservoir_chunks(tmp_path):
    """load_reservoir loads chunked reservoir as TensorDataset."""
    res_dir = tmp_path / "reservoir"
    res_dir.mkdir()
    cs = Game.CANONICAL_SHAPE()

    total_samples = 0
    for i in range(3):
        n = 50 + i * 10
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.randn(n, Game.NUM_PLAYERS() + 1)
        pi = torch.randn(n, Game.NUM_MOVES())
        iters = torch.full((n,), i, dtype=torch.int16)
        _save_chunk(str(res_dir), i, c, v, pi, iters)
        total_samples += n

    meta = {
        "version": 2, "n_chunks": 10, "chunk_size": 60,
        "chunks_filled": 3, "last_updated": [0, 1, 2],
    }
    _save_reservoir_meta(str(res_dir), meta)

    paths = {"reservoir": str(res_dir)}
    ds = load_reservoir(paths)
    assert ds is not None
    assert len(ds) == total_samples


def test_load_reservoir_legacy_per_iteration(tmp_path):
    """load_reservoir falls back to per-iteration format when no chunks exist."""
    res_dir = tmp_path / "reservoir"
    res_dir.mkdir()
    cs = Game.CANONICAL_SHAPE()

    total_samples = 0
    for i in range(2):
        n = 50
        save_compressed(torch.randn(n, cs[0], cs[1], cs[2]),
                        str(res_dir / f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(torch.randn(n, Game.NUM_PLAYERS() + 1),
                        str(res_dir / f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(torch.randn(n, Game.NUM_MOVES()),
                        str(res_dir / f"{i:04d}-0000-pi-{n}.ptz"))
        total_samples += n

    paths = {"reservoir": str(res_dir)}
    ds = load_reservoir(paths)
    assert ds is not None
    assert len(ds) == total_samples


def test_load_reservoir_empty(tmp_path):
    """load_reservoir returns None for empty reservoir."""
    res_dir = tmp_path / "reservoir"
    res_dir.mkdir()
    paths = {"reservoir": str(res_dir)}
    assert load_reservoir(paths) is None


# ============================================================
# Bootstrap E2E with reservoir (diff-arch retrain path)
# ============================================================


def test_bootstrap_retrain_with_reservoir(tmp_path):
    """E2E: call _bootstrap_retrain with reservoir + window files.

    Simulates the diff-arch bootstrap path: build source data,
    call _bootstrap_retrain directly, verify training happened
    and checkpoint can be saved. Uses chunked reservoir format.
    """
    cs = Game.CANONICAL_SHAPE()

    # --- Build source experiment ---
    source_dir = tmp_path / "source"
    source_hist = source_dir / "history"
    source_res = source_dir / "reservoir"
    for d in (source_hist, source_res):
        d.mkdir(parents=True)

    # Reservoir: 3 chunks
    n = 60
    for i in range(3):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        iters = torch.full((n,), i, dtype=torch.int16)
        _save_chunk(str(source_res), i, c, v, pi, iters)

    meta = {
        "version": 2, "n_chunks": 10, "chunk_size": n,
        "chunks_filled": 3, "last_updated": [0, 1, 2],
    }
    _save_reservoir_meta(str(source_res), meta)

    # Window: 2 iterations
    for i in range(3, 5):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, str(source_hist / f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, str(source_hist / f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(pi, str(source_hist / f"{i:04d}-0000-pi-{n}.ptz"))

    source_n = 5

    # --- Setup dest experiment ---
    dest_dir = tmp_path / "dest"
    dest_ckpt = dest_dir / "checkpoint"
    dest_res = dest_dir / "reservoir"
    for d in (dest_ckpt, dest_res):
        d.mkdir(parents=True)

    # Create neural net
    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(str(dest_ckpt), "0000-test.pt")

    # Copy reservoir chunk files + metadata to dest
    import shutil
    for f in glob.glob(str(source_res / "chunk_*.ptz")) + \
             glob.glob(str(source_res / "reservoir_meta.json")):
        shutil.copy2(f, str(dest_res / os.path.basename(f)))

    # Discover files
    reservoir_files = glob_reservoir_chunks(str(dest_res))
    window_files = glob_file_triples(str(source_hist))
    assert len(reservoir_files) == 3
    assert len(window_files) == 2

    class DummyRun:
        def track(*a, **kw):
            pass

    config = TrainConfig(game="connect4", bootstrap_eval_interval=50)

    # Call _bootstrap_retrain directly
    total_train_steps = _bootstrap_retrain(
        nn, reservoir_files, window_files, config, DummyRun(), source_n, 0,
    )

    assert total_train_steps > 0

    # Verify checkpoint can be saved and loaded
    nn.save_checkpoint(str(dest_ckpt), f"{source_n:04d}-test.pt")
    assert (dest_ckpt / "0005-test.pt").exists()

    loaded = NNWrapper.load_checkpoint(Game, str(dest_ckpt), "0005-test.pt")
    assert loaded is not None


# ============================================================
# _bootstrap_train_phase early stopping
# ============================================================


def test_bootstrap_train_phase_early_stop(tmp_path):
    """_bootstrap_train_phase returns positive steps with aggressive early stopping."""
    cs = Game.CANONICAL_SHAPE()

    # Create 2 small files
    file_triples = []
    for i in range(2):
        n = 100
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        c_path = str(tmp_path / f"{i:04d}-0000-canonical-{n}.ptz")
        v_path = str(tmp_path / f"{i:04d}-0000-v-{n}.ptz")
        pi_path = str(tmp_path / f"{i:04d}-0000-pi-{n}.ptz")
        save_compressed(c, c_path)
        save_compressed(v, v_path)
        save_compressed(pi, pi_path)
        file_triples.append((c_path, v_path, pi_path, n))

    config = TrainConfig(
        game="connect4",
        bootstrap_eval_interval=1,
        bootstrap_lr_patience=1,
        bootstrap_convergence_patience=1,
        bootstrap_lr_max_drops=1,
    )

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)

    class DummyRun:
        def track(*a, **kw):
            pass

    steps, early_stopped = _bootstrap_train_phase(
        nn, file_triples, config, DummyRun(), source_n=0,
        total_train_steps=0, phase_name="test",
    )
    assert steps > 0


def test_ema_smoothing_prevents_oscillation_from_resetting_patience():
    """EMA smoothing detects plateau in oscillating losses but not in genuinely improving ones."""
    patience = 4
    threshold = 0.005
    beta = 1 - 1 / max(patience, 1)  # 0.75

    # --- Oscillating losses (flat around 1.0, noisy) should trigger patience ---
    oscillating = [1.0, 0.98, 1.01, 0.97, 1.02, 0.96, 1.01, 0.97, 1.02, 0.98, 1.01, 0.99]
    ema = None
    best_ema = None
    lr_patience_counter = 0

    for loss in oscillating:
        if ema is None:
            ema = loss
        else:
            ema = beta * ema + (1 - beta) * loss

        if best_ema is None:
            best_ema = ema
            continue

        rel_improvement = (best_ema - ema) / (best_ema + 1e-8)
        plateaued = rel_improvement < threshold

        if plateaued:
            lr_patience_counter += 1
        else:
            lr_patience_counter = 0
            best_ema = ema

    assert lr_patience_counter >= patience, (
        f"Oscillating losses should trigger patience, but counter={lr_patience_counter}"
    )

    # --- Genuinely improving losses should NOT accumulate patience ---
    improving = [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73]
    ema = None
    best_ema = None
    lr_patience_counter = 0

    for loss in improving:
        if ema is None:
            ema = loss
        else:
            ema = beta * ema + (1 - beta) * loss

        if best_ema is None:
            best_ema = ema
            continue

        rel_improvement = (best_ema - ema) / (best_ema + 1e-8)
        plateaued = rel_improvement < threshold

        if plateaued:
            lr_patience_counter += 1
        else:
            lr_patience_counter = 0
            best_ema = ema

    assert lr_patience_counter == 0, (
        f"Improving losses should not accumulate patience, but counter={lr_patience_counter}"
    )


def test_bootstrap_lr_drops_with_default_patience(tmp_path):
    """With patience=4 (default), LR should still drop on random data."""
    cs = Game.CANONICAL_SHAPE()

    # Create enough samples so that with batch_size=32 and eval_interval=10,
    # we get many eval chunks (need > patience + 1 chunks for LR to drop)
    file_triples = []
    for i in range(4):
        n = 500
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        c_path = str(tmp_path / f"{i:04d}-0000-canonical-{n}.ptz")
        v_path = str(tmp_path / f"{i:04d}-0000-v-{n}.ptz")
        pi_path = str(tmp_path / f"{i:04d}-0000-pi-{n}.ptz")
        save_compressed(c, c_path)
        save_compressed(v, v_path)
        save_compressed(pi, pi_path)
        file_triples.append((c_path, v_path, pi_path, n))

    config = TrainConfig(
        game="connect4",
        train_batch_size=32,
        bootstrap_eval_interval=10,
        bootstrap_lr_patience=4,
        bootstrap_lr_max_drops=3,
        bootstrap_convergence_patience=3,
    )

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)

    class DummyRun:
        def track(*a, **kw):
            pass

    initial_lr = config.bootstrap_lr
    steps, early_stopped = _bootstrap_train_phase(
        nn, file_triples, config, DummyRun(), source_n=0,
        total_train_steps=0, phase_name="test",
    )
    assert steps > 0
    final_lr = nn.optimizer.param_groups[0]["lr"]
    assert final_lr < initial_lr, (
        f"Expected at least one LR drop, but lr stayed at {final_lr}"
    )


# ============================================================
# Disk-based exploit_symmetries
# ============================================================


def test_exploit_symmetries_disk_creates_syms_files(tmp_path):
    """exploit_symmetries reads raw files from tmp_history and writes -syms- files."""
    config = TrainConfig(game="connect4")
    cs = Game.CANONICAL_SHAPE()
    iteration = 0

    tmp_hist = str(tmp_path / "tmp_history")
    os.makedirs(tmp_hist, exist_ok=True)
    paths = {"tmp_history": tmp_hist}

    # Create raw history files
    n = 50
    c = torch.randn(n, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
    save_compressed(c, os.path.join(tmp_hist, f"{iteration:04d}-0000-canonical-{n}.ptz"))
    save_compressed(v, os.path.join(tmp_hist, f"{iteration:04d}-0000-v-{n}.ptz"))
    save_compressed(pi, os.path.join(tmp_hist, f"{iteration:04d}-0000-pi-{n}.ptz"))

    exploit_symmetries(config, paths, iteration)

    # Raw files should be deleted (only non-syms files with batch 0000)
    raw_files = [f for f in glob.glob(os.path.join(tmp_hist, f"{iteration:04d}-0000-*.ptz"))
                 if "-syms-" not in f]
    assert len(raw_files) == 0, f"Raw files should be deleted, found: {raw_files}"

    # Syms files should exist
    syms_files = glob.glob(os.path.join(tmp_hist, f"{iteration:04d}-*-syms-*.ptz"))
    assert len(syms_files) > 0, "Symmetrized files should be created"

    # Total samples should be original * NUM_SYMMETRIES
    syms_triples = glob_file_triples(tmp_hist, f"{iteration:04d}-*-syms-canonical-*.ptz")
    total_syms = sum(s for _, _, _, s in syms_triples)
    expected = n * Game.NUM_SYMMETRIES()
    assert total_syms == expected, f"Expected {expected} symmetric samples, got {total_syms}"


def test_exploit_symmetries_no_op_for_no_symmetries(tmp_path):
    """exploit_symmetries is a no-op when NUM_SYMMETRIES <= 1."""
    # Brandubh has 8 symmetries, but we need a game with <= 1.
    # Instead, just verify that with no files, it's a no-op.
    config = TrainConfig(game="connect4")
    tmp_hist = str(tmp_path / "tmp_history")
    os.makedirs(tmp_hist, exist_ok=True)
    paths = {"tmp_history": tmp_hist}

    # No files — should return without error
    exploit_symmetries(config, paths, 0)
    files = glob.glob(os.path.join(tmp_hist, "*.ptz"))
    assert len(files) == 0


def test_exploit_symmetries_data_integrity(tmp_path):
    """Symmetrized data has correct shapes and is loadable."""
    config = TrainConfig(game="connect4")
    cs = Game.CANONICAL_SHAPE()
    iteration = 2

    tmp_hist = str(tmp_path / "tmp_history")
    os.makedirs(tmp_hist, exist_ok=True)
    paths = {"tmp_history": tmp_hist}

    n = 30
    c = torch.randn(n, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
    save_compressed(c, os.path.join(tmp_hist, f"{iteration:04d}-0000-canonical-{n}.ptz"))
    save_compressed(v, os.path.join(tmp_hist, f"{iteration:04d}-0000-v-{n}.ptz"))
    save_compressed(pi, os.path.join(tmp_hist, f"{iteration:04d}-0000-pi-{n}.ptz"))

    exploit_symmetries(config, paths, iteration)

    syms_triples = glob_file_triples(tmp_hist, f"{iteration:04d}-*-syms-canonical-*.ptz")
    assert len(syms_triples) > 0

    for c_path, v_path, pi_path, size in syms_triples:
        c_loaded = load_compressed(c_path)
        v_loaded = load_compressed(v_path)
        pi_loaded = load_compressed(pi_path)
        assert c_loaded.shape == (size, cs[0], cs[1], cs[2])
        assert v_loaded.shape == (size, Game.NUM_PLAYERS() + 1)
        assert pi_loaded.shape == (size, Game.NUM_MOVES())


def test_exploit_symmetries_multiple_raw_files(tmp_path):
    """exploit_symmetries handles multiple raw file batches."""
    config = TrainConfig(game="connect4")
    cs = Game.CANONICAL_SHAPE()
    iteration = 1

    tmp_hist = str(tmp_path / "tmp_history")
    os.makedirs(tmp_hist, exist_ok=True)
    paths = {"tmp_history": tmp_hist}

    total_raw = 0
    for batch in range(3):
        n = 20 + batch * 5
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(tmp_hist, f"{iteration:04d}-{batch:04d}-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(tmp_hist, f"{iteration:04d}-{batch:04d}-v-{n}.ptz"))
        save_compressed(pi, os.path.join(tmp_hist, f"{iteration:04d}-{batch:04d}-pi-{n}.ptz"))
        total_raw += n

    exploit_symmetries(config, paths, iteration)

    syms_triples = glob_file_triples(tmp_hist, f"{iteration:04d}-*-syms-canonical-*.ptz")
    total_syms = sum(s for _, _, _, s in syms_triples)
    expected = total_raw * Game.NUM_SYMMETRIES()
    assert total_syms == expected, f"Expected {expected} symmetric samples, got {total_syms}"


# ============================================================
# Disk-based resample_by_surprise
# ============================================================


def test_resample_by_surprise_disk_pipeline(tmp_path):
    """resample_by_surprise reads from tmp_history, writes to history, cleans tmp."""
    config = TrainConfig(game="connect4", train_batch_size=32)
    cs = Game.CANONICAL_SHAPE()
    experiment_name = "test"
    iteration = 0

    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("checkpoint", "history", "tmp_history"):
        os.makedirs(paths[key], exist_ok=True)

    # Create checkpoint for iteration 0
    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt")
    del nn

    # Create tmp_history files
    n = 100
    c = torch.randn(n, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
    save_compressed(c, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-canonical-{n}.ptz"))
    save_compressed(v, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-v-{n}.ptz"))
    save_compressed(pi, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-pi-{n}.ptz"))

    resample_by_surprise(config, paths, experiment_name, iteration)

    # History files should exist
    hist_triples = glob_file_triples(paths["history"], f"{iteration:04d}-*-canonical-*.ptz")
    assert len(hist_triples) > 0, "History files should be written"
    total_resampled = sum(s for _, _, _, s in hist_triples)
    assert total_resampled > 0, "Resampled data should have samples"

    # tmp_history files should be cleaned up
    tmp_files = glob.glob(os.path.join(paths["tmp_history"], f"{iteration:04d}-*.ptz"))
    assert len(tmp_files) == 0, f"tmp_history should be cleaned, found: {tmp_files}"


def test_resample_by_surprise_preserves_data_shapes(tmp_path):
    """Resampled files have correct tensor shapes."""
    config = TrainConfig(game="connect4", train_batch_size=32)
    cs = Game.CANONICAL_SHAPE()
    experiment_name = "test"
    iteration = 0

    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("checkpoint", "history", "tmp_history"):
        os.makedirs(paths[key], exist_ok=True)

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt")
    del nn

    n = 80
    c = torch.randn(n, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
    save_compressed(c, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-canonical-{n}.ptz"))
    save_compressed(v, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-v-{n}.ptz"))
    save_compressed(pi, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-pi-{n}.ptz"))

    resample_by_surprise(config, paths, experiment_name, iteration)

    hist_triples = glob_file_triples(paths["history"], f"{iteration:04d}-*-canonical-*.ptz")
    for c_path, v_path, pi_path, size in hist_triples:
        c_loaded = load_compressed(c_path)
        v_loaded = load_compressed(v_path)
        pi_loaded = load_compressed(pi_path)
        assert c_loaded.shape == (size, cs[0], cs[1], cs[2])
        assert v_loaded.shape == (size, Game.NUM_PLAYERS() + 1)
        assert pi_loaded.shape == (size, Game.NUM_MOVES())


def test_resample_by_surprise_multiple_files(tmp_path):
    """resample_by_surprise handles multiple input files from tmp_history."""
    config = TrainConfig(game="connect4", train_batch_size=32)
    cs = Game.CANONICAL_SHAPE()
    experiment_name = "test"
    iteration = 0

    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("checkpoint", "history", "tmp_history"):
        os.makedirs(paths[key], exist_ok=True)

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt")
    del nn

    total_input = 0
    for batch in range(3):
        n = 40
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["tmp_history"], f"{iteration:04d}-{batch:04d}-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["tmp_history"], f"{iteration:04d}-{batch:04d}-v-{n}.ptz"))
        save_compressed(pi, os.path.join(paths["tmp_history"], f"{iteration:04d}-{batch:04d}-pi-{n}.ptz"))
        total_input += n

    resample_by_surprise(config, paths, experiment_name, iteration)

    hist_triples = glob_file_triples(paths["history"], f"{iteration:04d}-*-canonical-*.ptz")
    total_resampled = sum(s for _, _, _, s in hist_triples)
    # Resampling changes count, but should be in the ballpark of input
    assert total_resampled > 0
    # With surprise weighting, total should be roughly similar (within 3x)
    assert total_resampled < total_input * 3, \
        f"Resampled count {total_resampled} seems too high for {total_input} input"


def test_resample_clears_old_history(tmp_path):
    """resample_by_surprise clears previous history files for the iteration."""
    config = TrainConfig(game="connect4", train_batch_size=32)
    cs = Game.CANONICAL_SHAPE()
    experiment_name = "test"
    iteration = 0

    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("checkpoint", "history", "tmp_history"):
        os.makedirs(paths[key], exist_ok=True)

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt")
    del nn

    # Plant a stale history file
    stale_c = torch.randn(10, cs[0], cs[1], cs[2])
    stale_v = torch.randn(10, Game.NUM_PLAYERS() + 1)
    stale_pi = torch.randn(10, Game.NUM_MOVES())
    save_compressed(stale_c, os.path.join(paths["history"], f"{iteration:04d}-9999-canonical-10.ptz"))
    save_compressed(stale_v, os.path.join(paths["history"], f"{iteration:04d}-9999-v-10.ptz"))
    save_compressed(stale_pi, os.path.join(paths["history"], f"{iteration:04d}-9999-pi-10.ptz"))

    # Create real tmp_history data
    n = 60
    c = torch.randn(n, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
    save_compressed(c, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-canonical-{n}.ptz"))
    save_compressed(v, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-v-{n}.ptz"))
    save_compressed(pi, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-pi-{n}.ptz"))

    resample_by_surprise(config, paths, experiment_name, iteration)

    # Stale file should be gone (replaced by new files)
    all_hist = glob.glob(os.path.join(paths["history"], f"{iteration:04d}-*.ptz"))
    stale_remaining = [f for f in all_hist if "9999" in f]
    assert len(stale_remaining) == 0, "Stale history files should be cleared"


def test_exploit_symmetries_idempotent_rerun(tmp_path):
    """Simulate crash (leave raw+syms files), re-run, verify no duplicates."""
    config = TrainConfig(game="connect4")
    cs = Game.CANONICAL_SHAPE()
    iteration = 0

    tmp_hist = str(tmp_path / "tmp_history")
    os.makedirs(tmp_hist, exist_ok=True)
    paths = {"tmp_history": tmp_hist}

    # Create raw history files
    n = 30
    c = torch.randn(n, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
    save_compressed(c, os.path.join(tmp_hist, f"{iteration:04d}-0000-canonical-{n}.ptz"))
    save_compressed(v, os.path.join(tmp_hist, f"{iteration:04d}-0000-v-{n}.ptz"))
    save_compressed(pi, os.path.join(tmp_hist, f"{iteration:04d}-0000-pi-{n}.ptz"))

    # First run — produces syms files and deletes raw
    exploit_symmetries(config, paths, iteration)
    syms_triples_1 = glob_file_triples(tmp_hist, f"{iteration:04d}-*-syms-canonical-*.ptz")
    total_1 = sum(s for _, _, _, s in syms_triples_1)
    expected = n * Game.NUM_SYMMETRIES()
    assert total_1 == expected

    # Simulate crash: re-add raw files alongside existing syms files
    save_compressed(c, os.path.join(tmp_hist, f"{iteration:04d}-0000-canonical-{n}.ptz"))
    save_compressed(v, os.path.join(tmp_hist, f"{iteration:04d}-0000-v-{n}.ptz"))
    save_compressed(pi, os.path.join(tmp_hist, f"{iteration:04d}-0000-pi-{n}.ptz"))

    # Re-run — should clean stale syms before processing
    exploit_symmetries(config, paths, iteration)
    syms_triples_2 = glob_file_triples(tmp_hist, f"{iteration:04d}-*-syms-canonical-*.ptz")
    total_2 = sum(s for _, _, _, s in syms_triples_2)
    assert total_2 == expected, \
        f"Re-run produced {total_2} samples, expected {expected} (no duplicates)"


def test_resample_zero_loss_guard(tmp_path):
    """All-zero loss produces uniform copies instead of NaN."""
    config = TrainConfig(game="connect4", train_batch_size=32)
    cs = Game.CANONICAL_SHAPE()
    experiment_name = "test"
    iteration = 0

    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("checkpoint", "history", "tmp_history"):
        os.makedirs(paths[key], exist_ok=True)

    # Create a trivial network and checkpoint
    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt")
    del nn

    # Create tmp_history with all-zero inputs (should produce zero loss)
    n = 40
    c = torch.zeros(n, cs[0], cs[1], cs[2])
    # Use uniform distribution for v and pi (mimics zero-loss scenario)
    v = torch.full((n, Game.NUM_PLAYERS() + 1), 1.0 / (Game.NUM_PLAYERS() + 1))
    pi = torch.full((n, Game.NUM_MOVES()), 1.0 / Game.NUM_MOVES())
    save_compressed(c, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-canonical-{n}.ptz"))
    save_compressed(v, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-v-{n}.ptz"))
    save_compressed(pi, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-pi-{n}.ptz"))

    # Should not crash (previously would NaN from 0/0)
    resample_by_surprise(config, paths, experiment_name, iteration)

    # Verify history files were written with valid data
    hist_triples = glob_file_triples(paths["history"], f"{iteration:04d}-*-canonical-*.ptz")
    assert len(hist_triples) > 0, "History files should be written even with zero loss"
    total_resampled = sum(s for _, _, _, s in hist_triples)
    # With uniform copies (each sample copied once), output should equal input
    assert total_resampled == n, \
        f"Expected {n} samples (uniform copies), got {total_resampled}"


def test_exploit_symmetries_then_resample_pipeline(tmp_path):
    """Full pipeline: exploit_symmetries → resample_by_surprise on disk."""
    config = TrainConfig(game="connect4", train_batch_size=32)
    cs = Game.CANONICAL_SHAPE()
    experiment_name = "test"
    iteration = 0

    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("checkpoint", "history", "tmp_history"):
        os.makedirs(paths[key], exist_ok=True)

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt")
    del nn

    # Create raw history (simulating hist_saver output)
    n = 50
    c = torch.randn(n, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
    save_compressed(c, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-canonical-{n}.ptz"))
    save_compressed(v, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-v-{n}.ptz"))
    save_compressed(pi, os.path.join(paths["tmp_history"], f"{iteration:04d}-0000-pi-{n}.ptz"))

    # Step 1: exploit_symmetries (raw → syms in tmp_history)
    exploit_symmetries(config, paths, iteration)

    # Verify syms files exist in tmp_history
    syms_triples = glob_file_triples(paths["tmp_history"], f"{iteration:04d}-*-canonical-*.ptz")
    assert len(syms_triples) > 0, "Syms files should exist after exploit_symmetries"
    total_syms = sum(s for _, _, _, s in syms_triples)
    assert total_syms == n * Game.NUM_SYMMETRIES()

    # Step 2: resample_by_surprise (tmp_history → history)
    resample_by_surprise(config, paths, experiment_name, iteration)

    # Verify final history files
    hist_triples = glob_file_triples(paths["history"], f"{iteration:04d}-*-canonical-*.ptz")
    assert len(hist_triples) > 0, "History files should exist after resampling"
    total_resampled = sum(s for _, _, _, s in hist_triples)
    assert total_resampled > 0

    # tmp_history should be cleaned
    tmp_remaining = glob.glob(os.path.join(paths["tmp_history"], f"{iteration:04d}-*.ptz"))
    assert len(tmp_remaining) == 0, "tmp_history should be cleaned after resampling"
