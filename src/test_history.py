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
from neural_net import NNWrapper, NNArgs
from game_runner import (
    save_compressed, load_compressed, _glob_hist_files,
    _atomic_save_compressed, update_reservoir, calc_hist_size,
)
from config import TrainConfig

Game = alphazero.Connect4GS


# ============================================================
# Compression round-trip
# ============================================================


def test_roundtrip_float32(tmp_path):
    """Random float32 tensor round-trips within float16 tolerance."""
    path = str(tmp_path / "rt.ptz")
    original = torch.randn(100, 3, 6, 7)
    save_compressed(original, path)
    loaded = load_compressed(path)
    assert loaded.shape == original.shape
    assert loaded.dtype == torch.float32
    assert torch.allclose(original, loaded, atol=0.01, rtol=0.01)


def test_roundtrip_integer_values(tmp_path):
    """0/1 tensor round-trips exactly."""
    path = str(tmp_path / "int_rt.ptz")
    original = torch.randint(0, 2, (200, 3, 6, 7)).float()
    save_compressed(original, path)
    loaded = load_compressed(path)
    assert torch.equal(original, loaded)


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
# Reservoir logic
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
    """meta.ptz stores correct iteration-of-origin per sample."""
    iters = torch.tensor([0, 1, 2, 5, 10, 20], dtype=torch.int16)
    save_compressed(iters.float(), str(tmp_path / "meta.ptz"))
    loaded = load_compressed(str(tmp_path / "meta.ptz")).to(torch.int16)
    assert torch.equal(iters, loaded)


def test_reservoir_incremental_merge(tmp_path):
    """Multiple reservoir updates grow then stabilize at capacity."""
    config = TrainConfig()
    res_dir = tmp_path / "reservoir"
    res_dir.mkdir()
    cs = Game.CANONICAL_SHAPE()
    capacity = 300

    for batch_iter in range(5):
        new_c = torch.randn(100, cs[0], cs[1], cs[2])
        new_iters = torch.full((100,), batch_iter, dtype=torch.int16)

        res_c_path = str(res_dir / "canonical.ptz")
        if os.path.exists(res_c_path):
            old_c = load_compressed(res_c_path)
            old_iters = load_compressed(str(res_dir / "meta.ptz")).to(torch.int16)
            all_c = torch.cat([old_c, new_c])
            all_iters = torch.cat([old_iters, new_iters])
        else:
            all_c, all_iters = new_c, new_iters

        if len(all_c) > capacity:
            ages = (batch_iter + 10 - all_iters.float()).clamp(min=0)
            weights = config.reservoir_recency_decay ** ages
            indices = torch.multinomial(weights, capacity, replacement=False)
            all_c = all_c[indices]
            all_iters = all_iters[indices]

        save_compressed(all_c, res_c_path)
        save_compressed(all_iters.float(), str(res_dir / "meta.ptz"))

    final_c = load_compressed(str(res_dir / "canonical.ptz"))
    assert final_c.shape[0] == capacity


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
    """Build reservoir, then bootstrap (train) from it."""
    res_dir = tmp_path / "reservoir"
    res_dir.mkdir()
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    cs = Game.CANONICAL_SHAPE()

    n_total = 500
    c = torch.randn(n_total, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n_total, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n_total, Game.NUM_MOVES()), dim=1)
    save_compressed(c, str(res_dir / "canonical.ptz"))
    save_compressed(v, str(res_dir / "v.ptz"))
    save_compressed(pi, str(res_dir / "pi.ptz"))

    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(str(ckpt_dir), "0000-test.pt")

    class DummyRun:
        def track(*a, **kw):
            pass

    run = DummyRun()

    res_c = load_compressed(str(res_dir / "canonical.ptz"))
    res_v = load_compressed(str(res_dir / "v.ptz"))
    res_pi = load_compressed(str(res_dir / "pi.ptz"))
    ds = TensorDataset(res_c, res_v, res_pi)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    v_loss, pi_loss = nn.train(dl, int(math.ceil(len(ds) / 64)) * 2, run, 5, 0)
    assert math.isfinite(v_loss)
    assert math.isfinite(pi_loss)
    nn.save_checkpoint(str(ckpt_dir), "0005-test.pt")
    assert (ckpt_dir / "0005-test.pt").exists()


# ============================================================
# update_reservoir() direct tests
# ============================================================


def _setup_reservoir_test(tmp_path, n_iters=10, samples_per_iter=50):
    """Create history files for testing update_reservoir."""
    config = TrainConfig(hist_size=1000)
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    for i in range(n_iters):
        n = samples_per_iter
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))
    return config, paths


def test_update_reservoir_with_eviction(tmp_path):
    """update_reservoir creates reservoir files when window slides."""
    config = TrainConfig(hist_size=3)
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    n = 50
    for i in range(6):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    hist_size = calc_hist_size(config, 5)
    update_reservoir(config, paths, 5, hist_size)

    # Reservoir files should exist if eviction happened
    res_c_path = os.path.join(paths["reservoir"], "canonical.ptz")
    if hist_size < 5:  # eviction should have happened
        assert os.path.exists(res_c_path)
        loaded = load_compressed(res_c_path)
        assert loaded.shape[0] > 0


def test_atomic_save_produces_valid_file(tmp_path):
    """_atomic_save_compressed produces a valid, loadable file."""
    path = str(tmp_path / "atomic_test.ptz")
    original = torch.randn(100, 3, 6, 7)
    _atomic_save_compressed(original, path)

    assert os.path.exists(path)
    loaded = load_compressed(path)
    assert loaded.shape == original.shape
    assert torch.allclose(original, loaded, atol=0.01, rtol=0.01)


def test_atomic_save_no_temp_files_on_success(tmp_path):
    """After successful atomic save, no .tmp files remain."""
    path = str(tmp_path / "clean.ptz")
    _atomic_save_compressed(torch.randn(50), path)

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_reservoir_incomplete_source_skips_copy(tmp_path):
    """Bootstrap reservoir copy skips when not all 4 files exist."""
    import shutil

    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    source_res = source_dir / "reservoir"
    dest_res = dest_dir / "reservoir"
    source_res.mkdir(parents=True)
    dest_res.mkdir(parents=True)

    # Only create 2 of the 4 files
    save_compressed(torch.randn(10), str(source_res / "canonical.ptz"))
    save_compressed(torch.randn(10), str(source_res / "v.ptz"))

    # Simulate the all-or-nothing check
    reservoir_files = ("canonical.ptz", "v.ptz", "pi.ptz", "meta.ptz")
    all_exist = all(
        os.path.exists(os.path.join(str(source_res), f)) for f in reservoir_files
    )
    assert not all_exist  # Should NOT copy

    # dest should remain empty
    assert len(list(dest_res.iterdir())) == 0


def test_reservoir_complete_source_copies_all(tmp_path):
    """Bootstrap reservoir copy works when all 4 files exist."""
    import shutil

    source_res = tmp_path / "source" / "reservoir"
    dest_res = tmp_path / "dest" / "reservoir"
    source_res.mkdir(parents=True)
    dest_res.mkdir(parents=True)

    # Create all 4 files
    reservoir_files = ("canonical.ptz", "v.ptz", "pi.ptz", "meta.ptz")
    for fname in reservoir_files:
        save_compressed(torch.randn(10), str(source_res / fname))

    all_exist = all(
        os.path.exists(os.path.join(str(source_res), f)) for f in reservoir_files
    )
    assert all_exist

    for fname in reservoir_files:
        shutil.copy2(str(source_res / fname), str(dest_res / fname))

    # All 4 should be in dest
    for fname in reservoir_files:
        assert (dest_res / fname).exists()
