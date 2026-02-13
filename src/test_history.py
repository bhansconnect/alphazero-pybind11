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
    glob_file_triples, _replace_size_in_path,
    StreamingCompressedDataset, load_reservoir, train,
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
    assert loaded.dtype == get_storage_dtype()  # bfloat16 or float16
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


def test_replace_size_in_path():
    """_replace_size_in_path correctly replaces the size component."""
    assert _replace_size_in_path("/data/0003-0001-canonical-30000.ptz", 25000) == \
        "/data/0003-0001-canonical-25000.ptz"
    assert _replace_size_in_path("/data/0003-0001-v-30000.ptz", 0) == \
        "/data/0003-0001-v-0.ptz"


# ============================================================
# Reservoir logic (per-iteration format)
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
    """Integer values round-trip through half-precision storage."""
    iters = torch.tensor([0, 1, 2, 5, 10, 20], dtype=torch.int16)
    save_compressed(iters.float(), str(tmp_path / "meta.ptz"))
    loaded = load_compressed(str(tmp_path / "meta.ptz")).float().to(torch.int16)
    assert torch.equal(iters, loaded)


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


def test_update_reservoir_moves_evicted_files(tmp_path):
    """update_reservoir moves evicted files to reservoir directory."""
    # window_size_scalar=1.0 produces small windows so eviction triggers quickly:
    # calc_hist_size(config, 3) = 2, calc_hist_size(config, 4) = 2
    # iter 3: evicts iter 0, iter 4: evicts iter 1
    config = TrainConfig(window_size_scalar=1.0)
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

    for iteration in range(3, 5):
        hist_size = calc_hist_size(config, iteration)
        update_reservoir(config, paths, iteration, hist_size)

    # Reservoir should contain evicted iterations (0 and 1)
    reservoir_triples = glob_file_triples(paths["reservoir"])
    assert len(reservoir_triples) >= 1, "Reservoir should have evicted files"

    reservoir_iters = set()
    for c_path, _, _, _ in reservoir_triples:
        iter_num = int(os.path.basename(c_path).split("-")[0])
        reservoir_iters.add(iter_num)
    assert 0 in reservoir_iters, "Iteration 0 should be in reservoir"

    # Verify files are proper per-iteration format
    for c_path, v_path, pi_path, size in reservoir_triples:
        assert "-canonical-" in c_path
        assert "-v-" in v_path
        assert "-pi-" in pi_path
        assert size > 0

    # Evicted files should no longer be in history
    for evicted_iter in reservoir_iters:
        hist_files = glob.glob(os.path.join(paths["history"], f"{evicted_iter:04d}-*.ptz"))
        assert len(hist_files) == 0, f"Iteration {evicted_iter} still in history after eviction"


def test_update_reservoir_thinning(tmp_path):
    """Reservoir thins down when exceeding window capacity.

    With window_size_scalar=1.0 and 9 iterations (n=100 each):
    - Evictions at iters 3(→0), 4(→1), 6(→2), 7(→3), 8(→4)
    - After iter 8: reservoir has 5 iters × 100 = 500 samples
    - Window has 4 iters (5-8) = 400 capacity
    - excess = 100 → thinning triggers
    """
    config = TrainConfig(window_size_scalar=1.0)
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    n = 100

    # Create 9 iterations of history (0-8)
    for i in range(9):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    # Run reservoir updates sequentially (simulating the training loop)
    for iteration in range(3, 9):
        hist_size = calc_hist_size(config, iteration)
        update_reservoir(config, paths, iteration, hist_size)

    # Reservoir should be bounded: total reservoir <= window capacity
    reservoir_triples = glob_file_triples(paths["reservoir"])
    reservoir_total = sum(s for _, _, _, s in reservoir_triples)

    # Compute current window capacity
    final_iter = 8
    final_hist_size = calc_hist_size(config, final_iter)
    capacity = 0
    for i in range(max(0, final_iter - final_hist_size), final_iter + 1):
        for fn in glob.glob(os.path.join(paths["history"], f"{i:04d}-*-canonical-*.ptz")):
            capacity += int(fn.rsplit("-", 1)[-1].split(".")[0])

    assert capacity > 0, "Window should have data"
    assert reservoir_total > 0, "Reservoir should have data after evictions"
    assert reservoir_total <= capacity, \
        f"Reservoir ({reservoir_total}) exceeds capacity ({capacity})"


def test_update_reservoir_thinning_age_bias(tmp_path):
    """Age-weighted thinning removes more old samples than recent ones.

    Uses reservoir_recency_decay=0.9 (aggressive) so the age bias is clearly
    visible after thinning. With window_size_scalar=1.0 and 12 iterations,
    multiple thinning rounds occur with enough iterations in the reservoir
    to see the age-weighted effect.
    """
    config = TrainConfig(window_size_scalar=1.0, reservoir_recency_decay=0.9)
    exp_dir = str(tmp_path / "experiment")
    paths = config.resolve_paths(exp_dir)
    for key in ("history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    n = 200

    # Create 12 iterations with large sample counts to force multiple thinning rounds
    for i in range(12):
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        p = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        save_compressed(c, os.path.join(paths["history"], f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, os.path.join(paths["history"], f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(p, os.path.join(paths["history"], f"{i:04d}-0000-pi-{n}.ptz"))

    # Run updates — evictions happen at iters 3,4,6,7,8,10,11
    # Thinning triggers when reservoir exceeds window capacity
    for iteration in range(3, 12):
        hist_size = calc_hist_size(config, iteration)
        update_reservoir(config, paths, iteration, hist_size)

    # Check that oldest iterations have fewer or no samples
    reservoir_triples = glob_file_triples(paths["reservoir"])
    iter_samples = {}
    for c_path, _, _, size in reservoir_triples:
        iter_num = int(os.path.basename(c_path).split("-")[0])
        iter_samples[iter_num] = iter_samples.get(iter_num, 0) + size

    assert len(iter_samples) >= 2, \
        f"Expected at least 2 iterations in reservoir, got {len(iter_samples)}: {iter_samples}"

    sorted_iters = sorted(iter_samples.keys())
    oldest = sorted_iters[0]
    newest = sorted_iters[-1]
    # With recency bias (decay=0.9), newest should have >= oldest samples
    assert iter_samples[newest] >= iter_samples[oldest], \
        f"Newest iter {newest} ({iter_samples[newest]}) should have >= " \
        f"oldest iter {oldest} ({iter_samples[oldest]})"


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


def test_reservoir_source_copy_per_iteration(tmp_path):
    """Bootstrap reservoir copy works with per-iteration files."""
    import shutil

    source_res = tmp_path / "source" / "reservoir"
    dest_res = tmp_path / "dest" / "reservoir"
    source_res.mkdir(parents=True)
    dest_res.mkdir(parents=True)

    cs = Game.CANONICAL_SHAPE()
    n = 50
    # Create per-iteration reservoir files
    for i in range(3):
        save_compressed(torch.randn(n, cs[0], cs[1], cs[2]),
                        str(source_res / f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(torch.randn(n, 3),
                        str(source_res / f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(torch.randn(n, 7),
                        str(source_res / f"{i:04d}-0000-pi-{n}.ptz"))

    # Use glob_file_triples to discover and copy
    source_triples = glob_file_triples(str(source_res))
    assert len(source_triples) == 3

    all_files = []
    for c, v, p, _ in source_triples:
        all_files.extend([c, v, p])
    for src in all_files:
        shutil.copy2(src, str(dest_res / os.path.basename(src)))

    dest_triples = glob_file_triples(str(dest_res))
    assert len(dest_triples) == 3
    total_samples = sum(s for _, _, _, s in dest_triples)
    assert total_samples == n * 3


def test_reservoir_empty_source_skips_copy(tmp_path):
    """Bootstrap reservoir copy skips when source has no per-iteration files."""
    source_res = tmp_path / "source" / "reservoir"
    dest_res = tmp_path / "dest" / "reservoir"
    source_res.mkdir(parents=True)
    dest_res.mkdir(parents=True)

    # No per-iteration files → glob_file_triples returns empty
    triples = glob_file_triples(str(source_res))
    assert len(triples) == 0
    assert len(list(dest_res.iterdir())) == 0


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


def test_load_reservoir_per_iteration(tmp_path):
    """load_reservoir loads per-iteration .ptz files as TensorDataset."""
    res_dir = tmp_path / "reservoir"
    res_dir.mkdir()
    cs = Game.CANONICAL_SHAPE()

    total_samples = 0
    for i in range(3):
        n = 50 + i * 10
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
    """E2E: build source experiment with history + reservoir, bootstrap retrain with streaming.

    Simulates the diff-arch bootstrap path: glob reservoir + window files,
    create StreamingCompressedDataset, train phase 1 (all data) and phase 2
    (window only), verify finite loss and checkpoint creation.
    """
    cs = Game.CANONICAL_SHAPE()

    # --- Build source experiment ---
    source_dir = tmp_path / "source"
    source_hist = source_dir / "history"
    source_res = source_dir / "reservoir"
    source_ckpt = source_dir / "checkpoint"
    for d in (source_hist, source_res, source_ckpt):
        d.mkdir(parents=True)

    # Source has 5 iterations of history (iters 3-4 in window, 0-2 evicted to reservoir)
    for i in range(5):
        n = 60
        c = torch.randn(n, cs[0], cs[1], cs[2])
        v = torch.softmax(torch.randn(n, Game.NUM_PLAYERS() + 1), dim=1)
        pi = torch.softmax(torch.randn(n, Game.NUM_MOVES()), dim=1)
        dest = source_hist if i >= 3 else source_res  # iters 0-2 in reservoir, 3-4 in history
        save_compressed(c, str(dest / f"{i:04d}-0000-canonical-{n}.ptz"))
        save_compressed(v, str(dest / f"{i:04d}-0000-v-{n}.ptz"))
        save_compressed(pi, str(dest / f"{i:04d}-0000-pi-{n}.ptz"))

    source_n = 5  # source had 5 iterations

    # --- Setup dest experiment ---
    dest_dir = tmp_path / "dest"
    dest_ckpt = dest_dir / "checkpoint"
    dest_res = dest_dir / "reservoir"
    for d in (dest_ckpt, dest_res):
        d.mkdir(parents=True)

    # Create initial checkpoint
    nnargs = NNArgs(num_channels=8, depth=2, kernel_size=3, dense_net=True)
    nn = NNWrapper(Game, nnargs)
    nn.save_checkpoint(str(dest_ckpt), "0000-test.pt")

    # Copy reservoir from source
    import shutil
    source_triples = glob_file_triples(str(source_res))
    assert len(source_triples) == 3  # iters 0, 1, 2
    for c_path, v_path, pi_path, _ in source_triples:
        for p in (c_path, v_path, pi_path):
            shutil.copy2(p, str(dest_res / os.path.basename(p)))

    # --- Bootstrap retrain (diff-arch path) ---
    source_paths = {"reservoir": str(source_res), "history": str(source_hist)}

    # Discover all files
    reservoir_files = glob_file_triples(str(dest_res))
    window_files = []
    for wi in range(0, source_n):
        triples = glob_file_triples(str(source_hist), f"{wi:04d}-*-canonical-*.ptz")
        window_files.extend(triples)
    all_files = reservoir_files + window_files
    assert len(all_files) == 5  # 3 reservoir + 2 window

    class DummyRun:
        def track(*a, **kw):
            pass

    run = DummyRun()
    total_train_steps = 0
    bs = 64

    # Phase 1: train on all data (reservoir + window)
    total_samples = sum(s for _, _, _, s in all_files)
    assert total_samples == 300  # 5 files × 60 samples
    full_passes = 2
    streaming_ds = StreamingCompressedDataset(all_files, bs, passes=full_passes)
    dl = DataLoader(streaming_ds, batch_size=None, num_workers=0)
    steps_p1 = int(math.ceil(total_samples / bs)) * full_passes
    v_loss, pi_loss = nn.train(dl, steps_p1, run, source_n, total_train_steps)
    total_train_steps += steps_p1
    assert math.isfinite(v_loss)
    assert math.isfinite(pi_loss)

    # Phase 2: train on window only
    if window_files:
        window_total = sum(s for _, _, _, s in window_files)
        assert window_total == 120  # 2 files × 60 samples
        window_passes = 2
        streaming_ds2 = StreamingCompressedDataset(window_files, bs, passes=window_passes)
        dl2 = DataLoader(streaming_ds2, batch_size=None, num_workers=0)
        steps_p2 = int(math.ceil(window_total / bs)) * window_passes
        v_loss2, pi_loss2 = nn.train(dl2, steps_p2, run, source_n, total_train_steps)
        total_train_steps += steps_p2
        assert math.isfinite(v_loss2)
        assert math.isfinite(pi_loss2)

    # Save final checkpoint
    nn.save_checkpoint(str(dest_ckpt), f"{source_n:04d}-test.pt")
    assert (dest_ckpt / "0005-test.pt").exists()
    assert total_train_steps > 0
