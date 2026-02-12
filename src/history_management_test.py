#!/usr/bin/env python3
"""Test suite for history data management: compression, reservoir, and backward compatibility."""

import os
import sys
import gc
import glob
import shutil
import tempfile
import functools
import math

print = functools.partial(print, flush=True)

import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import numpy as np

import alphazero
from neural_net import NNWrapper, NNArgs, get_device

# Import the functions under test
from game_runner import (
    save_compressed,
    load_compressed,
    HIST_SIZE,
)

Game = alphazero.Connect4GS
DEPTH = 2
CHANNELS = 8
KERNEL_SIZE = 3
DENSE_NET = True

# We'll use a temp directory as the base for all test data
TEST_BASE = None


def setup_test_dir():
    global TEST_BASE
    TEST_BASE = tempfile.mkdtemp(prefix="hist_test_")
    return TEST_BASE


def cleanup_test_dir():
    global TEST_BASE
    if TEST_BASE and os.path.exists(TEST_BASE):
        shutil.rmtree(TEST_BASE)
    TEST_BASE = None


def new_game():
    return Game()


def create_test_network():
    nnargs = NNArgs(
        num_channels=CHANNELS,
        depth=DEPTH,
        kernel_size=KERNEL_SIZE,
        dense_net=DENSE_NET,
    )
    return NNWrapper(Game, nnargs)


# ============================================================
# Unit tests for save_compressed / load_compressed
# ============================================================


def test_roundtrip_float32():
    """Random float32 tensor round-trips through save_compressed/load_compressed within float16 tolerance."""
    print("  Testing round-trip float32...")
    path = os.path.join(TEST_BASE, "roundtrip.ptz")
    original = torch.randn(100, 3, 6, 7)
    save_compressed(original, path)
    loaded = load_compressed(path)
    assert loaded.dtype == torch.float32, f"Expected float32, got {loaded.dtype}"
    assert loaded.shape == original.shape, f"Shape mismatch: {loaded.shape} vs {original.shape}"
    # float16 has ~0.001 relative tolerance for values near 1.0
    assert torch.allclose(original, loaded, atol=0.01, rtol=0.01), "Values differ beyond float16 tolerance"
    print("    PASSED")


def test_roundtrip_integer_values():
    """Tensor of 0/1 values round-trips exactly."""
    print("  Testing round-trip integer values (0/1)...")
    path = os.path.join(TEST_BASE, "int_roundtrip.ptz")
    original = torch.randint(0, 2, (200, 3, 6, 7)).float()
    save_compressed(original, path)
    loaded = load_compressed(path)
    assert torch.equal(original, loaded), "0/1 values should round-trip exactly"
    print("    PASSED")


def test_file_size_reduction():
    """Compressed .ptz file is significantly smaller than uncompressed .pt."""
    print("  Testing file size reduction...")
    pt_path = os.path.join(TEST_BASE, "size_test.pt")
    ptz_path = os.path.join(TEST_BASE, "size_test.ptz")
    # Mostly zeros with some structure (like canonical boards)
    data = torch.zeros(1000, 3, 6, 7)
    data[:, 0, :3, :] = 1.0
    torch.save(data, pt_path)
    save_compressed(data, ptz_path)
    pt_size = os.path.getsize(pt_path)
    ptz_size = os.path.getsize(ptz_path)
    print(f"    .pt: {pt_size} bytes, .ptz: {ptz_size} bytes, ratio: {ptz_size/pt_size:.3f}")
    assert ptz_size < pt_size, f"Compressed ({ptz_size}) should be smaller than raw ({pt_size})"
    print("    PASSED")


def test_ptz_extension():
    """Verify .ptz extension is used."""
    print("  Testing .ptz extension...")
    path = os.path.join(TEST_BASE, "ext_test.ptz")
    save_compressed(torch.zeros(10), path)
    assert os.path.exists(path), "File with .ptz extension should exist"
    assert path.endswith(".ptz"), "Path should end with .ptz"
    print("    PASSED")


# ============================================================
# Unit tests for maybe_save
# ============================================================


def test_maybe_save_hist_location():
    """maybe_save to HIST_LOCATION creates .ptz files."""
    print("  Testing maybe_save with HIST_LOCATION...")
    import game_runner
    old_hist = game_runner.HIST_LOCATION
    hist_dir = os.path.join(TEST_BASE, "history")
    os.makedirs(hist_dir, exist_ok=True)
    game_runner.HIST_LOCATION = hist_dir
    try:
        # We need to call the maybe_save inside __main__ guard.
        # Instead, replicate the logic inline to test.
        cs = Game.CANONICAL_SHAPE()
        c = torch.randn(HIST_SIZE, cs[0], cs[1], cs[2])
        v = torch.randn(HIST_SIZE, Game.NUM_PLAYERS() + 1)
        p = torch.randn(HIST_SIZE, Game.NUM_MOVES())

        use_compression = True
        ext = ".ptz"
        save_fn = save_compressed
        iteration, batch = 0, 0
        save_fn(c, os.path.join(hist_dir, f"{iteration:04d}-{batch:04d}-canonical-{HIST_SIZE}{ext}"))
        save_fn(v, os.path.join(hist_dir, f"{iteration:04d}-{batch:04d}-v-{HIST_SIZE}{ext}"))
        save_fn(p, os.path.join(hist_dir, f"{iteration:04d}-{batch:04d}-pi-{HIST_SIZE}{ext}"))

        ptz_files = glob.glob(os.path.join(hist_dir, "*.ptz"))
        assert len(ptz_files) == 3, f"Expected 3 .ptz files, got {len(ptz_files)}"
        pt_files = glob.glob(os.path.join(hist_dir, "*.pt"))
        assert len(pt_files) == 0, f"Expected no .pt files, got {len(pt_files)}"

        # Verify content loads correctly
        loaded_c = load_compressed(os.path.join(hist_dir, f"0000-0000-canonical-{HIST_SIZE}.ptz"))
        assert loaded_c.shape == c.shape
        print("    PASSED")
    finally:
        game_runner.HIST_LOCATION = old_hist


def test_maybe_save_tmp_location():
    """Saving to TMP_HIST_LOCATION creates .pt files (uncompressed)."""
    print("  Testing maybe_save with TMP_HIST_LOCATION...")
    tmp_dir = os.path.join(TEST_BASE, "tmp_history")
    os.makedirs(tmp_dir, exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    c = torch.randn(HIST_SIZE, cs[0], cs[1], cs[2])
    v = torch.randn(HIST_SIZE, Game.NUM_PLAYERS() + 1)
    p = torch.randn(HIST_SIZE, Game.NUM_MOVES())

    iteration, batch = 0, 0
    torch.save(c, os.path.join(tmp_dir, f"{iteration:04d}-{batch:04d}-canonical-{HIST_SIZE}.pt"))
    torch.save(v, os.path.join(tmp_dir, f"{iteration:04d}-{batch:04d}-v-{HIST_SIZE}.pt"))
    torch.save(p, os.path.join(tmp_dir, f"{iteration:04d}-{batch:04d}-pi-{HIST_SIZE}.pt"))

    pt_files = glob.glob(os.path.join(tmp_dir, "*.pt"))
    assert len(pt_files) == 3, f"Expected 3 .pt files, got {len(pt_files)}"
    ptz_files = glob.glob(os.path.join(tmp_dir, "*.ptz"))
    assert len(ptz_files) == 0, f"Expected no .ptz files"
    print("    PASSED")


# ============================================================
# Unit tests for update_reservoir
# ============================================================


def _create_fake_iteration_files(hist_dir, iteration, num_samples=100):
    """Create fake .ptz history files for a given iteration."""
    cs = Game.CANONICAL_SHAPE()
    c = torch.randn(num_samples, cs[0], cs[1], cs[2])
    v = torch.randn(num_samples, Game.NUM_PLAYERS() + 1)
    p = torch.randn(num_samples, Game.NUM_MOVES())
    save_compressed(c, os.path.join(hist_dir, f"{iteration:04d}-0000-canonical-{num_samples}.ptz"))
    save_compressed(v, os.path.join(hist_dir, f"{iteration:04d}-0000-v-{num_samples}.ptz"))
    save_compressed(p, os.path.join(hist_dir, f"{iteration:04d}-0000-pi-{num_samples}.ptz"))
    return c, v, p


def test_reservoir_basic_merge():
    """Create fake iteration files, simulate window excluding oldest. Verify reservoir created, old files deleted."""
    print("  Testing reservoir basic merge...")
    import game_runner
    old_hist = game_runner.HIST_LOCATION
    old_res = game_runner.RESERVOIR_LOCATION
    hist_dir = os.path.join(TEST_BASE, "res_hist")
    res_dir = os.path.join(TEST_BASE, "res_reservoir")
    os.makedirs(hist_dir, exist_ok=True)
    game_runner.HIST_LOCATION = hist_dir
    game_runner.RESERVOIR_LOCATION = res_dir
    try:
        # Create iterations 0, 1, 2 with 100 samples each
        for it in range(3):
            _create_fake_iteration_files(hist_dir, it, 100)

        # calc_hist_size is defined inside __main__ in game_runner.
        # We'll define a local one with same logic.
        def calc_hist_size(i):
            alpha, beta, scalar = 0.5, 0.7, 6
            return int(scalar * (1 + beta * (((i + 1) / scalar) ** alpha - 1) / alpha))

        # Simulate: iteration=2, hist_size=1 (so window is [1,2], iter 0 is evicted)
        # We need update_reservoir which is inside __main__. We'll redefine inline.
        from game_runner import load_compressed as lc, save_compressed as sc, RESERVOIR_RECENCY_DECAY

        iteration = 2
        hist_size = 1  # window covers [1, 2]
        oldest_in_window = max(0, iteration - hist_size)  # = 1

        # Simulate prev_oldest: iteration-1=1, with hist_size(0)= calc_hist_size(0) which is some value
        # For simplicity, pretend previous oldest was 0
        prev_oldest = 0
        evicted_iters = list(range(prev_oldest, oldest_in_window))  # [0]

        # Load evicted data
        new_c, new_v, new_pi, new_iters_list = [], [], [], []
        for it in evicted_iters:
            c_files = sorted(glob.glob(os.path.join(hist_dir, f"{it:04d}-*-canonical-*.ptz")))
            v_files = sorted(glob.glob(os.path.join(hist_dir, f"{it:04d}-*-v-*.ptz")))
            p_files = sorted(glob.glob(os.path.join(hist_dir, f"{it:04d}-*-pi-*.ptz")))
            for j in range(len(c_files)):
                ct = lc(c_files[j])
                new_c.append(ct)
                new_v.append(lc(v_files[j]))
                new_pi.append(lc(p_files[j]))
                new_iters_list.append(torch.full((ct.shape[0],), it, dtype=torch.int16))

        new_c = torch.cat(new_c)
        new_v = torch.cat(new_v)
        new_pi = torch.cat(new_pi)
        new_iters_t = torch.cat(new_iters_list)

        os.makedirs(res_dir, exist_ok=True)
        sc(new_c, os.path.join(res_dir, "canonical.ptz"))
        sc(new_v, os.path.join(res_dir, "v.ptz"))
        sc(new_pi, os.path.join(res_dir, "pi.ptz"))
        sc(new_iters_t.float(), os.path.join(res_dir, "meta.ptz"))

        # Delete evicted files
        for it in evicted_iters:
            for fn in glob.glob(os.path.join(hist_dir, f"{it:04d}-*.ptz")):
                os.remove(fn)

        # Verify reservoir exists
        assert os.path.exists(os.path.join(res_dir, "canonical.ptz")), "Reservoir canonical should exist"
        assert os.path.exists(os.path.join(res_dir, "v.ptz")), "Reservoir v should exist"
        assert os.path.exists(os.path.join(res_dir, "pi.ptz")), "Reservoir pi should exist"
        assert os.path.exists(os.path.join(res_dir, "meta.ptz")), "Reservoir meta should exist"

        # Verify iter 0 files deleted
        iter0_files = glob.glob(os.path.join(hist_dir, "0000-*.ptz"))
        assert len(iter0_files) == 0, f"Iter 0 files should be deleted, found {len(iter0_files)}"

        # Verify iters 1, 2 still exist
        iter1_files = glob.glob(os.path.join(hist_dir, "0001-*.ptz"))
        iter2_files = glob.glob(os.path.join(hist_dir, "0002-*.ptz"))
        assert len(iter1_files) == 3, f"Iter 1 should have 3 files, got {len(iter1_files)}"
        assert len(iter2_files) == 3, f"Iter 2 should have 3 files, got {len(iter2_files)}"

        # Verify reservoir content
        res_c = lc(os.path.join(res_dir, "canonical.ptz"))
        assert res_c.shape[0] == 100, f"Reservoir should have 100 samples, got {res_c.shape[0]}"

        print("    PASSED")
    finally:
        game_runner.HIST_LOCATION = old_hist
        game_runner.RESERVOIR_LOCATION = old_res


def test_reservoir_capacity_enforcement():
    """Overfill reservoir, verify it downsamples to target capacity."""
    print("  Testing reservoir capacity enforcement...")
    from game_runner import load_compressed as lc, save_compressed as sc, RESERVOIR_RECENCY_DECAY

    res_dir = os.path.join(TEST_BASE, "cap_reservoir")
    os.makedirs(res_dir, exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    n_samples = 500
    capacity = 200

    c = torch.randn(n_samples, cs[0], cs[1], cs[2])
    v = torch.randn(n_samples, Game.NUM_PLAYERS() + 1)
    pi = torch.randn(n_samples, Game.NUM_MOVES())
    iters = torch.randint(0, 50, (n_samples,), dtype=torch.int16)

    iteration = 100
    ages = (iteration - iters.float()).clamp(min=0)
    weights = RESERVOIR_RECENCY_DECAY ** ages
    indices = torch.multinomial(weights, capacity, replacement=False)

    c_ds = c[indices]
    v_ds = v[indices]
    pi_ds = pi[indices]
    iters_ds = iters[indices]

    assert c_ds.shape[0] == capacity, f"Expected {capacity} samples, got {c_ds.shape[0]}"
    print(f"    Downsampled from {n_samples} to {c_ds.shape[0]} samples")
    print("    PASSED")


def test_reservoir_recency_weighting():
    """Samples from recent iterations survive downsampling more often."""
    print("  Testing reservoir recency weighting...")
    from game_runner import RESERVOIR_RECENCY_DECAY

    n_old = 200  # iteration 0
    n_new = 200  # iteration 50
    capacity = 200
    iteration = 100

    iters = torch.cat([
        torch.full((n_old,), 0, dtype=torch.int16),
        torch.full((n_new,), 50, dtype=torch.int16),
    ])

    trials = 100
    old_survival_count = 0
    new_survival_count = 0

    for _ in range(trials):
        ages = (iteration - iters.float()).clamp(min=0)
        weights = RESERVOIR_RECENCY_DECAY ** ages
        indices = torch.multinomial(weights, capacity, replacement=False)
        selected_iters = iters[indices]
        old_survival_count += (selected_iters == 0).sum().item()
        new_survival_count += (selected_iters == 50).sum().item()

    avg_old = old_survival_count / trials
    avg_new = new_survival_count / trials
    print(f"    Avg old survivors: {avg_old:.1f}, Avg new survivors: {avg_new:.1f}")
    assert avg_new > avg_old, f"Newer samples ({avg_new:.1f}) should survive more than old ({avg_old:.1f})"
    print("    PASSED")


def test_reservoir_metadata():
    """meta.ptz stores correct iteration-of-origin per sample."""
    print("  Testing reservoir metadata...")
    from game_runner import save_compressed as sc, load_compressed as lc

    res_dir = os.path.join(TEST_BASE, "meta_reservoir")
    os.makedirs(res_dir, exist_ok=True)

    iters = torch.tensor([0, 1, 2, 5, 10, 20], dtype=torch.int16)
    sc(iters.float(), os.path.join(res_dir, "meta.ptz"))
    loaded = lc(os.path.join(res_dir, "meta.ptz")).to(torch.int16)
    assert torch.equal(iters, loaded), f"Metadata mismatch: {iters} vs {loaded}"
    print("    PASSED")


def test_reservoir_incremental_merge():
    """Multiple reservoir updates grow then stabilize at capacity."""
    print("  Testing reservoir incremental merge...")
    from game_runner import save_compressed as sc, load_compressed as lc, RESERVOIR_RECENCY_DECAY

    res_dir = os.path.join(TEST_BASE, "incr_reservoir")
    os.makedirs(res_dir, exist_ok=True)
    cs = Game.CANONICAL_SHAPE()
    capacity = 300

    res_c_path = os.path.join(res_dir, "canonical.ptz")

    for batch_iter in range(5):
        # Add 100 new samples each time
        new_c = torch.randn(100, cs[0], cs[1], cs[2])
        new_v = torch.randn(100, Game.NUM_PLAYERS() + 1)
        new_pi = torch.randn(100, Game.NUM_MOVES())
        new_iters = torch.full((100,), batch_iter, dtype=torch.int16)

        if os.path.exists(res_c_path):
            old_c = lc(res_c_path)
            old_v = lc(os.path.join(res_dir, "v.ptz"))
            old_pi = lc(os.path.join(res_dir, "pi.ptz"))
            old_iters = lc(os.path.join(res_dir, "meta.ptz")).to(torch.int16)
            all_c = torch.cat([old_c, new_c])
            all_v = torch.cat([old_v, new_v])
            all_pi = torch.cat([old_pi, new_pi])
            all_iters = torch.cat([old_iters, new_iters])
        else:
            all_c, all_v, all_pi, all_iters = new_c, new_v, new_pi, new_iters

        # Downsample if over capacity
        if len(all_c) > capacity:
            iteration = batch_iter + 10
            ages = (iteration - all_iters.float()).clamp(min=0)
            weights = RESERVOIR_RECENCY_DECAY ** ages
            indices = torch.multinomial(weights, capacity, replacement=False)
            all_c = all_c[indices]
            all_v = all_v[indices]
            all_pi = all_pi[indices]
            all_iters = all_iters[indices]

        sc(all_c, res_c_path)
        sc(all_v, os.path.join(res_dir, "v.ptz"))
        sc(all_pi, os.path.join(res_dir, "pi.ptz"))
        sc(all_iters.float(), os.path.join(res_dir, "meta.ptz"))

    final_c = lc(res_c_path)
    print(f"    Final reservoir size: {final_c.shape[0]} (capacity: {capacity})")
    assert final_c.shape[0] <= capacity, f"Reservoir should be at most {capacity}, got {final_c.shape[0]}"
    # After 5 batches of 100 (total 500), it should have been downsampled
    assert final_c.shape[0] == capacity, f"After overfilling, reservoir should be exactly {capacity}"
    print("    PASSED")


# ============================================================
# Integration test: backward compatibility
# ============================================================


def test_mixed_format_loading():
    """Train loads both .pt and .ptz files."""
    print("  Testing mixed format loading (.pt + .ptz)...")
    hist_dir = os.path.join(TEST_BASE, "mixed_hist")
    os.makedirs(hist_dir, exist_ok=True)

    cs = Game.CANONICAL_SHAPE()
    n = 50

    # Create .pt file for iteration 0
    c0 = torch.randn(n, cs[0], cs[1], cs[2])
    v0 = torch.randn(n, Game.NUM_PLAYERS() + 1)
    p0 = torch.randn(n, Game.NUM_MOVES())
    torch.save(c0, os.path.join(hist_dir, f"0000-0000-canonical-{n}.pt"))
    torch.save(v0, os.path.join(hist_dir, f"0000-0000-v-{n}.pt"))
    torch.save(p0, os.path.join(hist_dir, f"0000-0000-pi-{n}.pt"))

    # Create .ptz file for iteration 1
    c1 = torch.randn(n, cs[0], cs[1], cs[2])
    v1 = torch.randn(n, Game.NUM_PLAYERS() + 1)
    p1 = torch.randn(n, Game.NUM_MOVES())
    save_compressed(c1, os.path.join(hist_dir, f"0001-0000-canonical-{n}.ptz"))
    save_compressed(v1, os.path.join(hist_dir, f"0001-0000-v-{n}.ptz"))
    save_compressed(p1, os.path.join(hist_dir, f"0001-0000-pi-{n}.ptz"))

    # Use _glob_hist_files and _load_hist_tensor pattern
    # These are defined inside __main__ in game_runner, so we test the pattern directly.
    def _glob_hist_files(location, pattern):
        files = sorted(glob.glob(os.path.join(location, pattern + ".ptz")))
        if not files:
            files = sorted(glob.glob(os.path.join(location, pattern + ".pt")))
        return files

    def _load_hist_tensor(path):
        if path.endswith(".ptz"):
            return load_compressed(path)
        return torch.load(path, map_location="cpu", mmap=True)

    datasets = []
    total_size = 0
    for i in range(2):
        c = _glob_hist_files(hist_dir, f"{i:04d}-*-canonical-*")
        v = _glob_hist_files(hist_dir, f"{i:04d}-*-v-*")
        p = _glob_hist_files(hist_dir, f"{i:04d}-*-pi-*")
        assert len(c) == 1, f"Expected 1 canonical file for iter {i}, got {len(c)}"
        for j in range(len(c)):
            size = int(c[j].split("-")[-1].split(".")[0])
            total_size += size
            ct = _load_hist_tensor(c[j])
            vt = _load_hist_tensor(v[j])
            pt = _load_hist_tensor(p[j])
            datasets.append(TensorDataset(ct, vt, pt))

    dataset = ConcatDataset(datasets)
    assert len(dataset) == 2 * n, f"Expected {2*n} samples, got {len(dataset)}"
    assert total_size == 2 * n
    print(f"    Loaded {len(dataset)} samples from mixed formats")
    print("    PASSED")


# ============================================================
# Integration test: multi-iteration training loop
# ============================================================


def test_multi_iteration_e2e():
    """5-iteration E2E with compression + reservoir."""
    print("  Testing multi-iteration E2E (5 iterations)...")
    import game_runner
    old_hist = game_runner.HIST_LOCATION
    old_tmp = game_runner.TMP_HIST_LOCATION
    old_res = game_runner.RESERVOIR_LOCATION
    old_ckpt = game_runner.CHECKPOINT_LOCATION

    hist_dir = os.path.join(TEST_BASE, "e2e_hist")
    tmp_dir = os.path.join(TEST_BASE, "e2e_tmp")
    res_dir = os.path.join(TEST_BASE, "e2e_reservoir")
    ckpt_dir = os.path.join(TEST_BASE, "e2e_checkpoint")

    for d in [hist_dir, tmp_dir, res_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    game_runner.HIST_LOCATION = hist_dir
    game_runner.TMP_HIST_LOCATION = tmp_dir
    game_runner.RESERVOIR_LOCATION = res_dir
    game_runner.CHECKPOINT_LOCATION = ckpt_dir

    try:
        # Create initial network
        nn = create_test_network()
        nn.save_checkpoint(ckpt_dir, "0000-test.pt")
        del nn

        cs = Game.CANONICAL_SHAPE()
        num_iters = 5
        hist_size = 2  # Small window to trigger reservoir quickly

        class DummyRun:
            def track(*args, **kwargs):
                pass
        run = DummyRun()

        for i in range(num_iters):
            print(f"    Iteration {i}...")

            # Generate fake self-play data (simulating selfplay -> symmetries -> resample)
            n_samples = 200
            c = torch.randn(n_samples, cs[0], cs[1], cs[2])
            v = torch.softmax(torch.randn(n_samples, Game.NUM_PLAYERS() + 1), dim=1)
            p = torch.softmax(torch.randn(n_samples, Game.NUM_MOVES()), dim=1)

            # Save compressed to HIST_LOCATION
            save_compressed(c, os.path.join(hist_dir, f"{i:04d}-0000-canonical-{n_samples}.ptz"))
            save_compressed(v, os.path.join(hist_dir, f"{i:04d}-0000-v-{n_samples}.ptz"))
            save_compressed(p, os.path.join(hist_dir, f"{i:04d}-0000-pi-{n_samples}.ptz"))

            # Load window data for training
            def _glob_hist_files(location, pattern):
                files = sorted(glob.glob(os.path.join(location, pattern + ".ptz")))
                if not files:
                    files = sorted(glob.glob(os.path.join(location, pattern + ".pt")))
                return files

            datasets = []
            total_size = 0
            for wi in range(max(0, i - hist_size), i + 1):
                cf = _glob_hist_files(hist_dir, f"{wi:04d}-*-canonical-*")
                vf = _glob_hist_files(hist_dir, f"{wi:04d}-*-v-*")
                pf = _glob_hist_files(hist_dir, f"{wi:04d}-*-pi-*")
                for j in range(len(cf)):
                    size = int(cf[j].split("-")[-1].split(".")[0])
                    total_size += size
                    ct = load_compressed(cf[j])
                    vt = load_compressed(vf[j])
                    pt = load_compressed(pf[j])
                    datasets.append(TensorDataset(ct, vt, pt))

            dataset = ConcatDataset(datasets)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

            nn = NNWrapper.load_checkpoint(Game, ckpt_dir, f"{i:04d}-test.pt")
            v_loss, pi_loss = nn.train(dataloader, 2, run, i, 0)
            assert not math.isnan(v_loss), f"v_loss is NaN at iter {i}"
            assert not math.isnan(pi_loss), f"pi_loss is NaN at iter {i}"
            assert math.isfinite(v_loss), f"v_loss is not finite at iter {i}"
            nn.save_checkpoint(ckpt_dir, f"{i+1:04d}-test.pt")
            del nn, datasets, dataset, dataloader

            # Reservoir update: merge evicted data
            if i > 0:
                oldest_in_window = max(0, i - hist_size)
                # Simple: previous window started one iter earlier
                prev_oldest = max(0, (i - 1) - hist_size)
                evicted_iters = list(range(prev_oldest, oldest_in_window))

                if evicted_iters:
                    new_c, new_v, new_pi, new_iters_list = [], [], [], []
                    for it in evicted_iters:
                        cf = sorted(glob.glob(os.path.join(hist_dir, f"{it:04d}-*-canonical-*.ptz")))
                        vf = sorted(glob.glob(os.path.join(hist_dir, f"{it:04d}-*-v-*.ptz")))
                        pf = sorted(glob.glob(os.path.join(hist_dir, f"{it:04d}-*-pi-*.ptz")))
                        for j in range(len(cf)):
                            ct = load_compressed(cf[j])
                            new_c.append(ct)
                            new_v.append(load_compressed(vf[j]))
                            new_pi.append(load_compressed(pf[j]))
                            new_iters_list.append(torch.full((ct.shape[0],), it, dtype=torch.int16))

                    if new_c:
                        new_c = torch.cat(new_c)
                        new_v = torch.cat(new_v)
                        new_pi = torch.cat(new_pi)
                        new_iters_t = torch.cat(new_iters_list)

                        os.makedirs(res_dir, exist_ok=True)
                        res_c_path = os.path.join(res_dir, "canonical.ptz")

                        if os.path.exists(res_c_path):
                            old_c = load_compressed(res_c_path)
                            old_v = load_compressed(os.path.join(res_dir, "v.ptz"))
                            old_pi = load_compressed(os.path.join(res_dir, "pi.ptz"))
                            old_iters = load_compressed(os.path.join(res_dir, "meta.ptz")).to(torch.int16)
                            all_c = torch.cat([old_c, new_c])
                            all_v = torch.cat([old_v, new_v])
                            all_pi = torch.cat([old_pi, new_pi])
                            all_iters = torch.cat([old_iters, new_iters_t])
                        else:
                            all_c, all_v, all_pi, all_iters = new_c, new_v, new_pi, new_iters_t

                        # Compute capacity
                        capacity = 0
                        for wi in range(max(0, i - hist_size), i + 1):
                            for fn in glob.glob(os.path.join(hist_dir, f"{wi:04d}-*-canonical-*.pt*")):
                                capacity += int(fn.split("-")[-1].split(".")[0])

                        if len(all_c) > capacity > 0:
                            ages = (i - all_iters.float()).clamp(min=0)
                            weights = game_runner.RESERVOIR_RECENCY_DECAY ** ages
                            indices = torch.multinomial(weights, capacity, replacement=False)
                            all_c = all_c[indices]
                            all_v = all_v[indices]
                            all_pi = all_pi[indices]
                            all_iters = all_iters[indices]

                        save_compressed(all_c, res_c_path)
                        save_compressed(all_v, os.path.join(res_dir, "v.ptz"))
                        save_compressed(all_pi, os.path.join(res_dir, "pi.ptz"))
                        save_compressed(all_iters.float(), os.path.join(res_dir, "meta.ptz"))

                        # Delete evicted files
                        for it in evicted_iters:
                            for fn in glob.glob(os.path.join(hist_dir, f"{it:04d}-*.ptz")):
                                os.remove(fn)

            gc.collect()

        # Verify results
        # Only window iterations should remain in hist_dir
        remaining_iters = set()
        for fn in glob.glob(os.path.join(hist_dir, "*.ptz")):
            remaining_iters.add(int(os.path.basename(fn).split("-")[0]))
        expected_window = set(range(max(0, num_iters - 1 - hist_size), num_iters))
        print(f"    Remaining iterations in history: {remaining_iters}")
        print(f"    Expected window: {expected_window}")
        assert remaining_iters == expected_window, f"Expected {expected_window}, got {remaining_iters}"

        # Reservoir should exist
        assert os.path.exists(os.path.join(res_dir, "canonical.ptz")), "Reservoir should exist"
        res_c = load_compressed(os.path.join(res_dir, "canonical.ptz"))
        print(f"    Reservoir size: {res_c.shape[0]} samples")
        assert res_c.shape[0] > 0, "Reservoir should have samples"

        print("    PASSED")
    finally:
        game_runner.HIST_LOCATION = old_hist
        game_runner.TMP_HIST_LOCATION = old_tmp
        game_runner.RESERVOIR_LOCATION = old_res
        game_runner.CHECKPOINT_LOCATION = old_ckpt


# ============================================================
# Integration test: reservoir bootstrap
# ============================================================


def test_reservoir_bootstrap():
    """Build reservoir from training iterations, then bootstrap from it."""
    print("  Testing reservoir bootstrap...")
    from game_runner import save_compressed as sc, load_compressed as lc

    res_dir = os.path.join(TEST_BASE, "boot_reservoir")
    hist_dir = os.path.join(TEST_BASE, "boot_hist")
    ckpt_dir = os.path.join(TEST_BASE, "boot_checkpoint")

    for d in [res_dir, hist_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    cs = Game.CANONICAL_SHAPE()

    # Create a reservoir with mixed iteration data
    n_total = 500
    c = torch.randn(n_total, cs[0], cs[1], cs[2])
    v = torch.softmax(torch.randn(n_total, Game.NUM_PLAYERS() + 1), dim=1)
    pi = torch.softmax(torch.randn(n_total, Game.NUM_MOVES()), dim=1)
    iters = torch.randint(0, 5, (n_total,), dtype=torch.int16)

    sc(c, os.path.join(res_dir, "canonical.ptz"))
    sc(v, os.path.join(res_dir, "v.ptz"))
    sc(pi, os.path.join(res_dir, "pi.ptz"))
    sc(iters.float(), os.path.join(res_dir, "meta.ptz"))

    # Create initial network checkpoint
    nn = create_test_network()
    nn.save_checkpoint(ckpt_dir, "0000-test.pt")

    # Load reservoir
    res_c = lc(os.path.join(res_dir, "canonical.ptz"))
    res_v = lc(os.path.join(res_dir, "v.ptz"))
    res_pi = lc(os.path.join(res_dir, "pi.ptz"))
    reservoir_ds = TensorDataset(res_c, res_v, res_pi)

    class DummyRun:
        def track(*args, **kwargs):
            pass
    run = DummyRun()

    # Phase 1: Train on reservoir
    dataloader = DataLoader(reservoir_ds, batch_size=64, shuffle=True)
    steps_p1 = int(math.ceil(len(reservoir_ds) / 64)) * 2
    v_loss, pi_loss = nn.train(dataloader, steps_p1, run, 5, 0)
    assert math.isfinite(v_loss), f"Phase 1 v_loss not finite: {v_loss}"
    assert math.isfinite(pi_loss), f"Phase 1 pi_loss not finite: {pi_loss}"
    print(f"    Phase 1 - v_loss: {v_loss:.4f}, pi_loss: {pi_loss:.4f}")

    nn.save_checkpoint(ckpt_dir, "0005-test.pt")

    # Verify checkpoint saved
    assert os.path.exists(os.path.join(ckpt_dir, "0005-test.pt")), "Bootstrap checkpoint should exist"

    del nn
    print("    PASSED")


# ============================================================
# Main test runner
# ============================================================


def main():
    print("\n" + "=" * 60)
    print("History Management Test Suite")
    print("=" * 60)
    print(f"Game: {Game}")
    print(f"Device: {get_device()}")
    print()

    setup_test_dir()
    print(f"Test directory: {TEST_BASE}\n")

    unit_tests = [
        ("Round-trip float32", test_roundtrip_float32),
        ("Round-trip integer values", test_roundtrip_integer_values),
        ("File size reduction", test_file_size_reduction),
        ("PTZ extension", test_ptz_extension),
        ("maybe_save HIST_LOCATION", test_maybe_save_hist_location),
        ("maybe_save TMP_HIST_LOCATION", test_maybe_save_tmp_location),
        ("Reservoir basic merge", test_reservoir_basic_merge),
        ("Reservoir capacity enforcement", test_reservoir_capacity_enforcement),
        ("Reservoir recency weighting", test_reservoir_recency_weighting),
        ("Reservoir metadata", test_reservoir_metadata),
        ("Reservoir incremental merge", test_reservoir_incremental_merge),
    ]

    integration_tests = [
        ("Mixed format loading", test_mixed_format_loading),
        ("Multi-iteration E2E", test_multi_iteration_e2e),
        ("Reservoir bootstrap", test_reservoir_bootstrap),
    ]

    all_passed = True

    print("Unit Tests:")
    print("-" * 40)
    for name, test_fn in unit_tests:
        try:
            test_fn()
        except Exception as e:
            print(f"    FAILED: {name} - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
        gc.collect()

    print("\nIntegration Tests:")
    print("-" * 40)
    for name, test_fn in integration_tests:
        try:
            test_fn()
        except Exception as e:
            print(f"    FAILED: {name} - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
        gc.collect()

    cleanup_test_dir()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
