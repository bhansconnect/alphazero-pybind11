"""Network capacity fitting benchmark.

Loads existing training data from an experiment directory, then benchmarks
multiple network architectures on identical batches to compare loss, inference
speed, and memory usage.

Usage:
    uv run python src/network_pareto.py data/star_gambit_skirmish/densenet-4d-16c-3k-120sims-board-size
    # Then interactively type configs: 4d16c, 6d24c, 4d32c-k5, go
"""

import argparse
import itertools
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import numpy as np

from config import load_config, find_latest_checkpoint
from game_runner import glob_file_triples, _load_and_select, calc_hist_size
from neural_net import NNArgs, NNWrapper, NNArch, get_device
from policy_metrics import (
    batch_top1_agreement, batch_top_k_agreement, batch_kl_divergence,
    batch_policy_entropy, batch_total_variation,
)


@dataclass
class BenchResult:
    label: str
    params: int
    mem_mb: float
    infer_ms: float
    v_loss: float
    pi_loss: float
    total_loss: float
    steps: int
    time_min: float
    losses_log: list = field(default_factory=list)
    # Validation metrics (None when no val set)
    val_v_loss: Optional[float] = None
    val_pi_loss: Optional[float] = None
    val_total_loss: Optional[float] = None
    top1_agree: Optional[float] = None
    top3_agree: Optional[float] = None
    kl_div: Optional[float] = None
    pi_entropy_net: Optional[float] = None
    pi_entropy_tgt: Optional[float] = None
    v_tv: Optional[float] = None


def parse_config_string(s):
    """Parse a config string like '6d24c-k5-hc48-resnet' into NNArgs kwargs.

    Format: {depth}d{channels}c[-k{N}][-hc{N}][-resnet]
    Returns (label, kwargs_dict) or raises ValueError.
    """
    s = s.strip()
    # Match the required depth/channels prefix
    m = re.match(r'^(\d+)d(\d+)c', s)
    if not m:
        raise ValueError(f"Bad format: expected {{depth}}d{{channels}}c, got '{s}'")
    depth = int(m.group(1))
    channels = int(m.group(2))
    rest = s[m.end():]

    kwargs = {
        'depth': depth,
        'num_channels': channels,
        'dense_net': True,
        'kernel_size': 3,
        'head_channels': 32,
    }

    # Parse optional modifiers
    while rest:
        if not rest.startswith('-'):
            raise ValueError(f"Unexpected modifier: '{rest}' (expected dash separator)")
        rest = rest[1:]

        km = re.match(r'k(\d+)', rest)
        if km:
            kwargs['kernel_size'] = int(km.group(1))
            rest = rest[km.end():]
            continue

        hm = re.match(r'hc(\d+)', rest)
        if hm:
            kwargs['head_channels'] = int(hm.group(1))
            rest = rest[hm.end():]
            continue

        if rest.startswith('resnet'):
            kwargs['dense_net'] = False
            rest = rest[6:]
            continue

        raise ValueError(f"Unknown modifier: '-{rest}'")

    return s, kwargs


def expand_config_string(s):
    """Expand comma-separated values and optional groups into a cross-product of configs.

    Comma-separated values in any position expand as cross-product:
        '4,6,8d16,24,32c' -> 9 configs
        '4,6d16c-k3,5'    -> 4 configs
        '4d16c'           -> 1 config (backward compatible)

    Parenthesized groups are optional (with and without):
        '4d16,32c(-hc64)' -> 4 configs: 4d16c, 4d32c, 4d16c-hc64, 4d32c-hc64
        '4d16c(-resnet)'  -> 2 configs: 4d16c, 4d16c-resnet
    """
    s = s.strip()

    # Extract optional groups: (-modifier) anywhere in string
    optional_groups = []
    base = s
    while True:
        m = re.search(r'\((-[^)]+)\)', base)
        if not m:
            break
        optional_groups.append(m.group(1))
        base = base[:m.start()] + base[m.end():]

    # Parse base: {depths}d{channels}c[-k{kernels}][-hc{heads}][-resnet]
    m = re.match(r'^([\d,]+)d([\d,]+)c', base)
    if not m:
        return [s]  # let parse_config_string handle the error

    depths = m.group(1).split(',')
    channels = m.group(2).split(',')
    rest = base[m.end():]

    # Parse modifiers from rest
    kernels = None
    heads = None
    resnet = False
    remaining = rest

    while remaining:
        if not remaining.startswith('-'):
            break
        tail = remaining[1:]

        km = re.match(r'k([\d,]+)', tail)
        if km:
            kernels = km.group(1).split(',')
            remaining = tail[km.end():]
            continue

        hm = re.match(r'hc([\d,]+)', tail)
        if hm:
            heads = hm.group(1).split(',')
            remaining = tail[hm.end():]
            continue

        if tail.startswith('resnet'):
            resnet = True
            remaining = tail[6:]
            continue

        break  # unknown modifier, let parse_config_string handle it

    # Build cross-product of value lists
    lists = [depths, channels]
    if kernels:
        lists.append(kernels)
    if heads:
        lists.append(heads)

    # Optional groups: each adds [("", group_str)] to the product
    for grp in optional_groups:
        lists.append(["", grp])

    results = []
    for combo in itertools.product(*lists):
        idx = 0
        cfg = f"{combo[idx]}d{combo[idx+1]}c"
        idx += 2
        if kernels:
            cfg += f"-k{combo[idx]}"
            idx += 1
        if heads:
            cfg += f"-hc{combo[idx]}"
            idx += 1
        if resnet:
            cfg += "-resnet"
        if remaining:
            cfg += remaining
        # Append optional group selections
        for grp in optional_groups:
            cfg += combo[idx]
            idx += 1
        results.append(cfg)

    return results


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_config_only(source_dir):
    """Load just the config from an experiment directory (no data)."""
    config_path = os.path.join(source_dir, "config.yaml")
    config = load_config(config_path, {}, warn=False)
    return config


def _load_iter_range(hist_dir, iter_range, max_samples, num_workers):
    """Load data from a range of iterations. Returns (c, v, pi) tensors or None if empty."""
    file_triples = []
    for i in iter_range:
        triples = glob_file_triples(hist_dir, f"{i:04d}-*-canonical-*.ptz")
        if not triples:
            triples = glob_file_triples(hist_dir, f"{i:04d}-*-canonical-*.pt")
        file_triples.extend(triples)

    if not file_triples:
        return None, None, None

    sizes = [size for _, _, _, size in file_triples]
    total_size = sum(sizes)

    # Pre-select sample indices
    if total_size <= max_samples:
        selected_per_file = [list(range(s)) for s in sizes]
        samples_selected = total_size
    else:
        rng = random.Random(42)
        all_indices = sorted(rng.sample(range(total_size), max_samples))
        cum_sizes = []
        cum = 0
        for s in sizes:
            cum_sizes.append(cum)
            cum += s
        selected_per_file = [[] for _ in range(len(file_triples))]
        fi = 0
        for idx_val in all_indices:
            while fi < len(cum_sizes) - 1 and idx_val >= cum_sizes[fi + 1]:
                fi += 1
            selected_per_file[fi].append(idx_val - cum_sizes[fi])
        samples_selected = max_samples

    # Load files and extract selected indices (parallel)
    work_items = [
        (c_path, v_path, pi_path, selected_per_file[i])
        for i, (c_path, v_path, pi_path, _) in enumerate(file_triples)
        if selected_per_file[i]
    ]

    acc_c, acc_v, acc_pi = [], [], []
    if num_workers <= 1:
        for item in work_items:
            c, v, pi = _load_and_select(item)
            acc_c.append(c); acc_v.append(v); acc_pi.append(pi)
    else:
        results = [None] * len(work_items)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(_load_and_select, item): i for i, item in enumerate(work_items)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
        for c, v, pi in results:
            acc_c.append(c); acc_v.append(v); acc_pi.append(pi)

    all_c = torch.cat(acc_c).float()
    all_v = torch.cat(acc_v).float()
    all_pi = torch.cat(acc_pi).float()
    return all_c, all_v, all_pi


def load_data(source_dir, max_samples, num_workers, config=None,
              val_iters=0, val_max_samples=0):
    """Load training data with optional validation split.

    When val_iters > 0, the most recent val_iters iterations are held out for
    validation, and the remaining earlier iterations form the training set.

    Returns (config, train_c, train_v, train_pi, val_c, val_v, val_pi).
    Val tensors are None when val_iters=0.
    """
    if config is None:
        config = load_config_only(source_dir)
    paths = config.resolve_paths(source_dir)

    # Find latest iteration
    iteration = find_latest_checkpoint(paths["checkpoint"])
    if iteration == 0:
        print(f"Error: no checkpoints found in {paths['checkpoint']}")
        sys.exit(1)

    hist_size = calc_hist_size(config, iteration)
    hist_dir = paths["history"]

    window_start = max(0, iteration - hist_size)
    window_end = iteration  # inclusive

    # Clamp val_iters
    val_iters = min(val_iters, hist_size // 2)

    if val_iters > 0:
        val_start = window_end - val_iters + 1
        train_range = range(window_start, val_start)
        val_range = range(val_start, window_end + 1)
        print(f"Train iters: {train_range.start}-{train_range.stop - 1}, "
              f"Val iters: {val_range.start}-{val_range.stop - 1}")
    else:
        train_range = range(window_start, window_end + 1)
        val_range = None

    # Load training data
    print(f"Loading training data...")
    train_c, train_v, train_pi = _load_iter_range(
        hist_dir, train_range, max_samples, num_workers)
    if train_c is None:
        print(f"Error: no training data files found in {source_dir}")
        sys.exit(1)
    print(f"Train: {train_c.shape[0]:,} samples (iters {train_range.start}-{train_range.stop - 1})")

    # Load validation data
    val_c = val_v = val_pi = None
    if val_range is not None:
        print(f"Loading validation data...")
        val_c, val_v, val_pi = _load_iter_range(
            hist_dir, val_range, val_max_samples, num_workers)
        if val_c is not None:
            print(f"Val: {val_c.shape[0]:,} samples (iters {val_range.start}-{val_range.stop - 1})")
        else:
            print(f"Warning: no validation data found in val iters, using train-only mode")

    return config, train_c, train_v, train_pi, val_c, val_v, val_pi


def pregenerate_batches(n_samples, batch_size, steps):
    """Pre-generate batch index tensors with a fixed seed."""
    rng = torch.Generator()
    rng.manual_seed(123)
    indices_list = []
    for _ in range(steps):
        idx = torch.randint(0, n_samples, (batch_size,), generator=rng)
        indices_list.append(idx)
    return indices_list


def benchmark_inference(nn_wrapper, game, batch_size):
    """Benchmark inference speed using CUDA events. Returns ms per batch."""
    device = nn_wrapper.device
    cs = game.CANONICAL_SHAPE()
    dummy = torch.randn(batch_size, *cs, dtype=torch.float32, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            nn_wrapper.process(dummy)

    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        reps = 200
        timings = []
        with torch.no_grad():
            for _ in range(reps):
                starter.record()
                nn_wrapper.process(dummy)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        return np.median(timings)
    else:
        # CPU/MPS fallback
        reps = 100
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(reps):
                nn_wrapper.process(dummy)
        elapsed = (time.perf_counter() - t0) / reps * 1000
        return elapsed


def train_config(nn_wrapper, all_c, all_v, all_pi, batch_indices, lr, steps,
                 log_interval, device):
    """Train a single config on pre-generated batches. Returns (losses_log, final_time)."""
    nn_wrapper.nnet.train()
    optimizer = torch.optim.SGD(
        nn_wrapper.nnet.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)

    non_blocking = device.type == 'cuda'
    losses_log = []  # (step, v_loss, pi_loss, total)
    v_sum = 0.0
    pi_sum = 0.0

    t0 = time.perf_counter()
    for step_i, idx in enumerate(batch_indices):
        c_batch = all_c[idx].to(device, non_blocking=non_blocking)
        v_batch = all_v[idx].to(device, non_blocking=non_blocking)
        pi_batch = all_pi[idx].to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        out_v, out_pi = nn_wrapper.nnet(c_batch)
        l_v = nn_wrapper.loss_v(v_batch, out_v)
        l_pi = nn_wrapper.loss_pi(pi_batch, out_pi)
        loss = l_v + l_pi
        loss.backward()
        optimizer.step()
        scheduler.step()

        v_val = l_v.item()
        pi_val = l_pi.item()
        v_sum += v_val
        pi_sum += pi_val

        step_num = step_i + 1
        if step_num % log_interval == 0 or step_num == steps:
            losses_log.append((step_num, v_sum / step_num, pi_sum / step_num, (v_sum + pi_sum) / step_num))

    elapsed = time.perf_counter() - t0
    return losses_log, elapsed


def eval_loss(nn_wrapper, all_c, all_v, all_pi, batch_size, device):
    """Full eval-mode pass over the dataset. Returns (v_loss, pi_loss)."""
    nn_wrapper.nnet.eval()
    non_blocking = device.type == 'cuda'
    n = all_c.shape[0]
    v_total = 0.0
    pi_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            c = all_c[start:end].to(device, non_blocking=non_blocking)
            v = all_v[start:end].to(device, non_blocking=non_blocking)
            pi = all_pi[start:end].to(device, non_blocking=non_blocking)

            out_v, out_pi = nn_wrapper.nnet(c)
            v_total += nn_wrapper.loss_v(v, out_v).item()
            pi_total += nn_wrapper.loss_pi(pi, out_pi).item()
            n_batches += 1

    return v_total / n_batches, pi_total / n_batches


def eval_with_metrics(nn_wrapper, val_c, val_v, val_pi, batch_size, device):
    """Full eval pass over validation data with quality metrics.

    Returns dict with val_v_loss, val_pi_loss, val_total_loss, and policy/value metrics.
    """
    nn_wrapper.nnet.eval()
    non_blocking = device.type == 'cuda'
    n = val_c.shape[0]
    v_total = 0.0
    pi_total = 0.0
    n_batches = 0

    # Collect outputs for metrics
    all_net_pi = []
    all_tgt_pi = []
    all_net_v = []
    all_tgt_v = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            c = val_c[start:end].to(device, non_blocking=non_blocking)
            v = val_v[start:end].to(device, non_blocking=non_blocking)
            pi = val_pi[start:end].to(device, non_blocking=non_blocking)

            out_v, out_pi = nn_wrapper.nnet(c)
            v_total += nn_wrapper.loss_v(v, out_v).item()
            pi_total += nn_wrapper.loss_pi(pi, out_pi).item()
            n_batches += 1

            # Convert log-softmax to probabilities
            net_pi_probs = torch.exp(out_pi).cpu().numpy()
            tgt_pi_probs = pi.cpu().numpy()
            all_net_pi.append(net_pi_probs)
            all_tgt_pi.append(tgt_pi_probs)

            # Value distributions
            net_v_probs = torch.exp(out_v).cpu().numpy()
            tgt_v_probs = v.cpu().numpy()
            all_net_v.append(net_v_probs)
            all_tgt_v.append(tgt_v_probs)

    val_v_loss = v_total / n_batches
    val_pi_loss = pi_total / n_batches

    # Concatenate all batches
    net_pi = np.concatenate(all_net_pi, axis=0)
    tgt_pi = np.concatenate(all_tgt_pi, axis=0)
    net_v = np.concatenate(all_net_v, axis=0)
    tgt_v = np.concatenate(all_tgt_v, axis=0)

    return {
        "val_v_loss": val_v_loss,
        "val_pi_loss": val_pi_loss,
        "val_total_loss": val_v_loss + val_pi_loss,
        "top1_agree": batch_top1_agreement(net_pi, tgt_pi),
        "top3_agree": batch_top_k_agreement(net_pi, tgt_pi, 3),
        "kl_div": batch_kl_divergence(tgt_pi, net_pi),
        "pi_entropy_net": batch_policy_entropy(net_pi),
        "pi_entropy_tgt": batch_policy_entropy(tgt_pi),
        "v_tv": batch_total_variation(net_v, tgt_v),
    }


def is_pareto_optimal(points):
    """Given Nx3 array (mem, infer_ms, loss), return boolean mask of Pareto-optimal points.

    A point is Pareto-optimal if no other point is <= on all 3 objectives and < on at least one.
    """
    n = len(points)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j <= i on all dims and j < i on at least one
            if (np.all(points[j] <= points[i]) and np.any(points[j] < points[i])):
                is_optimal[i] = False
                break
    return is_optimal


def collect_configs(config, args):
    """Interactive config input loop. Returns list of (label, NNArgs)."""
    Game = config.Game

    print("Config format: {depth}d{channels}c[-k{N}][-hc{N}][-resnet]")
    print("  Comma-separated values expand as cross-product:")
    print("    4,6d16,24c          -> 4 configs (2 depths x 2 channels)")
    print("    4,6,8d16,24,32c     -> 9 configs")
    print("    4,6d16c-k3,5        -> 4 configs")
    print("  Parenthesized modifiers are optional (with and without):")
    print("    4d16,32c(-hc64)     -> 4 configs: 4d16c, 4d32c, +hc64 variants")
    print("    4d16c(-resnet)      -> 2 configs: 4d16c, 4d16c-resnet")
    print("  Modifiers (all optional, dash-separated):")
    print("    -k{N}      kernel size (default: 3)")
    print("    -hc{N}     head channels (default: 32)")
    print("    -resnet    use ResNet instead of DenseNet")
    print("  Examples: 6d24c  4d32c-k5  6d24c-hc64  4,6,8d16,24,32c-resnet")
    print()
    print("Add configs (empty or 'go' to start):")

    configs = []
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() == 'go':
            break
        expanded = expand_config_string(line)
        for single in expanded:
            try:
                label, kwargs = parse_config_string(single)
                kwargs['star_gambit_spatial'] = config.star_gambit_spatial
                kwargs['head_pool'] = config.head_pool
                nn_args = NNArgs(lr=args.lr, cv=config.cv, **kwargs)
                tmp_model = NNArch(Game, nn_args)
                params = count_params(tmp_model)
                del tmp_model
                print(f"  Added: {label}  ({params:,} params)")
                configs.append((label, nn_args))
            except ValueError as e:
                print(f"  Error: {e}")
            except Exception as e:
                print(f"  Error creating network: {e}")

    return configs


def print_results_table(results, has_val=False):
    """Print the results table and learning curves."""
    # Filter out OOM entries for Pareto, but show them in table
    valid = [(i, r) for i, r in enumerate(results) if r.total_loss != float('inf')]
    oom = [(i, r) for i, r in enumerate(results) if r.total_loss == float('inf')]

    # Sort: valid by val total if available, else train total. OOM at end.
    if has_val:
        valid.sort(key=lambda x: x[1].val_total_loss if x[1].val_total_loss is not None else float('inf'))
    else:
        valid.sort(key=lambda x: x[1].total_loss)
    ordered = valid + oom

    sort_col = "val total" if has_val else "total loss"
    print(f"\n{'='*60}")
    print(f"RESULTS (sorted by {sort_col})")
    print(f"{'='*60}")

    # Pareto frontier (valid entries only) — use val_total if available
    if valid:
        loss_col = [(r.val_total_loss if has_val and r.val_total_loss is not None else r.total_loss) for _, r in valid]
        valid_points = np.array([[r.mem_mb, r.infer_ms, lc] for (_, r), lc in zip(valid, loss_col)])
        valid_pareto = is_pareto_optimal(valid_points)
    else:
        valid_pareto = np.array([], dtype=bool)

    header = f"{'Config':<22} {'Params':>10} {'Mem MB':>8} {'Infer ms':>9} {'V Loss':>8} {'Pi Loss':>9} {'Total':>8}"
    if has_val:
        header += f" {'ValTot':>8}"
    header += f" {'Steps':>6} {'Time':>6}"
    print(header)
    print("-" * len(header))

    vi = 0
    for orig_i, r in ordered:
        if r.total_loss == float('inf'):
            line = f"{r.label:<22} {r.params:>10,} {'OOM':>8s} {r.infer_ms:>9.1f} {'OOM':>8s} {'OOM':>9s} {'OOM':>8s}"
            if has_val:
                line += f" {'OOM':>8s}"
            line += f" {'-':>6s} {'-':>6s}"
            print(line)
        else:
            star = " ***" if valid_pareto[vi] else ""
            line = f"{r.label:<22} {r.params:>10,} {r.mem_mb:>8.1f} {r.infer_ms:>9.1f} {r.v_loss:>8.4f} {r.pi_loss:>9.4f} {r.total_loss:>8.4f}"
            if has_val:
                vt = r.val_total_loss
                line += f" {vt:>8.4f}" if vt is not None else f" {'N/A':>8s}"
            line += f" {r.steps:>6} {r.time_min:>5.1f}m{star}"
            print(line)
            vi += 1

    if any(valid_pareto):
        print(f"\n*** = Pareto-optimal (memory vs inference vs loss)")

    # Learning curves (valid entries only)
    valid_results = [r for _, r in valid]
    if not valid_results:
        return

    print(f"\n{'='*60}")
    print("LEARNING CURVES (running avg total loss)")
    print(f"{'='*60}")

    all_steps = sorted(set(s for r in valid_results for s, _, _, _ in r.losses_log))

    labels = [r.label for r in valid_results]
    hdr = f"{'Step':>6}"
    for lbl in labels:
        hdr += f"  {lbl:>12}"
    print(hdr)
    print("-" * len(hdr))

    logs = {}
    for r in valid_results:
        logs[r.label] = {s: total for s, _, _, total in r.losses_log}

    for step in all_steps:
        row = f"{step:>6}"
        for lbl in labels:
            val = logs[lbl].get(step)
            if val is not None:
                row += f"  {val:>12.4f}"
            else:
                row += f"  {'':>12}"
        print(row)


def print_val_metrics_table(results):
    """Print validation quality metrics table."""
    valid = [r for r in results if r.val_total_loss is not None]
    if not valid:
        return

    valid.sort(key=lambda r: r.val_total_loss)

    print(f"\n{'='*60}")
    print("VALIDATION QUALITY METRICS")
    print(f"{'='*60}")

    header = (f"{'Config':<22} {'ValTot':>8} {'Top1%':>7} {'Top3%':>7} "
              f"{'KL Div':>8} {'Net Ent':>8} {'Tgt Ent':>8} {'V TV':>7}")
    print(header)
    print("-" * len(header))

    for r in valid:
        top1 = f"{r.top1_agree * 100:>6.1f}%" if r.top1_agree is not None else f"{'N/A':>7s}"
        top3 = f"{r.top3_agree * 100:>6.1f}%" if r.top3_agree is not None else f"{'N/A':>7s}"
        kl = f"{r.kl_div:>8.4f}" if r.kl_div is not None else f"{'N/A':>8s}"
        ne = f"{r.pi_entropy_net:>8.4f}" if r.pi_entropy_net is not None else f"{'N/A':>8s}"
        te = f"{r.pi_entropy_tgt:>8.4f}" if r.pi_entropy_tgt is not None else f"{'N/A':>8s}"
        vtv = f"{r.v_tv:>7.4f}" if r.v_tv is not None else f"{'N/A':>7s}"
        print(f"{r.label:<22} {r.val_total_loss:>8.4f} {top1} {top3} {kl} {ne} {te} {vtv}")


def save_charts(results, save_dir):
    """Save analysis charts to save_dir. Skip gracefully if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping charts")
        return

    os.makedirs(save_dir, exist_ok=True)
    valid = [r for r in results if r.total_loss != float('inf')]
    if not valid:
        return

    has_val = any(r.val_total_loss is not None for r in valid)

    # Chart 1: Train vs Val loss grouped bars
    if has_val:
        val_valid = [r for r in valid if r.val_total_loss is not None]
        if val_valid:
            fig, ax = plt.subplots(figsize=(max(8, len(val_valid) * 1.2), 5))
            labels = [r.label for r in val_valid]
            x = np.arange(len(labels))
            width = 0.35
            train_vals = [r.total_loss for r in val_valid]
            val_vals = [r.val_total_loss for r in val_valid]
            ax.bar(x - width/2, train_vals, width, label='Train', color='tab:blue', alpha=0.8)
            ax.bar(x + width/2, val_vals, width, label='Val', color='tab:orange', alpha=0.8)
            ax.set_xlabel('Config')
            ax.set_ylabel('Total Loss')
            ax.set_title('Train vs Validation Loss')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            fig.tight_layout()
            path = os.path.join(save_dir, "pareto_train_val_loss.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved: {path}")

    # Chart 2: Top-1 / Top-3 agreement bars
    if has_val:
        val_valid = [r for r in valid if r.top1_agree is not None]
        if val_valid:
            fig, ax = plt.subplots(figsize=(max(8, len(val_valid) * 1.2), 5))
            labels = [r.label for r in val_valid]
            x = np.arange(len(labels))
            width = 0.35
            top1_vals = [r.top1_agree * 100 for r in val_valid]
            top3_vals = [r.top3_agree * 100 for r in val_valid]
            ax.bar(x - width/2, top1_vals, width, label='Top-1', color='tab:blue', alpha=0.8)
            ax.bar(x + width/2, top3_vals, width, label='Top-3', color='tab:green', alpha=0.8)
            ax.set_xlabel('Config')
            ax.set_ylabel('Agreement (%)')
            ax.set_title('Policy Agreement with MCTS Target')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            fig.tight_layout()
            path = os.path.join(save_dir, "pareto_agreement.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved: {path}")

    # Chart 3: Learning curves
    configs_with_logs = [r for r in valid if r.losses_log]
    if configs_with_logs:
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in configs_with_logs:
            steps_arr = [s for s, _, _, _ in r.losses_log]
            totals = [t for _, _, _, t in r.losses_log]
            ax.plot(steps_arr, totals, label=r.label, linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Running Avg Total Loss')
        ax.set_title('Learning Curves')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(save_dir, "pareto_learning_curves.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    # Chart 4: Params vs loss scatter with Pareto frontier
    loss_col = [(r.val_total_loss if has_val and r.val_total_loss is not None else r.total_loss) for r in valid]
    params_arr = np.array([r.params for r in valid])
    loss_arr = np.array(loss_col)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Compute Pareto-optimal for this 2D view
    pts_2d = np.column_stack([params_arr, loss_arr])
    n = len(pts_2d)
    pareto_2d = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(pts_2d[j] <= pts_2d[i]) and np.any(pts_2d[j] < pts_2d[i]):
                pareto_2d[i] = False
                break

    ax.scatter(params_arr[~pareto_2d], loss_arr[~pareto_2d],
               color='tab:blue', alpha=0.7, s=60, label='Dominated')
    ax.scatter(params_arr[pareto_2d], loss_arr[pareto_2d],
               color='tab:red', alpha=0.9, s=100, marker='*', label='Pareto-optimal')
    # Label points
    for i, r in enumerate(valid):
        ax.annotate(r.label, (params_arr[i], loss_arr[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    # Draw Pareto frontier line
    pareto_idx = np.where(pareto_2d)[0]
    if len(pareto_idx) > 1:
        order = np.argsort(params_arr[pareto_idx])
        ax.plot(params_arr[pareto_idx[order]], loss_arr[pareto_idx[order]],
                'r--', alpha=0.5, linewidth=1)

    loss_label = 'Val Total Loss' if has_val else 'Total Loss'
    ax.set_xlabel('Parameters')
    ax.set_ylabel(loss_label)
    ax.set_title(f'Parameters vs {loss_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "pareto_params_vs_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def save_results_npz(results, save_dir):
    """Save raw results data as npz."""
    os.makedirs(save_dir, exist_ok=True)
    data = {}
    data["labels"] = np.array([r.label for r in results])
    data["params"] = np.array([r.params for r in results])
    data["mem_mb"] = np.array([r.mem_mb for r in results])
    data["infer_ms"] = np.array([r.infer_ms for r in results])
    data["v_loss"] = np.array([r.v_loss for r in results])
    data["pi_loss"] = np.array([r.pi_loss for r in results])
    data["total_loss"] = np.array([r.total_loss for r in results])
    data["steps"] = np.array([r.steps for r in results])
    data["time_min"] = np.array([r.time_min for r in results])

    # Val metrics (use nan for None)
    for key in ["val_v_loss", "val_pi_loss", "val_total_loss",
                "top1_agree", "top3_agree", "kl_div",
                "pi_entropy_net", "pi_entropy_tgt", "v_tv"]:
        data[key] = np.array([getattr(r, key) if getattr(r, key) is not None else np.nan
                              for r in results])

    path = os.path.join(save_dir, "pareto_results.npz")
    np.savez(path, **data)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Network capacity fitting benchmark")
    parser.add_argument("source_dir", help="Experiment directory with training data")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per config (default: 2000)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--max-samples", type=int, default=200_000, help="Max samples to load (default: 200000)")
    parser.add_argument("--log-interval", type=int, default=100, help="Log loss every N steps (default: 100)")
    parser.add_argument("--val-iters", type=int, default=5, help="Most-recent iterations for validation (default: 5)")
    parser.add_argument("--val-max-samples", type=int, default=50_000, help="Cap validation samples (default: 50000)")
    parser.add_argument("--no-charts", action="store_true", help="Disable chart generation")
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"Error: {args.source_dir} is not a directory")
        sys.exit(1)

    # 1. Load config only (lightweight)
    config = load_config_only(args.source_dir)
    Game = config.Game
    cs = Game.CANONICAL_SHAPE()
    n_moves = Game.NUM_MOVES()
    print(f"\nGame: {config.game} ({cs[0]}x{cs[1]}x{cs[2]}, {n_moves} moves)")

    # 2. Interactive config input
    configs = collect_configs(config, args)
    if not configs:
        print("No configs queued. Exiting.")
        sys.exit(0)

    print(f"\nQueued {len(configs)} configs.")

    # 3. Inference benchmarks (fast, no data needed)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"INFERENCE BENCHMARKS (bs={args.batch_size})")
    print(f"{'='*60}")

    infer_results = {}  # label -> (params, infer_ms)
    for label, nn_args in configs:
        nn_infer = NNWrapper(Game, nn_args)
        params = count_params(nn_infer.nnet)
        nn_infer.enable_inference_optimizations()
        infer_ms = benchmark_inference(nn_infer, Game, args.batch_size)
        del nn_infer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        infer_results[label] = (params, infer_ms)
        print(f"  {label:<22} {params:>10,} params  {infer_ms:>7.1f} ms")

    # 4. Load training data with optional val split
    print()
    num_workers = os.cpu_count() or 4
    config, train_c, train_v, train_pi, val_c, val_v, val_pi = load_data(
        args.source_dir, args.max_samples, num_workers, config=config,
        val_iters=args.val_iters, val_max_samples=args.val_max_samples,
    )
    n_train = train_c.shape[0]
    has_val = val_c is not None
    print(f"Train dataset: {n_train:,} samples")
    if has_val:
        print(f"Val dataset: {val_c.shape[0]:,} samples")

    # 5. Pre-generate batch indices
    print(f"Pre-generating {args.steps} batches (bs={args.batch_size})...")
    batch_indices = pregenerate_batches(n_train, args.batch_size, args.steps)

    # Move data to pinned memory for faster transfers
    if device.type == 'cuda':
        train_c = train_c.pin_memory()
        train_v = train_v.pin_memory()
        train_pi = train_pi.pin_memory()
        if has_val:
            val_c = val_c.pin_memory()
            val_v = val_v.pin_memory()
            val_pi = val_pi.pin_memory()

    # 6. Train + eval each config (with OOM recovery)
    results: List[BenchResult] = []

    for label, nn_args in configs:
        params, infer_ms = infer_results[label]
        print(f"\n{'='*60}")
        print(f"Training: {label}  ({params:,} params, {infer_ms:.1f} ms inference)")
        print(f"{'='*60}")

        nn_train = None
        try:
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            nn_train = NNWrapper(Game, nn_args)
            losses_log, elapsed = train_config(
                nn_train, train_c, train_v, train_pi, batch_indices,
                args.lr, args.steps, args.log_interval, device
            )
            mem_mb = 0
            if device.type == 'cuda':
                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            time_min = elapsed / 60
            print(f"  Time: {time_min:.1f}m  Peak memory: {mem_mb:.0f} MB")

            print("  Final eval loss (train)...")
            v_loss, pi_loss = eval_loss(nn_train, train_c, train_v, train_pi, args.batch_size, device)
            total_loss = v_loss + pi_loss
            print(f"  V Loss: {v_loss:.4f}  Pi Loss: {pi_loss:.4f}  Total: {total_loss:.4f}")

            br = BenchResult(
                label=label, params=params, mem_mb=mem_mb, infer_ms=infer_ms,
                v_loss=v_loss, pi_loss=pi_loss, total_loss=total_loss,
                steps=args.steps, time_min=time_min, losses_log=losses_log,
            )

            # Validation metrics
            if has_val:
                print("  Validation eval + metrics...")
                val_metrics = eval_with_metrics(nn_train, val_c, val_v, val_pi, args.batch_size, device)
                br.val_v_loss = val_metrics["val_v_loss"]
                br.val_pi_loss = val_metrics["val_pi_loss"]
                br.val_total_loss = val_metrics["val_total_loss"]
                br.top1_agree = val_metrics["top1_agree"]
                br.top3_agree = val_metrics["top3_agree"]
                br.kl_div = val_metrics["kl_div"]
                br.pi_entropy_net = val_metrics["pi_entropy_net"]
                br.pi_entropy_tgt = val_metrics["pi_entropy_tgt"]
                br.v_tv = val_metrics["v_tv"]
                print(f"  Val Total: {br.val_total_loss:.4f}  Top1: {br.top1_agree*100:.1f}%  KL: {br.kl_div:.4f}")

            results.append(br)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM! Skipping {label}.")
                results.append(BenchResult(
                    label=label, params=params, mem_mb=float('inf'), infer_ms=infer_ms,
                    v_loss=float('inf'), pi_loss=float('inf'), total_loss=float('inf'),
                    steps=0, time_min=0.0,
                ))
            else:
                raise
        finally:
            del nn_train
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 7. Results
    print_results_table(results, has_val=has_val)
    print_val_metrics_table(results)

    # 8. Save to analysis dir
    analysis_dir = os.path.join(args.source_dir, "analysis")
    save_results_npz(results, analysis_dir)

    if not args.no_charts:
        save_charts(results, analysis_dir)


if __name__ == "__main__":
    main()
