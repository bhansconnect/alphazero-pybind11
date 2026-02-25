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
import os
import random
import re
import readline  # noqa: F401 - Enable line editing in input()
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
    top1_agree: Optional[float] = None
    top3_agree: Optional[float] = None
    kl_div: Optional[float] = None
    target_entropy: Optional[float] = None
    kl_gap: Optional[float] = None


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


def _expand_optional_group(grp):
    """Expand an optional group with comma-separated values.

    '-k5,7'    -> ['', '-k5', '-k7']
    '-hc32,64' -> ['', '-hc32', '-hc64']
    '-resnet'  -> ['', '-resnet']  (no comma values, unchanged)
    """
    m = re.match(r'^(-[a-zA-Z]+)([\d,]+)$', grp)
    if m and ',' in m.group(2):
        prefix = m.group(1)
        values = m.group(2).split(',')
        return [""] + [f"{prefix}{v}" for v in values]
    return ["", grp]


def expand_config_string(s):
    """Expand comma-separated values and optional groups into a cross-product of configs.

    Comma-separated values in any position expand as cross-product:
        '4,6,8d16,24,32c' -> 9 configs
        '4,6d16c-k3,5'    -> 4 configs
        '4d16c'           -> 1 config (backward compatible)

    Parenthesized groups are optional (with and without):
        '4d16,32c(-hc64)' -> 4 configs: 4d16c, 4d32c, 4d16c-hc64, 4d32c-hc64
        '4d16c(-resnet)'  -> 2 configs: 4d16c, 4d16c-resnet

    Comma-separated values inside optional groups expand independently:
        '4d16c(-k5,7)'    -> 3 configs: 4d16c, 4d16c-k5, 4d16c-k7
        '4d16c(-hc32,64)' -> 3 configs: 4d16c, 4d16c-hc32, 4d16c-hc64
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

    # Optional groups: each adds variants to the product
    for grp in optional_groups:
        lists.append(_expand_optional_group(grp))

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


def load_data(source_dir, max_samples, num_workers, config=None):
    """Load training data from an experiment directory.

    Returns (config, train_c, train_v, train_pi).
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

    train_range = range(window_start, window_end + 1)

    # Load training data (full window)
    print(f"Loading training data...")
    train_c, train_v, train_pi = _load_iter_range(
        hist_dir, train_range, max_samples, num_workers)
    if train_c is None:
        print(f"Error: no training data files found in {source_dir}")
        sys.exit(1)
    print(f"Train: {train_c.shape[0]:,} samples (iters {train_range.start}-{train_range.stop - 1})")

    return config, train_c, train_v, train_pi


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
                 log_interval, device, early_stop=True):
    """Train a single config on pre-generated batches. Returns (losses_log, elapsed, actual_steps)."""
    nn_wrapper.nnet.train()
    optimizer = torch.optim.SGD(
        nn_wrapper.nnet.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)

    non_blocking = device.type == 'cuda'
    losses_log = []  # (step, v_loss, pi_loss, total)
    v_sum = 0.0
    pi_sum = 0.0

    # Early stopping via EMA plateau detection
    convergence_threshold = 0.005
    convergence_patience = 4
    min_steps = steps // 4
    ema_beta = 0.75
    ema_loss = None
    best_ema_loss = None
    patience_counter = 0
    chunk_v = 0.0
    chunk_pi = 0.0
    chunk_count = 0

    actual_steps = steps
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
        chunk_v += v_val
        chunk_pi += pi_val
        chunk_count += 1

        step_num = step_i + 1
        if step_num % log_interval == 0 or step_num == steps:
            losses_log.append((step_num, v_sum / step_num, pi_sum / step_num, (v_sum + pi_sum) / step_num))

            # Early stopping check at log_interval boundaries
            if early_stop and step_num >= min_steps and step_num % log_interval == 0:
                chunk_avg = (chunk_v + chunk_pi) / chunk_count

                if ema_loss is None:
                    ema_loss = chunk_avg
                else:
                    ema_loss = ema_beta * ema_loss + (1 - ema_beta) * chunk_avg

                if best_ema_loss is None:
                    best_ema_loss = ema_loss
                else:
                    rel_improvement = (best_ema_loss - ema_loss) / (best_ema_loss + 1e-8)
                    if rel_improvement < convergence_threshold:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                        best_ema_loss = ema_loss

                    if patience_counter >= convergence_patience:
                        print(f"  Early stop at step {step_num}/{steps} "
                              f"(ema_loss={ema_loss:.4f})")
                        actual_steps = step_num
                        break

            # Reset chunk accumulators
            if step_num % log_interval == 0:
                chunk_v = 0.0
                chunk_pi = 0.0
                chunk_count = 0

    elapsed = time.perf_counter() - t0
    return losses_log, elapsed, actual_steps


def eval_loss(nn_wrapper, all_c, all_v, all_pi, batch_size, device):
    """Full eval-mode pass over the dataset with policy metrics.

    All metrics are computed incrementally on GPU with torch ops — no numpy
    concatenation, no per-batch GPU→CPU syncs. Only .item() at the very end.

    Returns dict: {v_loss, pi_loss, total_loss, top1_agree, top3_agree, kl_div,
                    target_entropy, kl_gap}.
    """
    nn_wrapper.enable_inference_optimizations()
    nn_wrapper.nnet.eval()
    non_blocking = device.type == 'cuda'
    use_amp = device.type in ('cuda', 'mps')
    n = all_c.shape[0]

    # Running accumulators (GPU tensors)
    v_total = torch.zeros(1, device=device)
    pi_total = torch.zeros(1, device=device)
    top1_matches = torch.zeros(1, device=device, dtype=torch.long)
    top3_matches = torch.zeros(1, device=device, dtype=torch.long)
    kl_sum = torch.zeros(1, device=device)
    entropy_sum = torch.zeros(1, device=device)
    n_total = 0
    n_batches = 0
    eps = 1e-10

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            c = all_c[start:end].to(device, non_blocking=non_blocking)
            v = all_v[start:end].to(device, non_blocking=non_blocking)
            pi = all_pi[start:end].to(device, non_blocking=non_blocking)
            bs = end - start

            if use_amp:
                with torch.amp.autocast(device.type, dtype=torch.float16):
                    out_v, out_pi = nn_wrapper.nnet(c)
                out_v = out_v.float()
                out_pi = out_pi.float()
            else:
                out_v, out_pi = nn_wrapper.nnet(c)

            v_total += nn_wrapper.loss_v(v, out_v)
            pi_total += nn_wrapper.loss_pi(pi, out_pi)
            n_batches += 1

            # net_prob = exp(log_softmax output)
            net_prob = torch.exp(out_pi)

            # Top-1 agreement
            top1_matches += (out_pi.argmax(1) == pi.argmax(1)).sum()

            # Top-3 agreement: count overlap between top-3 sets
            _, net_top3 = torch.topk(out_pi, 3, dim=1)  # (bs, 3)
            _, tgt_top3 = torch.topk(pi, 3, dim=1)       # (bs, 3)
            # (bs, 3, 1) == (bs, 1, 3) -> (bs, 3, 3) -> any over last -> (bs, 3)
            matches = (net_top3.unsqueeze(2) == tgt_top3.unsqueeze(1)).any(dim=2)
            top3_matches += matches.sum()

            # KL(target || network), masked where target > 0
            mask = pi > 0
            kl_sum += (pi[mask] * torch.log(pi[mask] / (net_prob[mask] + eps))).sum()

            # Target entropy H(pi), masked where pi > 0
            entropy_sum += (-pi[mask] * torch.log(pi[mask])).sum()

            n_total += bs

    v_loss = (v_total / n_batches).item()
    pi_loss = (pi_total / n_batches).item()
    top1 = top1_matches.item() / n_total
    top3 = top3_matches.item() / (n_total * 3)
    kl = kl_sum.item() / n_total
    target_entropy = entropy_sum.item() / n_total

    return {
        "v_loss": v_loss,
        "pi_loss": pi_loss,
        "total_loss": v_loss + pi_loss,
        "top1_agree": top1,
        "top3_agree": top3,
        "kl_div": kl,
        "target_entropy": target_entropy,
        "kl_gap": pi_loss - target_entropy,
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
    seen_labels = set()
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
                if label in seen_labels:
                    print(f"  Skipped duplicate: {label}")
                    continue
                kwargs['star_gambit_spatial'] = config.star_gambit_spatial
                kwargs['head_pool'] = config.head_pool
                nn_args = NNArgs(lr=args.lr, cv=config.cv, **kwargs)
                tmp_model = NNArch(Game, nn_args)
                params = count_params(tmp_model)
                del tmp_model
                print(f"  Added: {label}  ({params:,} params)")
                configs.append((label, nn_args))
                seen_labels.add(label)
            except ValueError as e:
                print(f"  Error: {e}")
            except Exception as e:
                print(f"  Error creating network: {e}")

    return configs


def print_results_table(results):
    """Print the results table and learning curves."""
    # Filter out OOM entries for Pareto, but show them in table
    valid = [(i, r) for i, r in enumerate(results) if r.total_loss != float('inf')]
    oom = [(i, r) for i, r in enumerate(results) if r.total_loss == float('inf')]

    valid.sort(key=lambda x: x[1].total_loss)
    ordered = valid + oom

    print(f"\n{'='*60}")
    print(f"RESULTS (sorted by total loss)")
    print(f"{'='*60}")

    # Pareto frontier (valid entries only)
    if valid:
        valid_points = np.array([[r.mem_mb, r.infer_ms, r.total_loss] for _, r in valid])
        valid_pareto = is_pareto_optimal(valid_points)
    else:
        valid_pareto = np.array([], dtype=bool)

    header = (f"{'Config':<22} {'Params':>10} {'Mem MB':>8} {'Infer ms':>9} "
              f"{'V Loss':>8} {'Pi Loss':>9} {'Total':>8} "
              f"{'Top1%':>7} {'Top3%':>7} {'KL':>8} "
              f"{'TgtEnt':>8} {'KLGap':>8} "
              f"{'Steps':>6} {'Time':>6}")
    print(header)
    print("-" * len(header))

    vi = 0
    for orig_i, r in ordered:
        if r.total_loss == float('inf'):
            line = (f"{r.label:<22} {r.params:>10,} {'OOM':>8s} {r.infer_ms:>9.1f} "
                    f"{'OOM':>8s} {'OOM':>9s} {'OOM':>8s} "
                    f"{'':>7s} {'':>7s} {'':>8s} "
                    f"{'':>8s} {'':>8s} "
                    f"{'-':>6s} {'-':>6s}")
            print(line)
        else:
            star = " ***" if valid_pareto[vi] else ""
            top1 = f"{r.top1_agree * 100:>6.1f}%" if r.top1_agree is not None else f"{'N/A':>7s}"
            top3 = f"{r.top3_agree * 100:>6.1f}%" if r.top3_agree is not None else f"{'N/A':>7s}"
            kl = f"{r.kl_div:>8.4f}" if r.kl_div is not None else f"{'N/A':>8s}"
            te = f"{r.target_entropy:>8.4f}" if r.target_entropy is not None else f"{'N/A':>8s}"
            kg = f"{r.kl_gap:>8.4f}" if r.kl_gap is not None else f"{'N/A':>8s}"
            line = (f"{r.label:<22} {r.params:>10,} {r.mem_mb:>8.1f} {r.infer_ms:>9.1f} "
                    f"{r.v_loss:>8.4f} {r.pi_loss:>9.4f} {r.total_loss:>8.4f} "
                    f"{top1} {top3} {kl} "
                    f"{te} {kg} "
                    f"{r.steps:>6} {r.time_min:>5.1f}m{star}")
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


def _pareto_2d(x, y, lower_better_y=True):
    """2D Pareto frontier: lower x is better, y direction set by flag.

    Returns boolean mask of Pareto-optimal points.
    """
    pts = np.column_stack([x, y if lower_better_y else -y])
    n = len(pts)
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                optimal[i] = False
                break
    return optimal


def _draw_pareto_scatter(ax, x, y, labels, pareto, x_label, y_label, title,
                         marker='o', color='tab:blue', lower_better_y=True):
    """Draw a scatter plot with Pareto frontier and labeled points."""
    ax.scatter(x[~pareto], y[~pareto], color='gray', alpha=0.5, s=50, label='Dominated')
    ax.scatter(x[pareto], y[pareto], color=color, alpha=0.9, s=80, marker=marker, label='Pareto-optimal')

    # Pareto frontier line
    pidx = np.where(pareto)[0]
    if len(pidx) > 1:
        order = np.argsort(x[pidx])
        ax.plot(x[pidx[order]], y[pidx[order]], '--', color=color, alpha=0.5, linewidth=1)

    # Label every point with a color-coded box
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (x[i], y[i]), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.15))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


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

    labels = [r.label for r in valid]
    speed = np.array([r.infer_ms for r in valid])

    # Chart 1: Speed vs Loss (3 series on one plot)
    fig, ax = plt.subplots(figsize=(10, 7))
    series = [
        ('Total', [r.total_loss for r in valid], 'tab:blue', 'o'),
        ('Value', [r.v_loss for r in valid], 'tab:red', 's'),
        ('Policy', [r.pi_loss for r in valid], 'tab:green', '^'),
    ]
    for name, vals, color, marker in series:
        y = np.array(vals)
        pareto = _pareto_2d(speed, y)
        ax.scatter(speed[~pareto], y[~pareto], color=color, alpha=0.3, s=40, marker=marker)
        ax.scatter(speed[pareto], y[pareto], color=color, alpha=0.9, s=80, marker=marker, label=f'{name}')
        pidx = np.where(pareto)[0]
        if len(pidx) > 1:
            order = np.argsort(speed[pidx])
            ax.plot(speed[pidx[order]], y[pidx[order]], '--', color=color, alpha=0.5, linewidth=1)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (speed[i], y[i]), textcoords="offset points", xytext=(5, 5),
                        fontsize=6, bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.15))
    ax.set_xlabel('Inference (ms)')
    ax.set_ylabel('Loss')
    ax.set_title('Inference Speed vs Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "pareto_speed_vs_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # Chart 2: Speed vs Quality (3 subplots)
    has_metrics = any(r.top1_agree is not None for r in valid)
    if has_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Top-1 agreement (higher is better)
        top1 = np.array([r.top1_agree * 100 if r.top1_agree is not None else 0 for r in valid])
        p = _pareto_2d(speed, top1, lower_better_y=False)
        _draw_pareto_scatter(axes[0], speed, top1, labels, p,
                             'Inference (ms)', 'Top-1 Agreement (%)',
                             'Speed vs Top-1%', 'o', 'tab:blue', False)

        # Top-3 agreement (higher is better)
        top3 = np.array([r.top3_agree * 100 if r.top3_agree is not None else 0 for r in valid])
        p = _pareto_2d(speed, top3, lower_better_y=False)
        _draw_pareto_scatter(axes[1], speed, top3, labels, p,
                             'Inference (ms)', 'Top-3 Agreement (%)',
                             'Speed vs Top-3%', 's', 'tab:green', False)

        # KL divergence (lower is better)
        kl = np.array([r.kl_div if r.kl_div is not None else 0 for r in valid])
        p = _pareto_2d(speed, kl)
        _draw_pareto_scatter(axes[2], speed, kl, labels, p,
                             'Inference (ms)', 'KL Divergence',
                             'Speed vs KL Div', '^', 'tab:red', True)

        fig.tight_layout()
        path = os.path.join(save_dir, "pareto_speed_vs_quality.png")
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

    for key in ["top1_agree", "top3_agree", "kl_div"]:
        data[key] = np.array([getattr(r, key) if getattr(r, key) is not None else np.nan
                              for r in results])

    path = os.path.join(save_dir, "pareto_results.npz")
    np.savez(path, **data)
    print(f"Saved: {path}")


def _parse_selection_input(text, n):
    """Parse user selection like '1,3' or '1-3' or 'all'. Returns list of 0-based indices."""
    text = text.strip().lower()
    if text == 'all':
        return list(range(n))
    indices = set()
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            lo, hi = int(lo.strip()), int(hi.strip())
            indices.update(range(lo - 1, hi))  # 1-based to 0-based
        else:
            indices.add(int(part) - 1)
    return sorted(i for i in indices if 0 <= i < n)


def select_configs_for_training(configs, infer_results, steps):
    """Show time estimates and let user select which configs to train.

    Returns filtered (configs, infer_results).
    """
    if not configs:
        return configs, infer_results

    # Estimate training time: forward+backward+optim ≈ 3x forward
    estimates = []
    for label, nn_args in configs:
        params, infer_ms = infer_results[label]
        total_min = steps * infer_ms * 3 / 1000 / 60
        estimates.append((label, params, infer_ms, total_min))

    total_all = sum(e[3] for e in estimates)
    print(f"\nEstimated total training time: {total_all:.1f}m")

    # Try simple-term-menu for interactive selection
    try:
        from simple_term_menu import TerminalMenu

        entries = []
        for label, params, infer_ms, total_min in estimates:
            entries.append(f"{label:<22} ({params:>10,} params, {infer_ms:.1f}ms, ~{total_min:.1f}m)")

        menu = TerminalMenu(
            entries,
            title="\nSelect configs to train (space=toggle, enter=confirm):",
            multi_select=True,
            show_multi_select_hint=True,
            preselected_entries=list(range(len(entries))),
        )
        sel = menu.show()

        if sel is None:
            return [], infer_results

        if isinstance(sel, int):
            selected = [sel]
        else:
            selected = list(sel)

        if not selected:
            return [], infer_results

    except (ImportError, Exception):
        # Fallback: text-based selection
        print("\nConfigs to train:")
        for i, (label, params, infer_ms, total_min) in enumerate(estimates, 1):
            print(f"  {i}. {label:<22} ({params:>10,} params, {infer_ms:.1f}ms, ~{total_min:.1f}m)")
        print(f"\nSelect configs (e.g. '1,3' or '1-3' or 'all', default=all):")
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return [], infer_results
        if not text or text.lower() == 'all':
            selected = list(range(len(configs)))
        else:
            selected = _parse_selection_input(text, len(configs))
            if not selected:
                return [], infer_results

    # Filter configs and infer_results
    new_configs = [configs[i] for i in selected]
    new_infer = {label: infer_results[label] for label, _ in new_configs}
    selected_labels = [configs[i][0] for i in selected]
    print(f"\nSelected {len(new_configs)} configs: {', '.join(selected_labels)}")
    return new_configs, new_infer


def main():
    parser = argparse.ArgumentParser(description="Network capacity fitting benchmark")
    parser.add_argument("source_dir", help="Experiment directory with training data")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per config (default: 2000)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--max-samples", type=int, default=200_000, help="Max samples to load (default: 200000)")
    parser.add_argument("--log-interval", type=int, default=100, help="Log loss every N steps (default: 100)")
    parser.add_argument("--no-charts", action="store_true", help="Disable chart generation")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping on plateau")
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

    # 3b. Interactive selection with time estimates
    configs, infer_results = select_configs_for_training(
        configs, infer_results, args.steps)
    if not configs:
        print("No configs selected. Exiting.")
        sys.exit(0)

    # 4. Load training data
    print()
    num_workers = os.cpu_count() or 4
    config, train_c, train_v, train_pi = load_data(
        args.source_dir, args.max_samples, num_workers, config=config)
    n_train = train_c.shape[0]
    print(f"Train dataset: {n_train:,} samples")

    # 5. Pre-generate batch indices
    print(f"Pre-generating {args.steps} batches (bs={args.batch_size})...")
    batch_indices = pregenerate_batches(n_train, args.batch_size, args.steps)

    # Move data to pinned memory for faster transfers
    if device.type == 'cuda':
        train_c = train_c.pin_memory()
        train_v = train_v.pin_memory()
        train_pi = train_pi.pin_memory()

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
            losses_log, elapsed, actual_steps = train_config(
                nn_train, train_c, train_v, train_pi, batch_indices,
                args.lr, args.steps, args.log_interval, device,
                early_stop=not args.no_early_stop,
            )
            mem_mb = 0
            if device.type == 'cuda':
                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            time_min = elapsed / 60
            print(f"  Time: {time_min:.1f}m  Peak memory: {mem_mb:.0f} MB")

            print("  Final eval + metrics (train)...")
            metrics = eval_loss(nn_train, train_c, train_v, train_pi, args.batch_size, device)
            print(f"  V Loss: {metrics['v_loss']:.4f}  Pi Loss: {metrics['pi_loss']:.4f}  "
                  f"Total: {metrics['total_loss']:.4f}  "
                  f"Top1: {metrics['top1_agree']*100:.1f}%  KL: {metrics['kl_div']:.4f}  "
                  f"TE: {metrics['target_entropy']:.4f}  KL gap: {metrics['kl_gap']:.4f}")

            br = BenchResult(
                label=label, params=params, mem_mb=mem_mb, infer_ms=infer_ms,
                v_loss=metrics['v_loss'], pi_loss=metrics['pi_loss'],
                total_loss=metrics['total_loss'],
                steps=actual_steps, time_min=time_min, losses_log=losses_log,
                top1_agree=metrics['top1_agree'], top3_agree=metrics['top3_agree'],
                kl_div=metrics['kl_div'],
                target_entropy=metrics['target_entropy'], kl_gap=metrics['kl_gap'],
            )
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
    print_results_table(results)

    # 8. Save to analysis dir
    analysis_dir = os.path.join(args.source_dir, "analysis")
    save_results_npz(results, analysis_dir)

    if not args.no_charts:
        save_charts(results, analysis_dir)


if __name__ == "__main__":
    main()
