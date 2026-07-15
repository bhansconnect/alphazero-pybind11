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
import queue
import re
import readline
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import tqdm as tqdm_module
import numpy as np
from torch.utils.data import DataLoader

from config import load_config, find_latest_checkpoint
from game_runner import (
    glob_file_triples,
    calc_hist_size,
    StreamingCompressedDataset,
    _parallel_load_triples,
)
from lr_scheduler import PlateauLRScheduler, ema_update
from neural_net import NNArgs, NNWrapper, NNArch, get_device, get_amp_dtype


@dataclass
class BenchResult:
    label: str
    params: int
    mem_mb: float
    infer_ms: float
    v_loss: float
    v_loss_raw: float
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
    eff_rank: Optional[float] = None
    # List of (step, eff_rank) — trunk effective rank measured on a fixed
    # probe batch every log_interval during training. Lets us see whether
    # rank collapses, stays flat, or improves under a given config.
    eff_rank_log: list = field(default_factory=list)


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

        vm = re.match(r'vconv(\d+)', rest)
        if vm:
            kwargs['v_head_convs'] = int(vm.group(1))
            rest = rest[vm.end():]
            continue

        pm = re.match(r'pconv(\d+)', rest)
        if pm:
            kwargs['pi_head_convs'] = int(pm.group(1))
            rest = rest[pm.end():]
            continue

        vfm = re.match(r'vfc(\d+)', rest)
        if vfm:
            kwargs['v_fc_layers'] = int(vfm.group(1))
            rest = rest[vfm.end():]
            continue

        pfm = re.match(r'pfc(\d+)', rest)
        if pfm:
            kwargs['pi_fc_layers'] = int(pfm.group(1))
            rest = rest[pfm.end():]
            continue

        cvm = re.match(r'cv([\d.]+)', rest)
        if cvm:
            kwargs['cv'] = float(cvm.group(1))
            rest = rest[cvm.end():]
            continue

        # -wd{F}: weight decay (e.g. -wd1e-4, -wd0.0001, -wd0)
        # Regex matches a non-negative float with optional exponent, but NOT
        # a trailing dash (which would be the next modifier's separator).
        _NUM = r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?'
        wdm = re.match(rf'wd({_NUM})', rest)
        if wdm:
            kwargs['weight_decay'] = float(wdm.group(1))
            rest = rest[wdm.end():]
            continue

        # -orth{F}: orthogonal regularization lambda (e.g. -orth0.01, -orth1e-2)
        orthm = re.match(rf'orth({_NUM})', rest)
        if orthm:
            kwargs['orth_reg_lambda'] = float(orthm.group(1))
            rest = rest[orthm.end():]
            continue

        # -ln / -bn: trunk normalization choice
        if rest.startswith('ln'):
            kwargs['trunk_norm'] = 'layer'
            rest = rest[2:]
            continue
        if rest.startswith('bn'):
            kwargs['trunk_norm'] = 'batch'
            rest = rest[2:]
            continue

        # -crelu: Concatenated ReLU trunk activation
        if rest.startswith('crelu'):
            kwargs['trunk_act'] = 'crelu'
            rest = rest[5:]
            continue

        # -spatial / -flatpolicy (aliases: -sgspatial / -fcpolicy): override the
        # spatial policy head (default comes from the experiment's config.yaml).
        # Lets a single run compare the spatial head against the standard FC head
        # on the same data.
        if rest.startswith('sgspatial'):
            kwargs['spatial_policy'] = 'on'
            rest = rest[9:]
            continue
        if rest.startswith('fcpolicy'):
            kwargs['spatial_policy'] = 'off'
            rest = rest[8:]
            continue
        if rest.startswith('spatial'):
            kwargs['spatial_policy'] = 'on'
            rest = rest[7:]
            continue
        if rest.startswith('flatpolicy'):
            kwargs['spatial_policy'] = 'off'
            rest = rest[10:]
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
    m = re.match(r'^(-[a-zA-Z]+)([\d.,]+)$', grp)
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


def load_file_triples(source_dir, config=None):
    """Find all training data file triples from the history window.

    Returns (config, file_triples, n_samples).
    """
    if config is None:
        config = load_config_only(source_dir)
    paths = config.resolve_paths(source_dir)
    iteration = find_latest_checkpoint(paths["checkpoint"])
    if iteration == 0:
        print(f"Error: no checkpoints found in {paths['checkpoint']}")
        sys.exit(1)

    hist_size = calc_hist_size(config, iteration)
    hist_dir = paths["history"]
    window_start = max(0, iteration - hist_size)

    file_triples = []
    for i in range(window_start, iteration + 1):
        triples = glob_file_triples(hist_dir, f"{i:04d}-*-canonical-*.ptz")
        if not triples:
            triples = glob_file_triples(hist_dir, f"{i:04d}-*-canonical-*.pt")
        file_triples.extend(triples)

    if not file_triples:
        print(f"Error: no training data files found in {source_dir}")
        sys.exit(1)

    n_samples = sum(s for _, _, _, s in file_triples)
    print(f"Window: iters {window_start}-{iteration}, {len(file_triples)} files, {n_samples:,} samples")
    return config, file_triples, n_samples


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


def train_config(nn_wrapper, file_triples, n_samples, epochs, batch_size,
                 lr, log_interval, device, label, early_stop=True, active_files=8,
                 lr_patience=3, lr_conv_patience=3, lr_cooldown=1,
                 lr_drop_factor=0.3, lr_max_drops=3, lr_threshold=0.002,
                 shuffle_seed=None, eff_rank_probe=None):
    """Train a single config by streaming through file triples.

    Uses PyTorch ReduceLROnPlateau-style LR schedule via PlateauLRScheduler.

    If `eff_rank_probe` (a tensor of canonical inputs) is provided, effective
    rank of the trunk is computed on that probe every `log_interval` steps
    and accumulated in eff_rank_log.

    Returns (losses_log, elapsed, actual_steps, eff_rank_log).
    """
    target_steps = int(epochs * n_samples / batch_size)
    passes = max(1, math.ceil(epochs))
    dataset = StreamingCompressedDataset(file_triples, batch_size, passes=passes,
                                         active_files=active_files, seed=shuffle_seed)
    dl = DataLoader(dataset, batch_size=None, num_workers=0)

    nn_wrapper.nnet.train()
    # Use the wrapper's weight_decay (threaded from NNArgs) so -wd modifier
    # actually affects training. Fall back to optimizer's stored value if
    # the wrapper attribute is missing (legacy).
    wd = getattr(nn_wrapper, "weight_decay", None)
    if wd is None:
        wd = next(iter(nn_wrapper.optimizer.param_groups))["weight_decay"]
    optimizer = torch.optim.SGD(
        nn_wrapper.nnet.parameters(), lr=lr, momentum=0.9, weight_decay=wd
    )

    non_blocking = device.type == 'cuda'
    losses_log = []  # (step, v_loss, pi_loss, total)
    v_sum = 0.0
    pi_sum = 0.0

    current_lr = lr

    def _set_lr(new_lr):
        nonlocal current_lr
        current_lr = new_lr
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr

    scheduler = PlateauLRScheduler(
        set_lr=_set_lr,
        initial_lr=lr,
        drop_factor=lr_drop_factor,
        max_drops=lr_max_drops,
        patience=log_interval * lr_patience,
        conv_patience=log_interval * lr_conv_patience,
        cooldown=log_interval * lr_cooldown,
        threshold=lr_threshold,
    ) if early_stop else None
    ema_loss = None

    actual_steps = target_steps
    eff_rank_log = []  # [(step, eff_rank), ...]
    pbar = tqdm_module.tqdm(total=target_steps, desc=f"  {label}", leave=False)
    t0 = time.perf_counter()
    for step_i, (c_batch, v_batch, pi_batch) in enumerate(_maybe_prefetch(dl, device)):
        if step_i >= target_steps:
            break

        c_batch = c_batch.to(device, non_blocking=non_blocking)
        v_batch = v_batch.to(device, non_blocking=non_blocking)
        pi_batch = pi_batch.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        out_v, out_pi = nn_wrapper.nnet(c_batch)
        l_v = nn_wrapper.loss_v(v_batch, out_v)
        l_pi = nn_wrapper.loss_pi(pi_batch, out_pi)
        loss = l_v + l_pi
        # Orthogonal regularization on trunk conv weights, if enabled.
        # No-op (returns int 0) when orth_reg_lambda == 0.
        orth = nn_wrapper.trunk_orth_reg()
        if torch.is_tensor(orth):
            loss = loss + nn_wrapper.orth_reg_lambda * orth
        loss.backward()
        optimizer.step()

        v_val = l_v.item()
        pi_val = l_pi.item()
        v_sum += v_val
        pi_sum += pi_val
        step_loss = v_val + pi_val
        ema_loss = ema_update(ema_loss, step_loss)

        step_num = step_i + 1
        pbar.update()
        pbar.set_postfix(
            ema=f"{ema_loss:.4f}",
            lr=f"{current_lr:.5f}",
        )

        if step_num % log_interval == 0 or step_num == target_steps:
            losses_log.append((step_num, v_sum / step_num, pi_sum / step_num, (v_sum + pi_sum) / step_num))
            if eff_rank_probe is not None:
                # effective_rank() handles eval()/no_grad/hook internally and
                # restores eval mode. Switch back to train() after.
                try:
                    er = nn_wrapper.effective_rank(eff_rank_probe)
                    eff_rank_log.append((step_num, float(er)))
                except Exception as e:
                    pbar.write(f"  effective_rank failed at step {step_num}: {e}")
                nn_wrapper.nnet.train()

        if scheduler is not None:
            event, _ = scheduler.step(ema_loss)
            if event == "drop":
                pbar.write(f"  LR drop #{scheduler.drops} at step {step_num}: "
                           f"lr={current_lr:.6f} (best={scheduler.best:.4f}, cur={ema_loss:.4f})")
            elif event == "stop":
                pbar.write(f"  Early stop at step {step_num}/{target_steps} "
                           f"(ema={ema_loss:.4f}, lr={current_lr:.6f})")
                actual_steps = step_num
                break

    pbar.close()
    elapsed = time.perf_counter() - t0
    return losses_log, elapsed, actual_steps, eff_rank_log


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
                with torch.amp.autocast(device.type, dtype=get_amp_dtype(device)):
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
    v_loss_raw = v_loss / nn_wrapper.cv
    pi_loss = (pi_total / n_batches).item()
    top1 = top1_matches.item() / n_total
    top3 = top3_matches.item() / (n_total * 3)
    kl = kl_sum.item() / n_total
    target_entropy = entropy_sum.item() / n_total

    return {
        "v_loss": v_loss,
        "v_loss_raw": v_loss_raw,
        "pi_loss": pi_loss,
        "total_loss": v_loss + pi_loss,
        "top1_agree": top1,
        "top3_agree": top3,
        "kl_div": kl,
        "target_entropy": target_entropy,
        "kl_gap": pi_loss - target_entropy,
    }


class _PinPrefetch:
    """Background prefetch: pulls (c, v, pi) batches from source, casts to
    fp32 + pins on a worker thread, queues them. Lets pin cost overlap
    with GPU compute on the previous batch.
    """
    _SENTINEL = object()

    def __init__(self, source_iter, queue_depth=2):
        self._q = queue.Queue(maxsize=queue_depth)
        self._stop = threading.Event()
        self._exc = None
        self._thread = threading.Thread(
            target=self._worker, args=(source_iter,), daemon=True)
        self._thread.start()

    def _worker(self, source_iter):
        try:
            for batch in source_iter:
                if self._stop.is_set():
                    break
                pinned = tuple(t.float().pin_memory() for t in batch)
                while not self._stop.is_set():
                    try:
                        self._q.put(pinned, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:
            self._exc = e
        finally:
            # Blocking put: the consumer WILL drain eventually unless it died.
            # A previous version used timeout=0.1, which silently dropped the
            # sentinel whenever the consumer was mid-batch >100ms — leading to
            # the main thread blocking forever on q.get() after the last real
            # batch. See test_pinprefetch_sentinel.py reproducer.
            self._q.put(self._SENTINEL)

    def __iter__(self):
        try:
            while True:
                item = self._q.get()
                if item is self._SENTINEL:
                    if self._exc is not None:
                        raise self._exc
                    return
                yield item
        finally:
            self._stop.set()


def _maybe_prefetch(batch_iter, device):
    """On CUDA, wrap iter in a background pin+float prefetch. Elsewhere,
    just float-cast lazily on the main thread."""
    if device.type == 'cuda':
        return _PinPrefetch(batch_iter)
    return ((c.float(), v.float(), pi.float()) for c, v, pi in batch_iter)


def _accumulate_eval_metrics(nn_wrapper, batch_iter, device, eff_rank_probe=None):
    """Run forward + 8 metric accumulators over a (c, v, pi) batch iterator.

    Returns the same dict as eval_loss_streaming / eval_loss_sampled, with
    an "eff_rank" key if `eff_rank_probe` is supplied.
    """
    nn_wrapper.enable_inference_optimizations()
    nn_wrapper.nnet.eval()
    non_blocking = device.type == 'cuda'
    use_amp = device.type in ('cuda', 'mps')

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
        for c, v, pi in _maybe_prefetch(batch_iter, device):
            c = c.to(device, non_blocking=non_blocking)
            v = v.to(device, non_blocking=non_blocking)
            pi = pi.to(device, non_blocking=non_blocking)
            bs = c.shape[0]

            if use_amp:
                with torch.amp.autocast(device.type, dtype=get_amp_dtype(device)):
                    out_v, out_pi = nn_wrapper.nnet(c)
                out_v = out_v.float()
                out_pi = out_pi.float()
            else:
                out_v, out_pi = nn_wrapper.nnet(c)

            v_total += nn_wrapper.loss_v(v, out_v)
            pi_total += nn_wrapper.loss_pi(pi, out_pi)
            n_batches += 1

            net_prob = torch.exp(out_pi)

            top1_matches += (out_pi.argmax(1) == pi.argmax(1)).sum()

            _, net_top3 = torch.topk(out_pi, 3, dim=1)
            _, tgt_top3 = torch.topk(pi, 3, dim=1)
            matches = (net_top3.unsqueeze(2) == tgt_top3.unsqueeze(1)).any(dim=2)
            top3_matches += matches.sum()

            mask = pi > 0
            kl_sum += (pi[mask] * torch.log(pi[mask] / (net_prob[mask] + eps))).sum()
            entropy_sum += (-pi[mask] * torch.log(pi[mask])).sum()

            n_total += bs

    v_loss = (v_total / n_batches).item()
    v_loss_raw = v_loss / nn_wrapper.cv
    pi_loss = (pi_total / n_batches).item()
    top1 = top1_matches.item() / n_total
    top3 = top3_matches.item() / (n_total * 3)
    kl = kl_sum.item() / n_total
    target_entropy = entropy_sum.item() / n_total

    out = {
        "v_loss": v_loss,
        "v_loss_raw": v_loss_raw,
        "pi_loss": pi_loss,
        "total_loss": v_loss + pi_loss,
        "top1_agree": top1,
        "top3_agree": top3,
        "kl_div": kl,
        "target_entropy": target_entropy,
        "kl_gap": pi_loss - target_entropy,
    }
    if eff_rank_probe is not None:
        # IMPORTANT: enable_inference_optimizations() may have JIT-traced
        # self.nnet (Pascal path). The forward hook in effective_rank()
        # silently no-ops on the traced module. We surface the failure
        # rather than swallow it.
        try:
            out["eff_rank"] = float(nn_wrapper.effective_rank(eff_rank_probe))
        except Exception as e:
            print(f"  WARNING: effective_rank failed in eval: {e}")
            out["eff_rank"] = None
    return out


def eval_loss_streaming(nn_wrapper, file_triples, batch_size, device, eff_rank_probe=None):
    """Full eval-mode pass over all data by streaming from file triples.

    Same metrics as eval_loss() but streams one file at a time to avoid
    loading the full dataset into memory.

    Returns dict: {v_loss, pi_loss, total_loss, top1_agree, top3_agree, kl_div,
                    target_entropy, kl_gap[, eff_rank]}.
    """
    dataset = StreamingCompressedDataset(file_triples, batch_size, passes=1)
    dl = DataLoader(dataset, batch_size=None, num_workers=0)
    return _accumulate_eval_metrics(nn_wrapper, dl, device, eff_rank_probe=eff_rank_probe)


def eval_loss_sampled(nn_wrapper, file_triples, batch_size, device, n_eval,
                      num_workers=8, seed=None, eff_rank_probe=None):
    """Eval over a uniform-random subsample of ~n_eval positions.

    Systematic (stratified) file-level sampling: pick k = round(q * F)
    files spread evenly across the F-file iteration range with a single
    random offset. Each file (and thus each position) has probability
    k/F ≈ q of being included — same per-position uniformity as plain
    Bernoulli, but guaranteed even coverage of the iteration range (no
    gaps wider than F/k).

    `seed` controls the offset RNG: passing the same seed across configs
    in one pareto run picks the same files for every config, so
    config-to-config metric comparisons have zero sampling noise.

    Selected files are loaded in parallel (with a tqdm progress bar) on
    the main thread — no generator-inside-a-thread, ctrl-C works, and
    failures surface immediately.
    """
    sizes = np.array([s for _, _, _, s in file_triples], dtype=np.int64)
    total = int(sizes.sum())
    n_eval = min(int(n_eval), total)
    q = n_eval / total
    F = len(file_triples)

    rng = np.random.default_rng(seed)
    k = max(1, min(F, int(round(q * F))))
    stride = F / k
    offset = rng.random() * stride
    indices = sorted({min(F - 1, int(offset + i * stride)) for i in range(k)})
    selected = [file_triples[i] for i in indices]

    parts = _parallel_load_triples(selected, num_workers, desc="  load sample")
    c_all = torch.cat([p[0] for p in parts])
    v_all = torch.cat([p[1] for p in parts])
    pi_all = torch.cat([p[2] for p in parts])
    del parts

    perm = torch.randperm(c_all.shape[0])
    c_all = c_all[perm]
    v_all = v_all[perm]
    pi_all = pi_all[perm]

    def batch_iter():
        n = c_all.shape[0]
        for start in range(0, n, batch_size):
            end = start + batch_size
            yield c_all[start:end], v_all[start:end], pi_all[start:end]

    return _accumulate_eval_metrics(nn_wrapper, batch_iter(), device,
                                    eff_rank_probe=eff_rank_probe)


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


_HISTORY_FILE = os.path.expanduser("~/.cache/network_pareto_history")


def _setup_readline():
    """Load persistent readline history for config input."""
    os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
    try:
        readline.read_history_file(_HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(200)


def _save_readline():
    """Save readline history."""
    try:
        readline.write_history_file(_HISTORY_FILE)
    except OSError:
        pass


def collect_configs(config, args):
    """Interactive config input loop. Returns list of (label, NNArgs)."""
    Game = config.Game

    _setup_readline()

    print("Config format: {depth}d{channels}c[-k{N}][-hc{N}][-vconv{N}][-pconv{N}][-vfc{N}][-pfc{N}][-cv{F}][-wd{F}][-orth{F}][-ln|-bn][-crelu][-spatial|-flatpolicy][-resnet]")
    print("  Comma-separated configs:       6d32c-vconv1,4d64c-resnet-vconv1")
    print("  Comma-separated cross-product: 4,6d16,24c -> 4 configs")
    print("    4,6,8d16,24,32c     -> 9 configs")
    print("    4,6d16c-k3,5        -> 4 configs")
    print("  Parenthesized modifiers are optional (with and without):")
    print("    4d16,32c(-hc64)     -> 4 configs: 4d16c, 4d32c, +hc64 variants")
    print("    4d16c(-resnet)      -> 2 configs: 4d16c, 4d16c-resnet")
    print("  Modifiers (all optional, dash-separated):")
    print("    -k{N}      kernel size (default: 3)")
    print("    -hc{N}     head channels (default: 32)")
    print("    -vconv{N}  extra value head conv layers (default: 0)")
    print("    -pconv{N}  extra policy head conv layers (default: 0)")
    print("    -vfc{N}    value FC hidden layers (default: 1)")
    print("    -pfc{N}    policy FC hidden layers (default: 0)")
    print("    -cv{F}     value loss coefficient (default: from config)")
    print("    -wd{F}     weight decay (default: 1e-4; e.g. -wd1e-3, -wd0)")
    print("    -orth{F}   orthogonal weight reg lambda (default: 0; e.g. -orth1e-2)")
    print("    -ln / -bn  trunk normalization: LayerNorm (GroupNorm(1,C)) or BatchNorm (default)")
    print("    -crelu     Concatenated ReLU trunk activation (preserves negative-direction features)")
    print("    -spatial   spatial conv policy head (override config default on; alias -sgspatial)")
    print("    -flatpolicy standard fully-connected policy head (override config default off; alias -fcpolicy)")
    print("    -resnet    use ResNet instead of DenseNet")
    print("  Examples: 6d24c  4d32c-k5  6d24c-vconv2-vfc2  4d16c-cv2.0  4d64c-resnet-ln-wd1e-4  4d16c-orth1e-2-crelu")
    print("  Compare spatial vs FC policy head: 4d32c-spatial,4d32c-flatpolicy")
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
        # Detect comma-separated list of complete configs vs cross-product syntax.
        # If splitting on comma yields multiple parts that each start with NdNc,
        # treat as a list of individual configs. Otherwise use cross-product expansion.
        parts = [p.strip() for p in line.split(',')]
        if len(parts) > 1 and all(re.match(r'^\d+d\d+c', p) for p in parts):
            expanded = []
            for p in parts:
                expanded.extend(expand_config_string(p))
        else:
            expanded = expand_config_string(line)
        for single in expanded:
            try:
                label, kwargs = parse_config_string(single)
                if label in seen_labels:
                    print(f"  Skipped duplicate: {label}")
                    continue
                # -spatial/-flatpolicy may have set this explicitly; only fall
                # back to the experiment's config default when it didn't.
                kwargs.setdefault('spatial_policy', config.spatial_policy)
                kwargs['head_pool'] = config.head_pool
                cv = kwargs.pop('cv', config.cv)
                nn_args = NNArgs(lr=args.lr, cv=cv, **kwargs)
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

    _save_readline()
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
              f"{'V Raw':>8} {'Pi Loss':>9} {'Total':>8} "
              f"{'Top1%':>7} {'Top3%':>7} {'KL':>8} "
              f"{'TgtEnt':>8} {'KLGap':>8} {'EffRank':>8} "
              f"{'Steps':>6} {'Time':>6}")
    print(header)
    print("-" * len(header))

    vi = 0
    for orig_i, r in ordered:
        if r.total_loss == float('inf'):
            line = (f"{r.label:<22} {r.params:>10,} {'OOM':>8s} {r.infer_ms:>9.1f} "
                    f"{'OOM':>8s} {'OOM':>9s} {'OOM':>8s} "
                    f"{'':>7s} {'':>7s} {'':>8s} "
                    f"{'':>8s} {'':>8s} {'':>8s} "
                    f"{'-':>6s} {'-':>6s}")
            print(line)
        else:
            star = " ***" if valid_pareto[vi] else ""
            top1 = f"{r.top1_agree * 100:>6.1f}%" if r.top1_agree is not None else f"{'N/A':>7s}"
            top3 = f"{r.top3_agree * 100:>6.1f}%" if r.top3_agree is not None else f"{'N/A':>7s}"
            kl = f"{r.kl_div:>8.4f}" if r.kl_div is not None else f"{'N/A':>8s}"
            te = f"{r.target_entropy:>8.4f}" if r.target_entropy is not None else f"{'N/A':>8s}"
            kg = f"{r.kl_gap:>8.4f}" if r.kl_gap is not None else f"{'N/A':>8s}"
            er = f"{r.eff_rank:>8.2f}" if r.eff_rank is not None else f"{'N/A':>8s}"
            line = (f"{r.label:<22} {r.params:>10,} {r.mem_mb:>8.1f} {r.infer_ms:>9.1f} "
                    f"{r.v_loss_raw:>8.4f} {r.pi_loss:>9.4f} {r.total_loss:>8.4f} "
                    f"{top1} {top3} {kl} "
                    f"{te} {kg} {er} "
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
        ('Value', [r.v_loss_raw for r in valid], 'tab:red', 's'),
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
    data["v_loss_raw"] = np.array([r.v_loss_raw for r in results])
    data["pi_loss"] = np.array([r.pi_loss for r in results])
    data["total_loss"] = np.array([r.total_loss for r in results])
    data["steps"] = np.array([r.steps for r in results])
    data["time_min"] = np.array([r.time_min for r in results])

    for key in ["top1_agree", "top3_agree", "kl_div",
                "target_entropy", "kl_gap", "eff_rank"]:
        data[key] = np.array([getattr(r, key) if getattr(r, key) is not None else np.nan
                              for r in results])

    # eff_rank trajectory per config — keep as a dict-of-arrays since lengths
    # may differ across configs (early stopping etc.).
    er_traj = {}
    for r in results:
        if getattr(r, "eff_rank_log", None):
            arr = np.array(r.eff_rank_log, dtype=np.float64)  # [(step, rank), ...]
            er_traj[f"eff_rank_log__{r.label}"] = arr
    data.update(er_traj)

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


def select_configs_for_training(configs, infer_results, steps, auto_accept=False):
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

    if auto_accept:
        for label, params, infer_ms, total_min in estimates:
            print(f"  {label:<22} ({params:>10,} params, {infer_ms:.1f}ms, ~{total_min:.1f}m)")
        print(f"Auto-accepted all {len(configs)} configs.")
        return configs, infer_results

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
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs over the window (default: 1.0)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--log-interval", type=int, default=100, help="Log loss every N steps (default: 100)")
    parser.add_argument("--active-files", type=int, default=8,
                        help="Cross-file shuffle pool size for StreamingCompressedDataset (default: 8)")
    parser.add_argument("--no-charts", action="store_true", help="Disable chart generation")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping on plateau")
    parser.add_argument("--lr-patience", type=int, default=3,
                        help="LR-drop patience in units of log-interval (default: 3)")
    parser.add_argument("--lr-conv-patience", type=int, default=3,
                        help="Post-final-drop convergence patience in units of log-interval (default: 3)")
    parser.add_argument("--lr-cooldown", type=int, default=1,
                        help="Post-drop cooldown in units of log-interval (default: 1)")
    parser.add_argument("--lr-drop-factor", type=float, default=0.3,
                        help="Multiplicative LR drop factor (default: 0.3)")
    parser.add_argument("--lr-max-drops", type=int, default=3,
                        help="Max number of LR drops before final convergence phase (default: 3)")
    parser.add_argument("--lr-threshold", type=float, default=0.002,
                        help="Relative improvement threshold to count as a new best (default: 0.002)")
    parser.add_argument("--eval-samples", type=int, default=None,
                        help="Final eval on a uniform random subsample of N positions "
                             "from the window (unbiased per-position). Default: full pass.")
    parser.add_argument("--eff-rank-batch", type=int, default=1024,
                        help="Probe batch size for trunk effective_rank logging. "
                             "Set to 0 to disable. Default: 1024.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Auto-accept all networks; skip the interactive subselect prompt.")
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

    # 3b. Load file triples (lightweight — just discovers files, no loading)
    print()
    config, file_triples, n_samples = load_file_triples(args.source_dir, config=config)
    target_steps = int(args.epochs * n_samples / args.batch_size)
    print(f"Training: {args.epochs} epochs = {target_steps} steps (bs={args.batch_size})")

    # 3c. Interactive selection with time estimates
    configs, infer_results = select_configs_for_training(
        configs, infer_results, target_steps, auto_accept=args.yes)
    if not configs:
        print("No configs selected. Exiting.")
        sys.exit(0)

    # 4. Train + eval each config (with OOM recovery)
    train_seed = int(np.random.SeedSequence().entropy & 0xFFFFFFFF)
    print(f"Train shuffle seed: {train_seed} (shared across configs this run)")
    eval_seed = None
    if args.eval_samples is not None:
        eval_seed = np.random.SeedSequence().entropy
        print(f"Eval sample seed: {eval_seed} (shared across configs this run)")

    # Pre-load a fixed probe batch once, reused across every config to give
    # apples-to-apples effective_rank measurements. Use a small batch since
    # effective_rank is just a per-feature SVD on the trunk output.
    eff_rank_probe = None
    if args.eff_rank_batch > 0:
        try:
            from neural_net import to_half_safe, get_storage_dtype
            probe_ds = StreamingCompressedDataset(
                file_triples, args.eff_rank_batch, passes=1, active_files=1,
                seed=12345)
            probe_dl = DataLoader(probe_ds, batch_size=None, num_workers=0)
            for c, _, _ in probe_dl:
                eff_rank_probe = c.float().contiguous()
                break
            if eff_rank_probe is not None:
                print(f"effective_rank probe: {eff_rank_probe.shape[0]} positions "
                      f"(shared across configs)")
        except Exception as e:
            print(f"Warning: failed to load effective_rank probe ({e}); skipping rank logging.")
            eff_rank_probe = None

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
            losses_log, elapsed, actual_steps, eff_rank_log = train_config(
                nn_train, file_triples, n_samples, args.epochs, args.batch_size,
                args.lr, args.log_interval, device, label,
                early_stop=not args.no_early_stop, active_files=args.active_files,
                lr_patience=args.lr_patience,
                lr_conv_patience=args.lr_conv_patience,
                lr_cooldown=args.lr_cooldown,
                lr_drop_factor=args.lr_drop_factor,
                lr_max_drops=args.lr_max_drops,
                lr_threshold=args.lr_threshold,
                shuffle_seed=train_seed,
                eff_rank_probe=eff_rank_probe,
            )
            mem_mb = 0
            if device.type == 'cuda':
                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            time_min = elapsed / 60
            print(f"  Time: {time_min:.1f}m  Peak memory: {mem_mb:.0f} MB")

            if args.eval_samples is not None:
                eval_n = min(args.eval_samples, n_samples)
                frac = eval_n / max(n_samples, 1)
                print(f"  Final eval + metrics... (~{eval_n:,} of {n_samples:,}; "
                      f"~{frac * len(file_triples):.0f}/{len(file_triples)} files)")
                metrics = eval_loss_sampled(
                    nn_train, file_triples, args.batch_size, device, eval_n,
                    seed=eval_seed, eff_rank_probe=eff_rank_probe)
            else:
                print("  Final eval + metrics...")
                metrics = eval_loss_streaming(nn_train, file_triples, args.batch_size, device,
                                              eff_rank_probe=eff_rank_probe)
            if nn_args.cv != 1.0:
                v_str = f"V Loss: {metrics['v_loss_raw']:.4f} (cv{nn_args.cv:g}: {metrics['v_loss']:.4f})"
            else:
                v_str = f"V Loss: {metrics['v_loss']:.4f}"
            er_str = ""
            if metrics.get("eff_rank") is not None:
                er_str = f"  EffRank: {metrics['eff_rank']:.2f}"
            print(f"  {v_str}  Pi Loss: {metrics['pi_loss']:.4f}  "
                  f"Total: {metrics['total_loss']:.4f}  "
                  f"Top1: {metrics['top1_agree']*100:.1f}%  KL: {metrics['kl_div']:.4f}  "
                  f"TE: {metrics['target_entropy']:.4f}  KL gap: {metrics['kl_gap']:.4f}{er_str}")

            br = BenchResult(
                label=label, params=params, mem_mb=mem_mb, infer_ms=infer_ms,
                v_loss=metrics['v_loss'], v_loss_raw=metrics['v_loss_raw'],
                pi_loss=metrics['pi_loss'],
                total_loss=metrics['total_loss'],
                steps=actual_steps, time_min=time_min, losses_log=losses_log,
                top1_agree=metrics['top1_agree'], top3_agree=metrics['top3_agree'],
                kl_div=metrics['kl_div'],
                target_entropy=metrics['target_entropy'], kl_gap=metrics['kl_gap'],
                eff_rank=metrics.get("eff_rank"),
                eff_rank_log=eff_rank_log,
            )
            results.append(br)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM! Skipping {label}.")
                results.append(BenchResult(
                    label=label, params=params, mem_mb=float('inf'), infer_ms=infer_ms,
                    v_loss=float('inf'), v_loss_raw=float('inf'),
                    pi_loss=float('inf'), total_loss=float('inf'),
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
