#!/usr/bin/env python3
"""One-shot machine benchmark for the AlphaZero workload — data-free & portable.

Purpose: quantify the per-machine performance of THIS project's real hot paths so
two machines (e.g. an old GTX 1060 box vs a new RTX 5070 box) can be compared on
identical code. It is deliberately:
  * DATA-FREE   — builds UNTRAINED nets from configs/ and lets self-play generate
                  its own games, so it runs from a fresh checkout with no data/.
  * STACK-AWARE — records the full hardware+software env (torch/cuda/gpu/arch/
                  cpu/threads/ram/driver). The two machines run *different* torch
                  builds (Pascal is EOL on modern torch) so the env is recorded to
                  keep the comparison honest.
  * QUICK       — short, self-converging windows; ~10 min total.

Measures the dimensions that actually differ between machines for this workload:
  1. GEMM precision (fp32/fp16/bf16) — raw tensor-core capability (usable precisions
     only; int8/fp8 omitted — no PyTorch CUDA conv, so they don't help self-play here)
  2. Conv-inference precision (fp32/fp16/bf16 autocast) — precision-matched conv
     inference speedup (isolates the tensor-core win on real convs)
  3. GPU inference vs NET SIZE  — GPU capability + how it scales with net size
  4. GPU inference vs BATCH     — saturation batch (∝ SM count)
  5. CPU MCTS scaling vs THREADS — the CPU limit on self-play (the real bottleneck)
  6. End-to-end self-play throughput vs NET SIZE — where each machine goes
     CPU-bound vs GPU-bound as the net grows

Emits bench/results/<tag>.json and prints a summary.

Run (see BENCHMARK.md):
  # modern GPU (sm>=7.5): uv sync + editable build, then
  uv run python bench/benchmark_machine.py
  # Pascal (sm 6.x): pin torch==2.6.0+cu124 first, then
  .venv/bin/python bench/benchmark_machine.py

NOTE: untrained nets make self-play search bushier than a trained net, so the
ABSOLUTE self-play numbers aren't comparable to production; the point is the
cross-machine RATIO on identical untrained nets.
"""
import os
import sys
import json
import time
import platform
import subprocess
import socket
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import numpy as np
import torch

from neural_net import NNArch, NNArgs, NNWrapper, get_device
from config import load_config
import network_pareto as NP           # reuse measure_selfplay_throughput
import alphazero                      # noqa: F401 (ensures bindings import)

CONFIG_PATH = os.path.join(_HERE, "..", "configs", "star_gambit_unified.yaml")

# (depth, channels, head_channels) spanning tiny -> large; head scales with width.
NET_SIZES = [(4, 64, 64), (6, 96, 96), (8, 128, 128), (10, 192, 128), (12, 256, 128)]
# self-play is expensive -> a smaller subset for the end-to-end sweep
SP_NET_SIZES = [(4, 64, 64), (6, 96, 96), (8, 128, 128)]


# --------------------------------------------------------------------------- env
def _read_cpu_model():
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _physical_cores():
    try:
        ids = set()
        cur = {}
        for ln in open("/proc/cpuinfo"):
            if ln.strip() == "":
                if "physical id" in cur and "core id" in cur:
                    ids.add((cur["physical id"], cur["core id"]))
                cur = {}
            elif ":" in ln:
                k, v = ln.split(":", 1)
                cur[k.strip()] = v.strip()
        return len(ids) or None
    except Exception:
        return None


def _ram_gb():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemTotal"):
                return round(int(ln.split()[1]) / 1024 / 1024, 1)
    except Exception:
        return None


def _driver():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
    except Exception:
        return None


def env_info():
    cuda = torch.cuda.is_available()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": _read_cpu_model(),
        "cpu_physical_cores": _physical_cores(),
        "cpu_logical_threads": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name() if cuda else None,
        "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability())) if cuda else None,
        "torch_arch_list": torch.cuda.get_arch_list() if cuda else [],
        "nvidia_driver": _driver(),
    }


# --------------------------------------------------------------------------- util
def _timeit(fn, iters, warmup, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


# ---------------------------------------------------------------------- 1. GEMM
def bench_gemm(device):
    """fp32/fp16/bf16 GEMM TFLOP/s at N=4096 — raw tensor-core capability.

    Only the precisions that this conv workload can actually USE are measured:
    fp16/bf16 map to real conv via autocast (see section 2). int8/fp8 are omitted
    on purpose — PyTorch has no int8/fp8 CUDA conv, so their GEMM throughput doesn't
    translate to any self-play perf here (would need TensorRT).
    """
    if device.type != "cuda":
        return {}
    N = 4096
    flop = 2 * N ** 3
    cuda = True
    out = {}
    try:
        a = torch.randn(N, N, device=device)
        b = torch.randn(N, N, device=device)
        out["fp32_tflops"] = round(flop / _timeit(lambda: a @ b, 30, 10, cuda) / 1e12, 1)
        for name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            try:
                x, y = a.to(dt), b.to(dt)
                out[name + "_tflops"] = round(flop / _timeit(lambda: x @ y, 30, 10, cuda) / 1e12, 1)
            except Exception:
                out[name + "_tflops"] = None
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


# ------------------------------------------------------------- net construction
def bench_inference_precision(game, in_shape, device, batch=256):
    """Real conv-inference latency by precision (fp32 vs fp16/bf16 autocast) on the
    SAME baseline net — makes the cross-machine comparison precision-matched and
    directly measures the tensor-core speedup the new GPU gets (and the 1060 doesn't).

    Uses a plain eager forward under autocast (not process(), which hardcodes fp16),
    so each precision is isolated. fp8 conv isn't reachable via autocast -> see the
    GEMM section for fp8 capability.
    """
    import contextlib
    if device.type != "cuda":
        return {}
    args = NNArgs(num_channels=64, depth=4, kernel_size=3, dense_net=False,
                  head_channels=64, head_pool=True, v_head_convs=1, pi_head_convs=1,
                  trunk_norm="batch", trunk_act="relu")
    net = NNArch(game, args).to(device).eval()
    x = torch.randn(batch, *in_shape, device=device)

    def t_fwd(make_ctx):
        with torch.no_grad():
            for _ in range(10):
                with make_ctx():
                    net(x)
            torch.cuda.synchronize()
            s = time.perf_counter()
            for _ in range(50):
                with make_ctx():
                    net(x)
            torch.cuda.synchronize()
        return (time.perf_counter() - s) / 50 * 1000

    out = {"net": "4d64c", "batch": batch}
    fp32 = t_fwd(contextlib.nullcontext)
    out["fp32_ms"] = round(fp32, 3)
    for name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        try:
            ms = t_fwd(lambda: torch.amp.autocast(device.type, dtype=dt))
            out[name + "_ms"] = round(ms, 3)
            out[name + "_speedup_vs_fp32"] = round(fp32 / ms, 2)
        except Exception:
            out[name + "_ms"] = None
    del net
    torch.cuda.empty_cache()
    return out


def build_net(game, depth, ch, hc, device):
    args = NNArgs(num_channels=ch, depth=depth, kernel_size=3, dense_net=False,
                  head_channels=hc, head_pool=True, v_head_convs=1, pi_head_convs=1,
                  trunk_norm="batch", trunk_act="relu")
    nn = NNWrapper(game, args)
    nn.enable_inference_optimizations()
    params = sum(p.numel() for p in nn._eager_nnet.parameters())
    return nn, params


# ----------------------------------------------------- 2. GPU inference vs size
def bench_gpu_netsize(game, in_shape, device, batches=(64, 256, 1024)):
    """Peak inference kpos/s per net size (real process() path)."""
    rows = []
    for d, c, hc in NET_SIZES:
        try:
            nn, params = build_net(game, d, c, hc, device)
        except Exception as e:
            rows.append({"net": f"{d}d{c}c", "error": str(e)[:80]})
            continue
        best, best_bs = 0.0, 0
        for bs in batches:
            try:
                x = torch.randn(bs, *in_shape, dtype=torch.float32, device=device)
                with torch.no_grad():
                    for _ in range(5):
                        nn.process(x)
                    ms = _timeit(lambda: nn.process(x), 30, 0, device.type == "cuda") * 1000
                kpps = bs / ms
                if kpps > best:
                    best, best_bs = kpps, bs
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    break
                raise
        rows.append({"net": f"{d}d{c}c", "params": params,
                     "peak_kpos_s": round(best, 1), "at_batch": best_bs})
        del nn
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


# --------------------------------------------------- 3. GPU inference vs batch
def bench_batch_saturation(game, in_shape, device,
                           batches=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)):
    """Throughput vs batch for the baseline net -> saturation batch."""
    nn, params = build_net(game, 4, 64, 64, device)
    rows = []
    for bs in batches:
        try:
            x = torch.randn(bs, *in_shape, dtype=torch.float32, device=device)
            with torch.no_grad():
                for _ in range(5):
                    nn.process(x)
                ms = _timeit(lambda: nn.process(x), 50, 0, device.type == "cuda") * 1000
            rows.append({"batch": bs, "ms": round(ms, 3), "kpos_s": round(bs / ms, 1)})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    del nn
    if device.type == "cuda":
        torch.cuda.empty_cache()
    peak = max((r["kpos_s"] for r in rows), default=0)
    knee = next((r["batch"] for r in rows if r["kpos_s"] >= 0.95 * peak), None)
    return {"net": "4d64c", "params": params, "curve": rows,
            "peak_kpos_s": round(peak, 1), "saturates_at_batch": knee}


# ---------------------------------------------------------- 4. CPU MCTS scaling
def bench_cpu_scaling(game, config, device, max_seconds=25.0):
    """Self-play sims/s vs mcts worker threads (baseline net) — the CPU limit."""
    nn, params = build_net(game, 4, 64, 64, device)
    logical = os.cpu_count() or 4
    counts = sorted(set([1, 2, 4, 8, 16, max(1, logical - 1), logical]))
    counts = [c for c in counts if c <= logical]
    rows = []
    base = None
    for w in counts:
        r = NP.measure_selfplay_throughput(nn, config, 0, device,
                                           max_seconds=max_seconds, workers=w)
        if not r:
            continue
        if base is None:
            base = r["ksims_s"]
        rows.append({"workers": w, "ksims_s": round(r["ksims_s"], 1),
                     "kevals_s": round(r["kevals_s"], 1), "hit": round(r["hit"], 3),
                     "scaling": round(r["ksims_s"] / base, 2) if base else None,
                     "converged": r["converged"]})
    del nn
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"net": "4d64c", "params": params, "by_workers": rows}


# ---------------------------------------- 5. end-to-end self-play vs net size
def bench_selfplay_netsize(game, config, device, max_seconds=25.0):
    """Self-play sims/s per net size at full worker count -> CPU/GPU crossover."""
    rows = []
    for d, c, hc in SP_NET_SIZES:
        try:
            nn, params = build_net(game, d, c, hc, device)
        except Exception as e:
            rows.append({"net": f"{d}d{c}c", "error": str(e)[:80]})
            continue
        r = NP.measure_selfplay_throughput(nn, config, 0, device, max_seconds=max_seconds)
        if r:
            rows.append({"net": f"{d}d{c}c", "params": params,
                         "ksims_s": round(r["ksims_s"], 1),
                         "kevals_s": round(r["kevals_s"], 1),
                         "hit": round(r["hit"], 3), "converged": r["converged"]})
        del nn
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=None, help="results filename tag (default: hostname)")
    ap.add_argument("--sp-seconds", type=float, default=25.0,
                    help="max seconds per self-play measurement (default 25)")
    ap.add_argument("--selfplay", dest="selfplay", action="store_true", default=None,
                    help="opt IN to the slower CPU-scaling + self-play sections "
                         "(non-interactive/scripted runs). Interactively you're asked.")
    ap.add_argument("--no-selfplay", dest="selfplay", action="store_false",
                    help="opt OUT of the CPU-scaling + self-play sections.")
    args = ap.parse_args()

    device = get_device()
    config = load_config(CONFIG_PATH, {}, warn=False)
    game = config.Game
    in_shape = tuple(game.CANONICAL_SHAPE())

    env = env_info()
    print("=" * 70)
    print(f"MACHINE BENCHMARK — {env['hostname']}")
    print("=" * 70)
    for k in ["cpu_model", "cpu_physical_cores", "cpu_logical_threads", "ram_gb",
              "gpu", "gpu_capability", "torch", "torch_cuda", "nvidia_driver"]:
        print(f"  {k:22} {env[k]}")
    print(f"  {'game':22} {config.game}  shape={in_shape}  moves={game.NUM_MOVES()}")
    print()

    results = {"env": env, "game": config.game, "input_shape": list(in_shape),
               "timestamp_note": "stamp externally"}

    print("[1/6] GEMM precision (N=4096)...", flush=True)
    results["gemm"] = bench_gemm(device)
    print("     ", results["gemm"])

    print("[2/6] Conv-inference precision (fp32/fp16/bf16, 4d64c)...", flush=True)
    results["inference_precision"] = bench_inference_precision(game, in_shape, device)
    print("     ", results["inference_precision"])

    print("[3/6] GPU inference vs net size...", flush=True)
    results["gpu_netsize"] = bench_gpu_netsize(game, in_shape, device)
    for r in results["gpu_netsize"]:
        print("     ", r)

    print("[4/6] GPU inference vs batch (saturation)...", flush=True)
    results["batch_saturation"] = bench_batch_saturation(game, in_shape, device)
    print(f"      peak {results['batch_saturation']['peak_kpos_s']} kpos/s, "
          f"saturates @ batch {results['batch_saturation']['saturates_at_batch']}")

    # Sections 4-5 (CPU scaling + self-play) are the slow ~5-6 min part -> opt-in.
    run_sp = args.selfplay
    if run_sp is None:                    # not set by flag -> ask if interactive
        if sys.stdin.isatty():
            try:
                ans = input("\nRun the CPU-scaling + self-play sections too? "
                            "(the slow ~5-6 min part; needed for the CPU-limit "
                            "comparison) [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            run_sp = ans in ("y", "yes")
        else:
            run_sp = False                # non-interactive default: opt-out
    if run_sp:
        print("[5/6] CPU MCTS scaling vs threads (self-play)...", flush=True)
        results["cpu_scaling"] = bench_cpu_scaling(game, config, device, args.sp_seconds)
        for r in results["cpu_scaling"]["by_workers"]:
            print("     ", r)

        print("[6/6] End-to-end self-play vs net size...", flush=True)
        results["selfplay_netsize"] = bench_selfplay_netsize(game, config, device, args.sp_seconds)
        for r in results["selfplay_netsize"]:
            print("     ", r)
    else:
        print("[5-6/6] CPU/self-play sections skipped "
              "(opt in with --selfplay or answer 'y' interactively)")

    # Headline self-play throughput (peak MCTS sims/s across thread counts) + a
    # caveat so the results file isn't misread as a production rate.
    if run_sp and results.get("cpu_scaling", {}).get("by_workers"):
        peak = max(r["ksims_s"] for r in results["cpu_scaling"]["by_workers"])
        results["peak_selfplay_ksims_s"] = round(peak, 1)
        results["selfplay_note"] = (
            "self-play throughput = real MCTS simulations/s, but measured with an "
            "UNTRAINED net (search is bushier & cache hit-rate differs from a trained "
            "net), so the ABSOLUTE rate is not the production rate — use the "
            "cross-machine RATIO. For the true trained-net rate, run "
            "`network_pareto --sp` on an experiment dir that has a checkpoint.")
        print(f"\n>>> Peak self-play throughput: {results['peak_selfplay_ksims_s']} "
              f"ksims/s (~{int(peak * 1000):,} MCTS sims/s) "
              f"[untrained net — cross-machine ratio, not production rate]")

    outdir = os.path.join(_HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    tag = args.tag or env["hostname"]
    path = os.path.join(outdir, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
