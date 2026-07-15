# Machine benchmark — AlphaZero workload

Portable, **data-free** benchmark to compare machines on this project's real hot
paths (GPU inference, CPU MCTS/self-play, precision throughput). Runs from a fresh
checkout — no `data/` (training data) transfer needed: it builds **untrained** nets
from `configs/` and self-play generates its own games.

Script: [`benchmark_machine.py`](benchmark_machine.py) (run it from the repo root)
→ writes `bench/results/<tag>.json` and prints a summary.

> **Why untrained nets?** Inference/GEMM timing is weight-value-independent, and an
> untrained net still exercises the full MCTS + batched-eval pipeline. Self-play
> search is *bushier* untrained, so absolute self-play numbers aren't comparable to
> production — but the **cross-machine ratio on the identical untrained net is**.

---

## 1. Setup

The two reference machines run **different torch builds on purpose** — modern torch
(≥2.9 / CUDA 13) has **dropped Pascal (sm_61)**, so an old GTX 1060 cannot run the
same stack as a new RTX 5070. Pick by GPU compute capability
(`nvidia-smi --query-gpu=compute_cap --format=csv` or
`python -c "import torch;print(torch.cuda.get_device_capability())"`):

### Modern GPU — compute capability ≥ 7.5 (Turing/Ampere/Ada/**Blackwell/5070**)
Standard project setup (from `CLAUDE.md`) — the checked-in `pyproject` targets cu130:
```bash
uv sync --no-install-project
NINJA=$(pwd)/.venv/bin/ninja uv pip install --python .venv/bin/python \
    --no-build-isolation --no-cache -e .
uv run python bench/benchmark_machine.py --selfplay      # non-interactive full run
```

### Pascal GPU — compute capability 6.x (legacy, e.g. GTX 10xx)
Modern torch has no sm_61 kernels, so pin the last Pascal-capable stack
(torch 2.6 / cu124) **and use `.venv/bin/python` directly** (not `uv run`, which
would re-sync to cu130 and break the GPU):
```bash
uv sync --no-install-project
# override torch + its CUDA deps to the last Pascal-capable build:
uv pip install --python .venv/bin/python --reinstall "torch==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124
NINJA=$(pwd)/.venv/bin/ninja .venv/bin/python -m pip install \
    --no-build-isolation --no-deps --no-cache -e .
.venv/bin/python bench/benchmark_machine.py --selfplay
```
(Confirm: `.venv/bin/python -c "import torch;print(torch.cuda.is_available())"` → `True`.)

---

## 2. Running

```bash
# interactive: runs GEMM + GPU-inference sections, then ASKS before the
# slower (~5-6 min) CPU-scaling + self-play sections
python bench/benchmark_machine.py --tag <machine-name>

# non-interactive / scripted: opt in to everything
python bench/benchmark_machine.py --selfplay --tag <machine-name>

# quick GPU-only (skip CPU/self-play)
python bench/benchmark_machine.py --no-selfplay --tag <machine-name>
```
Total ≈ 10-12 min with self-play, ≈ 2-3 min without. Results → `bench/results/<tag>.json`.

The six sections:
1. **GEMM precision** (fp32/fp16/bf16 at N=4096) — raw tensor-core capability. *(int8/fp8
   omitted — no PyTorch CUDA conv for them, so their GEMM throughput doesn't map to any
   self-play perf here; would need TensorRT.)*
2. **Conv-inference precision** (fp32/fp16/bf16 autocast, one net) — precision-matched
   conv speedup on *both* machines (isolates the tensor-core win; fp8 is GEMM-only).
3. **GPU inference vs net size** — peak kpos/s for nets 0.5M→15M params.
4. **GPU inference vs batch** — saturation batch (∝ SM count).
5. **CPU MCTS scaling vs threads** — self-play sims/s as threads grow (the CPU limit).
6. **End-to-end self-play vs net size** — where the machine goes CPU- vs GPU-bound.

> **Precision, per machine:** section 2 explicitly runs fp32/fp16/bf16 on *both*
> machines (matched). Sections 3-6 use each machine's *real serving default*
> (`process()` / `enable_inference_optimizations`): **fp32 on Pascal** (graph path),
> **bf16 on cc≥8** (Ampere/Ada/Blackwell — main uses bf16 when the GPU has native
> bf16 tensor cores), fp16 on Turing. So the 5070's sections 3-6 reflect bf16 serving.

---

## 3. Baseline — `geo-gtx1060` (old machine)

**Env:** Intel i7-8750H (6C/12T, mobile), 31 GB RAM, **GTX 1060** (Pascal, cc 6.1),
torch 2.6.0+cu124, driver 550.163. *Laptop — thermally throttles under sustained
multicore load, so CPU-scaling shape is pessimistic vs a desktop.*

**1. GEMM (N=4096):** fp32 **3.3** TFLOP/s · fp16 **2.8** · bf16 **1.9** (emulated, slow).
→ No tensor cores; fp16/bf16 not faster than fp32. (int8/fp8 omitted — GEMM-only, no
usable conv path here.)

**2. Conv-inference precision (4d64c, batch 256):** fp32 **15.7 ms** · fp16 **0.97×**
(16.2 ms — no tensor cores) · bf16 **0.08×** (191 ms — software-emulated on Pascal).
→ the 1060 gets **zero precision speedup**; the new machine's bf16 serving path should
show fp16/bf16 well above 1× here (this is the isolated tensor-core comparison).

**3. GPU inference vs net size (peak kpos/s):**

| net | params | peak kpos/s |
|---|---|---|
| 4d64c | 478k | 17.4 |
| 6d96c | 1.38M | 7.2 |
| 8d128c | 3.02M | 4.2 |
| 10d192c | 7.34M | 2.0 |
| 12d256c | 14.9M | 1.1 |

→ GPU throughput collapses ~16× as params grow ~31×.

**4. Batch saturation (4d64c):** peak **17.3 kpos/s**, saturates by **batch ~16-64**
(only 10 SMs → fills at tiny batch; larger batch = pure latency).

**5. CPU MCTS scaling vs threads** (self-play sims/s, untrained 4d64c):

| threads | ksims/s | scaling |
|---|---|---|
| 1 | 21.0 | 1.00× |
| 2 | 26.6 | 1.27× |
| 4 | 42.4 | 2.03× |
| 8 | 52.8 | 2.52× |
| 11 | **54.9** | **2.62×** |
| 12 | 52.8 | 2.52× |

→ **peak ~54.9 ksims/s (~54,900 MCTS sims/s) at 11 threads.** *Caveats: untrained net
→ bushier, noisy search — the 1-thread baseline (and thus the scaling ratio) wanders
run-to-run, so read the **peak**, not the ratio; laptop thermal-throttles under
sustained load. This is the section most improved by the new machine's 32 threads +
desktop cooling.*

**6. End-to-end self-play vs net size** (untrained, all threads):

| net | params | ksims/s | kevals/s (GPU) | cache hit |
|---|---|---|---|---|
| 4d64c | 478k | 51.7 | 6.6 | 87% |
| 6d96c | 1.38M | 30.7 | 6.9 | 77% |
| 8d128c | 3.02M | 22.1 | 4.1 | 81% |

→ self-play throughput falls as the net grows (untrained-net numbers are noisy — ~±30%
run-to-run — but the downward trend is consistent). The new machine's 5070 should hold
high throughput to much larger nets — the key thing to compare.

---

## 4. Comparing machines

1. Run the script on each machine (`--tag oldbox`, `--tag newbox`) → commit/collect
   the `bench/results/*.json`.
2. Key cross-machine ratios to look at:
   - **GEMM**: fp16/bf16 TFLOP/s — the new machine's tensor cores vs the 1060's none
     (raw capability; the biggest hardware gap).
   - **Conv-inference precision (§2)**: fp16/bf16 speedup vs fp32 on real convs — the
     5070's tensor-core win where the 1060 baseline shows 0.97×/0.08× (i.e. none).
   - **GPU inference vs net size**: how much faster the new GPU is, and whether its
     advantage *grows* with net size (bigger nets are more compute-bound).
   - **Batch saturation**: the new GPU (more SMs) saturates at a *larger* batch — tells
     you the batch size needed to actually use it.
   - **CPU scaling**: threads×clock — the new 32-thread CPU's self-play sims/s ceiling.
   - **Self-play vs net size**: the crossover net size where each machine flips from
     CPU-bound to GPU-bound (the practical "how big a net is free" answer per machine).

The `env` block in each JSON records the exact stack so cross-stack differences
(torch/cuda/driver) are explicit.
