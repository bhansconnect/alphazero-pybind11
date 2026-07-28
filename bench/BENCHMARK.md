# Machine benchmark — AlphaZero workload

Portable, **data-free** benchmark to compare machines on this project's real hot
paths (GPU inference, CPU MCTS/self-play, precision throughput). Runs from a fresh
checkout — no `data/` (training data) transfer needed: it builds **untrained** nets
from `configs/` and self-play generates its own games.

Script: [`benchmark_machine.py`](benchmark_machine.py) (run it from the repo root)
→ writes `bench/results/<tag>.json` and prints a summary. `bench/results/` is
gitignored — results are local-only and should never be committed (they embed
hardware/environment details).

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

### Modern GPU — compute capability ≥ 7.5 (Turing/Ampere/Ada/Blackwell)
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
Total ≈ 10-12 min with self-play, ≈ 2-3 min without. Results → `bench/results/<tag>.json`
(local only — gitignored, never commit).

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
> bf16 tensor cores), fp16 on Turing.

---

## 3. Comparing machines

1. Run the script on each machine (`--tag oldbox`, `--tag newbox`) and compare the
   printed summaries / saved `bench/results/*.json` locally — don't commit them.
2. Key cross-machine ratios to look at:
   - **GEMM**: fp16/bf16 TFLOP/s — tensor-core capability (raw hardware gap).
   - **Conv-inference precision (§2)**: fp16/bf16 speedup vs fp32 on real convs —
     whether the GPU has usable tensor cores for this workload's precisions.
   - **GPU inference vs net size**: how much faster one GPU is, and whether its
     advantage *grows* with net size (bigger nets are more compute-bound).
   - **Batch saturation**: a GPU with more SMs saturates at a *larger* batch — tells
     you the batch size needed to actually use it.
   - **CPU scaling**: threads × clock — the self-play sims/s ceiling per machine.
   - **Self-play vs net size**: the crossover net size where a machine flips from
     CPU-bound to GPU-bound (the practical "how big a net is free" answer per machine).

The `env` block in each JSON records the exact stack so cross-stack differences
(torch/cuda/driver) are explicit.
