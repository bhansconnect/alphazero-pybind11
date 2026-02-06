# AlphaZero Pybind11 - Project Guide

## Overview

This is a C++/Python AlphaZero implementation. The core game logic and MCTS are in C++ for performance, exposed to Python via pybind11. Python handles neural network training with PyTorch.

## Environment Setup

This project uses a conda environment called `torch`:

```bash
conda activate torch
```

A SessionStart hook (`.claude/hooks/activate-conda.sh`) automatically activates this environment in Claude Code sessions via `CLAUDE_ENV_FILE`.

## Build Commands

```bash
# Initial setup
meson setup build --buildtype release

# Build and test C++
ninja -C build test

# Install Python package (editable mode)
pip install --no-build-isolation -e .
```

## Key Files

### Python Entry Points
- `src/game_runner.py` - Main training loop
- `src/neural_net.py` - Neural network implementation
- `src/monrad.py` / `src/roundrobin.py` - Tournament runners
- `src/play_agent.py` - Play against a trained network

### C++ Core
- `src/game_state.h` - Game interface (implement this to add new games)
- `src/*_gs.cc` - Game implementations (connect4, onitama, brandubh, etc.)
- `src/mcts.cc` - Monte Carlo Tree Search
- `src/play_manager.cc` - Manages concurrent game play and batching
- `src/py_wrapper.cc` - Python bindings

### Build
- `meson.build` - Root build config
- `src/meson.build` - Source build config (edit when adding games)
- `pyproject.toml` - Python package config (meson-python)

## Adding a New Game

1. Create `src/newgame_gs.h` and `src/newgame_gs.cc` implementing the `GameState` interface
2. Add to `src/meson.build`:
   - Create library target
   - Add to `link_with` in `extension_module`
3. Export in `src/py_wrapper.cc`
4. Rebuild: `ninja -C build`

## Testing

```bash
# C++ tests
ninja -C build test

# Python training test (Star Gambit)
python src/star_gambit_train_test.py
```

## Data Directories

- `data/` - Training data and checkpoints
- `data/bench/` - Networks for tournament evaluation

## Tracy Profiler

The codebase includes optional Tracy profiler integration for performance analysis.

### Building with Tracy

```bash
# Reconfigure with Tracy enabled
meson setup build -Dtracy_enabled=true --buildtype=release --reconfigure

# Build
ninja -C build

# Reinstall Python package
pip install --no-build-isolation -e .
```

### Running Tracy

```bash
# Install Tracy (macOS)
brew install tracy

# Run Tracy profiler GUI (connects automatically to running process)
tracy-profiler

# Or capture to file for later analysis
tracy-capture -o trace.tracy
```

### Python Usage

The `@tracy_zone` decorator, `TracyZone` context manager, and helpers are available:

```python
from tracy_utils import tracy_zone, tracy_thread, tracy_frame, TracyZone

@tracy_zone
def my_function():
    # Function is automatically instrumented with name "my_function"
    pass

def worker_thread():
    tracy_thread("worker")  # Name this thread in Tracy
    # ... thread work

# Inline zones using context manager
with TracyZone("my_operation"):
    # code to profile
    pass

# Mark frame boundaries (e.g., per training iteration)
tracy_frame()
```

### Instrumented Components

**C++:** play_manager (high-level thread zones only)
**Python:** game_runner (with training stage zones), neural_net, play_agent, monrad, roundrobin

**Training Stages:** Each training iteration has Tracy zones for: history, elo, selfplay, symmetries, resampling, training, gating

GPU synchronization is automatically enabled when Tracy is active for accurate timing.
