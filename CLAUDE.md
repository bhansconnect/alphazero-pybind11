# AlphaZero Pybind11 - Project Guide

## Overview

This is a C++/Python AlphaZero implementation. The core game logic and MCTS are in C++ for performance, exposed to Python via pybind11. Python handles neural network training with PyTorch.

## Environment Setup

This project uses a conda environment called `torch`:

```bash
conda activate torch
```

The `.claude/settings.local.json` is configured to use this environment's Python.

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
