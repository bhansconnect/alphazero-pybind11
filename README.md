# AlphaZero Pybind11

This is a modified implementation of AlphaZero. It takes some inspiration from [Learning From Scratch by Thinking Fast and Slow with Deep Learning and Tree Search](https://davidbarber.github.io/blog/2017/11/07/Learning-From-Scratch-by-Thinking-Fast-and-Slow-with-Deep-Learning-and-Tree-Search/) and some of my own modifications for caching positions. The actual core loops of games and MCTS are implemented in C++ for performance and memory reasons. Python is used for the Neural Network and data packaging.

## How to Install

### Prereqs

This project has a few prereqs, but should be usable on Linux or Mac.

1. You need a C++ toolchain (Xcode on macOS, gcc on Linux).

1. If using an nvidia gpu with cuda, you need the required video drivers.
This is a complex topic that is system specific. Google it for your OS.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) — the Python package and project manager.

### Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd alphazero-pybind11

# Install all dependencies + build tools (Python 3.11, torch, numpy, etc.)
uv sync

# Editable install (builds the C++ extension)
NINJA=$(pwd)/.venv/bin/ninja uv pip install --python .venv/bin/python --no-build-isolation -e .
```

That's it. `uv sync` automatically downloads Python 3.11 and installs all dependencies (including PyTorch with CUDA on Linux or MPS on macOS). The editable install builds the C++ extension.

### Build the C++ Tests

```bash
# Build and run C++ tests (build/cp311 is created by the editable install above)
uv run ninja -C build/cp311 test
```

## How to Use

All commands use `uv run` to ensure the correct environment is active.

### Training

```bash
# Start training
uv run python src/train.py --config configs/connect4.yaml

# See all options
uv run python src/train.py --help
```

### Tournaments

After training some networks, have them compete:

```bash
uv run python src/tournament.py
```

### Interactive Play

```bash
uv run python src/play.py
```

### Running Tests

```bash
# All tests (C++ + Python)
uv run python test_all.py

# C++ tests only
uv run ninja -C build/cp311 test

# Python unit tests only
uv run python -m pytest src/test_config.py src/test_game_ui.py src/test_mcts_analysis.py src/test_history.py -v

# Full integration tests (slow, ~3-5 min)
uv run python -m pytest src/test_train.py -v
```

### Optional Dependencies

```bash
# Aim (training visualization)
uv sync --extra aim

# Matplotlib (plotting)
uv sync --extra plot
```

## Important Files

 - `src/train.py`: CLI entry point for training.
 - `src/game_runner.py`: Training loop implementation.
 - `src/neural_net.py`: The neural network implementations and config.
 - `src/tournament.py`: Runs tournaments between networks in `data/bench`.
 - `src/play.py`: Unified interactive play agent.
 - `src/mcts_analysis.py`: MCTS threshold analysis.
 - `src/game_state.h`: This is the interface that must be implemented to add a new game.
 - `src/*_gs.(h|cc)`: Implementations of various games.
 - `src/py_wrapper.cc`: The definition of the python api that gets exposed.
 - `src/meson.build`: The build definition that must be edited when adding a new game.

In general if you want to add something new, just start by copying from some other game and then editing. That will probably make life easier. Also, don't forget to rebuild before running the python app again.
