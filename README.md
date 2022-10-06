# AlphaZero Pybind11

This is a modified implementation of AlphaZero. It takes some inspiration from [Learning From Scratch by Thinking Fast and Slow with Deep Learning and Tree Search](https://davidbarber.github.io/blog/2017/11/07/Learning-From-Scratch-by-Thinking-Fast-and-Slow-with-Deep-Learning-and-Tree-Search/) and some of my own modifications for caching positions. The actual core loops of games and MCTS are implemented in C++ for performance and memory reasons. Python is used for the Neural Network and data packaging.

## How to Install

### Prereqs

This project has a few prereqs, but should be usable on Linux, Windows, or Mac (note, M1 may have some architecture problems between python and the built c++).

1. You need a C++ toolchain, ninja, and the meson build system.
All of this information can be found in [the meson installation guide](https://mesonbuild.com/SimpleStart.html).
Just follow along for whichever platform you use.

1. If using an nvidia gpu with cuda, you need the required video drivers.
This is a complex topic that is system specific. Google it for your OS.

1. You need a python3 setup. You can use the system built in python with pip or virtual environments.
I am not really a fan of that. I would advise [getting Anaconda](https://www.anaconda.com/).

### Python Env Setup

If using conda, you can follow this to get all of the python libaries installed:

1. Create and activate a new environment (on windows make sure to use the Anaconda prompt if not in path)
```
conda create -n torch python=3.9
conda activate torch
```

1. Install [pytorch](https://pytorch.org/get-started/locally/) as recomended for your system. Just select your OS, Conda, and the compute platform.

1. Install aim, it is used for visualization of training. Sadly it is not yet supported by conda, so just: `pip install aim`

### Build the Project

Building the project is pretty simple once you have the dependencies.

1. Setup the project with `meson setup build --buildtype release`

1. Build the C++ `ninja -C build test`. There will be warnings, but my minimal tests should pass.

## How to Use

### Just running something

To start training a network, just run: `python src/game_runner.py`

### Base structure
