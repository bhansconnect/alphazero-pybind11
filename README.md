# AlphaZero Pybind11

This is a modified implementation of AlphaZero. It takes some inspiration from [Learning From Scratch by Thinking Fast and Slow with Deep Learning and Tree Search](https://davidbarber.github.io/blog/2017/11/07/Learning-From-Scratch-by-Thinking-Fast-and-Slow-with-Deep-Learning-and-Tree-Search/) and some of my own modifications for caching positions. The actual core loops of games and MCTS are implemented in C++ for performance and memory reasons. Python is used for the Neural Network and data packaging.

For release build:

```
meson setup build --buildtype release -Db_lto=true
ninja -C build test
python ...
```
