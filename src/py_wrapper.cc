#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "connect4_gs.h"

// This file deals with exposing C++ to Python.
// It uses Pybind11 for this.

namespace alphazero {

namespace py = pybind11;
using connect4_gs::Connect4GS;

// NOLINTNEXTLINE
PYBIND11_MODULE(alphazero, m) {
  m.doc() = "the c++ parts of an alphazero implementation";

  py::class_<Connect4GS>(m, "Connect4GS")
      .def(py::init<>())
      .def(py::init([](const py::array_t<int8_t>& board, int8_t player) {
        if (board.ndim() != connect4_gs::BOARD_SHAPE.size() ||
            board.shape(0) != connect4_gs::BOARD_SHAPE[0] ||
            board.shape(1) != connect4_gs::BOARD_SHAPE[1] ||
            board.shape(2) != connect4_gs::BOARD_SHAPE[2]) {
          throw std::runtime_error{"Improper connect 4 board shape"};
        }
        auto np_unchecked = board.unchecked<3>();
        auto tensor_board = connect4_gs::BoardTensor{};
        for (auto i = 0; i < connect4_gs::BOARD_SHAPE[0]; ++i) {
          for (auto j = 0; j < connect4_gs::BOARD_SHAPE[1]; ++j) {
            for (auto k = 0; k < connect4_gs::BOARD_SHAPE[2]; ++k) {
              tensor_board(i, j, k) = np_unchecked(i, j, k);
            }
          }
        }
        return Connect4GS(tensor_board, player);
      }))
      .def("copy", &Connect4GS::copy, py::call_guard<py::gil_scoped_release>())
      .def("__eq__", &Connect4GS::operator==,
           py::call_guard<py::gil_scoped_release>())
      .def("__str__", &Connect4GS::dump,
           py::call_guard<py::gil_scoped_release>())
      .def("current_player", &Connect4GS::current_player)
      .def("num_moves", &Connect4GS::num_moves)
      .def("valid_moves", &Connect4GS::valid_moves,
           py::call_guard<py::gil_scoped_release>())
      .def("play_move", &Connect4GS::play_move,
           py::call_guard<py::gil_scoped_release>())
      .def("scores", &Connect4GS::scores,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "canonicalized",
          [](const Connect4GS& gs) {
            auto out_tensor = gs.canonicalized();
            py::gil_scoped_acquire acquire;
            return py::array_t<float, py::array::c_style>(
                connect4_gs::CANONICAL_SHAPE, out_tensor.data());
          },
          py::call_guard<py::gil_scoped_release>());
  // m.def("play_game", &play_game, py::call_guard<py::gil_scoped_release>());
  // m.def("play_games", &play_games,
  // py::call_guard<py::gil_scoped_release>()); m.def("dumb_eval", &dumb_eval,
  // py::call_guard<py::gil_scoped_release>()); m.def("mutli_dumb_eval",
  // &multi_dumb_eval,
  //       py::call_guard<py::gil_scoped_release>());
}
}  // namespace alphazero
