#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>

#include "connect4_gs.h"
#include "play_manager.h"

// This file deals with exposing C++ to Python.
// It uses Pybind11 for this.

namespace alphazero {

namespace py = pybind11;
using connect4_gs::Connect4GS;

// NOLINTNEXTLINE
PYBIND11_MODULE(alphazero, m) {
  m.doc() = "the c++ parts of an alphazero implementation";

  py::class_<GameState>(m, "GameState")
      .def("copy", &GameState::copy, py::call_guard<py::gil_scoped_release>())
      .def("__eq__", &GameState::operator==,
           py::call_guard<py::gil_scoped_release>())
      .def("__str__", &GameState::dump,
           py::call_guard<py::gil_scoped_release>())
      .def("current_player", &GameState::current_player)
      .def("num_moves", &GameState::num_moves)
      .def("valid_moves", &GameState::valid_moves,
           py::call_guard<py::gil_scoped_release>())
      .def("play_move", &GameState::play_move,
           py::call_guard<py::gil_scoped_release>())
      .def("scores", &GameState::scores,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "canonicalized",
          [](const GameState* gs) {
            auto out_tensor = gs->canonicalized();
            py::gil_scoped_acquire acquire;
            return py::array_t<float, py::array::c_style>(
                out_tensor.dimensions(), out_tensor.data());
          },
          py::call_guard<py::gil_scoped_release>());

  py::class_<GameData>(m, "GameData")
      .def(
          "valid_moves",
          [](const GameData& gd) { return gd.gs->valid_moves(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "v", [](GameData& gd) { return &gd.v; },
          py::return_value_policy::reference_internal)
      .def(
          "pi", [](GameData& gd) { return &gd.pi; },
          py::return_value_policy::reference_internal)
      .def(
          "canonical",
          [](GameData& gd) {
            const auto dims = gd.canonical.dimensions();
            const auto size = sizeof(float);
            return py::memoryview::from_buffer(
                gd.canonical.data(), dims,
                {size * dims[1] * dims[2], size * dims[2], size});
          },
          py::return_value_policy::reference_internal);

  py::class_<PlayParams>(m, "PlayParams")
      .def(py::init<>())
      .def_readwrite("games_to_play", &PlayParams::games_to_play)
      .def_readwrite("concurrent_games", &PlayParams::concurrent_games)
      .def_readwrite("max_batch_size", &PlayParams::max_batch_size)
      .def_readwrite("max_cache_size", &PlayParams::max_cache_size)
      .def_readwrite("mcts_depth", &PlayParams::mcts_depth)
      .def_readwrite("cpuct", &PlayParams::cpuct)
      .def_readwrite("history_enabled", &PlayParams::history_enabled);

  py::class_<PlayManager>(m, "PlayManager")
      .def(py::init([](const GameState* gs, PlayParams params) {
             return std::make_unique<PlayManager>(gs->copy(), params);
           }),
           py::arg().none(false), py::arg())
      .def("game_data", &PlayManager::game_data,
           py::return_value_policy::reference_internal)
      .def("params", &PlayManager::params,
           py::return_value_policy::reference_internal)
      .def("scores", &PlayManager::scores)
      .def("games_completed", &PlayManager::games_completed)
      .def("remaining_games", &PlayManager::remaining_games)
      .def("awaiting_inference_count", &PlayManager::awaiting_inference_count)
      .def("awaiting_mcts_count", &PlayManager::awaiting_mcts_count)
      .def("cache_hits", &PlayManager::cache_hits)
      .def("cache_misses", &PlayManager::cache_misses)
      .def("cache_size", &PlayManager::cache_size)
      .def("play", &PlayManager::play, py::call_guard<py::gil_scoped_release>())
      .def("pop_game", &PlayManager::pop_game,
           py::call_guard<py::gil_scoped_release>())
      .def("pop_games_upto", &PlayManager::pop_games_upto,
           py::call_guard<py::gil_scoped_release>())
      .def("push_inference", &PlayManager::push_inference,
           py::call_guard<py::gil_scoped_release>())
      .def("update_inferences", &PlayManager::update_inferences,
           py::call_guard<py::gil_scoped_release>())
      .def("dumb_inference", &PlayManager::dumb_inference,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "build_batch",
          [](PlayManager& pm, py::array_t<float>& batch,
             uint32_t concurrent_batches) {
            const auto mbs = pm.params().max_batch_size;
            const auto max_bs = [&]() {
              auto remaining_ratio =
                  static_cast<float>(pm.remaining_games()) / concurrent_batches;
              return std::min(
                  mbs, static_cast<uint32_t>(std::ceil(remaining_ratio)));
            };
            auto out = std::vector<uint32_t>{};
            out.reserve(mbs);
            auto raw = batch.mutable_unchecked<4>();
            auto dimensions_checked = false;
            auto current = 0U;
            while (current < max_bs()) {
              const auto indices = pm.pop_games_upto(max_bs() - current);
              for (const auto i : indices) {
                const auto& canonical = pm.game_data(i).canonical;
                if (!dimensions_checked) {
                  if (batch.ndim() != 4 ||
                      batch.shape(1) != canonical.dimension(0) ||
                      batch.shape(2) != canonical.dimension(1) ||
                      batch.shape(3) != canonical.dimension(2)) {
                    throw std::runtime_error{"Improper batch size"};
                  } else {
                    dimensions_checked = true;
                  }
                }
                for (auto j = 0L; j < canonical.dimension(0); ++j) {
                  for (auto k = 0L; k < canonical.dimension(1); ++k) {
                    for (auto l = 0L; l < canonical.dimension(2); ++l) {
                      raw(current, j, k, l) = canonical(j, k, l);
                    }
                  }
                }
                out.push_back(i);
                ++current;
              }
            }
            return out;
          },
          py::call_guard<py::gil_scoped_release>());

  py::class_<Connect4GS, GameState>(m, "Connect4GS")
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
      .def_static("NUM_PLAYERS", [] { return connect4_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return connect4_gs::NUM_MOVES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return connect4_gs::CANONICAL_SHAPE; });
}

}  // namespace alphazero
