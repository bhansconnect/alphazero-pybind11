#include <cmath>

#include "brandubh_gs.h"
#include "connect4_gs.h"
#include "photosynthesis_gs.h"
#include "play_manager.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tawlbwrdd_gs.h"

// This file deals with exposing C++ to Python.
// It uses Pybind11 for this.

namespace alphazero {

namespace py = pybind11;
using brandubh_gs::BrandubhGS;
using connect4_gs::Connect4GS;
using photosynthesis_gs::PhotosynthesisGS;
using tawlbwrdd_gs::TawlbwrddGS;

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
      .def("num_players", &GameState::num_players)
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

  py::class_<MCTS>(m, "MCTS")
      .def(py::init<float, uint32_t, uint32_t>())
      .def("update_root", &MCTS::update_root)
      .def("find_leaf", &MCTS::find_leaf)
      .def("process_result", &MCTS::process_result)
      .def("root_value", &MCTS::root_value)
      .def("counts", &MCTS::counts)
      .def("probs", &MCTS::probs)
      .def("depth", &MCTS::depth)
      .def_static("pick_move", &MCTS::pick_move);

  py::class_<GameData>(m, "GameData")
      .def(
          "gs", [](const GameData& gd) { return gd.gs->copy(); },
          py::call_guard<py::gil_scoped_release>())
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
      .def_readwrite("playout_cap_randomization",
                     &PlayParams::playout_cap_randomization)
      .def_readwrite("playout_cap_depth", &PlayParams::playout_cap_depth)
      .def_readwrite("playout_cap_percent", &PlayParams::playout_cap_percent)
      .def_readwrite("start_temp", &PlayParams::start_temp)
      .def_readwrite("final_temp", &PlayParams::final_temp)
      .def_readwrite("temp_decay_half_life", &PlayParams::temp_decay_half_life)
      .def_readwrite("history_enabled", &PlayParams::history_enabled)
      .def_readwrite("self_play", &PlayParams::self_play)
      .def_readwrite("add_noise", &PlayParams::add_noise)
      .def_readwrite("epsilon", &PlayParams::epsilon);

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
      .def("hist_count", &PlayManager::hist_count)
      .def("cache_hits", &PlayManager::cache_hits)
      .def("cache_misses", &PlayManager::cache_misses)
      .def("avg_game_length", &PlayManager::avg_game_length)
      .def("play", &PlayManager::play, py::call_guard<py::gil_scoped_release>())
      .def("pop_game", &PlayManager::pop_game,
           py::call_guard<py::gil_scoped_release>())
      .def("pop_games_upto", &PlayManager::pop_games_upto,
           py::call_guard<py::gil_scoped_release>())
      .def("push_inference", &PlayManager::push_inference,
           py::call_guard<py::gil_scoped_release>())
      .def("update_inferences", &PlayManager::update_inferences,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "build_history_batch",
          [](PlayManager& pm, py::array_t<float>& canonical,
             py::array_t<float>& v, py::array_t<float>& pi) {
            auto current = 0U;
            auto rc = canonical.mutable_unchecked<4>();
            auto rv = v.mutable_unchecked<2>();
            auto rpi = pi.mutable_unchecked<2>();
            while (current < canonical.shape(0) &&
                   (pm.remaining_games() > 0 || pm.hist_count() > 0)) {
              auto hist = pm.pop_hist();
              if (!hist.has_value()) {
                continue;
              }
              for (auto i = 0L; i < canonical.shape(1); ++i) {
                for (auto j = 0L; j < canonical.shape(2); ++j) {
                  for (auto k = 0L; k < canonical.shape(3); ++k) {
                    rc(current, i, j, k) = hist->canonical(i, j, k);
                  }
                }
              }
              for (auto i = 0L; i < v.shape(1); ++i) {
                rv(current, i) = hist->v(i);
              }
              for (auto i = 0L; i < pi.shape(1); ++i) {
                rpi(current, i) = hist->pi(i);
              }
              ++current;
            }
            return current;
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "build_batch",
          [](PlayManager& pm, uint32_t player, py::array_t<float>& batch,
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
              const auto indices =
                  pm.pop_games_upto(player, max_bs() - current);
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

  py::class_<BrandubhGS, GameState>(m, "BrandubhGS")
      .def(py::init<>())
      .def(py::init(
          [](const py::array_t<int8_t>& board, int8_t player, int32_t turn) {
            if (board.ndim() != brandubh_gs::BOARD_SHAPE.size() ||
                board.shape(0) != brandubh_gs::BOARD_SHAPE[0] ||
                board.shape(1) != brandubh_gs::BOARD_SHAPE[1] ||
                board.shape(2) != brandubh_gs::BOARD_SHAPE[2]) {
              throw std::runtime_error{"Improper brandubh board shape"};
            }
            auto np_unchecked = board.unchecked<3>();
            auto tensor_board = brandubh_gs::BoardTensor{};
            for (auto i = 0; i < brandubh_gs::BOARD_SHAPE[0]; ++i) {
              for (auto j = 0; j < brandubh_gs::BOARD_SHAPE[1]; ++j) {
                for (auto k = 0; k < brandubh_gs::BOARD_SHAPE[2]; ++k) {
                  tensor_board(i, j, k) = np_unchecked(i, j, k);
                }
              }
            }
            return BrandubhGS(tensor_board, player, turn);
          }))
      .def_static("NUM_PLAYERS", [] { return brandubh_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return brandubh_gs::NUM_MOVES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return brandubh_gs::CANONICAL_SHAPE; });

  py::class_<TawlbwrddGS, GameState>(m, "BrandubhGS")
      .def(py::init<>())
      .def(py::init(
          [](const py::array_t<int8_t>& board, int8_t player, int32_t turn) {
            if (board.ndim() != tawlbwrdd_gs::BOARD_SHAPE.size() ||
                board.shape(0) != tawlbwrdd_gs::BOARD_SHAPE[0] ||
                board.shape(1) != tawlbwrdd_gs::BOARD_SHAPE[1] ||
                board.shape(2) != tawlbwrdd_gs::BOARD_SHAPE[2]) {
              throw std::runtime_error{"Improper tawlbwrdd board shape"};
            }
            auto np_unchecked = board.unchecked<3>();
            auto tensor_board = tawlbwrdd_gs::BoardTensor{};
            for (auto i = 0; i < tawlbwrdd_gs::BOARD_SHAPE[0]; ++i) {
              for (auto j = 0; j < tawlbwrdd_gs::BOARD_SHAPE[1]; ++j) {
                for (auto k = 0; k < tawlbwrdd_gs::BOARD_SHAPE[2]; ++k) {
                  tensor_board(i, j, k) = np_unchecked(i, j, k);
                }
              }
            }
            return TawlbwrddGS(tensor_board, player, turn);
          }))
      .def_static("NUM_PLAYERS", [] { return tawlbwrdd_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return tawlbwrdd_gs::NUM_MOVES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return tawlbwrdd_gs::CANONICAL_SHAPE; });

  py::class_<Connect4GS, GameState>(m, "Connect4GS")
      .def(py::init<>())
      .def(py::init(
          [](const py::array_t<int8_t>& board, int8_t player, int32_t turn) {
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
            return Connect4GS(tensor_board, player, turn);
          }))
      .def_static("NUM_PLAYERS", [] { return connect4_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return connect4_gs::NUM_MOVES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return connect4_gs::CANONICAL_SHAPE; });

  py::class_<PhotosynthesisGS<2>, GameState>(m, "PhotosynthesisGS2")
      .def(py::init<>())
      .def_static("NUM_PLAYERS", [] { return 2; })
      .def_static("NUM_MOVES", [] { return photosynthesis_gs::NUM_MOVES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return PhotosynthesisGS<2>::CANONICAL_SHAPE; });

  py::class_<PhotosynthesisGS<3>, GameState>(m, "PhotosynthesisGS3")
      .def(py::init<>())
      .def_static("NUM_PLAYERS", [] { return 3; })
      .def_static("NUM_MOVES", [] { return photosynthesis_gs::NUM_MOVES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return PhotosynthesisGS<3>::CANONICAL_SHAPE; });

  py::class_<PhotosynthesisGS<4>, GameState>(m, "PhotosynthesisGS4")
      .def(py::init<>())
      .def_static("NUM_PLAYERS", [] { return 4; })
      .def_static("NUM_MOVES", [] { return photosynthesis_gs::NUM_MOVES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return PhotosynthesisGS<4>::CANONICAL_SHAPE; });
}

}  // namespace alphazero
