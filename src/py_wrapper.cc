#include <cmath>
#include <vector>

#include "brandubh_gs.h"
#include "connect4_gs.h"
#include "onitama_gs.h"
#include "opentafl_gs.h"
#include "photosynthesis_gs.h"
#include "play_manager.h"
#include "star_gambit_gs.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tawlbwrdd_gs.h"
#include "tracy_zones.h"

// This file deals with exposing C++ to Python.
// It uses Pybind11 for this.

// Tracy Python bindings
#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>

// Thread-local zone storage for Python decorator pattern
// Uses a simple struct to track active zones
struct PythonZone {
  tracy::ScopedZone* zone;
  const char* name_storage;
  const char* file_storage;
};
thread_local std::vector<PythonZone> py_zone_stack;

// Start a zone with dynamic name (for decorator)
void py_tracy_zone_begin(const std::string& name, const std::string& file, uint32_t line) {
  // Allocate copies of the strings since Tracy needs them to persist
  auto* name_copy = new char[name.size() + 1];
  std::strcpy(name_copy, name.c_str());
  auto* file_copy = new char[file.size() + 1];
  std::strcpy(file_copy, file.c_str());

  auto* zone = new tracy::ScopedZone(
    line, file_copy, file.size(), nullptr, 0, name_copy, name.size(), true
  );
  py_zone_stack.push_back({zone, name_copy, file_copy});
}

void py_tracy_zone_end() {
  if (!py_zone_stack.empty()) {
    auto& pz = py_zone_stack.back();
    delete pz.zone;
    delete[] pz.name_storage;
    delete[] pz.file_storage;
    py_zone_stack.pop_back();
  }
}

void py_tracy_frame_mark() {
  FrameMark;
}

void py_tracy_set_thread_name(const std::string& name) {
  tracy::SetThreadName(name.c_str());
}
#endif

namespace alphazero {

namespace py = pybind11;
using brandubh_gs::BrandubhGS;
using connect4_gs::Connect4GS;
using onitama_gs::OnitamaGS;
using opentafl_gs::OpenTaflGS;
using photosynthesis_gs::PhotosynthesisGS;
using star_gambit_gs::StarGambitSkirmishGS;
using star_gambit_gs::StarGambitClashGS;
using star_gambit_gs::StarGambitBattleGS;
using star_gambit_gs::SkirmishConfig;
using star_gambit_gs::ClashConfig;
using star_gambit_gs::BattleConfig;
using star_gambit_gs::ActionSpace;
using star_gambit_gs::UnitInfo;
using star_gambit_gs::FireInfo;
using tawlbwrdd_gs::TawlbwrddGS;

// NOLINTNEXTLINE
PYBIND11_MODULE(alphazero, m) {
  m.doc() = "the c++ parts of an alphazero implementation";

  py::class_<PlayHistory>(m, "PlayHistory")
      .def(py::init([](py::array_t<float>& canonical, py::array_t<float>& v,
                       py::array_t<float>& pi) {
             PlayHistory ph;
             ph.canonical = Tensor<float, 3>(
                 canonical.shape(0), canonical.shape(1), canonical.shape(2));
             ph.v = Vector<float>(v.shape(0));
             ph.pi = Vector<float>(pi.shape(0));

             auto rc = canonical.mutable_unchecked<3>();
             auto rv = v.mutable_unchecked<1>();
             auto rpi = pi.mutable_unchecked<1>();
             for (auto i = 0L; i < canonical.shape(0); ++i) {
               for (auto j = 0L; j < canonical.shape(1); ++j) {
                 for (auto k = 0L; k < canonical.shape(2); ++k) {
                   ph.canonical(i, j, k) = rc(i, j, k);
                 }
               }
             }
             for (auto i = 0L; i < v.shape(0); ++i) {
               ph.v(i) = rv(i);
             }
             for (auto i = 0L; i < pi.shape(0); ++i) {
               ph.pi(i) = rpi(i);
             }
             return ph;
           }),
           py::arg().none(false), py::arg().none(false), py::arg().none(false))
      .def(
          "v", [](PlayHistory& ph) { return &ph.v; },
          py::return_value_policy::reference_internal)
      .def(
          "pi", [](PlayHistory& ph) { return &ph.pi; },
          py::return_value_policy::reference_internal)
      .def(
          "canonical",
          [](PlayHistory& ph) {
            const auto dims = ph.canonical.dimensions();
            const auto size = sizeof(float);
            return py::memoryview::from_buffer(
                ph.canonical.data(), dims,
                {size * dims[1] * dims[2], size * dims[2], size});
          },
          py::return_value_policy::reference_internal);

  py::class_<GameState>(m, "GameState")
      .def("copy", &GameState::copy, py::call_guard<py::gil_scoped_release>())
      .def("__eq__", &GameState::operator==,
           py::call_guard<py::gil_scoped_release>())
      .def("__str__", &GameState::dump,
           py::call_guard<py::gil_scoped_release>())
      .def("current_turn", &GameState::current_turn)
      .def("current_player", &GameState::current_player)
      .def("num_players", &GameState::num_players)
      .def("num_moves", &GameState::num_moves)
      .def("num_symmetries", &GameState::num_symmetries)
      .def("symmetries", &GameState::symmetries,
           py::call_guard<py::gil_scoped_release>())
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
      .def(py::init<float, uint32_t, uint32_t, float, float, float>())
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
      .def_readwrite("cache_shards", &PlayParams::cache_shards)
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
      .def_readwrite("tree_reuse", &PlayParams::tree_reuse)
      .def_readwrite("self_play", &PlayParams::self_play)
      .def_readwrite("add_noise", &PlayParams::add_noise)
      .def_readwrite("epsilon", &PlayParams::epsilon)
      .def_readwrite("fpu_reduction", &PlayParams::fpu_reduction)
      .def_readwrite("resign_percent", &PlayParams::resign_percent)
      .def_readwrite("resign_playthrough_percent",
                     &PlayParams::resign_playthrough_percent);

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
      .def("resign_scores", &PlayManager::resign_scores)
      .def("games_completed", &PlayManager::games_completed)
      .def("remaining_games", &PlayManager::remaining_games)
      .def("awaiting_inference_count", &PlayManager::awaiting_inference_count)
      .def("awaiting_mcts_count", &PlayManager::awaiting_mcts_count)
      .def("hist_count", &PlayManager::hist_count)
      .def("cache_hits", &PlayManager::cache_hits)
      .def("cache_misses", &PlayManager::cache_misses)
      .def("avg_game_length", &PlayManager::avg_game_length)
      .def("avg_leaf_depth", &PlayManager::avg_leaf_depth)
      .def("avg_search_entropy", &PlayManager::avg_search_entropy)
      .def("avg_moves_per_turn", &PlayManager::avg_moves_per_turn)
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

  py::class_<onitama_gs::Card>(m, "OnitamaCard")
      .def_readonly("name", &onitama_gs::Card::name)
      .def_readonly("movements", &onitama_gs::Card::movements)
      .def_readonly("starting_player", &onitama_gs::Card::starting_player);

  py::class_<OnitamaGS, GameState>(m, "OnitamaGS")
      .def(py::init<>())
      .def(py::init<uint8_t>())
      .def(py::init<uint8_t, uint16_t>())
      .def("p0_card0", &onitama_gs::OnitamaGS::p0_card0)
      .def("p0_card1", &onitama_gs::OnitamaGS::p0_card1)
      .def("p1_card0", &onitama_gs::OnitamaGS::p1_card0)
      .def("p1_card1", &onitama_gs::OnitamaGS::p1_card1)
      .def("waiting_card", &onitama_gs::OnitamaGS::waiting_card)
      .def_static("NUM_PLAYERS", [] { return onitama_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return onitama_gs::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES", [] { return onitama_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return onitama_gs::CANONICAL_SHAPE; });

  py::class_<BrandubhGS, GameState>(m, "BrandubhGS")
      .def(py::init<>())
      .def(py::init<uint16_t>())
      .def_static("NUM_PLAYERS", [] { return brandubh_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return brandubh_gs::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES", [] { return brandubh_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return brandubh_gs::CANONICAL_SHAPE; });

  py::class_<OpenTaflGS, GameState>(m, "OpenTaflGS")
      .def(py::init<>())
      .def(py::init<uint16_t>())
      .def_static("NUM_PLAYERS", [] { return opentafl_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return opentafl_gs::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES", [] { return opentafl_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return opentafl_gs::CANONICAL_SHAPE; });

  py::class_<TawlbwrddGS, GameState>(m, "TawlbwrddGS")
      .def(py::init<>())
      .def(py::init<uint16_t>())
      .def_static("NUM_PLAYERS", [] { return tawlbwrdd_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return tawlbwrdd_gs::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES", [] { return tawlbwrdd_gs::NUM_SYMMETRIES; })
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
      .def_static("NUM_SYMMETRIES", [] { return connect4_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return connect4_gs::CANONICAL_SHAPE; });

  // Star Gambit support structs
  py::class_<UnitInfo>(m, "UnitInfo")
      .def_readonly("player", &UnitInfo::player)
      .def_readonly("type", &UnitInfo::type)
      .def_readonly("slot", &UnitInfo::slot)
      .def_readonly("hp", &UnitInfo::hp)
      .def_readonly("anchor_q", &UnitInfo::anchor_q)
      .def_readonly("anchor_r", &UnitInfo::anchor_r)
      .def_readonly("facing", &UnitInfo::facing)
      .def_readonly("moves_left", &UnitInfo::moves_left);

  py::class_<FireInfo>(m, "FireInfo")
      .def_readonly("has_target", &FireInfo::has_target)
      .def_readonly("target_player", &FireInfo::target_player)
      .def_readonly("target_type", &FireInfo::target_type)
      .def_readonly("target_slot", &FireInfo::target_slot)
      .def_readonly("damage", &FireInfo::damage);

  // Star Gambit - Skirmish (3F, 1C, 0D, 5-side board)
  py::class_<StarGambitSkirmishGS, GameState>(m, "StarGambitSkirmishGS")
      .def(py::init<>())
      .def("get_units", &StarGambitSkirmishGS::get_units)
      .def("get_fire_info", &StarGambitSkirmishGS::get_fire_info)
      .def_static("NUM_PLAYERS", [] { return star_gambit_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return ActionSpace<SkirmishConfig>::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES", [] { return star_gambit_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return ActionSpace<SkirmishConfig>::CANONICAL_SHAPE; });

  // Star Gambit - Clash (3F, 2C, 1D, 5-side board)
  py::class_<StarGambitClashGS, GameState>(m, "StarGambitClashGS")
      .def(py::init<>())
      .def("get_units", &StarGambitClashGS::get_units)
      .def("get_fire_info", &StarGambitClashGS::get_fire_info)
      .def_static("NUM_PLAYERS", [] { return star_gambit_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return ActionSpace<ClashConfig>::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES", [] { return star_gambit_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return ActionSpace<ClashConfig>::CANONICAL_SHAPE; });

  // Star Gambit - Battle (4F, 3C, 2D, 6-side board)
  py::class_<StarGambitBattleGS, GameState>(m, "StarGambitBattleGS")
      .def(py::init<>())
      .def("get_units", &StarGambitBattleGS::get_units)
      .def("get_fire_info", &StarGambitBattleGS::get_fire_info)
      .def_static("NUM_PLAYERS", [] { return star_gambit_gs::NUM_PLAYERS; })
      .def_static("NUM_MOVES", [] { return ActionSpace<BattleConfig>::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES", [] { return star_gambit_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return ActionSpace<BattleConfig>::CANONICAL_SHAPE; });

  py::class_<PhotosynthesisGS<2>, GameState>(m, "PhotosynthesisGS2")
      .def(py::init<>())
      .def_static("NUM_PLAYERS", [] { return 2; })
      .def_static("NUM_MOVES", [] { return photosynthesis_gs::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES",
                  [] { return photosynthesis_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return PhotosynthesisGS<2>::CANONICAL_SHAPE; });

  py::class_<PhotosynthesisGS<3>, GameState>(m, "PhotosynthesisGS3")
      .def(py::init<>())
      .def_static("NUM_PLAYERS", [] { return 3; })
      .def_static("NUM_MOVES", [] { return photosynthesis_gs::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES",
                  [] { return photosynthesis_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return PhotosynthesisGS<3>::CANONICAL_SHAPE; });

  py::class_<PhotosynthesisGS<4>, GameState>(m, "PhotosynthesisGS4")
      .def(py::init<>())
      .def_static("NUM_PLAYERS", [] { return 4; })
      .def_static("NUM_MOVES", [] { return photosynthesis_gs::NUM_MOVES; })
      .def_static("NUM_SYMMETRIES",
                  [] { return photosynthesis_gs::NUM_SYMMETRIES; })
      .def_static("CANONICAL_SHAPE",
                  [] { return PhotosynthesisGS<4>::CANONICAL_SHAPE; });

  // Tracy profiler bindings
#ifdef TRACY_ENABLE
  m.def("_tracy_zone_begin", &py_tracy_zone_begin,
        py::arg("name"), py::arg("file"), py::arg("line"));
  m.def("_tracy_zone_end", &py_tracy_zone_end);
  m.def("tracy_frame_mark", &py_tracy_frame_mark);
  m.def("_tracy_set_thread_name", &py_tracy_set_thread_name, py::arg("name"));
  m.def("tracy_is_enabled", []() { return true; });
#else
  // No-op stubs when Tracy disabled
  m.def("_tracy_zone_begin", [](const std::string&, const std::string&, uint32_t) {});
  m.def("_tracy_zone_end", []() {});
  m.def("tracy_frame_mark", []() {});
  m.def("_tracy_set_thread_name", [](const std::string&) {});
  m.def("tracy_is_enabled", []() { return false; });
#endif
}

}  // namespace alphazero
