py_mod = import('python')
py3_install = py_mod.find_installation('python3')
py3_dep = dependency('python3')
pybind11_proj = subproject('pybind11')
pybind11_dep = pybind11_proj.get_variable('pybind11_dep')

absl_proj = subproject('abseil-cpp')
absl_container_dep = absl_proj.get_variable('absl_container_dep')
absl_hash_dep = absl_proj.get_variable('absl_hash_dep')

eigen_proj = subproject('eigen')
eigen_dep = eigen_proj.get_variable('eigen_dep')

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_main_dep')


brandubh_gs = library(
    'brandubh_gs',
    'brandubh_gs.cc',
    dependencies: [eigen_dep, absl_hash_dep, absl_container_dep],
)

brandubh_gs_test = executable(
  'brandubh_gs_test',
  'brandubh_gs_test.cc',
  dependencies: [gtest_dep, eigen_dep, absl_hash_dep, absl_container_dep],
  link_with: [brandubh_gs],
)
test('gtest tests', brandubh_gs_test)

connect4_gs = library(
    'connect4_gs',
    'connect4_gs.cc',
    dependencies: [eigen_dep, absl_hash_dep],
)

connect4_gs_test = executable(
  'connect4_gs_test',
  'connect4_gs_test.cc',
  dependencies: [gtest_dep, eigen_dep, absl_hash_dep],
  link_with: [connect4_gs],
)
test('gtest tests', connect4_gs_test)

photosynthesis_gs_test = executable(
  'photosynthesis_gs_test',
  'photosynthesis_gs_test.cc',
  dependencies: [gtest_dep, eigen_dep, absl_hash_dep],
)
test('gtest tests', photosynthesis_gs_test)

mcts = library(
    'mcts',
    'mcts.cc',
    dependencies: [eigen_dep, absl_hash_dep],
)

mcts_test = executable(
  'mcts_test',
  'mcts_test.cc',
  dependencies: [gtest_dep, eigen_dep, absl_hash_dep],
  link_with: [mcts, connect4_gs],
)
test('gtest tests', mcts_test)

concurrent_queue_test = executable(
  'concurrent_queue_test',
  'concurrent_queue_test.cc',
  dependencies: [gtest_dep],
  link_with: [],
)
test('gtest tests', concurrent_queue_test)

lru_cache_test = executable(
  'lru_cache_test',
  'lru_cache_test.cc',
  dependencies: [gtest_dep, absl_container_dep],
  link_with: [],
)
test('gtest tests', lru_cache_test)

play_manager = library(
  'play_manager',
  'play_manager.cc',
  dependencies: [eigen_dep, absl_container_dep, absl_hash_dep],
  link_with: [mcts],
)

play_manager_test = executable(
  'play_manager_test',
  'play_manager_test.cc',
  dependencies: [gtest_dep, eigen_dep, absl_container_dep, absl_hash_dep],
  link_with: [play_manager, connect4_gs],
)
test('gtest tests', play_manager_test)

py3_install.extension_module(
  'alphazero',
  sources: ['py_wrapper.cc'],
  link_with: [connect4_gs, brandubh_gs, play_manager],
  dependencies : [pybind11_dep, py3_dep, eigen_dep, absl_container_dep, absl_hash_dep],
)
