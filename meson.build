project(
  'alphazero-pybind11',
  'cpp',
  version : '0.1',
  default_options : [
    'warning_level=2',
    'cpp_std=c++17',
  ]
)

add_global_arguments('-march=native', language : 'c++')

subdir('src')

sources = run_command('sources.sh').stdout().strip().split('\n')
clangtidy = find_program('clang-tidy', required: false)
if clangtidy.found()
    run_target(
        'tidy',
        command: [
            clangtidy,
            '-p', meson.build_root(),
        ]+sources)
endif

cppcheck = find_program('cppcheck', required: false)
if cppcheck.found()
    run_target(
        'cppcheck',
        command: [
            cppcheck,
            '--enable=all',
            '--language=c++',
        ]+sources)
endif
