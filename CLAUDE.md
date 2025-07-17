If the build directory is missing, configure with `meson setup build --buildtype release`.
Build all c++ with `ninja -C build`.
Run all tests with `ninja -C build test`.
Run a single test with a command like `ninja -C build src/tak_gs_test && build/src/tak_gs_test`.
Always ensure that c++ code builds and tests pass when editing c++ code.
Always make sure `ruff check` passes when working on python code.
Only create c++ tests using gtest. Do not create other random files. Do not manually build things.
Keep everything in the meson build system.
