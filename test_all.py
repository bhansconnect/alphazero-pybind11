#!/usr/bin/env python
"""Run all C++ and Python tests."""
import subprocess
import sys


def main():
    failed = False

    print("=== C++ Tests ===")
    if subprocess.run(["ninja", "-C", "build/cp311/", "test"]).returncode:
        failed = True

    print("\n=== Python Tests ===")
    if subprocess.run([sys.executable, "-m", "pytest", "src/", "-v"]).returncode:
        failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
