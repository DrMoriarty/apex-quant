#!/usr/bin/env python3
"""Run the full APEX test suite.

Two kinds of test live under tests/:

  * unittest modules (tests/test_generate_config.py, tests/test_generate_dense_config.py,
    tests/test_estimate_size.py, tests/test_reduce_calibration.py) — collected via
    stdlib unittest discovery;
  * standalone script-style tests (tests/test_config_collisions.py,
    tests/test_cloudcp_stream_retry.py) — no TestCase classes inside, they carry
    their own entry point and output, so the runner executes them as subprocesses
    and folds their exit codes into the final result.

No package manager, no CI, no third-party test deps — stdlib unittest only.

Usage: python3 tests/run_all.py
"""
import os
import subprocess
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Script-style tests: module-level scripts with `if __name__ == "__main__"` blocks.
SCRIPT_TESTS = [
    "test_config_collisions.py",
    "test_cloudcp_stream_retry.py",
]


def run_unittest_suite():
    loader = unittest.defaultTestLoader
    suite = loader.discover(TESTS_DIR, top_level_dir=TESTS_DIR)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def run_script_tests():
    rc = 0
    for script in SCRIPT_TESTS:
        path = os.path.join(TESTS_DIR, script)
        print(f"\n--- {script} " + "-" * max(0, 56 - len(script)))
        proc = subprocess.run([sys.executable, path])
        if proc.returncode != 0:
            print(f"FAILED: {script} exited with rc={proc.returncode}")
            rc = 1
    return rc


def main():
    rc = run_unittest_suite()
    rc |= run_script_tests()
    print("\n" + "=" * 66)
    print("SUITE:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
