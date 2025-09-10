"""
Test runner script for comprehensive testing.
This can be used to run different test suites individually or together.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def run_unit_tests():
    """Run only unit tests."""
    return pytest.main(["tests/unit/", "-v", "--tb=short", "-m", "not slow"])


def run_integration_tests():
    """Run only integration tests."""
    return pytest.main(["tests/integration/", "-v", "--tb=short"])


def run_all_tests():
    """Run all tests."""
    return pytest.main(
        [
            "tests/",
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )


def run_quick_tests():
    """Run quick tests only (excluding slow tests)."""
    return pytest.main(
        ["tests/", "-v", "--tb=short", "-m", "not slow and not requires_ollama"]
    )


def run_tests_with_ollama():
    """Run tests that require Ollama service."""
    return pytest.main(["tests/", "-v", "--tb=short", "-m", "requires_ollama"])


if __name__ == "__main__":
    """Command line interface for test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Support Deflect Bot Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick tests (no slow/ollama tests)"
    )
    parser.add_argument(
        "--ollama", action="store_true", help="Run tests requiring Ollama"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")

    args = parser.parse_args()

    if args.unit:
        exit_code = run_unit_tests()
    elif args.integration:
        exit_code = run_integration_tests()
    elif args.quick:
        exit_code = run_quick_tests()
    elif args.ollama:
        exit_code = run_tests_with_ollama()
    else:  # default to all tests
        exit_code = run_all_tests()

    sys.exit(exit_code)
