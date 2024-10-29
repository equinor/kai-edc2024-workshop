import argparse
import os
import sys
import unittest


def run_unit_tests():
    """
    Runs unit tests.
    """
    print('Running unit tests ... ')
    tests = os.path.join('kai', 'tests')
    suite = unittest.defaultTestLoader.discover(tests, pattern='test_*.py')
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run tests for kai.',
        epilog='To run individual unit tests, use e.g.'
               ' $ python kai/tests/{test file}',
    )

    # Unit tests
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run all unit tests using the `python` interpreter.',)

    # Parse!
    args = parser.parse_args()

    # Run tests
    has_run = False

    # Unit tests
    if args.unit:
        has_run = True
        run_unit_tests()

    # Help
    if not has_run:
        parser.print_help()