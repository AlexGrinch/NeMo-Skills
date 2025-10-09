#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Format Verification Script

This script allows users to import format checkers and verify JSONL files containing
both src and generation fields line by line to ensure they follow the expected format.

Usage:
    python verify_format.py --checker format1 --file data.jsonl
    python verify_format.py --checker custom.MyChecker --file data.jsonl --verbose
    python verify_format.py --list-checkers
"""

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add the config directory to Python path to import format checkers
config_dir = Path(__file__).parent.parent.parent / "config" / "format-following"
sys.path.insert(0, str(config_dir))

try:
    from format import FormatChecker
except ImportError:
    # Fallback: try to import from the current directory structure
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
    from recipes.translation.config.format_following.format import FormatChecker


class FormatVerificationResult:
    """Container for verification results."""

    def __init__(self):
        self.total_lines = 0
        self.passed_lines = 0
        self.failed_lines = 0
        self.errors: List[Tuple[int, str, str]] = []  # (line_num, input_text, error_msg)

    def add_result(self, line_num: int, passed: bool, input_text: str = "", error_msg: str = ""):
        """Add a verification result for a line."""
        self.total_lines += 1
        if passed:
            self.passed_lines += 1
        else:
            self.failed_lines += 1
            self.errors.append(
                (line_num, input_text[:100] + "..." if len(input_text) > 100 else input_text, error_msg)
            )

    def get_summary(self) -> str:
        """Get a summary of the verification results."""
        pass_rate = (self.passed_lines / self.total_lines * 100) if self.total_lines > 0 else 0
        return (
            f"Verification Summary:\n"
            f"  Total lines: {self.total_lines}\n"
            f"  Passed: {self.passed_lines} ({pass_rate:.1f}%)\n"
            f"  Failed: {self.failed_lines}\n"
        )


class FormatVerifier:
    """Main class for format verification."""

    def __init__(self, checker: FormatChecker, verbose: bool = False):
        self.checker = checker
        self.verbose = verbose
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the verifier."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Enable debug logging for format checkers when verbose is on
        if self.verbose:
            # Set root logger to DEBUG to capture all format checker logs
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)

            # Set up logging for all format checker modules
            format_logger = logging.getLogger("recipes.translation.config.format-following")
            format_logger.setLevel(logging.DEBUG)

            # Also enable logging for any format checker modules that might be imported
            checker_loggers = [
                logging.getLogger("format1"),
                logging.getLogger("format2"),
                logging.getLogger("format6"),
                logging.getLogger("format11"),
            ]

            for checker_logger in checker_loggers:
                checker_logger.setLevel(logging.DEBUG)
                if not checker_logger.handlers:
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter("  FORMAT_CHECKER %(levelname)s: %(message)s")
                    handler.setFormatter(formatter)
                    checker_logger.addHandler(handler)

        return logger

    def verify_file(self, file_path: str) -> FormatVerificationResult:
        """
        Verify a JSONL file containing both src and generation fields line by line.

        Args:
            file_path: Path to JSONL file containing both src and generation fields

        Returns:
            FormatVerificationResult with verification results
        """
        result = FormatVerificationResult()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

                self.logger.info(f"Verifying {len(lines)} lines...")

                for line_num, line in enumerate(lines, 1):
                    try:
                        line = line.strip()

                        if not line:
                            if self.verbose:
                                self.logger.warning(f"Line {line_num}: Empty line found")
                            result.add_result(line_num, False, line, "Empty line")
                            continue

                        # Parse JSON line
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as e:
                            error_msg = f"JSON parse error: {str(e)}"
                            result.add_result(line_num, False, line, error_msg)
                            if self.verbose:
                                self.logger.error(f"Line {line_num}: {error_msg}")
                            continue

                        # Extract src field (input)
                        if "src" not in data:
                            error_msg = "Missing 'src' field in data"
                            result.add_result(line_num, False, line, error_msg)
                            if self.verbose:
                                self.logger.error(f"Line {line_num}: {error_msg}")
                            continue

                        # Extract generation field (output)
                        if "generation" not in data:
                            error_msg = "Missing 'generation' field in data"
                            result.add_result(line_num, False, line, error_msg)
                            if self.verbose:
                                self.logger.error(f"Line {line_num}: {error_msg}")
                            continue

                        # Extract content for format checking
                        input_text = data["src"]
                        output_text = data["generation"]

                        # Perform format check
                        is_valid = self.checker.check(input_text, output_text)

                        if is_valid:
                            result.add_result(line_num, True)
                            if self.verbose:
                                self.logger.debug(f"Line {line_num}: ✓ PASSED")
                        else:
                            error_msg = "Format validation failed"
                            result.add_result(line_num, False, line, error_msg)
                            if self.verbose:
                                self.logger.warning(f"Line {line_num}: ✗ FAILED - {error_msg}")

                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        result.add_result(line_num, False, line, error_msg)
                        self.logger.error(f"Line {line_num}: {error_msg}")

        except FileNotFoundError as e:
            self.logger.error(f"File not found: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error reading files: {str(e)}")

        return result


def load_format_checker(checker_name: str) -> Optional[FormatChecker]:
    """
    Load a format checker by name.

    Args:
        checker_name: Name of the checker (e.g., 'format1', 'format2', or 'module.ClassName')

    Returns:
        FormatChecker instance or None if not found
    """
    try:
        # Handle built-in format checkers
        if checker_name.startswith("format") and checker_name[6:].isdigit():
            module_name = checker_name
            class_name = f"Format{checker_name[6:]}Checker"
        elif "." in checker_name:
            # Handle custom format checkers (module.ClassName)
            module_name, class_name = checker_name.rsplit(".", 1)
        else:
            raise ValueError(f"Invalid checker name format: {checker_name}")

        # Import the module
        module = importlib.import_module(module_name)

        # Get the checker class
        checker_class = getattr(module, class_name)

        # Verify it's a FormatChecker subclass
        if not issubclass(checker_class, FormatChecker):
            raise ValueError(f"{class_name} is not a FormatChecker subclass")

        # Instantiate and return
        return checker_class()

    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error loading format checker '{checker_name}': {str(e)}")
        return None


def list_available_checkers() -> List[str]:
    """List all available format checkers in the config directory."""
    checkers = []

    # Look for format*.py files in the config directory
    for file_path in config_dir.glob("format*.py"):
        if file_path.name == "format.py":  # Skip base class
            continue

        module_name = file_path.stem
        try:
            module = importlib.import_module(module_name)

            # Look for FormatChecker subclasses
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, FormatChecker) and attr != FormatChecker:
                    checkers.append(module_name)
                    break
        except ImportError:
            continue

    return sorted(checkers)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Verify format of input/output file pairs using format checkers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --checker format1 --file data.jsonl
  %(prog)s --checker format2 --file data.jsonl --verbose
  %(prog)s --list-checkers
        """,
    )

    parser.add_argument(
        "--checker", type=str, help="Format checker to use (e.g., 'format1', 'format2', or 'module.ClassName')"
    )

    parser.add_argument("--file", type=str, help="Path to JSONL file containing both src and generation fields")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output with detailed logging")

    parser.add_argument("--list-checkers", action="store_true", help="List all available format checkers")

    parser.add_argument(
        "--max-errors", type=int, default=10, help="Maximum number of detailed errors to display (default: 10)"
    )

    args = parser.parse_args()

    # Handle list checkers command
    if args.list_checkers:
        checkers = list_available_checkers()
        if checkers:
            print("Available format checkers:")
            for checker in checkers:
                print(f"  {checker}")
        else:
            print("No format checkers found.")
        return 0

    # Validate required arguments
    if not all([args.checker, args.file]):
        parser.error("--checker and --file are required (unless using --list-checkers)")

    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        return 1

    # Load format checker
    print(f"Loading format checker: {args.checker}")
    checker = load_format_checker(args.checker)
    if not checker:
        return 1

    # Create verifier and run verification
    verifier = FormatVerifier(checker, verbose=args.verbose)
    print(f"Verifying file: {args.file}")

    result = verifier.verify_file(args.file)

    # Print results
    print("\n" + result.get_summary())

    if result.failed_lines > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
