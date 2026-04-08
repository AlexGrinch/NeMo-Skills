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
Filter JSONL by a format checker and keep only cleaned translated outputs.

Input JSONL is expected to contain:
  {"src": "...", "generation": "...", ...}

For each non-empty line:
1) validate with the requested checker
2) if valid, extract JSON from generation (including ```json ... ``` fenced output)
3) write only the extracted translated output JSON to output JSONL
"""

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add the config directory to Python path so checker modules can be imported.
config_dir = Path(__file__).parent.parent.parent / "config" / "formatted-translation"
sys.path.insert(0, str(config_dir))

try:
    from format import FormatChecker
except ImportError:
    # Fallback for alternate package layouts.
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
    from recipes.translation.config.format_following.format import FormatChecker


def load_format_checker(checker_name: str) -> Optional[FormatChecker]:
    """Load a format checker by name."""
    try:
        if checker_name.startswith("format") and checker_name[6:].isdigit():
            module_name = checker_name
            class_name = f"Format{checker_name[6:]}Checker"
        elif checker_name.startswith("format_"):
            module_name = checker_name
            class_name = "".join(part.capitalize() for part in checker_name.split("_")) + "Checker"
        elif "." in checker_name:
            module_name, class_name = checker_name.rsplit(".", 1)
        else:
            raise ValueError(f"Invalid checker name format: {checker_name}")

        module = importlib.import_module(module_name)
        checker_class = getattr(module, class_name)
        if not issubclass(checker_class, FormatChecker):
            raise ValueError(f"{class_name} is not a FormatChecker subclass")
        return checker_class()
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error loading format checker '{checker_name}': {e}")
        return None


def list_available_checkers() -> List[str]:
    """List all available format checkers."""
    checkers = []
    for file_path in config_dir.glob("format*.py"):
        if file_path.name == "format.py":
            continue
        module_name = file_path.stem
        try:
            module = importlib.import_module(module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, FormatChecker) and attr != FormatChecker:
                    checkers.append(module_name)
                    break
        except ImportError:
            continue
    return sorted(checkers)


def extract_first_valid_json(text: str) -> Optional[str]:
    """Extract JSON from raw model output text."""
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    # Prefer fenced JSON if available.
    code_block_pattern = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Fallback: scan for balanced objects/arrays and pick the largest valid one.
    candidates: List[str] = []
    for open_char, close_char in (("{", "}"), ("[", "]")):
        depth = 0
        start_idx = None
        for i, ch in enumerate(text):
            if ch == open_char:
                if start_idx is None:
                    start_idx = i
                depth += 1
            elif ch == close_char and depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidate = text[start_idx : i + 1]
                    try:
                        json.loads(candidate)
                        candidates.append(candidate)
                    except json.JSONDecodeError:
                        pass
                    start_idx = None
    if not candidates:
        return None
    return max(candidates, key=len)


def filter_file(
    input_file: str,
    output_file: str,
    checker: FormatChecker,
    verbose: bool = False,
    preserve_lines: bool = False,
) -> Tuple[int, int, int, int]:
    """Filter valid rows and output cleaned translated JSON objects.

    If preserve_lines is True, write one output line per input line.
    Dropped/invalid input lines are emitted as "{}".
    """
    total = 0
    kept = 0
    dropped = 0
    placeholders = 0

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line_num, raw_line in enumerate(infile, 1):
            line = raw_line.strip()
            if not line:
                if preserve_lines:
                    outfile.write("{}\n")
                    placeholders += 1
                continue

            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                if preserve_lines:
                    outfile.write("{}\n")
                    placeholders += 1
                if verbose:
                    print(f"Line {line_num}: dropped (invalid JSONL line)")
                continue

            src = row.get("src")
            generation = row.get("generation")
            if not isinstance(src, str) or not isinstance(generation, str):
                dropped += 1
                if preserve_lines:
                    outfile.write("{}\n")
                    placeholders += 1
                if verbose:
                    print(f"Line {line_num}: dropped (missing string 'src'/'generation')")
                continue

            if not checker.check(src, generation):
                dropped += 1
                if preserve_lines:
                    outfile.write("{}\n")
                    placeholders += 1
                if verbose:
                    print(f"Line {line_num}: dropped (format checker failed)")
                continue

            extracted = extract_first_valid_json(generation)
            if extracted is None:
                dropped += 1
                if preserve_lines:
                    outfile.write("{}\n")
                    placeholders += 1
                if verbose:
                    print(f"Line {line_num}: dropped (could not extract JSON from generation)")
                continue

            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError:
                dropped += 1
                if preserve_lines:
                    outfile.write("{}\n")
                    placeholders += 1
                if verbose:
                    print(f"Line {line_num}: dropped (extracted JSON invalid)")
                continue

            outfile.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            kept += 1

    return total, kept, dropped, placeholders


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter JSONL by checker and keep only translated output JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --checker format_nano1 --input data.jsonl --output filtered.jsonl
  %(prog)s --checker format1 --input data.jsonl --output filtered.jsonl --verbose
  %(prog)s --list-checkers
        """,
    )
    parser.add_argument("--checker", type=str, help="Checker name (e.g., format1, format_nano1)")
    parser.add_argument("--input", type=str, help="Input JSONL with 'src' and 'generation'")
    parser.add_argument("--output", type=str, help="Output JSONL for cleaned translated outputs")
    parser.add_argument("--verbose", action="store_true", help="Print per-line drop reasons")
    parser.add_argument(
        "--preserve-lines",
        action="store_true",
        help="Preserve input line count by writing {} for dropped/invalid lines",
    )
    parser.add_argument("--list-checkers", action="store_true", help="List all available format checkers")

    args = parser.parse_args()

    if args.list_checkers:
        checkers = list_available_checkers()
        if checkers:
            print("Available format checkers:")
            for checker in checkers:
                print(f"  {checker}")
        else:
            print("No format checkers found.")
        return 0

    if not all([args.checker, args.input, args.output]):
        parser.error("--checker, --input and --output are required (unless using --list-checkers)")

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1

    checker = load_format_checker(args.checker)
    if not checker:
        return 1

    total, kept, dropped, placeholders = filter_file(
        args.input, args.output, checker, verbose=args.verbose, preserve_lines=args.preserve_lines
    )

    print("Filtering Summary:")
    print(f"  Total non-empty lines: {total}")
    print(f"  Kept: {kept}")
    print(f"  Dropped: {dropped}")
    if args.preserve_lines:
        print(f"  Placeholder {{}} lines written: {placeholders}")
    print(f"  Output file: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
