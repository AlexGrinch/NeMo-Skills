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
Merge content/reasoning JSONL outputs after format verification.

For each aligned line index from two JSONL files:
  - content file: validate with content checker and extract:
      .translation.problem, .translation.generation
  - reasoning file: validate with reasoning checker and extract:
      .reasoning_content

Output one JSONL row per line index:
  {"problem": "...", "generation": "...", "reasoning_content": "..."}

If any parsing/extraction step fails for a field, that field is emitted as "".
"""

import argparse
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, List, Optional

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


def read_non_empty_lines(file_path: str) -> Iterator[str]:
    """Yield stripped non-empty lines from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                yield line


def parse_jsonl_row(raw_line: Optional[str]) -> Optional[dict]:
    """Parse one JSONL row; return None on any failure."""
    if raw_line is None:
        return None
    try:
        row = json.loads(raw_line)
    except json.JSONDecodeError:
        return None
    if not isinstance(row, dict):
        return None
    return row


def parse_generation_payload(row: Optional[dict], checker: FormatChecker) -> Optional[dict]:
    """
    Parse a model row with checker validation.

    Expected row fields:
      - src: str
      - generation: str
    """
    if row is None:
        return None

    src = row.get("src")
    generation = row.get("generation")
    if not isinstance(src, str) or not isinstance(generation, str):
        return None

    if not checker.check(src, generation):
        return None

    extracted = extract_first_valid_json(generation)
    if extracted is None:
        return None

    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    return parsed


@dataclass
class MergeStats:
    total_rows: int = 0
    content_rows_ok: int = 0
    reasoning_rows_ok: int = 0
    problem_ok: int = 0
    generation_ok: int = 0
    reasoning_content_ok: int = 0
    all_fields_ok: int = 0


def pct(numer: int, denom: int) -> float:
    return (100.0 * numer / denom) if denom else 0.0


def sanitize_utf8(text: str) -> str:
    """Repair invalid Unicode (e.g., lone surrogates) for safe UTF-8 output."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def merge_files(
    content_file: str,
    reasoning_file: str,
    output_file: str,
    content_checker: FormatChecker,
    reasoning_checker: FormatChecker,
    verbose: bool = False,
) -> MergeStats:
    """Merge two JSONL files into one with requested fields and blank fallbacks."""
    stats = MergeStats()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_lines = read_non_empty_lines(content_file)
    reasoning_lines = read_non_empty_lines(reasoning_file)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for line_idx, (content_line, reasoning_line) in enumerate(zip_longest(content_lines, reasoning_lines), 1):
            stats.total_rows += 1

            out_problem = ""
            out_generation = ""
            out_reasoning_content = ""
            problem_parsed = False
            generation_parsed = False
            reasoning_content_parsed = False

            content_row = parse_jsonl_row(content_line)
            reasoning_row = parse_jsonl_row(reasoning_line)

            content_payload = parse_generation_payload(content_row, content_checker)
            reasoning_payload = parse_generation_payload(reasoning_row, reasoning_checker)

            if content_payload is not None:
                stats.content_rows_ok += 1
                translation = content_payload.get("translation")
                if isinstance(translation, dict):
                    problem = translation.get("problem")
                    generation = translation.get("generation")
                    if isinstance(problem, str):
                        out_problem = problem
                        stats.problem_ok += 1
                        problem_parsed = True
                    if isinstance(generation, str):
                        out_generation = generation
                        stats.generation_ok += 1
                        generation_parsed = True

            if reasoning_payload is not None:
                stats.reasoning_rows_ok += 1
                reasoning_content = reasoning_payload.get("reasoning_content")
                if isinstance(reasoning_content, str):
                    out_reasoning_content = reasoning_content
                    stats.reasoning_content_ok += 1
                    reasoning_content_parsed = True

            if problem_parsed and generation_parsed and reasoning_content_parsed:
                stats.all_fields_ok += 1

            merged = {
                "problem": sanitize_utf8(out_problem),
                "generation": sanitize_utf8(out_generation),
                "reasoning_content": sanitize_utf8(out_reasoning_content),
            }
            outfile.write(json.dumps(merged, ensure_ascii=False) + "\n")

            if verbose and (content_payload is None or reasoning_payload is None):
                print(
                    f"Line {line_idx}: partial parse "
                    f"(content_ok={content_payload is not None}, reasoning_ok={reasoning_payload is not None})"
                )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge content/reasoning JSONL using format checkers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --content-checker format_nano1 --reasoning-checker format_nano2 \\
           --content-input content.jsonl --reasoning-input reasoning.jsonl \\
           --output merged.jsonl
  %(prog)s --list-checkers
        """,
    )
    parser.add_argument("--content-checker", type=str, help="Checker for content JSONL")
    parser.add_argument("--reasoning-checker", type=str, help="Checker for reasoning JSONL")
    parser.add_argument("--content-input", type=str, help="Input JSONL for content")
    parser.add_argument("--reasoning-input", type=str, help="Input JSONL for reasoning")
    parser.add_argument("--output", type=str, help="Output merged JSONL")
    parser.add_argument("--verbose", action="store_true", help="Print per-line partial parse information")
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

    if not all([args.content_checker, args.reasoning_checker, args.content_input, args.reasoning_input, args.output]):
        parser.error(
            "--content-checker, --reasoning-checker, --content-input, --reasoning-input, "
            "and --output are required (unless using --list-checkers)"
        )

    if not os.path.exists(args.content_input):
        print(f"Error: Content input file '{args.content_input}' not found")
        return 1
    if not os.path.exists(args.reasoning_input):
        print(f"Error: Reasoning input file '{args.reasoning_input}' not found")
        return 1

    content_checker = load_format_checker(args.content_checker)
    if not content_checker:
        return 1

    reasoning_checker = load_format_checker(args.reasoning_checker)
    if not reasoning_checker:
        return 1

    stats = merge_files(
        content_file=args.content_input,
        reasoning_file=args.reasoning_input,
        output_file=args.output,
        content_checker=content_checker,
        reasoning_checker=reasoning_checker,
        verbose=args.verbose,
    )

    print("Merge Summary:")
    print(f"  Total aligned rows: {stats.total_rows}")
    print(
        f"  Content parse success: {stats.content_rows_ok}/{stats.total_rows} "
        f"({pct(stats.content_rows_ok, stats.total_rows):.1f}%)"
    )
    print(
        f"  Reasoning parse success: {stats.reasoning_rows_ok}/{stats.total_rows} "
        f"({pct(stats.reasoning_rows_ok, stats.total_rows):.1f}%)"
    )
    print(
        f"  Field success - problem: {stats.problem_ok}/{stats.total_rows} ({pct(stats.problem_ok, stats.total_rows):.1f}%)"
    )
    print(
        f"  Field success - generation: {stats.generation_ok}/{stats.total_rows} "
        f"({pct(stats.generation_ok, stats.total_rows):.1f}%)"
    )
    print(
        f"  Field success - reasoning_content: {stats.reasoning_content_ok}/{stats.total_rows} "
        f"({pct(stats.reasoning_content_ok, stats.total_rows):.1f}%)"
    )
    print(
        f"  Complete rows (all three fields non-empty): {stats.all_fields_ok}/{stats.total_rows} "
        f"({pct(stats.all_fields_ok, stats.total_rows):.1f}%)"
    )
    print(f"  Output file: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
