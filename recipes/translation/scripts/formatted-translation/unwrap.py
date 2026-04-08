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
Extract the translated JSON object from the model's `generation` field.

Input JSONL is expected to contain (at minimum):
    {"src": "...", "generation": "...", ...}

For each line, the script extracts JSON from the `generation` field (handling
optional ```json ... ``` fences).  Any metadata fields (names starting with "_",
e.g. _src_id) present in the generation record are carried through into the
output so downstream steps can join on them.  Failed lines are dropped.

Usage:
    python unwrap.py --input generate/output-rs0.jsonl --output unwrapped.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from filter_by_format import extract_first_valid_json  # type: ignore


def unwrap_file(input_file: str, output_file: str, verbose: bool = False):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    total = failed = 0
    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue

            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                failed += 1
                if verbose:
                    print(f"Line {line_num}: dropped (invalid JSONL)", file=sys.stderr)
                continue

            generation = row.get("generation", "")
            extracted = extract_first_valid_json(generation) if isinstance(generation, str) else None

            if extracted is None:
                failed += 1
                if verbose:
                    print(f"Line {line_num}: dropped (could not extract JSON from generation)", file=sys.stderr)
                continue

            # Carry any metadata fields (e.g. _translation_src_id) into the output.
            metadata = {k: v for k, v in row.items() if k.startswith("_translation_")}
            if metadata:
                output = {**json.loads(extracted), **metadata}
                fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            else:
                fout.write(extracted + "\n")

    print(f"Unwrapped {total - failed}/{total} lines. Failed: {failed}.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract translated JSON from the model generation field.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to generation JSONL (src + generation fields)")
    parser.add_argument("--output", required=True, help="Path to output JSONL with extracted JSON objects")
    parser.add_argument("--verbose", action="store_true", help="Print per-line extraction failures to stderr")
    args = parser.parse_args()

    unwrap_file(args.input, args.output, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
