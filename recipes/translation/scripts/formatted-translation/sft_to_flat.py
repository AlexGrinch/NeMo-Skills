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
Convert SFT-format data (with a ``messages`` array) to flat format by
extracting the last assistant turn into a top-level ``generation`` field.

Records that already have a top-level ``generation`` field are passed through
unchanged.  Records without ``messages`` are also passed through unchanged.

Usage:
    python sft_to_flat.py --input data.jsonl --output flat.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def extract_generation(messages: list[dict]) -> str | None:
    """Return the content of the last assistant message, or None."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def sft_to_flat(input_file: str, output_file: str, verbose: bool = False):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    total = extracted = already_flat = no_messages = 0
    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if verbose:
                    print(f"Line {line_num}: skipped (invalid JSON)", file=sys.stderr)
                continue

            if "generation" in record:
                already_flat += 1
            elif "messages" in record and isinstance(record["messages"], list):
                gen = extract_generation(record["messages"])
                if gen is not None:
                    record["generation"] = gen
                    extracted += 1
                elif verbose:
                    print(f"Line {line_num}: messages present but no assistant turn", file=sys.stderr)
            else:
                no_messages += 1
                if verbose:
                    print(f"Line {line_num}: no 'generation' or 'messages' field", file=sys.stderr)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Wrote {total} records to '{output_file}': "
        f"{extracted} extracted from messages, "
        f"{already_flat} already had generation, "
        f"{no_messages} had neither."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract assistant content from SFT messages into a top-level generation field.",
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--verbose", action="store_true", help="Print per-line warnings to stderr")
    args = parser.parse_args()
    sft_to_flat(args.input, args.output, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
