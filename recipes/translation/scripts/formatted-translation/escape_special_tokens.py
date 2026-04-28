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
Escape or unescape model special tokens in JSONL files.

In escape mode, loads the model's tokenizer to discover all special tokens,
replaces them with <<ESC:...>> placeholders.
In unescape mode, reverses <<ESC:...>> placeholders back to the original tokens.

Escape format:
  <think>        → <<ESC:think>>
  </think>       → <<ESC:/think>>
  <|im_start|>   → <<ESC:|im_start|>>
  [PAD]          → <<ESC:[PAD]>>

The transformation is reversible: the content inside <<ESC:...>> is the original
token with outer <> stripped (if present). No mapping file needed.

Usage:
    python escape_special_tokens.py --model deepseek-ai/DeepSeek-V4-Flash \\
        --input wrapped.jsonl --output escaped.jsonl --mode escape
    python escape_special_tokens.py --model deepseek-ai/DeepSeek-V4-Flash \\
        --input output.jsonl --output unescaped.jsonl --mode unescape
"""

import argparse
import sys
from pathlib import Path


def get_special_tokens(model: str) -> list[str]:
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except (ValueError, AttributeError, KeyError):
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
    tokens = set(tokenizer.all_special_tokens)
    tokens.update(tokenizer.added_tokens_encoder.keys())
    return list(tokens)


def _escape_token(token: str) -> str:
    if token.startswith("<") and token.endswith(">"):
        inner = token[1:-1]
    else:
        inner = token
    return f"<<ESC:{inner}>>"


def _build_replacements(special_tokens: list[str], mode: str) -> list[tuple[str, str]]:
    pairs = [(token, _escape_token(token)) for token in special_tokens]
    if mode == "unescape":
        pairs = [(escaped, token) for token, escaped in pairs]
    # Longest first to avoid partial matches
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def process_file(input_file: str, output_file: str, replacements: list[tuple[str, str]]):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            for src, dst in replacements:
                line = line.replace(src, dst)
            fout.write(line)
            count += 1

    print(f"Processed {count} lines ({len(replacements)} token replacements).")


def main():
    parser = argparse.ArgumentParser(description="Escape/unescape model special tokens in JSONL files.")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--mode", required=True, choices=["escape", "unescape"])
    parser.add_argument("--model", required=True, help="Model name/path to load tokenizer")
    args = parser.parse_args()

    special_tokens = get_special_tokens(args.model)
    print(f"Found {len(special_tokens)} special tokens: {special_tokens}")

    replacements = _build_replacements(special_tokens, args.mode)
    process_file(args.input, args.output, replacements)


if __name__ == "__main__":
    sys.exit(main())
