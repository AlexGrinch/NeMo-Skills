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
Validate translated outputs with a format checker and merge passing translations
back into the original JSON objects.

Memory strategy (O(1), designed for very large files):
  Both files are streamed simultaneously.  The generation file is treated as an
  ordered subsequence of the original: for each generation record we advance the
  original file until _translation_src_id matches, writing any skipped original
  records according to --on-fail.  Once the IDs match we run the format check,
  extract the translated fields, and write the merged record.  Any remaining
  original records after the generation file is exhausted are also handled by
  --on-fail.

  This assumes the generation file preserves the relative order of the original.
  nemo_skills sorts generation output by _async_position, so this holds as long
  as the pipeline is run correctly.

Usage:
    python filter_and_merge.py \\
        --checker format_nano1 \\
        --original-file input.jsonl \\
        --generation-file generate/output-rs0.jsonl \\
        --fields-to-translate problem generation \\
        --output translated.jsonl
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from filter_by_format import extract_first_valid_json, load_format_checker  # type: ignore


def compute_src_id(record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def iter_nonempty(f):
    """Yield (line_num, raw_line, parsed_record) for each non-empty line."""
    for line_num, raw_line in enumerate(f, 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            yield line_num, raw_line, json.loads(stripped)
        except json.JSONDecodeError as e:
            print(f"Line {line_num}: skipped (invalid JSON: {e})", file=sys.stderr)


def write_original(fout, raw_line, on_fail):
    if on_fail == "keep":
        fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")


def filter_and_merge(
    original_file: str,
    generation_file: str,
    checker,
    fields_to_translate: list[str],
    output_file: str,
    on_fail: str,
    to_messages: bool = False,
    verbose: bool = False,
):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    total = merged = skipped_orig = failed_check = failed_extract = 0

    with (
        open(original_file, encoding="utf-8") as forig,
        open(generation_file, encoding="utf-8") as fgen,
        open(output_file, "w", encoding="utf-8") as fout,
    ):
        orig_iter = iter_nonempty(forig)
        gen_iter = iter_nonempty(fgen)

        # Peek at first generation record to decide pairing strategy.
        first_gen = next(gen_iter, None)
        if first_gen is not None:
            gen_iter = itertools.chain([first_gen], gen_iter)
            _, _, first_record = first_gen
            use_src_id = "_translation_src_id" in first_record
            if not use_src_id:
                print("Note: no '_translation_src_id' in generation; using positional matching.", file=sys.stderr)

        orig_item = next(orig_iter, None)

        for gen_line_num, gen_raw, gen_record in gen_iter:
            if use_src_id:
                # Advance original until IDs match, writing skipped records per --on-fail.
                gen_src_id = gen_record["_translation_src_id"]
                while orig_item is not None:
                    orig_line_num, orig_raw, orig_record = orig_item
                    total += 1
                    if compute_src_id(orig_record) == gen_src_id:
                        break
                    skipped_orig += 1
                    if verbose:
                        print(
                            f"Original line {orig_line_num}: no matching generation, applying --on-fail={on_fail}",
                            file=sys.stderr,
                        )
                    write_original(fout, orig_raw, on_fail)
                    orig_item = next(orig_iter, None)
                else:
                    print(
                        f"Warning: generation line {gen_line_num} (src_id={gen_src_id}) has no matching original record.",
                        file=sys.stderr,
                    )
                    continue
            else:
                # Positional: just take the next original line.
                if orig_item is None:
                    print(
                        f"Warning: generation line {gen_line_num} has no matching original (original exhausted).",
                        file=sys.stderr,
                    )
                    continue
                orig_line_num, orig_raw, orig_record = orig_item
                total += 1

            # Validate and merge.
            src = gen_record.get("src", "")
            generation = gen_record.get("generation", "")

            if not checker.check(src, generation):
                failed_check += 1
                if verbose:
                    print(f"Original line {orig_line_num}: failed format check", file=sys.stderr)
                write_original(fout, orig_raw, on_fail)
                orig_item = next(orig_iter, None)
                continue

            extracted = extract_first_valid_json(generation)
            if extracted is None:
                failed_extract += 1
                if verbose:
                    print(f"Original line {orig_line_num}: failed JSON extraction", file=sys.stderr)
                write_original(fout, orig_raw, on_fail)
                orig_item = next(orig_iter, None)
                continue

            translated = json.loads(extracted)
            translation_source = translated
            if isinstance(translated.get("translation"), dict):
                translation_source = translated["translation"]
            output_record = dict(orig_record)
            if to_messages and "messages" in output_record:
                messages = [dict(m) for m in output_record["messages"]]
                for i, field in enumerate(fields_to_translate):
                    if field in translation_source and i < len(messages):
                        messages[i]["content"] = translation_source[field]
                output_record["messages"] = messages
            else:
                output_record.update(
                    {k: translation_source[k] for k in fields_to_translate if k in translation_source}
                )
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            merged += 1
            orig_item = next(orig_iter, None)

        # Drain any remaining original records.
        while orig_item is not None:
            orig_line_num, orig_raw, orig_record = orig_item
            total += 1
            skipped_orig += 1
            if verbose:
                print(
                    f"Original line {orig_line_num}: no matching generation (exhausted), applying --on-fail={on_fail}",
                    file=sys.stderr,
                )
            write_original(fout, orig_raw, on_fail)
            orig_item = next(orig_iter, None)

    print(
        f"Done: {total} original records, {merged} merged, "
        f"{skipped_orig} skipped in generation, "
        f"{failed_check} failed format check, {failed_extract} failed extraction."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Filter by format checker and merge translated fields into original objects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checker", required=True, help="Format checker name (e.g. format_nano1)")
    parser.add_argument("--original-file", required=True, help="Original JSONL (pre-concise) to merge into")
    parser.add_argument("--generation-file", required=True, help="Generation JSONL (src + generation fields)")
    parser.add_argument(
        "--fields-to-translate",
        nargs="+",
        required=True,
        metavar="FIELD",
        help="Fields to replace in the original object with translated values",
    )
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument(
        "--on-fail",
        choices=["keep", "drop"],
        default="keep",
        help="What to do when format check fails or record is missing from generation (default: keep)",
    )
    parser.add_argument(
        "--to-messages",
        action="store_true",
        help="Write translated fields back into the messages array by position instead of top-level keys",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-line details to stderr")
    args = parser.parse_args()

    checker = load_format_checker(args.checker)
    if checker is None:
        sys.exit(1)

    filter_and_merge(
        original_file=args.original_file,
        generation_file=args.generation_file,
        checker=checker,
        fields_to_translate=args.fields_to_translate,
        output_file=args.output,
        on_fail=args.on_fail,
        to_messages=args.to_messages,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
