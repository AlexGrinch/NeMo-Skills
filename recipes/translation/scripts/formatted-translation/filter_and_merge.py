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
  The generation file is treated as an ordered subsequence of the original.
  Each original record may have one or more consecutive generation rows. For
  message-mode translation, make_concise emits one generation row per message
  text field, and this script groups those rows by _translation_source_index
  (or by _translation_src_id for older files) before writing one output record.

  This assumes the generation file preserves the relative order of the concise
  input. nemo_skills sorts generation output by _async_position, so this holds
  as long as the pipeline is run correctly.

Usage:
    python filter_and_merge.py \
        --checker format_messages \
        --original-file input.jsonl \
        --generation-file generate/output-rs0.jsonl \
        --output translated.jsonl \
        --to-messages
"""

import argparse
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).parent))

from filter_by_format import extract_first_valid_json, load_format_checker  # type: ignore


def compute_src_id(record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def iter_nonempty(f):
    """Yield (source_index, line_num, raw_line, parsed_record) for each non-empty line."""
    source_index = 0
    for line_num, raw_line in enumerate(f, 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            yield source_index, line_num, raw_line, json.loads(stripped)
            source_index += 1
        except json.JSONDecodeError as e:
            print(f"Line {line_num}: skipped (invalid JSON: {e})", file=sys.stderr)


def write_original(fout, raw_line, on_fail):
    if on_fail == "keep":
        fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")


def merge_translated_value(original: Any, translated: Any) -> Any:
    """Return original's shape with translated string leaves copied in."""
    if isinstance(original, str):
        return translated if isinstance(translated, str) else original
    if isinstance(original, list) and isinstance(translated, list):
        merged = copy.deepcopy(original)
        for index, translated_item in enumerate(translated[: len(merged)]):
            merged[index] = merge_translated_value(merged[index], translated_item)
        return merged
    if isinstance(original, dict) and isinstance(translated, dict):
        merged = copy.deepcopy(original)
        for key, translated_value in translated.items():
            if key in merged:
                merged[key] = merge_translated_value(merged[key], translated_value)
        return merged
    return original


def merge_message_translation(message: Any, translation: Any) -> Any:
    if not isinstance(message, dict):
        return message

    merged = copy.deepcopy(message)
    if isinstance(translation, str):
        if isinstance(merged.get("content"), str):
            merged["content"] = translation
        return merged

    if not isinstance(translation, dict):
        return merged

    for key, translated_value in translation.items():
        if key == "translation":
            continue
        if key in merged:
            merged[key] = merge_translated_value(merged[key], translated_value)
    return merged


def merge_translated_messages(
    original_messages: Any,
    translated_payload: dict,
    fields_to_translate: list[str],
    message_index: int | None = None,
) -> Any:
    if not isinstance(original_messages, list):
        return original_messages

    merged_messages = copy.deepcopy(original_messages)
    translated_messages = translated_payload.get("messages")

    if message_index is not None:
        if (
            isinstance(message_index, int)
            and not isinstance(message_index, bool)
            and 0 <= message_index < len(merged_messages)
            and isinstance(translated_messages, list)
            and translated_messages
            and isinstance(translated_messages[0], dict)
        ):
            translated_message = translated_messages[0]
            translation = translated_message.get("translation", translated_message)
            merged_messages[message_index] = merge_message_translation(merged_messages[message_index], translation)
        return merged_messages

    if isinstance(translated_messages, list):
        for index, translated_message in enumerate(translated_messages[: len(merged_messages)]):
            if not isinstance(translated_message, dict):
                continue
            translation = translated_message.get("translation", translated_message)
            merged_messages[index] = merge_message_translation(merged_messages[index], translation)
        return merged_messages

    # Backward-compatible fallback for the old from_messages behavior where
    # fields_to_translate mapped synthetic fields to messages by position.
    translation = translated_payload.get("translation")
    if isinstance(translation, dict):
        for index, field in enumerate(fields_to_translate):
            if index >= len(merged_messages):
                break
            if field in translation and isinstance(merged_messages[index], dict):
                merged_messages[index] = merge_message_translation(merged_messages[index], translation[field])

    return merged_messages


def generation_source_key(record: dict, match_mode: str):
    if match_mode == "source_index":
        return record.get("_translation_source_index")
    if match_mode == "src_id":
        return record.get("_translation_src_id")
    return None


def original_source_key(source_index: int, record: dict, match_mode: str):
    if match_mode == "source_index":
        return source_index
    if match_mode == "src_id":
        return compute_src_id(record)
    return None


def generation_group(first_item, gen_iter: Iterable, match_mode: str):
    group = [first_item]
    next_item = next(gen_iter, None)
    if match_mode == "positional":
        return group, next_item

    group_key = generation_source_key(first_item[3], match_mode)
    while next_item is not None and generation_source_key(next_item[3], match_mode) == group_key:
        group.append(next_item)
        next_item = next(gen_iter, None)
    return group, next_item


def extract_translated_payload(gen_record: dict, checker, verbose: bool, orig_line_num: int):
    src = gen_record.get("src", "")
    generation = gen_record.get("generation", "")

    if not checker.check(src, generation):
        if verbose:
            print(f"Original line {orig_line_num}: failed format check", file=sys.stderr)
        return None, "check"

    extracted = extract_first_valid_json(generation)
    if extracted is None:
        if verbose:
            print(f"Original line {orig_line_num}: failed JSON extraction", file=sys.stderr)
        return None, "extract"

    translated = json.loads(extracted)
    if not isinstance(translated, dict):
        if verbose:
            print(f"Original line {orig_line_num}: extracted JSON was not an object", file=sys.stderr)
        return None, "extract"

    return translated, None


def merge_generation_group(
    *,
    orig_record: dict,
    orig_line_num: int,
    gen_group: list[tuple[int, int, str, dict]],
    checker,
    fields_to_translate: list[str],
    to_messages: bool,
    verbose: bool,
):
    output_record = copy.deepcopy(orig_record)
    failed_check = failed_extract = 0

    for _, _, _, gen_record in gen_group:
        translated, error_kind = extract_translated_payload(gen_record, checker, verbose, orig_line_num)
        if error_kind == "check":
            failed_check += 1
            continue
        if error_kind == "extract":
            failed_extract += 1
            continue

        if gen_record.get("_translation_noop"):
            continue

        if to_messages and "messages" in output_record:
            message_index = gen_record.get("_translation_message_index")
            output_record["messages"] = merge_translated_messages(
                output_record["messages"],
                translated,
                fields_to_translate,
                message_index=message_index,
            )
        else:
            translation_source = translated
            if isinstance(translated.get("translation"), dict):
                translation_source = translated["translation"]
            output_record.update(
                {k: translation_source[k] for k in fields_to_translate if k in translation_source}
            )

    if failed_check or failed_extract:
        return None, failed_check, failed_extract
    return output_record, failed_check, failed_extract


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

        first_gen = next(gen_iter, None)
        if first_gen is None:
            pending_gen = None
            match_mode = "positional"
        else:
            gen_iter = itertools.chain([first_gen], gen_iter)
            _, _, _, first_record = first_gen
            if "_translation_source_index" in first_record:
                match_mode = "source_index"
            elif "_translation_src_id" in first_record:
                match_mode = "src_id"
            else:
                match_mode = "positional"
                print("Note: no translation source metadata in generation; using positional matching.", file=sys.stderr)
            pending_gen = next(gen_iter, None)

        orig_item = next(orig_iter, None)

        while pending_gen is not None:
            gen_group, pending_gen = generation_group(pending_gen, gen_iter, match_mode)
            group_key = generation_source_key(gen_group[0][3], match_mode)

            if match_mode == "positional":
                if orig_item is None:
                    print(
                        f"Warning: generation line {gen_group[0][1]} has no matching original (original exhausted).",
                        file=sys.stderr,
                    )
                    continue
                orig_source_index, orig_line_num, orig_raw, orig_record = orig_item
                total += 1
            else:
                while orig_item is not None:
                    orig_source_index, orig_line_num, orig_raw, orig_record = orig_item
                    total += 1
                    if original_source_key(orig_source_index, orig_record, match_mode) == group_key:
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
                        f"Warning: generation line {gen_group[0][1]} (source={group_key}) has no matching original record.",
                        file=sys.stderr,
                    )
                    continue

            output_record, group_failed_check, group_failed_extract = merge_generation_group(
                orig_record=orig_record,
                orig_line_num=orig_line_num,
                gen_group=gen_group,
                checker=checker,
                fields_to_translate=fields_to_translate,
                to_messages=to_messages,
                verbose=verbose,
            )
            failed_check += group_failed_check
            failed_extract += group_failed_extract

            if output_record is None:
                write_original(fout, orig_raw, on_fail)
            else:
                fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                merged += 1

            orig_item = next(orig_iter, None)

        while orig_item is not None:
            _, orig_line_num, orig_raw, _ = orig_item
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
        nargs="*",
        default=[],
        metavar="FIELD",
        help="Fields to replace in the original object with translated values. Optional with --to-messages.",
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
        help="Merge translated message text fields back into the original messages array",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-line details to stderr")
    args = parser.parse_args()
    if not args.to_messages and not args.fields_to_translate:
        parser.error("--fields-to-translate is required unless --to-messages is set")

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
