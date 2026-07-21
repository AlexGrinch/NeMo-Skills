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
Produce a concise version of an input JSONL by keeping only the specified fields,
stamp each output record with a _translation_src_id derived from the original,
and metadata.dataset as _translation_dataset_id when present.

Fields prefixed with _translation_ are pipeline metadata. Downstream stages
carry them outside the model-visible source payload.

The ID is a SHA-256 hash (first 16 hex chars) of the canonically serialized
original record (sorted keys, no extra whitespace).  Two records with identical
content always get the same ID, which is fine — their translations will be
identical too.  The ID is used by downstream steps (filter_and_merge) to join
translations back to original records without relying on line alignment.

Usage:
    python make_concise.py --input data.jsonl --output concise.jsonl --fields problem generation
    python make_concise.py --input data.jsonl --output concise.jsonl --from-messages
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MESSAGE_TEXT_FIELDS = {"content", "reasoning_content", "text", "refusal"}


def compute_src_id(record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def is_message_text_field(field_name: str, message_text_fields: set[str] | None = None) -> bool:
    if message_text_fields is not None:
        return field_name in message_text_fields
    return field_name in DEFAULT_MESSAGE_TEXT_FIELDS or field_name.endswith("_content")


def extract_translatable_value(value: Any, message_text_fields: set[str] | None = None) -> Any | None:
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list):
        extracted_items = []
        has_translatable_text = False
        for item in value:
            extracted = extract_translatable_value(item, message_text_fields)
            if extracted is None:
                if isinstance(item, dict):
                    extracted = {}
                elif isinstance(item, list):
                    extracted = []
            else:
                has_translatable_text = True
            extracted_items.append(extracted)
        return extracted_items if has_translatable_text else None
    if isinstance(value, dict):
        extracted = {}
        for key, nested_value in value.items():
            if is_message_text_field(key, message_text_fields):
                nested_extracted = extract_translatable_value(nested_value, message_text_fields)
            elif isinstance(nested_value, (dict, list)):
                nested_extracted = extract_translatable_value(nested_value, message_text_fields)
            else:
                nested_extracted = None
            if nested_extracted is not None:
                extracted[key] = nested_extracted
        return extracted if extracted else None
    return None


def extract_message_text_fields(message: dict, message_text_fields: set[str] | None = None) -> dict:
    extracted = {}
    for key, value in message.items():
        if not is_message_text_field(key, message_text_fields):
            continue
        extracted_value = extract_translatable_value(value, message_text_fields)
        if extracted_value is not None:
            extracted[key] = extracted_value
    return extracted


def extract_from_messages(record: dict, fields: list[str], message_text_fields: set[str] | None = None) -> dict:
    """Extract all configured translatable text fields from messages.

    Message indexes are preserved so translated values can be merged back into
    the original record without changing any non-message or non-text fields.
    ``fields`` is accepted for CLI compatibility but is not used in this mode.
    """
    messages = record.get("messages", [])
    concise = {"messages": []}
    if not isinstance(messages, list):
        return concise

    for message in messages:
        if isinstance(message, dict):
            concise["messages"].append(extract_message_text_fields(message, message_text_fields))
        else:
            concise["messages"].append({})
    return concise


def make_concise_file(
    input_file: str,
    output_file: str,
    fields: list[str],
    from_messages: bool = False,
    message_text_fields: list[str] | None = None,
    add_src_id: bool = True,
    verbose: bool = False,
):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    message_text_field_set = set(message_text_fields) if message_text_fields is not None else None

    total = skipped = 0
    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue

            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if verbose:
                    print(f"Line {line_num}: skipped (invalid JSON)", file=sys.stderr)
                continue

            if from_messages:
                concise = extract_from_messages(record, fields, message_text_field_set)
            else:
                missing = [k for k in fields if k not in record]
                if missing and verbose:
                    print(f"Line {line_num}: missing fields {missing}", file=sys.stderr)
                concise = {k: record[k] for k in fields if k in record}

            if add_src_id:
                concise["_translation_src_id"] = compute_src_id(record)

            metadata = record.get("metadata")
            if isinstance(metadata, dict) and "dataset" in metadata:
                concise["_translation_dataset_id"] = metadata["dataset"]

            fout.write(json.dumps(concise, ensure_ascii=False) + "\n")

    if skipped:
        print(f"Warning: skipped {skipped}/{total} lines (invalid JSON).", file=sys.stderr)
    print(f"Wrote {total - skipped} concise records to '{output_file}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Keep only specified fields from each JSON object in a JSONL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--fields",
        nargs="*",
        default=[],
        metavar="FIELD",
        help="Top-level fields to keep (e.g. --fields problem generation). Optional with --from-messages.",
    )
    parser.add_argument(
        "--from-messages",
        action="store_true",
        help="Extract translatable text fields from the messages array instead of top-level keys",
    )
    parser.add_argument(
        "--message-text-fields",
        nargs="+",
        metavar="FIELD",
        help=(
            "Message field names to translate with --from-messages. Defaults to "
            "content, reasoning_content, text, refusal, and fields ending in _content."
        ),
    )
    parser.add_argument(
        "--no-src-id",
        action="store_true",
        help="Do not add _translation_src_id to output records",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-line warnings to stderr")
    args = parser.parse_args()
    if not args.from_messages and not args.fields:
        parser.error("--fields is required unless --from-messages is set")

    make_concise_file(
        args.input,
        args.output,
        args.fields,
        from_messages=args.from_messages,
        message_text_fields=args.message_text_fields,
        add_src_id=not args.no_src_id,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
