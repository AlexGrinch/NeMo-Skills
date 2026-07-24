#!/usr/bin/env python3
"""
Script to wrap JSONL data into a new format for translation tasks.

Each line in the input JSONL file will be wrapped into a new object with:
- source_lang: "English" (fixed)
- target_lang: selected from either one language or a weighted language list
- src: the original JSONL object serialized as a string (fields starting with
  "_translation_" are excluded from src and carried over as top-level fields instead)
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path


def _validate_language_weights(language_weights: list[dict]) -> list[tuple[str, float]]:
    """Validate and normalize a list of language/weight objects."""
    if not isinstance(language_weights, list) or not language_weights:
        raise ValueError("target languages must be a non-empty list")
    validated = []
    seen = set()
    for index, entry in enumerate(language_weights):
        if not isinstance(entry, dict):
            raise ValueError(f"target language entry {index} must be an object")
        language, weight = entry.get("language"), entry.get("weight")
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"target language entry {index} has an invalid language")
        language = language.strip()
        if language in seen:
            raise ValueError(f"duplicate target language: {language}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise ValueError(f"weight for {language} must be a finite number")
        if weight <= 0:
            raise ValueError(f"weight for {language} must be greater than zero")
        seen.add(language)
        validated.append((language, float(weight)))
    total_weight = sum(weight for _, weight in validated)
    return [(language, weight / total_weight) for language, weight in validated]


def _allocate_languages(num_records: int, language_weights: list[dict], seed: int = 0) -> list[str]:
    """Allocate exact, reproducibly shuffled language counts using largest remainders."""
    normalized = _validate_language_weights(language_weights)
    quotas = [num_records * weight for _, weight in normalized]
    counts = [math.floor(quota) for quota in quotas]
    remaining = num_records - sum(counts)
    remainder_order = sorted(range(len(normalized)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in remainder_order[:remaining]:
        counts[index] += 1
    assignments = [language for (language, _), count in zip(normalized, counts) for _ in range(count)]
    random.Random(seed).shuffle(assignments)
    return assignments


def _dataset_group_key(record: dict) -> tuple[str, str]:
    """Return a stable, hashable key for per-dataset language allocation."""
    if "_translation_dataset_id" not in record:
        return ("missing", "")
    serialized_id = json.dumps(record["_translation_dataset_id"], sort_keys=True, ensure_ascii=False)
    return ("dataset", serialized_id)


def _dataset_seed(seed: int, dataset_key: tuple[str, str]) -> int:
    """Derive an order-independent deterministic seed for one dataset."""
    value = f"{seed}\0{dataset_key[0]}\0{dataset_key[1]}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _source_group_key(record: dict, row_index: int) -> tuple[tuple[str, str], tuple[str, str]]:
    """Return the source-record key used for language assignment."""
    if "_translation_source_index" in record:
        source_value = json.dumps(record["_translation_source_index"], sort_keys=True, ensure_ascii=False)
        source_key = ("source_index", source_value)
    elif "_translation_src_id" in record:
        source_value = json.dumps(record["_translation_src_id"], sort_keys=True, ensure_ascii=False)
        source_key = ("src_id", source_value)
    else:
        source_key = ("row", str(row_index))
    return _dataset_group_key(record), source_key


def _allocate_languages_by_source_group(
    source_group_keys: list[tuple[tuple[str, str], tuple[str, str]]],
    language_weights: list[dict],
    seed: int = 0,
) -> list[str]:
    """Allocate language proportions per source record, not per split task."""
    _validate_language_weights(language_weights)
    groups_by_dataset: dict[tuple[str, str], list[tuple[tuple[str, str], tuple[str, str]]]] = {}
    seen: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for group_key in source_group_keys:
        dataset_key, _ = group_key
        if group_key in seen:
            continue
        groups_by_dataset.setdefault(dataset_key, []).append(group_key)
        seen.add(group_key)

    assignments_by_group: dict[tuple[tuple[str, str], tuple[str, str]], str] = {}
    for dataset_key, group_keys in groups_by_dataset.items():
        group_assignments = _allocate_languages(
            len(group_keys), language_weights, _dataset_seed(seed, dataset_key)
        )
        for group_key, language in zip(group_keys, group_assignments):
            assignments_by_group[group_key] = language

    return [assignments_by_group[group_key] for group_key in source_group_keys]


def _read_valid_records(input_path: Path, warn: bool = True):
    with open(input_path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if warn:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)


def wrap_jsonl_data(
    input_file: str,
    output_file: str,
    target_lang: str | None = None,
    target_langs: list[dict] | None = None,
    seed: int = 0,
) -> None:
    """
    Wrap JSONL data into the specified format.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        target_lang: A single target language (legacy mode)
        target_langs: Weighted target language objects, applied independently per dataset/source record
        seed: Random seed used to shuffle per-dataset weighted assignments
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if (target_lang is None) == (target_langs is None):
        raise ValueError("specify exactly one of target_lang or target_langs")

    source_group_keys = [
        _source_group_key(record, row_index)
        for row_index, record in enumerate(_read_valid_records(input_path))
    ]
    if target_langs is not None:
        assignments = _allocate_languages_by_source_group(source_group_keys, target_langs, seed)
    else:
        if not isinstance(target_lang, str) or not target_lang.strip():
            raise ValueError("target_lang must be a non-empty string")
        assignments = [target_lang.strip()] * len(source_group_keys)

    source_language_by_group = {}
    for group_key, language in zip(source_group_keys, assignments):
        source_language_by_group.setdefault(group_key, language)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_lines = 0

    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            for record, assigned_lang in zip(_read_valid_records(input_path, warn=False), assignments):
                # Fields starting with "_translation_" (e.g. _translation_src_id)
                # are pipeline metadata: carry them through as top-level fields
                # rather than embedding them in src so the model never sees them.
                payload = {k: v for k, v in record.items() if not k.startswith("_translation_")}
                metadata = {k: v for k, v in record.items() if k.startswith("_translation_")}

                wrapped_obj = {
                    "source_lang": "English",
                    "target_lang": assigned_lang,
                    "src": json.dumps(payload, ensure_ascii=False),
                    **metadata,
                }

                json.dump(wrapped_obj, outfile, ensure_ascii=False)
                outfile.write("\n")
                processed_lines += 1

    except IOError as e:
        print(f"Error processing files: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully processed {processed_lines} lines from '{input_file}' to '{output_file}'")
    task_counts = {language: assignments.count(language) for language in dict.fromkeys(assignments)}
    source_languages = list(source_language_by_group.values())
    source_counts = {language: source_languages.count(language) for language in dict.fromkeys(source_languages)}
    print(f"Target language task counts: {task_counts}")
    print(f"Target language source counts: {source_counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Wrap JSONL data into translation format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wrap_sft_data.py --input input.jsonl --output output.jsonl --target-lang "Spanish"
  python wrap_sft_data.py --input data/train.jsonl --output data/wrapped_train.jsonl -t "French"
  python wrap_sft_data.py --input input.jsonl --output output.jsonl \\
      --target-langs-json '[{"language":"German","weight":0.4},{"language":"French","weight":0.6}]'
        """,
    )

    parser.add_argument("--input", required=True, help="Path to input JSONL file")

    parser.add_argument("--output", required=True, help="Path to output JSONL file")

    parser.add_argument(
        "-t",
        "--target-lang",
        help='Target language for translation (e.g., "Spanish", "French", "German")',
    )
    parser.add_argument(
        "--target-langs-json",
        help="Weighted target language list encoded as JSON; mutually exclusive with --target-lang",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for weighted language assignment (default: 0)")

    args = parser.parse_args()

    if bool(args.target_lang) == bool(args.target_langs_json):
        parser.error("specify exactly one of --target-lang or --target-langs-json")
    try:
        target_langs = json.loads(args.target_langs_json) if args.target_langs_json else None
        wrap_jsonl_data(args.input, args.output, args.target_lang, target_langs, args.seed)
    except (json.JSONDecodeError, ValueError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
