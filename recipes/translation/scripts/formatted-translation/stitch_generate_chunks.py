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
Collapse raw generations from chunked formatted-translation input.

Input rows are the raw output of nemo_skills.inference.generate over the
expanded chunk JSONL from chunk_generate_input.py.  Output rows match the normal
generate/output.jsonl contract: one row per original source row, with the
original unchunked `src` and one stitched `generation`.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).parent))

from filter_by_format import extract_first_valid_json  # type: ignore


CHUNK_PREFIX = "_translation_chunk_"
SUM_STATS = ("num_generated_tokens", "num_input_tokens", "generation_time")
MIN_STATS = ("generation_start_time",)
MAX_STATS = ("generation_end_time",)


def _iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fin:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSONL: {exc}") from exc


def _strip_chunk_metadata(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith(CHUNK_PREFIX)}


def _json_generation(obj: Any) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```"


def _failure_generation(errors: list[str]) -> str:
    return "Chunked generation failed; could not stitch translation chunks:\n" + "\n".join(errors)


def _extract_generated_object(row: dict) -> Any | None:
    generation = row.get("generation")
    if not isinstance(generation, str):
        return None
    extracted = extract_first_valid_json(generation)
    if extracted is None:
        return None
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return None


def _extract_translated_text(generated_obj: Any, field: str) -> tuple[str, str] | None:
    if not isinstance(generated_obj, dict):
        return None

    translation = generated_obj.get("translation")
    if isinstance(translation, dict) and isinstance(translation.get(field), str):
        return "translation", translation[field]

    if isinstance(generated_obj.get(field), str):
        return "direct", generated_obj[field]

    return None


def _aggregate_stats(rows: list[dict]) -> dict:
    stats = {}
    for key in SUM_STATS:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            stats[key] = sum(values)
    for key in MIN_STATS:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            stats[key] = min(values)
    for key in MAX_STATS:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            stats[key] = max(values)
    return stats


def _write_failure(fout, original_row: dict, errors: list[str], raw_rows: list[dict]) -> None:
    output = _strip_chunk_metadata(dict(original_row))
    output["generation"] = _failure_generation(errors)
    output.update(_aggregate_stats(raw_rows))
    fout.write(json.dumps(output, ensure_ascii=False) + "\n")


def _write_passthrough(fout, map_entry: dict, raw_rows: list[dict]) -> None:
    if not raw_rows:
        _write_failure(fout, map_entry["row"], ["missing raw generation row"], raw_rows)
        return
    output = _strip_chunk_metadata(dict(raw_rows[0]))
    output["src"] = map_entry["row"]["src"]
    fout.write(json.dumps(output, ensure_ascii=False) + "\n")


def _stitch_chunked_row(fout, map_entry: dict, raw_rows: list[dict], fields_to_translate: list[str]) -> bool:
    original_row = map_entry["row"]
    try:
        source_obj = json.loads(original_row["src"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        _write_failure(fout, original_row, [f"invalid original src: {exc}"], raw_rows)
        return False

    errors = []
    expected_count = map_entry.get("chunk_count")
    if expected_count != len(raw_rows):
        errors.append(f"expected {expected_count} raw chunks, found {len(raw_rows)}")

    parts_by_field: dict[str, list[str | None]] = {}
    modes_by_field: dict[str, list[str]] = {}

    for raw_row in raw_rows:
        field = raw_row.get(f"{CHUNK_PREFIX}field")
        part_index = raw_row.get(f"{CHUNK_PREFIX}part_index")
        part_count = raw_row.get(f"{CHUNK_PREFIX}part_count")
        if not isinstance(field, str) or not isinstance(part_index, int) or not isinstance(part_count, int):
            errors.append("raw chunk missing field/part metadata")
            continue

        generated_obj = _extract_generated_object(raw_row)
        if generated_obj is None:
            errors.append(f"{field}[{part_index}]: could not extract JSON generation")
            continue

        extracted = _extract_translated_text(generated_obj, field)
        if extracted is None:
            errors.append(f"{field}[{part_index}]: generated JSON has no translated text for field")
            continue
        mode, translated_text = extracted

        parts = parts_by_field.setdefault(field, [None] * part_count)
        if len(parts) != part_count:
            errors.append(f"{field}: inconsistent part_count metadata")
            continue
        if part_index < 0 or part_index >= part_count:
            errors.append(f"{field}: invalid part_index {part_index}")
            continue
        parts[part_index] = translated_text
        modes_by_field.setdefault(field, []).append(mode)

    joined: dict[str, str] = {}
    for field, parts in parts_by_field.items():
        if any(part is None for part in parts):
            errors.append(f"{field}: missing translated parts")
            continue
        joined[field] = "".join(part for part in parts if part is not None)

    missing_fields = [
        field
        for field in fields_to_translate
        if isinstance(source_obj, dict) and isinstance(source_obj.get(field), str) and field not in joined
    ]
    for field in missing_fields:
        errors.append(f"{field}: missing stitched translation")

    if errors:
        _write_failure(fout, original_row, errors, raw_rows)
        return False

    use_translation_object = any(mode == "translation" for modes in modes_by_field.values() for mode in modes)
    stitched_obj = deepcopy(source_obj)
    if use_translation_object:
        translation = stitched_obj.get("translation")
        if not isinstance(translation, dict):
            translation = {}
        for field in fields_to_translate:
            if field in joined:
                translation[field] = joined[field]
        stitched_obj["translation"] = translation
    else:
        for field in fields_to_translate:
            if field in joined:
                stitched_obj[field] = joined[field]

    output = _strip_chunk_metadata(dict(original_row))
    output["generation"] = _json_generation(stitched_obj)
    output.update(_aggregate_stats(raw_rows))
    fout.write(json.dumps(output, ensure_ascii=False) + "\n")
    return True


def stitch_file(
    *,
    raw_generation_file: str,
    map_file: str,
    output_file: str,
    fields_to_translate: list[str],
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    raw_iter = iter(_iter_jsonl(raw_generation_file))
    next_raw = next(raw_iter, None)
    total = passthrough = stitched = failed = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for map_entry in _iter_jsonl(map_file):
            total += 1
            source_index = map_entry["source_index"]
            raw_rows = []
            while next_raw is not None and next_raw.get(f"{CHUNK_PREFIX}source_index") == source_index:
                raw_rows.append(next_raw)
                next_raw = next(raw_iter, None)

            if not map_entry.get("chunked", False):
                _write_passthrough(fout, map_entry, raw_rows)
                passthrough += 1
                continue

            if _stitch_chunked_row(fout, map_entry, raw_rows, fields_to_translate):
                stitched += 1
            else:
                failed += 1

    extra_rows = 0
    while next_raw is not None:
        extra_rows += 1
        next_raw = next(raw_iter, None)
    if extra_rows:
        print(f"Warning: ignored {extra_rows} raw generation rows not present in stitch map.", file=sys.stderr)

    print(f"Stitched {total} rows: {passthrough} passthrough, {stitched} chunked, {failed} failed.")


def main():
    parser = argparse.ArgumentParser(description="Stitch chunked formatted-translation generations.")
    parser.add_argument("--input", required=True, help="Raw generation JSONL over chunked input")
    parser.add_argument("--map", required=True, help="Chunk map JSONL from chunk_generate_input.py")
    parser.add_argument("--output", required=True, help="Final stitched generation JSONL")
    parser.add_argument("--fields-to-translate", nargs="+", required=True, help="Top-level fields to stitch")
    args = parser.parse_args()

    stitch_file(
        raw_generation_file=args.input,
        map_file=args.map,
        output_file=args.output,
        fields_to_translate=args.fields_to_translate,
    )


if __name__ == "__main__":
    sys.exit(main())
