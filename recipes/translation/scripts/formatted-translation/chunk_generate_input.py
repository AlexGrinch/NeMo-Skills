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
Expand formatted translation inputs into smaller generation rows.

The formatted translation pipeline normally sends one wrapped row to the model:

    {"source_lang": "English", "target_lang": "German", "src": "{\"field\": \"...\"}"}

For very long string fields this can exceed the model context window.  This
script writes a new JSONL where oversized rows are split into field chunks.  A
sidecar map records how to stitch the raw generations back into the original
one-row-per-source shape.
"""

import argparse
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a repository dependency
    yaml = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - only used when --tokenizer is set
    AutoTokenizer = None

try:
    from nemo_skills.prompt.utils import get_prompt, get_token_count
except Exception:  # pragma: no cover - fallback keeps the script usable for local checks
    get_prompt = None
    get_token_count = None


CHUNK_PREFIX = "_translation_chunk_"


def _load_yaml(path: str) -> dict:
    if yaml is None:
        return {}
    with open(path, "rt", encoding="utf-8") as fin:
        return yaml.safe_load(fin) or {}


def _prompt_text_for_fallback(prompt_config: dict, data_point: dict) -> str:
    user_template = prompt_config.get("user", "{src}")
    system = prompt_config.get("system") or ""
    try:
        user = user_template.format(examples="", **data_point)
    except KeyError:
        user = data_point.get("src", "")
    return f"{system}\n{user}" if system else user


def _messages_to_text(messages: str | list[dict]) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(message.get("content", "")) for message in messages)


class PromptTokenCounter:
    def __init__(
        self,
        prompt_config: str,
        tokenizer: str | None,
        endpoint_type: str,
        chars_per_token: float,
    ):
        self.endpoint_type = endpoint_type
        self.chars_per_token = chars_per_token
        self.prompt_config_dict = _load_yaml(prompt_config)
        self.prompt = None
        self.tokenizer = None

        if tokenizer:
            if AutoTokenizer is None:
                print("Warning: transformers is unavailable; using approximate token counts.", file=sys.stderr)
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                except Exception as exc:
                    print(f"Warning: could not load tokenizer '{tokenizer}'; using approximate token counts: {exc}", file=sys.stderr)
                    self.tokenizer = None

        if get_prompt is not None:
            try:
                prompt_tokenizer = self.tokenizer if endpoint_type == "text" and self.tokenizer is not None else None
                self.prompt = get_prompt(prompt_config=prompt_config, tokenizer=prompt_tokenizer)
            except Exception as exc:
                print(f"Warning: falling back to approximate prompt formatting: {exc}", file=sys.stderr)
                self.prompt = None

    def build_prompt(self, data_point: dict) -> str | list[dict]:
        if self.prompt is not None:
            return self.prompt.fill(
                data_point,
                format_as_string=(self.endpoint_type == "text"),
            )

        text = _prompt_text_for_fallback(self.prompt_config_dict, data_point)
        system = self.prompt_config_dict.get("system")
        if self.endpoint_type == "text" or not system:
            return text
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    def count(self, data_point: dict) -> int:
        prompt = self.build_prompt(data_point)
        if self.tokenizer is not None:
            if get_token_count is not None:
                try:
                    count = get_token_count(self.tokenizer, prompt)
                    if count is not None:
                        return count
                except Exception:
                    pass

            text = _messages_to_text(prompt)
            return len(self.tokenizer.encode(text, add_special_tokens=False))

        text = _messages_to_text(prompt)
        return max(1, math.ceil(len(text) / self.chars_per_token))


def _empty_like(value: Any) -> Any:
    if isinstance(value, str):
        return ""
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {}
    return value


def _minimized_source(source_obj: Any, fields_to_translate: list[str]) -> Any:
    if not isinstance(source_obj, dict):
        return source_obj
    minimized = {}
    for key, value in source_obj.items():
        minimized[key] = _empty_like(value)
    return minimized


def _build_chunk_row(original_row: dict, source_obj: dict, field: str, text: str, fields_to_translate: list[str]) -> dict:
    chunk_source = _minimized_source(source_obj, fields_to_translate)
    chunk_source[field] = text
    chunk_row = dict(original_row)
    chunk_row["src"] = json.dumps(chunk_source, ensure_ascii=False)
    return chunk_row


def _natural_boundary(text: str, start: int, limit: int) -> int:
    if limit >= len(text):
        return len(text)

    min_pos = start + max(1, (limit - start) // 2)
    boundary_patterns = [
        r"\n\n+",
        r"\n+",
        r"(?<=[.!?。！？])\s+",
        r"\s+",
    ]
    best = None
    window = text[start:limit]
    for pattern in boundary_patterns:
        for match in re.finditer(pattern, window, flags=re.DOTALL):
            candidate = start + match.end()
            if candidate >= min_pos:
                best = candidate
        if best is not None:
            return best
    return limit


def _split_oversized_prefix(text: str, start: int, limit: int) -> int:
    """Return a non-empty split point even when prompt overhead leaves no fit."""
    if limit > start:
        return limit
    return min(len(text), start + 1)


def split_text_by_prompt_budget(
    *,
    text: str,
    original_row: dict,
    source_obj: dict,
    field: str,
    fields_to_translate: list[str],
    counter: PromptTokenCounter,
    prompt_token_budget: int,
) -> list[str]:
    if text == "":
        return [text]

    def fits(candidate: str) -> bool:
        chunk_row = _build_chunk_row(original_row, source_obj, field, candidate, fields_to_translate)
        return counter.count(chunk_row) <= prompt_token_budget

    if fits(text):
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        lo = 1
        hi = len(text) - start
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[start : start + mid]
            if fits(candidate):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if best == 0:
            raise ValueError(
                f"Cannot fit even one character from field '{field}' within prompt token budget "
                f"{prompt_token_budget}. Increase --max-model-len or reduce prompt/tokens_to_generate."
            )

        raw_end = start + best
        end = _natural_boundary(text, start, raw_end)
        end = _split_oversized_prefix(text, start, end)
        chunks.append(text[start:end])
        start = end

    return chunks


def _with_chunk_metadata(
    row: dict,
    *,
    source_index: int,
    chunk_index: int,
    chunk_count: int,
    mode: str,
    field: str | None = None,
    part_index: int | None = None,
    part_count: int | None = None,
) -> dict:
    chunk_row = dict(row)
    chunk_row[f"{CHUNK_PREFIX}source_index"] = source_index
    chunk_row[f"{CHUNK_PREFIX}index"] = chunk_index
    chunk_row[f"{CHUNK_PREFIX}count"] = chunk_count
    chunk_row[f"{CHUNK_PREFIX}mode"] = mode
    if field is not None:
        chunk_row[f"{CHUNK_PREFIX}field"] = field
    if part_index is not None:
        chunk_row[f"{CHUNK_PREFIX}part_index"] = part_index
    if part_count is not None:
        chunk_row[f"{CHUNK_PREFIX}part_count"] = part_count
    return chunk_row


def chunk_file(
    *,
    input_file: str,
    output_file: str,
    map_output: str,
    fields_to_translate: list[str],
    prompt_config: str,
    max_model_len: int,
    tokens_to_generate: int,
    safety_margin: int,
    prompt_token_budget: int | None,
    tokenizer: str | None,
    endpoint_type: str,
    chars_per_token: float,
) -> None:
    if prompt_token_budget is None:
        prompt_token_budget = max_model_len - tokens_to_generate - safety_margin
    if prompt_token_budget <= 0:
        raise ValueError(
            f"Prompt token budget is {prompt_token_budget}. "
            "Increase --max-model-len or reduce --tokens-to-generate/--safety-margin."
        )

    counter = PromptTokenCounter(
        prompt_config=prompt_config,
        tokenizer=tokenizer,
        endpoint_type=endpoint_type,
        chars_per_token=chars_per_token,
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(map_output).parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    expanded_rows = 0
    chunked_rows = 0
    warnings = 0

    with (
        open(input_file, encoding="utf-8") as fin,
        open(output_file, "w", encoding="utf-8") as fout,
        open(map_output, "w", encoding="utf-8") as fmap,
    ):
        for source_index, raw_line in enumerate(fin):
            line = raw_line.strip()
            if not line:
                continue
            total_rows += 1
            row = json.loads(line)

            try:
                source_obj = json.loads(row["src"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Input line {source_index + 1} has invalid JSON string in 'src': {exc}") from exc

            if counter.count(row) <= prompt_token_budget:
                chunk_row = _with_chunk_metadata(
                    row,
                    source_index=source_index,
                    chunk_index=0,
                    chunk_count=1,
                    mode="full",
                )
                fout.write(json.dumps(chunk_row, ensure_ascii=False) + "\n")
                fmap.write(
                    json.dumps(
                        {
                            "source_index": source_index,
                            "row": row,
                            "chunked": False,
                            "chunk_count": 1,
                            "chunks": [{"mode": "full", "chunk_index": 0}],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                expanded_rows += 1
                continue

            if not isinstance(source_obj, dict):
                print(
                    f"Warning: line {source_index + 1} is oversized but src is not an object; leaving unchunked.",
                    file=sys.stderr,
                )
                warnings += 1
                chunk_row = _with_chunk_metadata(
                    row,
                    source_index=source_index,
                    chunk_index=0,
                    chunk_count=1,
                    mode="full",
                )
                fout.write(json.dumps(chunk_row, ensure_ascii=False) + "\n")
                fmap.write(
                    json.dumps(
                        {
                            "source_index": source_index,
                            "row": row,
                            "chunked": False,
                            "chunk_count": 1,
                            "chunks": [{"mode": "full", "chunk_index": 0}],
                            "warning": "oversized non-object src left unchunked",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                expanded_rows += 1
                continue

            field_parts: dict[str, list[str]] = {}
            for field in fields_to_translate:
                value = source_obj.get(field)
                if isinstance(value, str):
                    field_parts[field] = split_text_by_prompt_budget(
                        text=value,
                        original_row=row,
                        source_obj=source_obj,
                        field=field,
                        fields_to_translate=fields_to_translate,
                        counter=counter,
                        prompt_token_budget=prompt_token_budget,
                    )

            if not field_parts:
                print(
                    f"Warning: line {source_index + 1} is oversized but has no top-level string fields to chunk; "
                    "leaving unchunked.",
                    file=sys.stderr,
                )
                warnings += 1
                chunk_row = _with_chunk_metadata(
                    row,
                    source_index=source_index,
                    chunk_index=0,
                    chunk_count=1,
                    mode="full",
                )
                fout.write(json.dumps(chunk_row, ensure_ascii=False) + "\n")
                fmap.write(
                    json.dumps(
                        {
                            "source_index": source_index,
                            "row": row,
                            "chunked": False,
                            "chunk_count": 1,
                            "chunks": [{"mode": "full", "chunk_index": 0}],
                            "warning": "oversized row had no top-level string fields to chunk",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                expanded_rows += 1
                continue

            chunks = []
            for field, parts in field_parts.items():
                for part_index, part in enumerate(parts):
                    chunks.append(
                        {
                            "mode": "field",
                            "field": field,
                            "part_index": part_index,
                            "part_count": len(parts),
                        }
                    )

            for chunk_index, chunk in enumerate(chunks):
                chunk_row = _build_chunk_row(
                    row,
                    source_obj,
                    chunk["field"],
                    field_parts[chunk["field"]][chunk["part_index"]],
                    fields_to_translate,
                )
                chunk_row = _with_chunk_metadata(
                    chunk_row,
                    source_index=source_index,
                    chunk_index=chunk_index,
                    chunk_count=len(chunks),
                    mode="field",
                    field=chunk["field"],
                    part_index=chunk["part_index"],
                    part_count=chunk["part_count"],
                )
                fout.write(json.dumps(chunk_row, ensure_ascii=False) + "\n")
                expanded_rows += 1

            fmap.write(
                json.dumps(
                    {
                        "source_index": source_index,
                        "row": row,
                        "chunked": True,
                        "chunk_count": len(chunks),
                        "chunks": [
                            {**deepcopy(chunk), "chunk_index": chunk_index}
                            for chunk_index, chunk in enumerate(chunks)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            chunked_rows += 1

    print(
        f"Chunked {total_rows} source rows into {expanded_rows} generation rows "
        f"({chunked_rows} oversized rows split, prompt budget {prompt_token_budget} tokens, warnings {warnings})."
    )


def main():
    parser = argparse.ArgumentParser(description="Chunk long formatted-translation generation inputs.")
    parser.add_argument("--input", required=True, help="Wrapped JSONL input")
    parser.add_argument("--output", required=True, help="Expanded chunk JSONL output")
    parser.add_argument("--map-output", required=True, help="Sidecar JSONL stitch map")
    parser.add_argument("--fields-to-translate", nargs="+", required=True, help="Top-level string fields to chunk")
    parser.add_argument("--prompt-config", required=True, help="Prompt config used by generation")
    parser.add_argument("--max-model-len", type=int, required=True, help="Model context length")
    parser.add_argument("--tokens-to-generate", type=int, default=0, help="Completion token budget per request")
    parser.add_argument("--safety-margin", type=int, default=512, help="Reserved tokens below max model length")
    parser.add_argument(
        "--prompt-token-budget",
        type=int,
        default=None,
        help="Override derived prompt budget. Mostly useful for tests.",
    )
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path/name for accurate prompt token counting")
    parser.add_argument(
        "--endpoint-type",
        choices=["chat", "text"],
        default="chat",
        help="How generation will send prompts to the model",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Approximate characters per token when --tokenizer is omitted",
    )
    args = parser.parse_args()

    chunk_file(
        input_file=args.input,
        output_file=args.output,
        map_output=args.map_output,
        fields_to_translate=args.fields_to_translate,
        prompt_config=args.prompt_config,
        max_model_len=args.max_model_len,
        tokens_to_generate=args.tokens_to_generate,
        safety_margin=args.safety_margin,
        prompt_token_budget=args.prompt_token_budget,
        tokenizer=args.tokenizer,
        endpoint_type=args.endpoint_type,
        chars_per_token=args.chars_per_token,
    )


if __name__ == "__main__":
    sys.exit(main())
