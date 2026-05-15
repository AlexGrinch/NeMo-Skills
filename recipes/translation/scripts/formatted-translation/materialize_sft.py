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

"""Materialize OpenAI-style message JSONL into SFT message chunks.

Input records must contain a ``messages`` list. Output records contain a
``messages`` list where each message content is a rendered chat-template chunk
that can be consumed by downstream SFT tooling.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is available in normal pipeline containers.
    tqdm = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>\n"

tokenizer = None
skip_token_validation = True


def replace_json_args(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert JSON string tool-call arguments to dictionaries."""
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        for tool_call in message["tool_calls"]:
            function = tool_call.get("function", {})
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    return messages


def render_template(messages, enable_thinking=True, tools=None):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )


def parse_rendered_blocks(rendered_template: str) -> list[dict[str, str]]:
    """Parse top-level <|im_start|>role blocks from a rendered template."""
    blocks = []
    cursor = 0

    while cursor < len(rendered_template):
        if not rendered_template.startswith(_IM_START, cursor):
            snippet = rendered_template[cursor : cursor + 80]
            raise ValueError(f"Unexpected content outside top-level block at offset {cursor}: {snippet!r}")

        role_start = cursor + len(_IM_START)
        role_end = rendered_template.find("\n", role_start)
        if role_end == -1:
            raise ValueError("Malformed rendered template: missing role header newline")

        block_end = rendered_template.find(_IM_END, role_end + 1)
        if block_end == -1:
            raise ValueError("Malformed rendered template: missing <|im_end|> terminator")
        block_end += len(_IM_END)

        blocks.append({"role": rendered_template[role_start:role_end], "content": rendered_template[cursor:block_end]})
        cursor = block_end

    return blocks


def logical_chunks_from_rendered(messages, rendered_blocks):
    """Map rendered blocks back to logical roles, including grouped tool replies."""
    if not rendered_blocks:
        return []

    chunks = []
    block_idx = 0
    msg_idx = 0

    if rendered_blocks[0]["role"] == "system":
        chunks.append({"role": "system", "content": rendered_blocks[0]["content"]})
        block_idx = 1
        if messages and messages[0].get("role") == "system":
            msg_idx = 1

    while msg_idx < len(messages):
        message = messages[msg_idx]

        if message.get("role") == "tool":
            if block_idx >= len(rendered_blocks):
                raise ValueError("Missing rendered block for tool message group")
            if rendered_blocks[block_idx]["role"] != "user":
                raise ValueError("Expected synthetic user block for tool response group")
            while msg_idx < len(messages) and messages[msg_idx].get("role") == "tool":
                msg_idx += 1
            chunks.append({"role": "tool", "content": rendered_blocks[block_idx]["content"]})
            block_idx += 1
            continue

        if block_idx >= len(rendered_blocks):
            raise ValueError(f"Missing rendered block for message role {message.get('role')}")
        chunks.append(
            {
                "role": message.get("role", rendered_blocks[block_idx]["role"]),
                "content": rendered_blocks[block_idx]["content"],
            }
        )
        block_idx += 1
        msg_idx += 1

    if block_idx != len(rendered_blocks):
        raise ValueError("Unused rendered blocks remain after reconstructing logical chunks")

    return chunks


def split_template_into_messages(messages, start_from_last_user=True, enable_thinking=True, tools=None):
    rendered_blocks = parse_rendered_blocks(render_template(messages, enable_thinking=enable_thinking, tools=tools))
    logical_chunks = logical_chunks_from_rendered(messages, rendered_blocks)

    if not start_from_last_user:
        return logical_chunks

    user_indices = [i for i, chunk in enumerate(logical_chunks) if chunk["role"] == "user"]
    if not user_indices:
        raise ValueError("No user chunk found")

    last_user_chunk_idx = user_indices[-1]
    if logical_chunks and logical_chunks[0]["role"] == "system":
        return [
            logical_chunks[0],
            {
                "role": "user",
                "content": "".join(chunk["content"] for chunk in logical_chunks[1 : last_user_chunk_idx + 1]),
            },
            *logical_chunks[last_user_chunk_idx + 1 :],
        ]

    return [
        {
            "role": "user",
            "content": "".join(chunk["content"] for chunk in logical_chunks[: last_user_chunk_idx + 1]),
        },
        *logical_chunks[last_user_chunk_idx + 1 :],
    ]


def create_masked_messages(messages, tools=None):
    """Create one or more rendered message-chunk groups for an SFT sample."""
    has_thinking = any(msg.get("reasoning_content") for msg in messages if isinstance(msg, dict))

    if not has_thinking:
        chunks = split_template_into_messages(messages, start_from_last_user=False, enable_thinking=False, tools=tools)
        return [(chunks, messages)]

    user_idxs = [i for i, msg in enumerate(messages) if msg.get("role") == "user"]
    result = []
    for i in range(len(user_idxs)):
        messages_i = messages if i == len(user_idxs) - 1 else messages[: user_idxs[i + 1]]
        chunks = split_template_into_messages(messages_i, start_from_last_user=True, enable_thinking=True, tools=tools)
        result.append((chunks, messages_i))
    return result


def encode_plain_text(text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return list(encoded["input_ids"])


def validate_processed_chunks_tokenization(processed_chunks) -> tuple[bool, int]:
    concatenated = "".join(chunk["content"] for chunk in processed_chunks)
    conversation_tokens = encode_plain_text(concatenated)
    part_tokens = []
    for chunk in processed_chunks:
        part_tokens.extend(encode_plain_text(chunk["content"]))
    return conversation_tokens == part_tokens, len(conversation_tokens)


def has_tool_calls(messages) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return True
        for key in ("content", "reasoning_content"):
            value = message.get(key)
            if isinstance(value, str) and "<tool_call>" in value:
                return True
    return False


def process_line(args):
    line_num, line, extra_info = args
    try:
        data = json.loads(line.strip())
        messages = data.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return None, line_num, "Missing non-empty messages list"

        messages = replace_json_args([dict(message) for message in messages])
        conversation_tools = data.get("tools")
        if has_tool_calls(messages) and conversation_tools in (None, [], {}):
            return None, line_num, "Tool calls present but missing top-level tools key"

        masked_messages = create_masked_messages(messages, tools=conversation_tools)
        if not masked_messages:
            return None, line_num, "No masked messages created"

        results = []
        token_counts = []
        for chunks, _ in masked_messages:
            for idx, chunk in enumerate(chunks):
                if idx > 0 and chunk["role"] == "assistant" and chunks[idx - 1]["role"] == "assistant":
                    return None, line_num, "Multiple assistant turns detected"
                if chunk.get("content", "").endswith("assistant\n<think><"):
                    return None, line_num, "Cutoff thinking detected"

            if skip_token_validation:
                num_tokens = len(encode_plain_text("".join(chunk["content"] for chunk in chunks)))
            else:
                tokens_match, num_tokens = validate_processed_chunks_tokenization(chunks)
                if not tokens_match:
                    return None, line_num, "Chunk tokenization mismatch"

            if extra_info:
                output_data = {"messages": chunks, **{k: v for k, v in data.items() if k != "messages"}}
            else:
                output_data = {"messages": chunks}
            results.append(json.dumps(output_data, ensure_ascii=False))
            token_counts.append(num_tokens)

        return (results, token_counts), line_num, None
    except Exception as e:
        return None, line_num, str(e)


def process_batch(batch):
    return [process_line(item) for item in batch]


def read_completed_positions(async_path: str) -> set[int]:
    completed = set()
    if not Path(async_path).exists():
        return completed
    with open(async_path, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["_async_position"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                break
    return completed


def restore_async_order(async_path: str, output_file: str, tokens_file: str):
    with open(async_path, encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin if line.strip()]

    records.sort(key=lambda record: record["_async_position"])
    with open(output_file, "w", encoding="utf-8") as output_f, open(tokens_file, "w", encoding="utf-8") as tokens_f:
        for record in records:
            for output_line, token_count in zip(record["outputs"], record["token_counts"]):
                output_f.write(output_line + "\n")
                tokens_f.write(f"{token_count}\n")

    Path(async_path).unlink()


def batched_task_iter(input_file: str, batch_size: int, extra_info: bool, completed_positions: set[int]):
    batch = []
    with open(input_file, encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue
            if line_num in completed_positions:
                continue
            batch.append((line_num, line, extra_info))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def count_nonempty_lines(input_file: str) -> int:
    with open(input_file, "rb") as fin:
        return sum(1 for line in fin if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize OpenAI-style JSONL messages for SFT.")
    parser.add_argument("--input_file", required=True, help="Input JSONL file with a messages field")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    parser.add_argument("-m", "--model", required=True, help="HuggingFace model name or tokenizer path")
    parser.add_argument("--chat_template", default=None, help="Optional Jinja chat template file")
    parser.add_argument(
        "--tokens_file",
        default=None,
        help="Optional token-count output file. Defaults to <output_file>.tokens.jsonl",
    )
    parser.add_argument("--extra_info", action="store_true", help="Preserve non-message fields in output records")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoTokenizer"
    )
    parser.add_argument("-j", "--workers", type=int, default=mp.cpu_count(), help="Number of worker processes")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Input lines per worker batch")
    parser.add_argument(
        "--skip-token-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip per-chunk tokenization validation for speed (default: true)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_file = args.tokens_file or f"{args.output_file}.tokens.jsonl"
    Path(tokens_file).parent.mkdir(parents=True, exist_ok=True)

    total_lines = count_nonempty_lines(args.input_file)
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Tokens: {tokens_file}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total non-empty lines: {total_lines:,}")

    global tokenizer, skip_token_validation
    skip_token_validation = args.skip_token_validation
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if args.chat_template:
        with open(args.chat_template, encoding="utf-8") as fin:
            tokenizer.chat_template = fin.read()
    print("Tokenizer loaded.")

    start_time = time.time()
    passed_count = 0
    failed_count = 0

    async_path = args.output_file + "-async"
    completed_positions = read_completed_positions(async_path)
    if completed_positions:
        print(f"Resuming: {len(completed_positions):,} input records already completed")

    remaining_lines = total_lines - len(completed_positions)
    task_iter = batched_task_iter(args.input_file, args.batch_size, args.extra_info, completed_positions)
    total_batches = (remaining_lines + args.batch_size - 1) // args.batch_size

    with (
        open(async_path, "a", encoding="utf-8", buffering=1) as async_f,
        Pool(processes=args.workers) as pool,
    ):
        mapped = pool.imap_unordered(process_batch, task_iter, chunksize=1)
        if tqdm is not None:
            mapped = tqdm(mapped, total=total_batches, desc="Processing", unit="batch")

        for batch_results in mapped:
            for result, line_num, error in batch_results:
                if result is None:
                    failed_count += 1
                    if error and failed_count <= 10:
                        print(f"Line {line_num}: {error}", file=sys.stderr)
                    continue
                output_lines, token_counts = result
                async_f.write(
                    json.dumps(
                        {
                            "outputs": output_lines,
                            "token_counts": token_counts,
                            "_async_position": line_num,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                passed_count += 1

    restore_async_order(async_path, args.output_file, tokens_file)

    elapsed = time.time() - start_time
    total_passed = len(completed_positions) + passed_count
    success_rate = total_passed / total_lines * 100 if total_lines else 0.0
    process_rate = remaining_lines / elapsed if elapsed > 0 else 0.0

    print("Processing complete!")
    print(f"Total time: {elapsed:.1f}s")
    if completed_positions:
        print(f"Resumed from checkpoint: {len(completed_positions):,} previously completed")
    print(f"Processed: {passed_count:,} passed, {failed_count:,} failed")
    print(f"Total passed: {total_passed:,}/{total_lines:,} ({success_rate:.1f}%)")
    print(f"Processing rate: {process_rate:.1f} lines/sec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
