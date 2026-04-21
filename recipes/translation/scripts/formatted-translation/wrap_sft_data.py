#!/usr/bin/env python3
"""
Script to wrap JSONL data into a new format for translation tasks.

Each line in the input JSONL file will be wrapped into a new object with:
- source_lang: "English" (fixed)
- target_lang: specified via command line argument
- src: the original JSONL object serialized as a string (fields starting with
  "_translation_" are excluded from src and carried over as top-level fields instead)
"""

import argparse
import json
import sys
from pathlib import Path


def wrap_jsonl_data(input_file: str, output_file: str, target_lang: str) -> None:
    """
    Wrap JSONL data into the specified format.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        target_lang: Target language for translation
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_lines = 0

    try:
        with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)
                    continue

                # Fields starting with "_translation_" (e.g. _translation_src_id)
                # are pipeline metadata: carry them through as top-level fields
                # rather than embedding them in src so the model never sees them.
                payload = {k: v for k, v in record.items() if not k.startswith("_translation_")}
                metadata = {k: v for k, v in record.items() if k.startswith("_translation_")}

                wrapped_obj = {
                    "source_lang": "English",
                    "target_lang": target_lang,
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
    print(f"Target language: {target_lang}")


def main():
    parser = argparse.ArgumentParser(
        description="Wrap JSONL data into translation format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wrap_sft_data.py --input input.jsonl --output output.jsonl --target-lang "Spanish"
  python wrap_sft_data.py --input data/train.jsonl --output data/wrapped_train.jsonl -t "French"
        """,
    )

    parser.add_argument("--input", required=True, help="Path to input JSONL file")

    parser.add_argument("--output", required=True, help="Path to output JSONL file")

    parser.add_argument(
        "-t",
        "--target-lang",
        required=True,
        help='Target language for translation (e.g., "Spanish", "French", "German")',
    )

    args = parser.parse_args()

    wrap_jsonl_data(args.input, args.output, args.target_lang)


if __name__ == "__main__":
    main()
