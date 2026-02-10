# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

import json
import logging
import re

from format import FormatChecker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Format7Checker(FormatChecker):
    """Checker for format 7 (translate think content, preserve LaTeX)."""

    def check(self, input_text: str, output_text: str) -> bool:
        logger.debug("Starting format7 check...")
        try:
            input_json = self._extract_json_from_prompt(input_text)
            if input_json is None:
                return False
            output_json = self._extract_json_from_generation(output_text)
            if output_json is None:
                return False

            input_data = json.loads(input_json)
            output_data = json.loads(output_json)

            if not self._validate_structure(input_data, output_data):
                return False

            if not self._check_translation_constraints(input_data, output_data):
                return False

            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def _validate_structure(self, input_data: dict, output_data: dict) -> bool:
        required_fields = ["expected_answer", "input", "output"]
        for field in required_fields:
            if field not in input_data or field not in output_data:
                logger.debug(f"FAILED: Missing field '{field}'")
                return False
        if input_data["expected_answer"] != output_data["expected_answer"]:
            logger.debug("FAILED: expected_answer changed")
            return False
        return True

    def _check_translation_constraints(self, input_data: dict, output_data: dict) -> bool:
        # Think tags must be preserved in count, but content can differ (translated)
        if not self._check_think_tags(input_data["output"], output_data["output"]):
            return False

        # LaTeX preserved across fields
        for field in ["input", "output", "expected_answer"]:
            if not self._check_latex_preservation(input_data[field], output_data[field], field):
                return False

        # Require 'Answer:' prefix (outside think) in output
        if "Answer:" not in self._strip_think(output_data["output"]):
            logger.debug("FAILED: 'Answer:' prefix missing in output")
            return False

        return True

    def _check_think_tags(self, original: str, translated: str) -> bool:
        pattern = r"<think>(.*?)</think>"
        orig_thinks = re.findall(pattern, original, re.DOTALL)
        trans_thinks = re.findall(pattern, translated, re.DOTALL)
        if len(orig_thinks) != len(trans_thinks):
            logger.debug("FAILED: think tag count mismatch")
            return False
        # Require translated think content to be non-empty when original existed
        if orig_thinks:
            if any(not t.strip() for t in trans_thinks):
                logger.debug("FAILED: empty think translation")
                return False
        return True

    def _check_latex_preservation(self, original: str, translated: str, field_name: str) -> bool:
        latex_pattern = r"\\[a-zA-Z]+\{[^}]*\}|\$[^$]+\$|\\\(|\\\)|\\\[|\\\]"
        if re.findall(latex_pattern, original) != re.findall(latex_pattern, translated):
            logger.debug(f"FAILED: LaTeX differs in {field_name}")
            return False
        return True

    def _strip_think(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def _extract_json_from_prompt(self, prompt_text: str) -> str:
        try:
            json.loads(prompt_text.strip())
            return prompt_text.strip()
        except json.JSONDecodeError:
            pass
        pattern = r"[A-Za-z]+:\s*(\{.*\})"
        match = re.search(pattern, prompt_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return self._extract_first_json(prompt_text)

    def _extract_json_from_generation(self, generation_text: str) -> str:
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, generation_text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        return self._extract_first_json(generation_text)

    def _extract_first_json(self, text: str) -> str:
        brace_count = 0
        start_idx = None
        for i, char in enumerate(text):
            if char == "{":
                if start_idx is None:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    candidate = text[start_idx : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        start_idx = None
                        continue
        return None
