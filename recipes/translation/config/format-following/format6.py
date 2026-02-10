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

import json
import logging
import re

from format import FormatChecker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Format6Checker(FormatChecker):
    """Checker for format 6 (think tags replaced by Reasoning:, Solution label, no blank lines)."""

    def check(self, input_text: str, output_text: str) -> bool:
        logger.debug("Starting format6 check...")
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

        # expected_answer must remain identical
        if input_data["expected_answer"] != output_data["expected_answer"]:
            logger.debug("FAILED: expected_answer changed")
            return False

        return True

    def _check_translation_constraints(self, input_data: dict, output_data: dict) -> bool:
        original_output = input_data.get("output", "")
        translated_output = output_data.get("output", "")

        # Think tags must be removed and replaced by "Reasoning: <original content>"
        if "<think>" in translated_output or "</think>" in translated_output:
            logger.debug("FAILED: think tags still present in translated output")
            return False

        if not self._check_reasoning_block(original_output, translated_output):
            return False

        # Must contain "Solution:" label after reasoning
        if "Solution:" not in translated_output:
            logger.debug("FAILED: 'Solution:' label missing")
            return False

        if translated_output.find("Solution:") < translated_output.find("Reasoning:"):
            logger.debug("FAILED: 'Solution:' appears before reasoning")
            return False

        # No blank lines allowed (no consecutive newlines)
        if "\n\n" in translated_output:
            logger.debug("FAILED: blank lines found in output")
            return False

        # LaTeX must be preserved across all fields
        for field in ["input", "output", "expected_answer"]:
            if not self._check_latex_preservation(input_data[field], output_data[field], field):
                return False

        return True

    def _check_reasoning_block(self, original_output: str, translated_output: str) -> bool:
        think_pattern = r"<think>(.*?)</think>"
        original_thinks = re.findall(think_pattern, original_output, re.DOTALL)

        if original_thinks:
            # Require the first reasoning block to appear and match exactly
            reason_pattern = r"Reasoning:\s*(.*?)(?:\nSolution:|\Z)"
            match = re.search(reason_pattern, translated_output, re.DOTALL)
            if not match:
                logger.debug("FAILED: Reasoning block not found")
                return False
            reasoning_text = match.group(1).strip()
            if reasoning_text != original_thinks[0].strip():
                logger.debug("FAILED: Reasoning text differs from original think content")
                return False
        else:
            # No think content originally; ensure no reasoning label added
            if "Reasoning:" in translated_output:
                logger.debug("FAILED: Reasoning label present but no think content in original")
                return False

        return True

    def _check_latex_preservation(self, original: str, translated: str, field_name: str) -> bool:
        latex_pattern = r"\\[a-zA-Z]+\{[^}]*\}|\$[^$]+\$|\\\(|\\\)|\\\[|\\\]"
        orig_matches = re.findall(latex_pattern, original)
        trans_matches = re.findall(latex_pattern, translated)
        if orig_matches != trans_matches:
            logger.debug(f"FAILED: LaTeX differs in {field_name}")
            return False
        return True

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
