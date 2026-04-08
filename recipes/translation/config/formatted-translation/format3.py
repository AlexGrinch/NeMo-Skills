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

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Format3Checker(FormatChecker):
    """Checker for format 3."""

    def check(self, input_text: str, output_text: str) -> bool:
        logger.debug("Starting format3 check...")

        try:
            input_json = self._extract_json_from_prompt(input_text)
            if input_json is None:
                logger.debug("FAILED: Could not extract JSON from input prompt")
                return False

            output_json = self._extract_json_from_generation(output_text)
            if output_json is None:
                logger.debug("FAILED: Could not extract JSON from output generation")
                return False

            input_data = json.loads(input_json)
            output_data = json.loads(output_json)

            if not self._validate_structure(input_data, output_data):
                logger.debug("FAILED: Structure validation failed")
                return False

            input_conversations = input_data.get("conversations", [])
            output_conversations = output_data.get("conversations", [])
            if len(input_conversations) != len(output_conversations):
                logger.debug("FAILED: Conversation count mismatch")
                return False

            for idx, (input_conv, output_conv) in enumerate(zip(input_conversations, output_conversations)):
                if not self._check_conversation_entry(input_conv, output_conv):
                    logger.debug(f"FAILED: Conversation entry {idx + 1} validation failed")
                    return False

            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"FAILED: Unexpected error during format check: {e}")
            return False

    def _validate_structure(self, input_data: dict, output_data: dict) -> bool:
        required_fields = ["conversations", "system", "mask"]
        for field in required_fields:
            if field not in input_data or field not in output_data:
                logger.debug(f"FAILED: Missing required field '{field}'")
                return False

        if input_data["system"] != output_data["system"]:
            logger.debug("FAILED: System field changed")
            return False

        if input_data["mask"] != output_data["mask"]:
            logger.debug("FAILED: Mask field changed")
            return False

        return True

    def _check_conversation_entry(self, input_conv: dict, output_conv: dict) -> bool:
        required_fields = ["from", "value", "canonical_form", "label"]
        for field in required_fields:
            if field not in input_conv or field not in output_conv:
                logger.debug(f"FAILED: Missing required field '{field}' in conversation")
                return False
            if input_conv[field] != output_conv[field]:
                logger.debug(f"FAILED: Field '{field}' changed")
                return False

        if "translation" not in output_conv:
            logger.debug("FAILED: Missing translation field")
            return False

        translation = output_conv["translation"]
        if not isinstance(translation, dict):
            logger.debug("FAILED: translation is not a dict")
            return False

        for key in ["think_translation", "non_think_translation", "combined"]:
            if key not in translation:
                logger.debug(f"FAILED: Missing translation key '{key}'")
                return False
            if not isinstance(translation[key], str):
                logger.debug(f"FAILED: Translation key '{key}' is not a string")
                return False

        if not self._check_translation_constraints(
            input_conv["value"],
            translation["think_translation"],
            translation["non_think_translation"],
            translation["combined"],
        ):
            logger.debug("FAILED: Translation constraints not satisfied")
            return False

        return True

    def _check_translation_constraints(
        self, original: str, think_translation: str, non_think_translation: str, combined: str
    ) -> bool:
        think_pattern = r"<think>(.*?)</think>"
        code_pattern = r"```.*?```"

        original_thinks = re.findall(think_pattern, original, re.DOTALL)
        translated_thinks = re.findall(think_pattern, think_translation, re.DOTALL)
        if len(original_thinks) != len(translated_thinks):
            logger.debug("FAILED: Think tag count mismatch")
            return False

        if not original_thinks and think_translation.strip():
            logger.debug("FAILED: Think translation provided when no think tags present")
            return False

        original_codes = re.findall(code_pattern, original, re.DOTALL)
        non_think_codes = re.findall(code_pattern, non_think_translation, re.DOTALL)
        if original_codes != non_think_codes:
            logger.debug("FAILED: Code blocks not preserved in non_think_translation")
            return False

        combined_codes = re.findall(code_pattern, combined, re.DOTALL)
        if original_codes != combined_codes:
            logger.debug("FAILED: Code blocks not preserved in combined output")
            return False

        combined_thinks = re.findall(think_pattern, combined, re.DOTALL)
        if len(combined_thinks) != len(original_thinks):
            logger.debug("FAILED: Think tag count mismatch in combined")
            return False

        return True

    def _extract_json_from_prompt(self, prompt_text: str) -> str:
        logger.debug("  Trying to extract JSON from prompt...")
        try:
            json.loads(prompt_text.strip())
            return prompt_text.strip()
        except json.JSONDecodeError:
            pass

        pattern = r"[A-Za-z]+:\s*(\{.*\})"
        match = re.search(pattern, prompt_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        brace_count = 0
        start_idx = None
        for i, char in enumerate(prompt_text):
            if char == "{":
                if start_idx is None:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    candidate = prompt_text[start_idx : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        start_idx = None
                        continue
        return None

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

        brace_count = 0
        start_idx = None
        candidates = []
        for i, char in enumerate(generation_text):
            if char == "{":
                if start_idx is None:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    candidate = generation_text[start_idx : i + 1]
                    try:
                        json.loads(candidate)
                        candidates.append(candidate)
                    except json.JSONDecodeError:
                        pass
                    start_idx = None

        if candidates:
            return max(candidates, key=len)

        return None
