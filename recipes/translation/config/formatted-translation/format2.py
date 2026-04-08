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


class Format2Checker(FormatChecker):
    """Checker for format 2."""

    def check(self, input_text: str, output_text: str) -> bool:
        """Check if the given text follows the expected format."""
        logger.debug("Starting format2 check...")

        try:
            input_json = self._extract_json_from_prompt(input_text)
            if input_json is None:
                logger.debug("FAILED: Could not extract JSON from input prompt")
                return False

            output_json = self._extract_json_from_generation(output_text)
            if output_json is None:
                logger.debug("FAILED: Could not extract JSON from output generation")
                return False

            try:
                input_data = json.loads(input_json)
            except json.JSONDecodeError as e:
                logger.debug(f"FAILED: Could not parse input JSON: {e}")
                return False

            try:
                output_data = json.loads(output_json)
            except json.JSONDecodeError as e:
                logger.debug(f"FAILED: Could not parse output JSON: {e}")
                return False

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

        for key in ["translated_value", "notes"]:
            if key not in translation:
                logger.debug(f"FAILED: Missing translation key '{key}'")
                return False
            if not isinstance(translation[key], str):
                logger.debug(f"FAILED: Translation key '{key}' is not a string")
                return False

        original_value = input_conv["value"]
        translated_value = translation["translated_value"]
        notes = translation["notes"]

        if not self._check_translation_constraints(original_value, translated_value, notes):
            logger.debug("FAILED: Translation constraints not satisfied")
            return False

        return True

    def _check_translation_constraints(self, original: str, translated_value: str, notes: str) -> bool:
        # Preserve think tag contents
        think_pattern = r"<think>(.*?)</think>"
        original_thinks = re.findall(think_pattern, original, re.DOTALL)
        translated_thinks = re.findall(think_pattern, translated_value, re.DOTALL)
        if original_thinks != translated_thinks:
            logger.debug("FAILED: Think tag content differs")
            return False

        # Preserve code blocks
        code_pattern = r"```.*?```"
        original_codes = re.findall(code_pattern, original, re.DOTALL)
        translated_codes = re.findall(code_pattern, translated_value, re.DOTALL)
        if original_codes != translated_codes:
            logger.debug("FAILED: Code block content differs")
            return False

        # Notes requirements:
        # - If there was translatable content, notes must be non-empty and not the default placeholder
        # - If nothing was translated, notes must exactly be "Nothing was translated"
        original_clean = self._remove_protected_content(original)
        notes_stripped = notes.strip()

        if original_clean:
            if not notes_stripped:
                logger.debug("FAILED: Notes empty despite translatable content")
                return False
            if notes_stripped.lower() == "nothing was translated":
                logger.debug("FAILED: Notes used placeholder despite having translatable content")
                return False
        else:
            if notes_stripped != "Nothing was translated":
                logger.debug("FAILED: Notes must be 'Nothing was translated' when no content was translated")
                return False

        return True

    def _remove_protected_content(self, text: str) -> str:
        # Remove protected segments entirely to detect if any translatable content remains
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        return text.strip()

    def _extract_json_from_prompt(self, prompt_text: str) -> str:
        """Extract JSON object from the input prompt text."""
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
                    json_candidate = prompt_text[start_idx : i + 1]
                    try:
                        json.loads(json_candidate)
                        return json_candidate
                    except json.JSONDecodeError:
                        start_idx = None
                        continue

        return None

    def _extract_json_from_generation(self, generation_text: str) -> str:
        """Extract JSON object from the model generation text."""
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, generation_text, re.DOTALL)
        if match:
            json_candidate = match.group(1).strip()
            try:
                json.loads(json_candidate)
                return json_candidate
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
                    json_candidate = generation_text[start_idx : i + 1]
                    try:
                        json.loads(json_candidate)
                        candidates.append(json_candidate)
                    except json.JSONDecodeError:
                        pass
                    start_idx = None

        if candidates:
            return max(candidates, key=len)

        return None
