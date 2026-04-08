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


class Format1Checker(FormatChecker):
    """Checker for format 1."""

    def check(self, input_text: str, output_text: str) -> bool:
        """Check if the given text follows the expected format."""
        logger.debug("Starting format1 check...")

        try:
            # Extract JSON from input prompt and output generation
            logger.debug("Extracting JSON from input prompt...")
            input_json = self._extract_json_from_prompt(input_text)
            if input_json is None:
                logger.debug("FAILED: Could not extract JSON from input prompt")
                return False
            logger.debug("✓ Successfully extracted JSON from input prompt")

            logger.debug("Extracting JSON from output generation...")
            output_json = self._extract_json_from_generation(output_text)
            if output_json is None:
                logger.debug("FAILED: Could not extract JSON from output generation")
                return False
            logger.debug("✓ Successfully extracted JSON from output generation")

            # Parse extracted JSON strings
            logger.debug("Parsing extracted JSON strings...")
            try:
                input_data = json.loads(input_json)
                logger.debug("✓ Successfully parsed input JSON")
            except json.JSONDecodeError as e:
                logger.debug(f"FAILED: Could not parse input JSON: {e}")
                return False

            try:
                output_data = json.loads(output_json)
                logger.debug("✓ Successfully parsed output JSON")
            except json.JSONDecodeError as e:
                logger.debug(f"FAILED: Could not parse output JSON: {e}")
                return False

            # Check that output has the same structure as input
            logger.debug("Validating JSON structure...")
            if not self._validate_structure(input_data, output_data):
                logger.debug("FAILED: Structure validation failed")
                return False
            logger.debug("✓ Structure validation passed")

            # Check formatting constraints for each conversation entry
            input_conversations = input_data.get("conversations", [])
            output_conversations = output_data.get("conversations", [])

            logger.debug(
                f"Checking conversation entries (input: {len(input_conversations)}, output: {len(output_conversations)})..."
            )
            if len(input_conversations) != len(output_conversations):
                logger.debug(
                    f"FAILED: Conversation count mismatch - input: {len(input_conversations)}, output: {len(output_conversations)}"
                )
                return False

            for i, (input_conv, output_conv) in enumerate(zip(input_conversations, output_conversations)):
                logger.debug(f"Checking conversation entry {i + 1}/{len(input_conversations)}...")
                if not self._check_conversation_entry(input_conv, output_conv):
                    logger.debug(f"FAILED: Conversation entry {i + 1} validation failed")
                    return False
                logger.debug(f"✓ Conversation entry {i + 1} validation passed")

            logger.debug("✓ All format1 checks passed!")
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"FAILED: Unexpected error during format check: {e}")
            return False

    def _validate_structure(self, input_data: dict, output_data: dict) -> bool:
        """Validate that output maintains the same structure as input."""
        logger.debug("  Checking required fields...")
        # Check that all required fields exist
        required_fields = ["conversations", "system", "mask"]
        for field in required_fields:
            if field not in input_data:
                logger.debug(f"  FAILED: Missing required field '{field}' in input data")
                return False
            if field not in output_data:
                logger.debug(f"  FAILED: Missing required field '{field}' in output data")
                return False
        logger.debug("  ✓ All required fields present")

        # Check that system and mask fields are unchanged
        logger.debug("  Checking system field preservation...")
        if input_data["system"] != output_data["system"]:
            logger.debug("  FAILED: System field changed")
            return False
        logger.debug("  ✓ System field preserved")

        logger.debug("  Checking mask field preservation...")
        if input_data["mask"] != output_data["mask"]:
            logger.debug("  FAILED: Mask field changed")
            return False
        logger.debug("  ✓ Mask field preserved")

        return True

    def _check_conversation_entry(self, input_conv: dict, output_conv: dict) -> bool:
        """Check formatting constraints for a single conversation entry."""
        logger.debug("    Checking translation field presence...")
        # Check that output has translation field
        if "translation" not in output_conv:
            logger.debug("    FAILED: Missing 'translation' field in output conversation")
            return False
        logger.debug("    ✓ Translation field present")

        # Check that translation field has the required subfields
        translation = output_conv["translation"]
        if not isinstance(translation, dict):
            logger.debug("    FAILED: Translation field is not a dictionary")
            return False

        required_translation_fields = ["translation_trace", "translation_answer"]
        for field in required_translation_fields:
            if field not in translation:
                logger.debug(f"    FAILED: Missing '{field}' in translation field")
                return False
        logger.debug("    ✓ Translation subfields present")

        # Check that all original fields are preserved
        logger.debug("    Checking field preservation...")
        required_fields = ["from", "value", "canonical_form", "label"]
        for field in required_fields:
            if field not in input_conv:
                logger.debug(f"    FAILED: Missing required field '{field}' in input conversation")
                return False
            if field not in output_conv:
                logger.debug(f"    FAILED: Missing required field '{field}' in output conversation")
                return False
            if input_conv[field] != output_conv[field]:
                logger.debug(f"    FAILED: Field '{field}' changed")
                return False
        logger.debug("    ✓ All required fields preserved")

        # Check formatting constraints on the translation
        logger.debug("    Checking translation constraints...")
        input_value = input_conv["value"]
        translation_trace = translation["translation_trace"]
        translation_answer = translation["translation_answer"]

        if self._check_translation_constraints(input_value, translation_trace, translation_answer):
            logger.debug("    ✓ Translation constraints satisfied")
            return True
        else:
            logger.debug("    FAILED: Translation constraints not satisfied")
            return False

    def _check_translation_constraints(self, original: str, translation_trace: str, translation_answer: str) -> bool:
        """Check that translation follows the formatting constraints."""
        logger.debug("      Checking think tag handling...")

        # Extract think tags content from original
        think_pattern = r"<think>(.*?)</think>"
        original_thinks = re.findall(think_pattern, original, re.DOTALL)

        # Check translation_trace - should contain translated think tag content
        trace_thinks = re.findall(think_pattern, translation_trace, re.DOTALL)

        if len(original_thinks) != len(trace_thinks):
            logger.debug("      FAILED: Think tag count mismatch")
            return False

        # If there are think tags, translation_trace should contain them
        if original_thinks:
            if not translation_trace.strip():
                logger.debug("      FAILED: Translation trace is empty but original has think tags")
                return False
            logger.debug(f"      ✓ Think tags handled in translation_trace ({len(original_thinks)} found)")
        else:
            # If no think tags, translation_trace should be empty
            if translation_trace.strip():
                logger.debug("      FAILED: Translation trace should be empty when no think tags present")
                return False
            logger.debug("      ✓ No think tags found, translation_trace is empty")

        logger.debug("      Checking code block preservation...")
        # Extract code blocks - should be preserved in translation_answer
        code_pattern = r"```(.*?)```"
        original_codes = re.findall(code_pattern, original, re.DOTALL)
        answer_codes = re.findall(code_pattern, translation_answer, re.DOTALL)

        # Code blocks should be identical in translation_answer
        if original_codes != answer_codes:
            logger.debug(
                f"      FAILED: Code block content differs - original: {original_codes}, answer: {answer_codes}"
            )
            return False
        logger.debug(f"      ✓ Code blocks preserved in translation_answer ({len(original_codes)} found)")

        # Check that translation_answer contains the non-think content
        logger.debug("      Checking translation_answer content...")
        original_without_think = re.sub(r"<think>.*?</think>", "", original, flags=re.DOTALL).strip()

        if original_without_think and not translation_answer.strip():
            logger.debug("      FAILED: Translation answer is empty but original has non-think content")
            return False

        if not original_without_think and translation_answer.strip():
            logger.debug("      FAILED: Translation answer should be empty when original has no non-think content")
            return False

        logger.debug("      ✓ Translation answer content validated")

        return True

    def _extract_json_from_prompt(self, prompt_text: str) -> str:
        """Extract JSON object from the input prompt text."""
        logger.debug("  Trying to extract JSON from prompt...")

        # First, check if the entire input is already valid JSON
        logger.debug("  Checking if input is already pure JSON...")
        try:
            json.loads(prompt_text.strip())
            logger.debug("  Input is already valid JSON")
            return prompt_text.strip()
        except json.JSONDecodeError:
            logger.debug("  Input is not pure JSON, trying extraction methods...")

        # Look for the JSON object in the prompt, typically after source language label
        # The format should be something like "English: {json_object}"

        # Try to find JSON object after language label (e.g., "English:", "Chinese:", etc.)
        # Look for pattern: language_name: {json_content}
        logger.debug("  Trying language pattern extraction...")
        pattern = r"[A-Za-z]+:\s*(\{.*\})"
        match = re.search(pattern, prompt_text, re.DOTALL)
        if match:
            json_candidate = match.group(1).strip()
            logger.debug(f"  Found JSON candidate via language pattern: {json_candidate[:100]}...")
            return json_candidate
        logger.debug("  Language pattern extraction failed")

        # Fallback: look for any JSON object in the text
        logger.debug("  Trying fallback JSON extraction...")
        # Find the first complete JSON object
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
                    # Found complete JSON object
                    json_str = prompt_text[start_idx : i + 1]
                    # Validate it's actually JSON
                    try:
                        json.loads(json_str)
                        logger.debug(f"  Found valid JSON via fallback: {json_str[:100]}...")
                        return json_str
                    except json.JSONDecodeError:
                        # Continue searching
                        start_idx = None
                        continue

        logger.debug("  Could not extract any valid JSON from prompt")
        return None

    def _extract_json_from_generation(self, generation_text: str) -> str:
        """Extract JSON object from the model generation text."""
        logger.debug("  Trying to extract JSON from generation...")
        # The model generation might contain explanations before/after the JSON
        # Look for JSON object patterns

        # First, try to find JSON objects enclosed in code blocks
        logger.debug("  Trying code block extraction...")
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, generation_text, re.DOTALL)
        if match:
            json_candidate = match.group(1).strip()
            logger.debug(f"  Found JSON candidate in code block: {json_candidate[:100]}...")
            try:
                json.loads(json_candidate)
                logger.debug("  Code block JSON is valid")
                return json_candidate
            except json.JSONDecodeError as e:
                logger.debug(f"  Code block JSON is invalid: {e}")
        else:
            logger.debug("  No code block JSON found")

        # Look for standalone JSON objects
        logger.debug("  Trying standalone JSON extraction...")
        # Find all potential JSON objects and validate them
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
                    # Found complete JSON object
                    json_str = generation_text[start_idx : i + 1]
                    try:
                        json.loads(json_str)
                        candidates.append(json_str)
                        logger.debug(f"  Found valid JSON candidate: {json_str[:100]}...")
                    except json.JSONDecodeError:
                        pass
                    start_idx = None

        # Return the largest valid JSON object (most likely to be the complete response)
        if candidates:
            best_candidate = max(candidates, key=len)
            logger.debug(f"  Selected best candidate ({len(best_candidate)} chars): {best_candidate[:100]}...")
            return best_candidate

        logger.debug("  Could not extract any valid JSON from generation")
        return None
