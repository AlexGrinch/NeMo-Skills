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


class FormatNanoChecker(FormatChecker):
    """Checker for format nano."""

    def check(self, input_text: str, output_text: str) -> bool:
        """Check if the given text follows the expected format."""
        logger.debug("Starting format_nano check...")

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

            # Check formatting constraints for each message entry
            input_messages = input_data.get("messages", [])
            output_messages = output_data.get("messages", [])

            logger.debug(f"Checking message entries (input: {len(input_messages)}, output: {len(output_messages)})...")
            if len(input_messages) != len(output_messages):
                logger.debug(
                    f"FAILED: Message count mismatch - input: {len(input_messages)}, output: {len(output_messages)}"
                )
                return False

            for i, (input_msg, output_msg) in enumerate(zip(input_messages, output_messages)):
                logger.debug(f"Checking message entry {i + 1}/{len(input_messages)}...")
                if not self._check_message_entry(input_msg, output_msg, i):
                    logger.debug(f"FAILED: Message entry {i + 1} validation failed")
                    return False
                logger.debug(f"✓ Message entry {i + 1} validation passed")

            logger.debug("✓ All format_nano checks passed!")
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"FAILED: Unexpected error during format check: {e}")
            return False

    def _validate_structure(self, input_data: dict, output_data: dict) -> bool:
        """Validate that output maintains the same structure as input."""
        logger.debug("  Checking required fields...")
        # Check that all required fields exist
        required_fields = ["problem", "is_MCQ", "has_tool", "expected_answer", "messages", "metadata"]
        for field in required_fields:
            if field not in input_data:
                logger.debug(f"  FAILED: Missing required field '{field}' in input data")
                return False
            if field not in output_data:
                logger.debug(f"  FAILED: Missing required field '{field}' in output data")
                return False
        logger.debug("  ✓ All required fields present")

        # Check that non-translatable fields are unchanged
        logger.debug("  Checking non-translatable field preservation...")
        non_translatable_fields = ["problem", "is_MCQ", "has_tool", "expected_answer", "metadata"]
        for field in non_translatable_fields:
            if input_data[field] != output_data[field]:
                logger.debug(f"  FAILED: Field '{field}' changed")
                return False
        logger.debug("  ✓ Non-translatable fields preserved")

        return True

    def _check_message_entry(self, input_msg: dict, output_msg: dict, index: int) -> bool:
        """Check formatting constraints for a single message entry."""
        logger.debug("    Checking translation field presence...")
        # Check that output has translation field
        if "translation" not in output_msg:
            logger.debug("    FAILED: Missing 'translation' field in output message")
            return False
        logger.debug("    ✓ Translation field present")

        # Check that role field is preserved
        logger.debug("    Checking role field preservation...")
        if "role" not in input_msg or "role" not in output_msg:
            logger.debug("    FAILED: Missing 'role' field")
            return False
        if input_msg["role"] != output_msg["role"]:
            logger.debug("    FAILED: Role field changed")
            return False
        logger.debug("    ✓ Role field preserved")

        # Check that content field is preserved
        logger.debug("    Checking content field preservation...")
        if "content" not in input_msg or "content" not in output_msg:
            logger.debug("    FAILED: Missing 'content' field")
            return False
        if input_msg["content"] != output_msg["content"]:
            logger.debug("    FAILED: Content field changed")
            return False
        logger.debug("    ✓ Content field preserved")

        # Check translation structure based on message index
        translation = output_msg["translation"]

        if index == 0:
            # First message (user): translation should be a string
            logger.debug("    Checking user message translation (should be string)...")
            if not isinstance(translation, str):
                logger.debug("    FAILED: Translation field is not a string for user message")
                return False
            if not translation.strip():
                logger.debug("    FAILED: Translation field is empty for user message")
                return False
            logger.debug("    ✓ User message translation is valid string")

        elif index == 1:
            # Second message (assistant): translation should be an object with content and reasoning_content
            logger.debug("    Checking assistant message translation (should be object)...")
            if not isinstance(translation, dict):
                logger.debug("    FAILED: Translation field is not a dictionary for assistant message")
                return False

            required_translation_fields = ["content", "reasoning_content"]
            for field in required_translation_fields:
                if field not in translation:
                    logger.debug(f"    FAILED: Missing '{field}' in translation field")
                    return False
                if not isinstance(translation[field], str):
                    logger.debug(f"    FAILED: Translation '{field}' is not a string")
                    return False
            logger.debug("    ✓ Assistant message translation structure is valid")

            # Check that reasoning_content field is preserved in original message
            logger.debug("    Checking reasoning_content field preservation...")
            if "reasoning_content" not in input_msg or "reasoning_content" not in output_msg:
                logger.debug("    FAILED: Missing 'reasoning_content' field")
                return False
            if input_msg["reasoning_content"] != output_msg["reasoning_content"]:
                logger.debug("    FAILED: reasoning_content field changed")
                return False
            logger.debug("    ✓ reasoning_content field preserved")

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
