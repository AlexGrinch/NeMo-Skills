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


class Format13Checker(FormatChecker):
    """Checker for format 13."""

    def check(self, input_text: str, output_text: str) -> bool:
        """Check if the given text follows the expected format."""
        logger.debug("Starting format13 check...")

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

            # Check formatting constraints for the translatable fields
            logger.debug("Checking translation constraints...")
            if not self._check_translation_constraints(input_data, output_data):
                logger.debug("FAILED: Translation constraints not satisfied")
                return False
            logger.debug("✓ Translation constraints satisfied")

            logger.debug("✓ All format13 checks passed!")
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"FAILED: Unexpected error during format check: {e}")
            return False

    def _validate_structure(self, input_data: dict, output_data: dict) -> bool:
        """Validate that output maintains the same structure as input."""
        logger.debug("  Checking required fields...")

        # Check that all fields from input exist in output
        for field in input_data.keys():
            if field not in output_data:
                logger.debug(f"  FAILED: Missing field '{field}' in output data")
                return False
        logger.debug("  ✓ All input fields present in output")

        # Check that non-translatable fields are unchanged
        non_translatable_fields = ["majority_res", "topic", "subtopic", "difficulty"]
        for field in non_translatable_fields:
            if field in input_data:
                if input_data[field] != output_data[field]:
                    logger.debug(
                        f"  FAILED: Field '{field}' changed - input: '{input_data[field]}', output: '{output_data[field]}'"
                    )
                    return False
                logger.debug(f"  ✓ Field '{field}' preserved")

        return True

    def _check_translation_constraints(self, input_data: dict, output_data: dict) -> bool:
        """Check that translation follows the formatting constraints."""
        translatable_fields = ["problem", "generation", "expected_answer"]

        for field in translatable_fields:
            if field not in input_data:
                logger.debug(f"    Field '{field}' not found in input, skipping...")
                continue

            if field not in output_data:
                logger.debug(f"    FAILED: Field '{field}' missing in output")
                return False

            logger.debug(f"    Checking field '{field}'...")
            input_value = input_data[field]
            output_value = output_data[field]

            # Check that output is a list with exactly 2 elements
            if not isinstance(output_value, list):
                logger.debug(f"    FAILED: Field '{field}' is not a list in output")
                return False

            if len(output_value) != 2:
                logger.debug(f"    FAILED: Field '{field}' list doesn't have exactly 2 elements")
                return False

            # Check that first element matches the original input
            if output_value[0] != input_value:
                logger.debug(f"    FAILED: Field '{field}' first element doesn't match input")
                return False

            # Check that LaTeX formatting is preserved in second element (translation)
            if not self._check_latex_preservation(input_value, output_value[1], field):
                logger.debug(f"    FAILED: LaTeX preservation failed for field '{field}'")
                return False

            # Check that whitespace and newlines are preserved in second element
            if not self._check_whitespace_preservation(input_value, output_value[1], field):
                logger.debug(f"    FAILED: Whitespace preservation failed for field '{field}'")
                return False

            logger.debug(f"    ✓ Field '{field}' validation passed")

        return True

    def _check_latex_preservation(self, original: str, translation: str, field_name: str) -> bool:
        """Check that LaTeX formatting is preserved."""
        logger.debug(f"      Checking LaTeX preservation for {field_name}...")

        # Extract LaTeX patterns like \boxed{}, \text{}, etc.
        latex_patterns = [
            r"\\boxed\{[^}]*\}",
            r"\\text\{[^}]*\}",
            r"\\[a-zA-Z]+\{[^}]*\}",  # General LaTeX commands
        ]

        for pattern in latex_patterns:
            original_matches = re.findall(pattern, original)
            translation_matches = re.findall(pattern, translation)

            # LaTeX commands should be preserved exactly
            if original_matches != translation_matches:
                logger.debug(
                    f"      FAILED: LaTeX pattern '{pattern}' differs - original: {original_matches}, translation: {translation_matches}"
                )
                return False

        logger.debug(f"      ✓ LaTeX formatting preserved for {field_name}")
        return True

    def _check_whitespace_preservation(self, original: str, translation: str, field_name: str) -> bool:
        """Check that whitespace and newline structure is preserved."""
        logger.debug(f"      Checking whitespace preservation for {field_name}...")

        # Count newlines
        original_newlines = original.count("\n")
        translation_newlines = translation.count("\n")

        if original_newlines != translation_newlines:
            logger.debug(
                f"      FAILED: Newline count differs - original: {original_newlines}, translation: {translation_newlines}"
            )
            return False

        # Check that leading/trailing whitespace patterns are similar
        original_lines = original.split("\n")
        translation_lines = translation.split("\n")

        if len(original_lines) != len(translation_lines):
            logger.debug(
                f"      FAILED: Line count differs - original: {len(original_lines)}, translation: {len(translation_lines)}"
            )
            return False

        # Check that empty lines are preserved
        for i, (orig_line, trans_line) in enumerate(zip(original_lines, translation_lines)):
            if orig_line.strip() == "" and trans_line.strip() != "":
                logger.debug(f"      FAILED: Empty line {i} not preserved in translation")
                return False
            if orig_line.strip() != "" and trans_line.strip() == "":
                logger.debug(f"      FAILED: Non-empty line {i} became empty in translation")
                return False

        logger.debug(f"      ✓ Whitespace structure preserved for {field_name}")
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

        # Look for JSON object in the prompt
        # Try to find JSON object after the instruction text
        logger.debug("  Trying instruction-based extraction...")
        lines = prompt_text.split("\n")
        json_start_idx = None

        for i, line in enumerate(lines):
            if "source json object" in line.lower() or "json input" in line.lower():
                # Look for the JSON starting from the next non-empty line
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("{"):
                        json_start_idx = j
                        break
                break

        if json_start_idx is not None:
            # Extract from the JSON start to the end, looking for complete JSON
            json_lines = lines[json_start_idx:]
            json_text = "\n".join(json_lines)

            # Find the complete JSON object
            brace_count = 0
            end_idx = 0

            for i, char in enumerate(json_text):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if end_idx > 0:
                json_candidate = json_text[:end_idx]
                try:
                    json.loads(json_candidate)
                    logger.debug(f"  Found valid JSON via instruction method: {json_candidate[:100]}...")
                    return json_candidate
                except json.JSONDecodeError:
                    pass

        # Fallback: look for any JSON object in the text
        logger.debug("  Trying fallback JSON extraction...")
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
