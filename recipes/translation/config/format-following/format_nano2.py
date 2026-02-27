import json
import logging
import re

from format import FormatChecker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FormatNano2Checker(FormatChecker):
    """Checker for format_nano2 with {reasoning_content, metadata} output."""

    def check(self, input_text: str, output_text: str) -> bool:
        logger.debug("Starting format_nano2 check...")
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

            logger.debug("All format_nano2 checks passed!")
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"FAILED: Unexpected error during format check: {e}")
            return False

    def _validate_structure(self, input_data: dict, output_data: dict) -> bool:
        required_fields = {"reasoning_content", "metadata"}

        if set(input_data.keys()) != required_fields:
            logger.debug("FAILED: Input must contain exactly {'reasoning_content', 'metadata'}")
            return False

        if set(output_data.keys()) != required_fields:
            logger.debug("FAILED: Output must contain exactly {'reasoning_content', 'metadata'}")
            return False

        if input_data["metadata"] != output_data["metadata"]:
            logger.debug("FAILED: metadata field changed")
            return False

        if not isinstance(output_data["reasoning_content"], str) or not output_data["reasoning_content"].strip():
            logger.debug("FAILED: reasoning_content must be a non-empty string")
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

        return self._extract_largest_valid_json(prompt_text)

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

        return self._extract_largest_valid_json(generation_text)

    def _extract_largest_valid_json(self, text: str) -> str:
        brace_count = 0
        start_idx = None
        candidates = []

        for i, ch in enumerate(text):
            if ch == "{":
                if start_idx is None:
                    start_idx = i
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    candidate = text[start_idx : i + 1]
                    try:
                        json.loads(candidate)
                        candidates.append(candidate)
                    except json.JSONDecodeError:
                        pass
                    start_idx = None

        if not candidates:
            return None
        return max(candidates, key=len)
