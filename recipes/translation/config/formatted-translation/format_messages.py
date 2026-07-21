import json
import re
from typing import Any

from format import FormatChecker


class FormatMessagesChecker(FormatChecker):
    """Checker for schema-preserving message translation.

    Expected input (src): a JSON object with a ``messages`` list. In normal
    pipeline use this object contains only the message text fields that should
    be translated, with message indexes preserved.

    Expected output: the same JSON shape, with the string leaves translated.
    """

    def check(self, input_text: str, output_text: str) -> bool:
        try:
            input_json = self._extract_json(input_text)
            output_json = self._extract_json(output_text)
            if input_json is None or output_json is None:
                return False

            input_data = json.loads(input_json)
            output_data = json.loads(output_json)

            if not isinstance(input_data, dict) or not isinstance(output_data, dict):
                return False
            if "messages" not in input_data or "messages" not in output_data:
                return False
            if set(input_data.keys()) != set(output_data.keys()):
                return False
            if not isinstance(input_data["messages"], list) or not isinstance(output_data["messages"], list):
                return False

            return self._same_shape_with_translated_strings(input_data, output_data)
        except (json.JSONDecodeError, TypeError):
            return False

    def _same_shape_with_translated_strings(self, input_value: Any, output_value: Any) -> bool:
        if isinstance(input_value, dict):
            if not isinstance(output_value, dict):
                return False
            if set(input_value.keys()) != set(output_value.keys()):
                return False
            return all(
                self._same_shape_with_translated_strings(input_value[key], output_value[key])
                for key in input_value
            )

        if isinstance(input_value, list):
            if not isinstance(output_value, list) or len(input_value) != len(output_value):
                return False
            return all(
                self._same_shape_with_translated_strings(input_item, output_item)
                for input_item, output_item in zip(input_value, output_value)
            )

        if isinstance(input_value, str):
            if not isinstance(output_value, str):
                return False
            return not input_value.strip() or bool(output_value.strip())

        return output_value == input_value

    def _extract_json(self, text: str):
        # Direct parse
        try:
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            pass

        # ```json ... ``` fence
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Largest valid JSON object
        brace_count = 0
        start = None
        candidates = []
        for i, ch in enumerate(text):
            if ch == "{":
                if start is None:
                    start = i
                brace_count += 1
            elif ch == "}" and brace_count > 0:
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        candidates.append(candidate)
                    except json.JSONDecodeError:
                        pass
                    start = None

        return max(candidates, key=len) if candidates else None
