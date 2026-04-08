import json
import re

from format import FormatChecker


class FormatMessagesChecker(FormatChecker):
    """
    Checker for message-list style data.

    Expected input (src):  {"messages": [{"role": "...", "content": "..."}, ...]}
    Expected output:       same structure, with translated content fields.

    Checks:
      - Both src and generation parse to JSON with a "messages" list.
      - Output has the same number of messages as the input.
      - Each output message has a non-empty "content" string.
    """

    def check(self, input_text: str, output_text: str) -> bool:
        try:
            input_json = self._extract_json(input_text)
            output_json = self._extract_json(output_text)
            if input_json is None or output_json is None:
                return False

            input_data = json.loads(input_json)
            output_data = json.loads(output_json)

            if not isinstance(input_data.get("messages"), list):
                return False
            if not isinstance(output_data.get("messages"), list):
                return False
            if len(input_data["messages"]) != len(output_data["messages"]):
                return False

            for msg in output_data["messages"]:
                if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                    return False

            return True
        except (json.JSONDecodeError, TypeError):
            return False

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
