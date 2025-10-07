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

import logging

LOGGING_LEVEL = logging.DEBUG

logging.basicConfig(level=LOGGING_LEVEL, format="%(levelname)s: %(message)s", force=True)


def get_format1_test_data():
    """Get Format1 test data as JSON strings."""

    import json

    from nemo_skills.prompt.few_shot_examples.format_translation import format1_data

    input_text = json.dumps(format1_data["input"])

    generation_correct = f"Here is the translation: {json.dumps(format1_data['correct_output'])}\n"

    generation_incorrect = f"""Ok, I have to translate the text to German. I need to follow the formatting rules:
1. For text between <think> and </think> tags: keep the original text, do not translate
2. For code blocks between ```python and ``` tags: keep the original code, do not translate
3. For all other text: translate it to German
4. Return the translation as a json object, with an extra field "translation" in each conversation entry, keeping all other fields unchanged

Here is the translation:
{json.dumps(format1_data["incorrect_output"])}
"""

    return input_text, generation_correct, generation_incorrect


def get_format2_test_data():
    """Get Format2 test data as JSON strings."""

    import json

    from nemo_skills.prompt.few_shot_examples.format_translation import format2_data

    input_text = json.dumps(format2_data["input"])

    generation_correct = f"Here is the translation: {json.dumps(format2_data['correct_output'])}\n"

    generation_incorrect = f"""I need to translate the text to German while separating think tags and other content.

Here is the translation:
{json.dumps(format2_data["incorrect_output"])}
"""

    return input_text, generation_correct, generation_incorrect


def get_format6_test_data():
    """Get format6 test data as JSON strings."""

    import json

    from nemo_skills.prompt.few_shot_examples.format_translation import format6_data

    input_text = json.dumps(format6_data["input"])

    generation_correct = f"Here is the translation: {json.dumps(format6_data['correct_output'])}\n"

    generation_incorrect = f"""I need to translate the text to Spanish while preserving think tags and LaTeX formulas.

Here is the translation:
{json.dumps(format6_data["incorrect_output"])}
"""

    return input_text, generation_correct, generation_incorrect


def get_format11_test_data():
    """Get Format11 test data as JSON strings."""

    import json

    from nemo_skills.prompt.few_shot_examples.format_translation import format11_data

    input_text = json.dumps(format11_data["input"])

    generation_correct = f"Here is the translation: {json.dumps(format11_data['correct_output'])}\n"

    generation_incorrect = f"""I need to translate the text to Spanish while preserving LaTeX formatting.

Here is the translation:
{json.dumps(format11_data["incorrect_output"])}
"""

    return input_text, generation_correct, generation_incorrect


if __name__ == "__main__":
    from format1 import Format1Checker
    from format2 import Format2Checker
    from format6 import Format6Checker
    from format11 import Format11Checker

    print("=" * 80)
    print("TESTING FORMAT1 CHECKER")
    print("=" * 80)

    checker = Format1Checker()
    input_text1, fake_generation_1, fake_generation_2 = get_format1_test_data()

    print("=" * 80)
    print("TESTING FAKE_GENERATION_1 (should pass):")
    print("=" * 80)
    result1 = checker.check(input_text1, fake_generation_1)
    print(f"Result: {result1}")

    print("\n" + "=" * 80)
    print("TESTING FAKE_GENERATION_2 (should fail):")
    print("=" * 80)
    result2 = checker.check(input_text1, fake_generation_2)
    print(f"Result: {result2}")

    print("\n" + "=" * 80)
    print("TESTING FORMAT2 CHECKER")
    print("=" * 80)

    checker2 = Format2Checker()
    input_text2, fake_generation2_1, fake_generation2_2 = get_format2_test_data()

    print("=" * 80)
    print("TESTING FORMAT2 FAKE_GENERATION_1 (should pass):")
    print("=" * 80)
    result2_1 = checker2.check(input_text2, fake_generation2_1)
    print(f"Result: {result2_1}")

    print("\n" + "=" * 80)
    print("TESTING FORMAT2 FAKE_GENERATION_2 (should fail):")
    print("=" * 80)
    result2_2 = checker2.check(input_text2, fake_generation2_2)
    print(f"Result: {result2_2}")

    print("\n" + "=" * 80)
    print("TESTING FORMAT6 CHECKER")
    print("=" * 80)

    checker6 = Format6Checker()
    input_text6, fake_generation6_1, fake_generation6_2 = get_format6_test_data()

    print("=" * 80)
    print("TESTING FORMAT6 FAKE_GENERATION_1 (should pass):")
    print("=" * 80)
    result6_1 = checker6.check(input_text6, fake_generation6_1)
    print(f"Result: {result6_1}")

    print("\n" + "=" * 80)
    print("TESTING FORMAT6 FAKE_GENERATION_2 (should fail):")
    print("=" * 80)
    result6_2 = checker6.check(input_text6, fake_generation6_2)
    print(f"Result: {result6_2}")

    print("\n" + "=" * 80)
    print("TESTING FORMAT11 CHECKER")
    print("=" * 80)

    checker11 = Format11Checker()
    input_text11, fake_generation11_1, fake_generation11_2 = get_format11_test_data()

    print("=" * 80)
    print("TESTING FORMAT11 FAKE_GENERATION_1 (should pass):")
    print("=" * 80)
    result11_1 = checker11.check(input_text11, fake_generation11_1)
    print(f"Result: {result11_1}")

    print("\n" + "=" * 80)
    print("TESTING FORMAT11 FAKE_GENERATION_2 (should fail):")
    print("=" * 80)
    result11_2 = checker11.check(input_text11, fake_generation11_2)
    print(f"Result: {result11_2}")
