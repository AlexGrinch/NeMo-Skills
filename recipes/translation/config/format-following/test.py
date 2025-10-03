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

if __name__ == "__main__":
    from format1 import Format1Checker
    from format2 import Format2Checker
    from format6 import Format6Checker
    from format11 import Format11Checker
    from test_data import get_format1_test_data, get_format2_test_data, get_format6_test_data, get_format11_test_data

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
