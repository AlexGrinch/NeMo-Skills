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

# Format1 test data
format1_input_data = {
    "conversations": [
        {"from": "User", "value": "Print hello world with Python.", "canonical_form": "", "label": ""},
        {
            "from": "Assistant",
            "value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
            "canonical_form": "",
            "label": "",
        },
    ],
    "system": "",
    "mask": "User",
}

format1_generation_correct = {
    "conversations": [
        {
            "from": "User",
            "value": "Print hello world with Python.",
            "translation": "Drucken Sie Hallo Welt mit Python.",
            "canonical_form": "",
            "label": "",
        },
        {
            "from": "Assistant",
            "value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
            "translation": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
            "canonical_form": "",
            "label": "",
        },
    ],
    "system": "",
    "mask": "User",
}

format1_generation_incorrect = {
    "conversations": [
        {"from": "User", "value": "Drucken Sie Hallo Welt mit Python.", "canonical_form": "", "label": ""},
        {
            "from": "Assistant",
            "value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
            "canonical_form": "",
            "label": "",
        },
    ],
    "system": "",
    "mask": "User",
}

# format6 test data
format6_input_data = {
    "conversations": [
        {
            "from": "User",
            "value": "Calculate the area of a circle with radius 5. Show your work.",
            "canonical_form": "",
            "label": "",
        },
        {
            "from": "Assistant",
            "value": "<think>\nI need to calculate the area of a circle with radius 5.\nThe formula is $A = \\pi r^2$\nGiven: $r = 5$\nSubstitution: $A = \\pi \\cdot 5^2 = \\pi \\cdot 25 = 25\\pi$\nNumerically: $A \\approx 3.14159 \\times 25 \\approx 78.54$\n</think>\n\nI'll calculate the area using the formula $A = \\pi r^2$.\n\nGiven $r = 5$:\n$A = \\pi \\cdot 5^2 = 25\\pi \\approx 78.54$ square units.",
            "canonical_form": "",
            "label": "",
        },
    ],
    "system": "",
    "mask": "User",
}

format6_generation_correct = {
    "conversations": [
        {
            "from": "User",
            "value": "Calculate the area of a circle with radius 5. Show your work.",
            "translation": "Calcule el área de un círculo con radio 5. Muestre su trabajo.",
            "canonical_form": "",
            "label": "",
        },
        {
            "from": "Assistant",
            "value": "<think>\nI need to calculate the area of a circle with radius 5.\nThe formula is $A = \\pi r^2$\nGiven: $r = 5$\nSubstitution: $A = \\pi \\cdot 5^2 = \\pi \\cdot 25 = 25\\pi$\nNumerically: $A \\approx 3.14159 \\times 25 \\approx 78.54$\n</think>\n\nI'll calculate the area using the formula $A = \\pi r^2$.\n\nGiven $r = 5$:\n$A = \\pi \\cdot 5^2 = 25\\pi \\approx 78.54$ square units.",
            "translation": "<think>\nI need to calculate the area of a circle with radius 5.\nThe formula is $A = \\pi r^2$\nGiven: $r = 5$\nSubstitution: $A = \\pi \\cdot 5^2 = \\pi \\cdot 25 = 25\\pi$\nNumerically: $A \\approx 3.14159 \\times 25 \\approx 78.54$\n</think>\n\nCalcularé el área usando la fórmula $A = \\pi r^2$.\n\nDado $r = 5$:\n$A = \\pi \\cdot 5^2 = 25\\pi \\approx 78.54$ unidades cuadradas.",
            "canonical_form": "",
            "label": "",
        },
    ],
    "system": "",
    "mask": "User",
}

format6_generation_incorrect = {
    "conversations": [
        {
            "from": "User",
            "value": "Calculate the area of a circle with radius 5. Show your work.",
            "translation": "Calcule el área de un círculo con radio 5. Muestre su trabajo.",
            "canonical_form": "",
            "label": "",
        },
        {
            "from": "Assistant",
            "value": "<think>\nI need to calculate the area of a circle with radius 5.\nThe formula is $A = \\pi r^2$\nGiven: $r = 5$\nSubstitution: $A = \\pi \\cdot 5^2 = \\pi \\cdot 25 = 25\\pi$\nNumerically: $A \\approx 3.14159 \\times 25 \\approx 78.54$\n</think>\n\nI'll calculate the area using the formula $A = \\pi r^2$.\n\nGiven $r = 5$:\n$A = \\pi \\cdot 5^2 = 25\\pi \\approx 78.54$ square units.",
            "translation": "<think>\nNecesito calcular el área de un círculo con radio 5.\nLa fórmula es $A = \\pi r^2$\nDado: $r = 5$\nSustitución: $A = \\pi \\cdot 5^2 = \\pi \\cdot 25 = 25\\pi$\nNuméricamente: $A \\approx 3.14159 \\times 25 \\approx 78.54$\n</think>\n\nCalcularé el área usando la fórmula $A = \\pi r^2$.\n\nDado $r = 5$:\n$A = \\pi \\cdot 5^2 = 25\\pi \\approx 78.54$ unidades cuadradas.",
            "canonical_form": "",
            "label": "",
        },
    ],
    "system": "",
    "mask": "User",
}

# Format2 test data
format2_input_data = {
    "conversations": [
        {"from": "User", "value": "Print hello world with Python.", "canonical_form": "", "label": ""},
        {
            "from": "Assistant",
            "value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
            "canonical_form": "",
            "label": "",
        },
    ],
    "system": "",
    "mask": "User",
}

format2_generation_correct = {
    "conversations": [
        {
            "from": "User",
            "value": "Print hello world with Python.",
            "translation": {"translation_trace": "", "translation_answer": "Drucken Sie Hallo Welt mit Python."},
            "canonical_form": "",
            "label": "",
        },
        {
            "from": "Assistant",
            "value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
            "translation": {
                "translation_trace": "<think>\nDrucken Sie Hallo Welt mit Python.\n</think>",
                "translation_answer": "\n\n```python\nprint('hello world')\n```",
            },
            "canonical_form": "",
            "label": "",
        },
    ],
    "system": "",
    "mask": "User",
}

format2_generation_incorrect = {
    "conversations": [
        {
            "from": "User",
            "value": "Print hello world with Python.",
            "translation": {"translation_trace": "Drucken Sie Hallo Welt mit Python.", "translation_answer": ""},
            "canonical_form": "",
            "label": "",
        },
        {
            "from": "Assistant",
            "value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
            "translation": {
                "translation_trace": "<think>\nPrint hello world with Python.\n</think>",
                "translation_answer": "\n\n```python\nprint('hello world')\n```",
            },
            "canonical_form": "",
            "label": "",
        },
    ],
    "system": "",
    "mask": "User",
}


# Helper functions to generate test strings
def get_format1_test_data():
    """Get Format1 test data as JSON strings."""
    input_text = json.dumps(format1_input_data)

    generation_correct = f"Here is the translation: {json.dumps(format1_generation_correct)}\n"

    generation_incorrect = f"""Ok, I have to translate the text to German. I need to follow the formatting rules:
1. For text between <think> and </think> tags: keep the original text, do not translate
2. For code blocks between ```python and ``` tags: keep the original code, do not translate
3. For all other text: translate it to German
4. Return the translation as a json object, with an extra field "translation" in each conversation entry, keeping all other fields unchanged

Here is the translation:
{json.dumps(format1_generation_incorrect)}
"""

    return input_text, generation_correct, generation_incorrect


def get_format6_test_data():
    """Get format6 test data as JSON strings."""
    input_text = json.dumps(format6_input_data)

    generation_correct = f"Here is the translation: {json.dumps(format6_generation_correct)}\n"

    generation_incorrect = f"""I need to translate the text to Spanish while preserving think tags and LaTeX formulas.

Here is the translation:
{json.dumps(format6_generation_incorrect)}
"""

    return input_text, generation_correct, generation_incorrect


def get_format2_test_data():
    """Get Format2 test data as JSON strings."""
    input_text = json.dumps(format2_input_data)

    generation_correct = f"Here is the translation: {json.dumps(format2_generation_correct)}\n"

    generation_incorrect = f"""I need to translate the text to German while separating think tags and other content.

Here is the translation:
{json.dumps(format2_generation_incorrect)}
"""

    return input_text, generation_correct, generation_incorrect


# Format11 test data
format11_input_data = {
    "problem": "Which of the following best describes the concept of the digital divide and its impact on global inequality in the 21st century?\nA: The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities.\nB: The digital divide is the difference in technological skills between younger and older generations, leading to a generational gap in employment opportunities.",
    "generation": '<think>\nI need to solve this problem about the digital divide and its impact on global inequality. The question asks which option best describes the concept and its impact.\n\nOption A says: "The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities."\n\nThis sounds right. It mentions access to computers and the internet, and it talks about exacerbating social and economic inequalities.\n\nI think A is the answer.\n</think>\nA\n\n\\boxed{\\text{A}}',
    "expected_answer": "\\text{A}",
    "majority_res": "8/8",
    "topic": "history",
    "subtopic": "neo_colonialism",
    "difficulty": "complex",
}

format11_generation_correct = {
    "problem": "¿Cuál de las siguientes opciones describe mejor el concepto de la brecha digital y su impacto en la desigualdad global en el siglo XXI?\nA: La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas.\nB: La brecha digital es la diferencia en habilidades tecnológicas entre generaciones más jóvenes y mayores, llevando a una brecha generacional en oportunidades de empleo.",
    "generation": '<think>\nNecesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.\n</think>\nA\n\n\\boxed{\\text{A}}',
    "expected_answer": "\\text{A}",
    "majority_res": "8/8",
    "topic": "history",
    "subtopic": "neo_colonialism",
    "difficulty": "complex",
}

format11_generation_incorrect = {
    "problem": "¿Cuál de las siguientes opciones describe mejor el concepto de la brecha digital y su impacto en la desigualdad global en el siglo XXI?\nA: La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas.\nB: La brecha digital es la diferencia en habilidades tecnológicas entre generaciones más jóvenes y mayores, llevando a una brecha generacional en oportunidades de empleo.",
    "generation": '<think>\nNecesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.\n</think>\nA\n\n\\boxed{A}',
    "expected_answer": "A",
    "majority_res": "8/8",
    "topic": "history",
    "subtopic": "neo_colonialism",
    "difficulty": "complex",
}


def get_format11_test_data():
    """Get Format11 test data as JSON strings."""
    input_text = json.dumps(format11_input_data)

    generation_correct = f"Here is the translation: {json.dumps(format11_generation_correct)}\n"

    generation_incorrect = f"""I need to translate the text to Spanish while preserving LaTeX formatting.

Here is the translation:
{json.dumps(format11_generation_incorrect)}
"""

    return input_text, generation_correct, generation_incorrect
