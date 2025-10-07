# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Translation test data and few-shot examples for format-following translation tasks."""

import json

# =============================================================================
# FORMAT 1 TEST DATA
# =============================================================================

format1_data = {
    "input": {
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
    },
    "correct_output": {
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
    },
    "incorrect_output": {
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
    },
}

# =============================================================================
# FORMAT 2 TEST DATA
# =============================================================================

format2_data = {
    "input": {
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
    },
    "correct_output": {
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
    },
    "incorrect_output": {
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
    },
}

# =============================================================================
# FORMAT 6 TEST DATA
# =============================================================================

format6_data = {
    "input": {
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
    },
    "correct_output": {
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
    },
    "incorrect_output": {
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
    },
}

# =============================================================================
# FORMAT 11 TEST DATA
# =============================================================================

format11_data = {
    "input": {
        "problem": "Which of the following best describes the concept of the digital divide and its impact on global inequality in the 21st century?\nA: The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities.\nB: The digital divide is the difference in technological skills between younger and older generations, leading to a generational gap in employment opportunities.",
        "generation": '<think>\nI need to solve this problem about the digital divide and its impact on global inequality. The question asks which option best describes the concept and its impact.\n\nOption A says: "The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities."\n\nThis sounds right. It mentions access to computers and the internet, and it talks about exacerbating social and economic inequalities.\n\nI think A is the answer.\n</think>\nA\n\n\\boxed{\\text{A}}',
        "expected_answer": "\\text{A}",
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
    "correct_output": {
        "problem": "¿Cuál de las siguientes opciones describe mejor el concepto de la brecha digital y su impacto en la desigualdad global en el siglo XXI?\nA: La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas.\nB: La brecha digital es la diferencia en habilidades tecnológicas entre generaciones más jóvenes y mayores, llevando a una brecha generacional en oportunidades de empleo.",
        "generation": '<think>\nNecesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.\n</think>\nA\n\n\\boxed{\\text{A}}',
        "expected_answer": "\\text{A}",
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
    "incorrect_output": {
        "problem": "¿Cuál de las siguientes opciones describe mejor el concepto de la brecha digital y su impacto en la desigualdad global en el siglo XXI?\nA: La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas.\nB: La brecha digital es la diferencia en habilidades tecnológicas entre generaciones más jóvenes y mayores, llevando a una brecha generacional en oportunidades de empleo.",
        "generation": '<think>\nNecesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.\n</think>\nA\n\n\\boxed{A}',
        "expected_answer": "A",
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
}

# =============================================================================
# FEW-SHOT EXAMPLES FOR NEMO-SKILLS
# =============================================================================

# Convert format1 data to few-shot example format
format1_translation_examples = [
    {
        "input_example": json.dumps(format1_data["input"], indent=2),
        "output_example": json.dumps(format1_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "German",
    }
]

# Examples map for nemo-skills integration
examples_map = {
    "format1_translation_few_shot": format1_translation_examples,
}
