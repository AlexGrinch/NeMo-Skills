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
# FORMAT 0 TEST DATA
# =============================================================================

format0_data = {
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
# FORMAT 5 TEST DATA
# =============================================================================

format5_data = {
    "input": {
        "expected_answer": "f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nSolve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\nFind all functions \\( f: \\mathbb{R} \\to \\mathbb{R} \\) such that:\n\\[\n(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)\n\\]<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>\nI need to find all functions $f: \\mathbb{R} \\to \\mathbb{R}$ that satisfy the functional equation $(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)$.\n\nLet me try some substitutions to understand the structure. First, let me set $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n$4f(0)^2 = 2f(0)$\n$2f(0)^2 = f(0)$\n$f(0)(2f(0) - 1) = 0$\n\nSo either $f(0) = 0$ or $f(0) = \\frac{1}{2}$.\n</think>\n\nI need to find all functions $f: \\mathbb{R} \\to \\mathbb{R}$ satisfying the functional equation.\n\nLet me start by setting $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n\nThis gives us $4f(0)^2 = 2f(0)$, so $f(0)(2f(0) - 1) = 0$.\n\nTherefore $f(0) = 0$ or $f(0) = \\frac{1}{2}$.\n\nAfter further analysis with various substitutions, I can show that the solutions are:\n\n\\boxed{f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}}<|im_end|>",
    },
    "correct_output": {
        "expected_answer": "f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nResuelve el siguiente problema de matemáticas. Asegúrate de poner la respuesta (y solo la respuesta) dentro de \\boxed{}.\n\nEncuentra todas las funciones \\( f: \\mathbb{R} \\to \\mathbb{R} \\) tales que:\n\\[\n(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)\n\\]<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>\nI need to find all functions $f: \\mathbb{R} \\to \\mathbb{R}$ that satisfy the functional equation $(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)$.\n\nLet me try some substitutions to understand the structure. First, let me set $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n$4f(0)^2 = 2f(0)$\n$2f(0)^2 = f(0)$\n$f(0)(2f(0) - 1) = 0$\n\nSo either $f(0) = 0$ or $f(0) = \\frac{1}{2}$.\n</think>\n\nNecesito encontrar todas las funciones $f: \\mathbb{R} \\to \\mathbb{R}$ que satisfacen la ecuación funcional.\n\nComencemos estableciendo $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n\nEsto nos da $4f(0)^2 = 2f(0)$, entonces $f(0)(2f(0) - 1) = 0$.\n\nPor lo tanto $f(0) = 0$ o $f(0) = \\frac{1}{2}$.\n\nDespués de un análisis adicional con varias sustituciones, puedo mostrar que las soluciones son:\n\n\\boxed{f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}}<|im_end|>",
    },
    "incorrect_output": {
        "expected_answer": "f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nResuelve el siguiente problema de matemáticas. Asegúrate de poner la respuesta (y solo la respuesta) dentro de \\boxed{}.\n\nEncuentra todas las funciones \\( f: \\mathbb{R} \\to \\mathbb{R} \\) tales que:\n\\[\n(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)\n\\]<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>\nNecesito encontrar todas las funciones $f: \\mathbb{R} \\to \\mathbb{R}$ que satisfacen la ecuación funcional $(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)$.\n\nPermíteme probar algunas sustituciones para entender la estructura. Primero, establecemos $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n$4f(0)^2 = 2f(0)$\n$2f(0)^2 = f(0)$\n$f(0)(2f(0) - 1) = 0$\n\nEntonces $f(0) = 0$ o $f(0) = \\frac{1}{2}$.\n</think>\n\nNecesito encontrar todas las funciones $f: \\mathbb{R} \\to \\mathbb{R}$ que satisfacen la ecuación funcional.\n\nComencemos estableciendo $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n\nEsto nos da $4f(0)^2 = 2f(0)$, entonces $f(0)(2f(0) - 1) = 0$.\n\nPor lo tanto $f(0) = 0$ o $f(0) = \\frac{1}{2}$.\n\nDespués de un análisis adicional con varias sustituciones, puedo mostrar que las soluciones son:\n\n\\boxed{f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}}<|im_end|>",
    },
}

# =============================================================================
# FORMAT 10 TEST DATA
# =============================================================================

format10_data = {
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

# Convert format0 data to few-shot example format
format0_translation_examples = [
    {
        "input_example": json.dumps(format0_data["input"], indent=2),
        "output_example": json.dumps(format0_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "German",
    }
]

# Convert format1 data to few-shot example format
format1_translation_examples = [
    {
        "input_example": json.dumps(format1_data["input"], indent=2),
        "output_example": json.dumps(format1_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "German",
    }
]

format5_translation_examples = [
    {
        "input_example": json.dumps(format5_data["input"], indent=2),
        "output_example": json.dumps(format5_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format10_translation_examples = [
    {
        "input_example": json.dumps(format10_data["input"], indent=2),
        "output_example": json.dumps(format10_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "German",
    }
]

# Examples map for nemo-skills integration
examples_map = {
    "format0_translation_few_shot": format0_translation_examples,
    "format1_translation_few_shot": format1_translation_examples,
    "format5_translation_few_shot": format5_translation_examples,
    "format10_translation_few_shot": format10_translation_examples,
}
