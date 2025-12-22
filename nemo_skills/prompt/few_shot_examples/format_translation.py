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
                "translation": {
                    "translated_value": "Drucken Sie Hallo Welt mit Python.",
                    "notes": 'Wählte den Imperativ "Drucken Sie" und übersetzte "hello world" wörtlich.',
                },
                "canonical_form": "",
                "label": "",
            },
            {
                "from": "Assistant",
                "value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
                "translation": {
                    "translated_value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
                    "notes": "Nothing was translated",
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
                "translation": {"translated_value": "Drucken Sie Hallo Welt mit Python.", "notes": ""},
                "canonical_form": "",
                "label": "",
            },
            {
                "from": "Assistant",
                "value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hello world')\n```",
                "translation": {
                    "translated_value": "<think>\nPrint hello world with Python.\n</think>\n\n```python\nprint('hi')\n```",
                    "notes": "",
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
# FORMAT 3 TEST DATA
# =============================================================================

format3_data = {
    "input": {
        "conversations": [
            {"from": "User", "value": "Explain how to greet someone politely.", "canonical_form": "", "label": ""},
            {
                "from": "Assistant",
                "value": "Intro text.\n<think>Inner reasoning</think>\nHere is code:\n```python\nprint('hello')\n```",
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
                "value": "Explain how to greet someone politely.",
                "translation": {
                    "think_translation": "",
                    "non_think_translation": "Erkläre, wie man jemanden höflich begrüßt.",
                    "combined": "Erkläre, wie man jemanden höflich begrüßt.",
                },
                "canonical_form": "",
                "label": "",
            },
            {
                "from": "Assistant",
                "value": "Intro text.\n<think>Inner reasoning</think>\nHere is code:\n```python\nprint('hello')\n```",
                "translation": {
                    "think_translation": "<think>Inneres Nachdenken</think>",
                    "non_think_translation": "Einleitungstext.\nHier ist Code:\n```python\nprint('hello')\n```",
                    "combined": "Einleitungstext.\n<think>Inneres Nachdenken</think>\nHier ist Code:\n```python\nprint('hello')\n```",
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
                "value": "Explain how to greet someone politely.",
                "translation": {
                    "think_translation": "",
                    "non_think_translation": "Erkläre, wie man jemanden höflich begrüßt.",
                    "combined": "Erkläre, wie man jemanden höflich begrüßt.",
                },
                "canonical_form": "",
                "label": "",
            },
            {
                "from": "Assistant",
                "value": "Intro text.\n<think>Inner reasoning</think>\nHere is code:\n```python\nprint('hello')\n```",
                "translation": {
                    # think translated, but combined puts it at the end (wrong order)
                    "think_translation": "<think>Inneres Nachdenken</think>",
                    "non_think_translation": "Einleitungstext.\nHier ist Code:\n```python\nprint('hello')\n```",
                    "combined": "Einleitungstext.\nHier ist Code:\n```python\nprint('hello')\n```\n<think>Inneres Nachdenken</think>",
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
# FORMAT 4 TEST DATA
# =============================================================================

format4_data = {
    "input": {
        "conversations": [
            {
                "from": "User",
                "value": "Plan:\n<think>First reflect</think>\nCode:\n```python\nprint('hi')\n```",
                "canonical_form": "",
                "label": "",
            },
            {"from": "Assistant", "value": "Here is the response.", "canonical_form": "", "label": ""},
        ],
        "system": "",
        "mask": "User",
    },
    "correct_output": {
        "conversations": [
            {
                "from": "User",
                "value": "Plan:\n<think>First reflect</think>\nCode:\n```python\nprint('hi')\n```",
                "translation": {
                    "source": "Plan:\n<think>First reflect</think>\nCode:\n```python\nprint('hi')\n```",
                    "target": "Plan:\n<think>Erste Überlegung</think>\nCode:\n```python\nprint('hi')\n```",
                    "language": "German",
                },
                "canonical_form": "",
                "label": "",
            },
            {
                "from": "Assistant",
                "value": "Here is the response.",
                "translation": {
                    "source": "Here is the response.",
                    "target": "Hier ist die Antwort.",
                    "language": "German",
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
                "value": "Plan:\n<think>First reflect</think>\nCode:\n```python\nprint('hi')\n```",
                "translation": {
                    "source": "Plan:\n<think>First reflect</think>\nCode:\n```python\nprint('hi')\n```",
                    # Code preserved, but think content left untranslated and language wrong
                    "target": "Plan:\n<think>First reflect</think>\nCode:\n```python\nprint('hi')\n```",
                    "language": "French",
                },
                "canonical_form": "",
                "label": "",
            },
            {
                "from": "Assistant",
                "value": "Here is the response.",
                "translation": {
                    "source": "Here is the response.",
                    "target": "Hier ist die Antwort.",
                    "language": "French",
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
        "output": "<think>\nI need to find all functions $f: \\mathbb{R} \\to \\mathbb{R}$ that satisfy the functional equation $(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)$.\n\nLet me try some substitutions to understand the structure. First, let me set $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n$4f(0)^2 = 2f(0)$\n$2f(0)^2 = f(0)$\nf(0)(2f(0) - 1) = 0\n\nSo either $f(0) = 0$ or $f(0) = \\frac{1}{2}$.\n</think>\n\nI need to find all functions $f: \\mathbb{R} \\to \\mathbb{R}$ satisfying the functional equation.\n\nLet me start by setting $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n\nThis gives us $4f(0)^2 = 2f(0)$, so $f(0)(2f(0) - 1) = 0$.\n\nTherefore $f(0) = 0$ or $f(0) = \\frac{1}{2}$.\n\nAfter further analysis with various substitutions, I can show that the solutions are:\n\n\\boxed{f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}}<|im_end|>",
    },
    "correct_output": {
        "expected_answer": "f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nResuelve el siguiente problema de matemáticas. Asegúrate de poner la respuesta (y solo la respuesta) dentro de \\boxed{}.\n\nEncuentra todas las funciones \\( f: \\mathbb{R} \\to \\mathbb{R} \\) tales que:\n\\[\n(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)\n\\]<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>\nI need to find all functions $f: \\mathbb{R} \\to \\mathbb{R}$ that satisfy the functional equation $(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)$.\n\nLet me try some substitutions to understand the structure. First, let me set $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n$4f(0)^2 = 2f(0)$\n$2f(0)^2 = f(0)$\nf(0)(2f(0) - 1) = 0\n\nSo either $f(0) = 0$ or $f(0) = \\frac{1}{2}$.\n</think>\n\nNecesito encontrar todas las funciones $f: \\mathbb{R} \\to \\mathbb{R}$ que satisfacen la ecuación funcional.\n\nComencemos estableciendo $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n\nEsto nos da $4f(0)^2 = 2f(0)$, entonces $f(0)(2f(0) - 1) = 0$.\n\nPor lo tanto $f(0) = 0$ o $f(0) = \\frac{1}{2}$.\n\nDespués de un análisis adicional con varias sustituciones, puedo mostrar que las soluciones son:\n\n\\boxed{f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}}<|im_end|>",
    },
    "incorrect_output": {
        "expected_answer": "f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nResuelve el siguiente problema de matemáticas. Asegúrate de poner la respuesta (y solo la respuesta) dentro de \\boxed{}.\n\nEncuentra todas las funciones \\( f: \\mathbb{R} \\to \\mathbb{R} \\) tales que:\n\\[\n(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)\n\\]<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>\nNecesito encontrar todas las funciones $f: \\mathbb{R} \\to \\mathbb{R}$ que satisfacen la ecuación funcional $(f(x) + f(y))(f(a) + f(b)) = f(xa + yb) + f(xb + ya)$.\n\nPermíteme probar algunas sustituciones para entender la estructura. Primero, establecemos $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n$4f(0)^2 = 2f(0)$\n$2f(0)^2 = f(0)$\nf(0)(2f(0) - 1) = 0$\n\nEntonces $f(0) = 0$ o $f(0) = \\frac{1}{2}$.\n</think>\n\nNecesito encontrar todas las funciones $f: \\mathbb{R} \\to \\mathbb{R}$ que satisfacen la ecuación funcional.\n\nComencemos estableciendo $x = y = a = b = 0$:\n$(f(0) + f(0))(f(0) + f(0)) = f(0) + f(0)$\n\nEsto nos da $4f(0)^2 = 2f(0)$, entonces $f(0)(2f(0) - 1) = 0$.\n\nPor lo tanto $f(0) = 0$ o $f(0) = \\frac{1}{2}$.\n\nDespués de un análisis adicional con varias sustituciones, puedo mostrar que las soluciones son:\n\n\\boxed{f(x) = 0 \\quad \\text{and} \\quad f(x) = \\dfrac{1}{3} \\quad \\text{and} \\quad f(x) = x^{2}}<|im_end|>",
    },
}

# =============================================================================
# FORMAT 6 TEST DATA
# =============================================================================

format6_data = {
    "input": {
        "expected_answer": "\\boxed{42}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nSolve for x in the equation $2^x = 42$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>We isolate x by taking logs.</think>Final answer: \\boxed{42}<|im_end|>",
    },
    "correct_output": {
        "expected_answer": "\\boxed{42}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nLöse nach x in der Gleichung $2^x = 42$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "Reasoning: We isolate x by taking logs.\nSolution: \\boxed{42}<|im_end|>",
    },
    "incorrect_output": {
        "expected_answer": "\\boxed{42}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nLöse nach x in der Gleichung $2^x = 42$.\n<|im_end|>\n<|im_start|>assistant\n",
        # Incorrect because think tags were not removed and no Reasoning: prefix
        "output": "<think>We isolate x by taking logs.</think>Solution: \\boxed{42}<|im_end|>",
    },
}

# =============================================================================
# FORMAT 7 TEST DATA
# =============================================================================

format7_data = {
    "input": {
        "expected_answer": "\\boxed{5}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nFind the integer $n$ such that $n+2=7$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>We subtract 2 from both sides.</think>The answer is \\boxed{5}<|im_end|>",
    },
    "correct_output": {
        "expected_answer": "\\boxed{5}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nEncuentra el entero $n$ tal que $n+2=7$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>Restamos 2 a ambos lados.</think>La respuesta es \\boxed{5}<|im_end|>",
    },
    "incorrect_output": {
        "expected_answer": "\\boxed{5}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nEncuentra el entero $n$ tal que $n+2=7$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>We subtract 2 from both sides.</think>La respuesta es \\boxed{5}<|im_end|>",
    },
}

# =============================================================================
# FORMAT 8 TEST DATA
# =============================================================================

format8_data = {
    "input": {
        "expected_answer": "\\boxed{7}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nCompute $3+4$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>Add 3 and 4.</think>The answer is \\boxed{7}<|im_end|>",
    },
    "correct_output": {
        "expected_answer": "\\boxed{7}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nCalculez $3+4$.\n<|im_end|>\n<|im_start|>assistant\n",
        # Inline, think removed, Explanation prefix, answer prefix
        "output": "Explanation: Ajouter 3 et 4. The answer is: \\boxed{7}<|im_end|>",
    },
    "incorrect_output": {
        "expected_answer": "\\boxed{7}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nCalculez $3+4$.\n<|im_end|>\n<|im_start|>assistant\n",
        # Incorrect because output has a newline and no Explanation prefix
        "output": "The answer is: \\boxed{7}\n<|im_end|>",
    },
}

# =============================================================================
# FORMAT 9 TEST DATA
# =============================================================================

format9_data = {
    "input": {
        "expected_answer": "The value is \\boxed{10}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nFind the value of $5+5$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>We add 5 and 5.</think>Answer: The value is \\boxed{10}<|im_end|>",
    },
    "correct_output": {
        "expected_answer": "Il valore è \\boxed{10}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nTrova il valore di $5+5$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>Sommiamo 5 e 5.</think>Risposta: Il valore è \\boxed{10}\nGlossary:\n- add: sommare\n- value: valore\n- answer: risposta<|im_end|>",
    },
    "incorrect_output": {
        "expected_answer": "The value is \\boxed{10}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nTrova il valore di $5+5$.\n<|im_end|>\n<|im_start|>assistant\n",
        # Incorrect because expected_answer not translated and glossary has <3 items
        "output": "<think>Sommiamo 5 e 5.</think>Risposta: The value is \\boxed{10}\nGlossary:\n- add: sommare<|im_end|>",
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

format0_translation_examples = [
    {
        "input_example": json.dumps(format0_data["input"], indent=2),
        "output_example": json.dumps(format0_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "German",
    }
]

format1_translation_examples = [
    {
        "input_example": json.dumps(format1_data["input"], indent=2),
        "output_example": json.dumps(format1_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "German",
    }
]

format2_translation_examples = [
    {
        "input_example": json.dumps(format2_data["input"], indent=2),
        "output_example": json.dumps(format2_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "German",
    }
]

format3_translation_examples = [
    {
        "input_example": json.dumps(format3_data["input"], indent=2),
        "output_example": json.dumps(format3_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "German",
    }
]

format4_translation_examples = [
    {
        "input_example": json.dumps(format4_data["input"], indent=2),
        "output_example": json.dumps(format4_data["correct_output"], indent=2),
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

format6_translation_examples = [
    {
        "input_example": json.dumps(format6_data["input"], indent=2),
        "output_example": json.dumps(format6_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "German",
    }
]

format7_translation_examples = [
    {
        "input_example": json.dumps(format7_data["input"], indent=2),
        "output_example": json.dumps(format7_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format8_translation_examples = [
    {
        "input_example": json.dumps(format8_data["input"], indent=2),
        "output_example": json.dumps(format8_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "French",
    }
]

format9_translation_examples = [
    {
        "input_example": json.dumps(format9_data["input"], indent=2),
        "output_example": json.dumps(format9_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Italian",
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

examples_map = {
    "format0_translation_few_shot": format0_translation_examples,
    "format1_translation_few_shot": format1_translation_examples,
    "format2_translation_few_shot": format2_translation_examples,
    "format3_translation_few_shot": format3_translation_examples,
    "format4_translation_few_shot": format4_translation_examples,
    "format5_translation_few_shot": format5_translation_examples,
    "format6_translation_few_shot": format6_translation_examples,
    "format7_translation_few_shot": format7_translation_examples,
    "format8_translation_few_shot": format8_translation_examples,
    "format9_translation_few_shot": format9_translation_examples,
    "format10_translation_few_shot": format10_translation_examples,
}
