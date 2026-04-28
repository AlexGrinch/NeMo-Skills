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
# FORMAT 109 TEST DATA
# =============================================================================

format109_data = {
    "input": {
        "expected_answer": "The value is \\boxed{10}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nFind the value of $5+5$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>We add 5 and 5.</think>Answer: The value is \\boxed{10}<|im_end|>",
    },
    "correct_output": {
        "expected_answer": "Il valore è \\boxed{10}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nTrova il valore di $5+5$.\n<|im_end|>\n<|im_start|>assistant\n",
        "output": "<think>Sommiamo 5 e 5.</think>Risposta: Il valore è \\boxed{10}<|im_end|>",
        "glossary": [
            "- add: sommare",
            "- value: valore",
            "- answer: risposta",
        ],
    },
    "incorrect_output": {
        "expected_answer": "The value is \\boxed{10}",
        "input": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nTrova il valore di $5+5$.\n<|im_end|>\n<|im_start|>assistant\n",
        # Incorrect: expected_answer not translated and glossary has <3 items
        "output": "<think>Sommiamo 5 e 5.</think>Risposta: The value is \\boxed{10}<|im_end|>",
        "glossary": [
            "- add: sommare",
            "- value: valore",
        ],
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
        "problem": {
            "source": "Which of the following best describes the concept of the digital divide and its impact on global inequality in the 21st century?\nA: The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities.\nB: The digital divide is the difference in technological skills between younger and older generations, leading to a generational gap in employment opportunities.",
            "target": "¿Cuál de las siguientes opciones describe mejor el concepto de la brecha digital y su impacto en la desigualdad global en el siglo XXI?\nA: La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas.\nB: La brecha digital es la diferencia en habilidades tecnológicas entre generaciones más jóvenes y mayores, llevando a una brecha generacional en oportunidades de empleo.",
        },
        "generation": {
            "source": '<think>\nI need to solve this problem about the digital divide and its impact on global inequality. The question asks which option best describes the concept and its impact.\n\nOption A says: "The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities."\n\nThis sounds right. It mentions access to computers and the internet, and it talks about exacerbating social and economic inequalities.\n\nI think A is the answer.\n</think>\nA\n\n\\boxed{\\text{A}}',
            "target": '<think>\nNecesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.\n</think>\nA\n\n\\boxed{\\text{A}}',
        },
        "expected_answer": {"source": "\\text{A}", "target": "\\text{A}"},
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
    "incorrect_output": {
        "problem": {
            "source": "Which of the following best describes the concept of the digital divide and its impact on global inequality in the 21st century?\nA: The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities.\nB: The digital divide is the difference in technological skills between younger and older generations, leading to a generational gap in employment opportunities.",
            "target": "¿Cuál de las siguientes opciones describe mejor el concepto de la brecha digital y su impacto en la desigualdad global en el siglo XXI?\nA: La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas.\nB: La brecha digital es la diferencia en habilidades tecnológicas entre generaciones más jóvenes y mayores, llevando a una brecha generacional en oportunidades de empleo.",
        },
        # Incorrect: LaTeX not preserved in target
        "generation": {
            "source": '<think>\nI need to solve this problem about the digital divide and its impact on global inequality. The question asks which option best describes the concept and its impact.\n\nOption A says: "The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities."\n\nThis sounds right. It mentions access to computers and the internet, and it talks about exacerbating social and economic inequalities.\n\nI think A is the answer.\n</think>\nA\n\n\\boxed{\\text{A}}',
            "target": '<think>\nNecesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.\n</think>\nA\n\n\\boxed{A}',
        },
        "expected_answer": {"source": "\\text{A}", "target": "A"},
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
}

# =============================================================================
# FORMAT 12 TEST DATA
# =============================================================================

format12_data = {
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
        "generation": {
            "reasoning_translation": 'Necesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.',
            "answer_translation": "A\n\n\\boxed{\\text{A}}",
        },
        "expected_answer": "\\text{A}",
        "problem_metadata": "Este es un problema complejo sobre historia, específicamente sobre neo_colonialism.",
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
    "incorrect_output": {
        "problem": "¿Cuál de las siguientes opciones describe mejor el concepto de la brecha digital y su impacto en la desigualdad global en el siglo XXI?\nA: La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas.\nB: La brecha digital es la diferencia en habilidades tecnológicas entre generaciones más jóvenes y mayores, llevando a una brecha generacional en oportunidades de empleo.",
        # Incorrect: missing problem_metadata field
        "generation": {
            "reasoning_translation": 'Necesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.',
            "answer_translation": "A\n\n\\boxed{\\text{A}}",
        },
        "expected_answer": "\\text{A}",
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
}

# =============================================================================
# FORMAT 13 TEST DATA
# =============================================================================

format13_data = {
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
        "problem": [
            "Which of the following best describes the concept of the digital divide and its impact on global inequality in the 21st century?\nA: The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities.\nB: The digital divide is the difference in technological skills between younger and older generations, leading to a generational gap in employment opportunities.",
            "¿Cuál de las siguientes opciones describe mejor el concepto de la brecha digital y su impacto en la desigualdad global en el siglo XXI?\nA: La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas.\nB: La brecha digital es la diferencia en habilidades tecnológicas entre generaciones más jóvenes y mayores, llevando a una brecha generacional en oportunidades de empleo.",
        ],
        "generation": [
            '<think>\nI need to solve this problem about the digital divide and its impact on global inequality. The question asks which option best describes the concept and its impact.\n\nOption A says: "The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities."\n\nThis sounds right. It mentions access to computers and the internet, and it talks about exacerbating social and economic inequalities.\n\nI think A is the answer.\n</think>\nA\n\n\\boxed{\\text{A}}',
            '<think>\nNecesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.\n</think>\nA\n\n\\boxed{\\text{A}}',
        ],
        "expected_answer": ["\\text{A}", "\\text{A}"],
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
    "incorrect_output": {
        # Incorrect: first element doesn't match input
        "problem": [
            "Wrong text here",
            "¿Cuál de las siguientes opciones describe mejor el concepto de la brecha digital y su impacto en la desigualdad global en el siglo XXI?\nA: La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas.\nB: La brecha digital es la diferencia en habilidades tecnológicas entre generaciones más jóvenes y mayores, llevando a una brecha generacional en oportunidades de empleo.",
        ],
        "generation": [
            '<think>\nI need to solve this problem about the digital divide and its impact on global inequality. The question asks which option best describes the concept and its impact.\n\nOption A says: "The digital divide refers to the gap between those who have ready access to computers and the internet and those who do not, often exacerbating social and economic inequalities."\n\nThis sounds right. It mentions access to computers and the internet, and it talks about exacerbating social and economic inequalities.\n\nI think A is the answer.\n</think>\nA\n\n\\boxed{\\text{A}}',
            '<think>\nNecesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.\n</think>\nA\n\n\\boxed{\\text{A}}',
        ],
        "expected_answer": ["\\text{A}", "\\text{A}"],
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
}

# =============================================================================
# FORMAT 14 TEST DATA
# =============================================================================

format14_data = {
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
        # Incorrect: content outside think tags was translated
        "generation": '<think>\nNecesito resolver este problema sobre la brecha digital y su impacto en la desigualdad global. La pregunta pide cuál opción describe mejor el concepto y su impacto.\n\nLa opción A dice: "La brecha digital se refiere a la diferencia entre aquellos que tienen acceso fácil a computadoras e internet y aquellos que no, a menudo exacerbando las desigualdades sociales y económicas."\n\nEsto suena correcto. Menciona el acceso a computadoras e internet, y habla de exacerbar las desigualdades sociales y económicas.\n\nCreo que A es la respuesta.\n</think>\nLa respuesta es A\n\n\\boxed{\\text{A}}',
        "expected_answer": "\\text{A}",
        "majority_res": "8/8",
        "topic": "history",
        "subtopic": "neo_colonialism",
        "difficulty": "complex",
    },
}

# =============================================================================
# FORMAT NANO TEST DATA
# =============================================================================

format_nano_data = {
    "input": {
        "problem": "**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus\n\nA) Scion\nB) Explant\nC) Stock\nD) Callus",
        "is_MCQ": True,
        "has_tool": False,
        "expected_answer": "B",
        "messages": [
            {
                "role": "user",
                "content": "Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus\n\nA) Scion\nB) Explant\nC) Stock\nD) Callus",
            },
            {
                "role": "assistant",
                "content": "Answer: B",
                "reasoning_content": 'We need to answer: "Part of plant used for culturing is called ..."\n\nWe need to consider plant tissue culture terminology. In plant tissue culture, the piece of plant tissue placed in culture medium is called an "explant". A "scion" is a piece of a plant used in grafting (the top part). "Stock" is the bottom part (rootstock). "Callus" is undifferentiated mass of cells that forms from an excised explant under culture conditions. So the correct answer is "Explant". So answer "B".\n\nBut we must respond with final answer line: "Answer: B". They gave both alphabet options and letter list; either "B" is correct. So final answer: "Answer: B".',
            },
        ],
        "metadata": {
            "uuid": "2a574afa-abc2-439d-83cb-6447270689f2",
            "turn_token_count": [
                {"reasoning": 0, "content": 88, "tool_calls": 0},
                {"reasoning": 151, "content": 3, "tool_calls": 0},
            ],
            "all_turns_token_count": {"reasoning": 151, "content": 91, "tool_calls": 0, "all": 242},
            "processing_steps": [
                "reasoning_extraction",
                "think_tag_cleaning",
                "identity_propaganda_check",
                "token_counting",
            ],
            "source_file": "vedantu_chemistry_mcq_filtered_all_decontaminated_kept.jsonl",
            "domain": "multilingual_stem",
        },
    },
    "correct_output": {
        "problem": "**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus\n\nA) Scion\nB) Explant\nC) Stock\nD) Callus",
        "is_MCQ": True,
        "has_tool": False,
        "expected_answer": "B",
        "messages": [
            {
                "role": "user",
                "content": "Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus\n\nA) Scion\nB) Explant\nC) Stock\nD) Callus",
                "translation": "Responda la siguiente pregunta de opción múltiple. La última línea de su respuesta debe estar en el siguiente formato: 'Respuesta: A/B/C/D' (por ejemplo, 'Respuesta: A').\n\n**Pregunta:** La parte de la planta utilizada para el cultivo se llama\n\nA. Injerto\nB. Explante\nC. Portainjerto\nD. Callo\n\nA) Injerto\nB) Explante\nC) Portainjerto\nD) Callo",
            },
            {
                "role": "assistant",
                "content": "Answer: B",
                "reasoning_content": 'We need to answer: "Part of plant used for culturing is called ..."\n\nWe need to consider plant tissue culture terminology. In plant tissue culture, the piece of plant tissue placed in culture medium is called an "explant". A "scion" is a piece of a plant used in grafting (the top part). "Stock" is the bottom part (rootstock). "Callus" is undifferentiated mass of cells that forms from an excised explant under culture conditions. So the correct answer is "Explant". So answer "B".\n\nBut we must respond with final answer line: "Answer: B". They gave both alphabet options and letter list; either "B" is correct. So final answer: "Answer: B".',
                "translation": {
                    "content": "Respuesta: B",
                    "reasoning_content": 'Necesitamos responder: "La parte de la planta utilizada para el cultivo se llama ..."\n\nNecesitamos considerar la terminología del cultivo de tejidos vegetales. En el cultivo de tejidos vegetales, el trozo de tejido vegetal colocado en medio de cultivo se llama "explante". Un "injerto" es un trozo de planta utilizado en injertos (la parte superior). "Portainjerto" es la parte inferior (portainjerto). "Callo" es una masa indiferenciada de células que se forma a partir de un explante escindido bajo condiciones de cultivo. Entonces la respuesta correcta es "Explante". Así que la respuesta es "B".\n\nPero debemos responder con la línea de respuesta final: "Respuesta: B". Dieron tanto opciones alfabéticas como lista de letras; cualquiera de las dos "B" es correcta. Así que la respuesta final: "Respuesta: B".',
                },
            },
        ],
        "metadata": {
            "uuid": "2a574afa-abc2-439d-83cb-6447270689f2",
            "turn_token_count": [
                {"reasoning": 0, "content": 88, "tool_calls": 0},
                {"reasoning": 151, "content": 3, "tool_calls": 0},
            ],
            "all_turns_token_count": {"reasoning": 151, "content": 91, "tool_calls": 0, "all": 242},
            "processing_steps": [
                "reasoning_extraction",
                "think_tag_cleaning",
                "identity_propaganda_check",
                "token_counting",
            ],
            "source_file": "vedantu_chemistry_mcq_filtered_all_decontaminated_kept.jsonl",
            "domain": "multilingual_stem",
        },
    },
    "incorrect_output": {
        "problem": "**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus\n\nA) Scion\nB) Explant\nC) Stock\nD) Callus",
        "is_MCQ": True,
        "has_tool": False,
        "expected_answer": "B",
        "messages": [
            {
                "role": "user",
                "content": "Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus\n\nA) Scion\nB) Explant\nC) Stock\nD) Callus",
                "translation": "Responda la siguiente pregunta de opción múltiple. La última línea de su respuesta debe estar en el siguiente formato: 'Respuesta: A/B/C/D' (por ejemplo, 'Respuesta: A').\n\n**Pregunta:** La parte de la planta utilizada para el cultivo se llama\n\nA. Injerto\nB. Explante\nC. Portainjerto\nD. Callo\n\nA) Injerto\nB) Explante\nC) Portainjerto\nD) Callo",
            },
            {
                "role": "assistant",
                "content": "Answer: B",
                "reasoning_content": 'We need to answer: "Part of plant used for culturing is called ..."\n\nWe need to consider plant tissue culture terminology. In plant tissue culture, the piece of plant tissue placed in culture medium is called an "explant". A "scion" is a piece of a plant used in grafting (the top part). "Stock" is the bottom part (rootstock). "Callus" is undifferentiated mass of cells that forms from an excised explant under culture conditions. So the correct answer is "Explant". So answer "B".\n\nBut we must respond with final answer line: "Answer: B". They gave both alphabet options and letter list; either "B" is correct. So final answer: "Answer: B".',
                # Incorrect: translation is a string instead of an object with content and reasoning_content
                "translation": "Respuesta: B",
            },
        ],
        "metadata": {
            "uuid": "2a574afa-abc2-439d-83cb-6447270689f2",
            "turn_token_count": [
                {"reasoning": 0, "content": 88, "tool_calls": 0},
                {"reasoning": 151, "content": 3, "tool_calls": 0},
            ],
            "all_turns_token_count": {"reasoning": 151, "content": 91, "tool_calls": 0, "all": 242},
            "processing_steps": [
                "reasoning_extraction",
                "think_tag_cleaning",
                "identity_propaganda_check",
                "token_counting",
            ],
            "source_file": "vedantu_chemistry_mcq_filtered_all_decontaminated_kept.jsonl",
            "domain": "multilingual_stem",
        },
    },
}

format_nano1_data = {
    "input": {
        "problem": "**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus",
        "generation": "Answer: B",
    },
    "correct_output": {
        "problem": "**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus",
        "generation": "Answer: B",
        "translation": {
            "problem": "**Pregunta:** La parte de la planta utilizada para el cultivo se llama\n\nA. Injerto\nB. Explante\nC. Portainjerto\nD. Callo",
            "generation": "Respuesta: B",
        },
    },
    "incorrect_output": {
        # Incorrect because original fields were overwritten instead of preserved
        "problem": "**Pregunta:** La parte de la planta utilizada para el cultivo se llama\n\nA. Injerto\nB. Explante\nC. Portainjerto\nD. Callo",
        "generation": "Respuesta: B",
        "translation": {
            "problem": "**Pregunta:** La parte de la planta utilizada para el cultivo se llama\n\nA. Injerto\nB. Explante\nC. Portainjerto\nD. Callo",
            "generation": "Respuesta: B",
        },
    },
}

format_nano1_escaped_data = {
    "input": {
        "problem": "**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus",
        "generation": '<<ESC:think>>\nWe need to identify the plant tissue culture term. The piece of plant tissue placed in culture medium is called an "explant".\n<<ESC:/think>>\nAnswer: B',
    },
    "correct_output": {
        "problem": "**Question:** Part of plant used for culturing is called\n\nA. Scion\nB. Explant\nC. Stock\nD. Callus",
        "generation": '<<ESC:think>>\nWe need to identify the plant tissue culture term. The piece of plant tissue placed in culture medium is called an "explant".\n<<ESC:/think>>\nAnswer: B',
        "translation": {
            "problem": "**Pregunta:** La parte de la planta utilizada para el cultivo se llama\n\nA. Injerto\nB. Explante\nC. Portainjerto\nD. Callo",
            "generation": '<<ESC:think>>\nNecesitamos identificar el término de cultivo de tejidos vegetales. El trozo de tejido vegetal colocado en medio de cultivo se llama "explante".\n<<ESC:/think>>\nRespuesta: B',
        },
    },
    "incorrect_output": {
        "problem": "**Pregunta:** La parte de la planta utilizada para el cultivo se llama\n\nA. Injerto\nB. Explante\nC. Portainjerto\nD. Callo",
        "generation": "<think>\nNecesitamos identificar el término de cultivo de tejidos vegetales.\n</think>\nRespuesta: B",
        "translation": {
            "problem": "**Pregunta:** La parte de la planta utilizada para el cultivo se llama\n\nA. Injerto\nB. Explante\nC. Portainjerto\nD. Callo",
            "generation": "<think>\nNecesitamos identificar el término de cultivo de tejidos vegetales.\n</think>\nRespuesta: B",
        },
    },
}

format_nano2_data = {
    "input": {
        "reasoning_content": ("We need to compute 12 * 7. Multiplying gives 84. So the final answer is 84."),
        "metadata": {
            "uuid": "f9baf1f7-59ba-4f4b-8ecf-4ec9fa7f7acc",
            "domain": "math",
            "source_file": "demo.jsonl",
        },
    },
    "correct_output": {
        "reasoning_content": (
            "Necesitamos calcular 12 * 7. Al multiplicar obtenemos 84. Por lo tanto, la respuesta final es 84."
        ),
        "metadata": {
            "uuid": "f9baf1f7-59ba-4f4b-8ecf-4ec9fa7f7acc",
            "domain": "math",
            "source_file": "demo.jsonl",
        },
    },
    "incorrect_output": {
        # Incorrect because metadata must remain unchanged
        "reasoning_content": (
            "Necesitamos calcular 12 * 7. Al multiplicar obtenemos 84. Por lo tanto, la respuesta final es 84."
        ),
        "metadata": {
            "uuid": "changed-id",
            "domain": "math",
            "source_file": "demo.jsonl",
        },
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


format109_translation_examples = [
    {
        "input_example": json.dumps(format109_data["input"], indent=2),
        "output_example": json.dumps(format109_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Italian",
    }
]

format10_translation_examples = [
    {
        "input_example": json.dumps(format10_data["input"], indent=2),
        "output_example": json.dumps(format10_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format11_translation_examples = [
    {
        "input_example": json.dumps(format11_data["input"], indent=2),
        "output_example": json.dumps(format11_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format12_translation_examples = [
    {
        "input_example": json.dumps(format12_data["input"], indent=2),
        "output_example": json.dumps(format12_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format13_translation_examples = [
    {
        "input_example": json.dumps(format13_data["input"], indent=2),
        "output_example": json.dumps(format13_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format14_translation_examples = [
    {
        "input_example": json.dumps(format14_data["input"], indent=2),
        "output_example": json.dumps(format14_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format_nano_translation_examples = [
    {
        "input_example": json.dumps(format_nano_data["input"], indent=2),
        "output_example": json.dumps(format_nano_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format_nano1_translation_examples = [
    {
        "input_example": json.dumps(format_nano1_data["input"], indent=2),
        "output_example": json.dumps(format_nano1_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format_nano1_escaped_translation_examples = [
    {
        "input_example": json.dumps(format_nano1_escaped_data["input"], indent=2),
        "output_example": json.dumps(format_nano1_escaped_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
    }
]

format_nano2_translation_examples = [
    {
        "input_example": json.dumps(format_nano2_data["input"], indent=2),
        "output_example": json.dumps(format_nano2_data["correct_output"], indent=2),
        "source_lang": "English",
        "target_lang": "Spanish",
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
    "format109_translation_few_shot": format109_translation_examples,
    "format10_translation_few_shot": format10_translation_examples,
    "format11_translation_few_shot": format11_translation_examples,
    "format12_translation_few_shot": format12_translation_examples,
    "format13_translation_few_shot": format13_translation_examples,
    "format14_translation_few_shot": format14_translation_examples,
    "format_nano_translation_few_shot": format_nano_translation_examples,
    "format_nano1_translation_few_shot": format_nano1_translation_examples,
    "format_nano1_escaped_translation_few_shot": format_nano1_escaped_translation_examples,
    "format_nano2_translation_few_shot": format_nano2_translation_examples,
}
