"""Single source of truth for the verifier prompt and verdict parsing."""
import re

VERIFIER_SYSTEM = "You are a helpful, strict, and precise math teacher."

VERIFIER_PROMPT_TEMPLATE = """You are an objective math evaluator.
Question: {question}

<student_solution>
{student_solution}
</student_solution>

Read the student's solution carefully. Is this solution correct or not?

Output exactly one line:
Final Conclusion: Correct (or Final Conclusion: Incorrect)"""

# The model sometimes repeats the instruction line verbatim; it contains both words and is not a verdict.
_ECHO = re.compile(r"(?i)Final\s+Conclusion\s*:\s*Correct\s*\(or\s*Final\s+Conclusion\s*:\s*Incorrect\)")

_PATTERNS = (
    r"(?i)(?:\*{1,2})?Final\s+Conclusion(?:\*{1,2})?\s*:\s*(?:\*{1,2})?\s*(Correct|Incorrect)\b",
    r"(?i)<final_conclusion>\s*(Correct|Incorrect)\s*</final_conclusion>",
    r"(?i)\b(?:the\s+student'?s?\s+|this\s+)?solution\s+is\s+(?:\*{1,2})?\s*(correct|incorrect)\b",
)


def parse_verdict(response):
    """
    True / False for an explicit verdict (the last one wins), None if the model did not give one.

    Deliberately no "re-solve and compare answers" fallback: a model that re-solves the problem
    has not judged anything, and counting that as a verdict inflated DeepSeek's coverage.
    """
    if not response:
        return None
    text = re.sub(r"<think>.*?(?:</think>|$)", "", response, flags=re.DOTALL)
    text = _ECHO.sub("", text)
    for pattern in _PATTERNS:
        found = re.findall(pattern, text)
        if found:
            return found[-1].lower() == "correct"
    return None
