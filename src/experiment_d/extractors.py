import re

_NUM = r'-?\d[\d,]*(?:\.\d+)?'


def _strip_think(text: str) -> str:
    # Strip <think> blocks (and unclosed <think> blocks) to avoid extracting from scratchpad
    return re.sub(r'<think>.*?(?:</think>|$)', '', text.strip(), flags=re.DOTALL).strip()


def _clean_number(value: str) -> str:
    return value.replace(',', '').rstrip('.')


def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the numeric answer from a GSM8K generation using multi-stage fallbacks.
    Returns the numeric string if found, else None.
    """
    text = _strip_think(text)

    # 1. Official format: #### N. The LAST marker wins (the model sometimes echoes the
    #    one-shot example first) and a short non-numeric gap is tolerated so that
    #    "#### **Answer:** \$18" is still read as 18.
    matches = re.findall(r'####[^\d\-\n]{0,40}?(' + _NUM + ')', text)
    if matches:
        return _clean_number(matches[-1])

    # 1b. \boxed{N} (reasoning models often use it regardless of the requested format)
    boxed = _last_boxed(text)
    if boxed is not None:
        boxed_num = re.search(r'(' + _NUM + ')', boxed.replace('\\$', ''))
        if boxed_num:
            return _clean_number(boxed_num.group(1))

    # 2. Equation end format: = N (take the last one)
    matches = re.findall(r'=\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if matches:
        return _clean_number(matches[-1])

    # 3. Textual "The answer is N" / "final answer is N" format
    match = re.search(
        r'(?:the answer is|final answer is|therefore,?\s*(?:the )?\w+ is|she makes|he makes|profit is|total is)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        text, flags=re.IGNORECASE
    )
    if match:
        return _clean_number(match.group(1))

    # 4. Dollar amount format: $N (common in GSM8K financial questions)
    matches = re.findall(r'\$\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if matches:
        return _clean_number(matches[-1])

    # 5. Fallback: absolute last standalone number in the text
    matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if matches:
        return _clean_number(matches[-1])

    return None


def extract_cqa_answer(text: str) -> str:
    """
    Extracts the multiple choice letter (A-E) from a CQA generation.
    Returns the uppercase letter if found, else None.
    """
    text = _strip_think(text)

    # 1. Official format: Answer: A  /  #### A
    match = re.search(r'(?:Answer|####)\s*:?\s*\(?([A-E])\)?', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. Parentheses/bracket format: (A), [A], A), Option A
    matches = re.findall(r'(?:\([A-E]\)|\[[A-E]\]|[A-E]\)|Option\s+[A-E])', text, flags=re.IGNORECASE)
    if matches:
        letter_match = re.search(r'([A-E])', matches[-1], flags=re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

    # 3. "The answer is A" / "correct answer is B" format
    match = re.search(
        r'(?:the answer is|correct answer is|best answer is)\s*\(?([A-E])\)?',
        text, flags=re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # 4. Fallback: The very last standalone capital letter A-E in the text
    matches = re.findall(r'\b([A-E])\b', text)
    if matches:
        return matches[-1].upper()

    return None


def _last_boxed(text: str):
    """Content of the last complete \\boxed{...} (brace-balanced), or None."""
    idx = text.rfind("\\boxed{")
    while idx != -1:
        start_idx = idx + len("\\boxed{")
        depth = 1
        for i in range(start_idx, len(text)):
            ch = text[i]
            escaped = i > 0 and text[i - 1] == '\\'
            if ch == '{' and not escaped:
                depth += 1
            elif ch == '}' and not escaped:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i].strip()
        # Unbalanced (truncated) box: look for an earlier one.
        idx = text.rfind("\\boxed{", 0, idx)
    return None


def extract_math500_answer(text: str) -> str:
    """
    Extracts the answer from a MATH-500 generation.
    It looks for \\boxed{} first, then falls back to other heuristics.
    """
    text = _strip_think(text)

    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed

    # Fallback to GSM8K logic
    return extract_gsm8k_answer(text)


def get_extractor(dataset_name: str):
    name = dataset_name.lower()
    if name == "gsm8k":
        return extract_gsm8k_answer
    elif name == "cqa":
        return extract_cqa_answer
    elif name == "math500":
        return extract_math500_answer
    else:
        # Generic fallback
        return extract_gsm8k_answer
