import re

def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the numeric answer from a GSM8K generation using multi-stage fallbacks.
    Returns the numeric string if found, else None.
    """
    text = text.strip()
    
    # 1. Official format: #### N
    match = re.search(r'####\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
        
    # 2. Equation end format: = N (take the last one)
    matches = re.findall(r'=\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if matches:
        return matches[-1].replace(',', '')
        
    # 3. Textual "The answer is N" / "final answer is N" format
    match = re.search(
        r'(?:the answer is|final answer is|therefore,?\s*(?:the )?\w+ is|she makes|he makes|profit is|total is)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        text, flags=re.IGNORECASE
    )
    if match:
        return match.group(1).replace(',', '')

    # 4. Dollar amount format: $N (common in GSM8K financial questions)
    matches = re.findall(r'\$\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if matches:
        return matches[-1].replace(',', '')
        
    # 5. Fallback: absolute last standalone number in the text
    matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if matches:
        return matches[-1].replace(',', '')
        
    return None

def extract_cqa_answer(text: str) -> str:
    """
    Extracts the multiple choice letter (A-E) from a CQA generation.
    Returns the uppercase letter if found, else None.
    """
    text = text.strip()
    
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
