import re

def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the numeric answer from a GSM8K generation using multi-stage fallbacks.
    Returns the numeric string if found, else None.
    """
    text = text.strip()
    
    # 1. Official format: #### N
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
        
    # 2. Equation end format: = N
    matches = re.findall(r'=\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if matches:
        return matches[-1].replace(',', '')
        
    # 3. Textual "The answer is N" format
    match = re.search(r'(?:the answer is|final answer is|therefore,.*?is)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
        
    # 4. Fallback: absolute last standalone number in the text
    matches = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if matches:
        return matches[-1].replace(',', '')
        
    return None

def extract_cqa_answer(text: str) -> str:
    """
    Extracts the multiple choice letter (A-E) from a CQA generation.
    Returns the uppercase letter if found, else None.
    """
    text = text.strip()
    
    # 1. Official format: Answer: A
    match = re.search(r'Answer:\s*([A-E])\b', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # 2. Parentheses format: (A) or Option A
    matches = re.findall(r'(?:\([A-E]\)|Option\s+[A-E])', text, flags=re.IGNORECASE)
    if matches:
        letter_match = re.search(r'([A-E])', matches[-1], flags=re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
            
    # 3. Fallback: The very last standalone capital letter A-E in the text.
    matches = re.findall(r'\b([A-E])\b', text)
    if matches:
        return matches[-1].upper()
        
    return None
