import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.extractors import extract_gsm8k_answer, extract_cqa_answer

def test_gsm8k_extractor():
    # 1. Standard #### format
    assert extract_gsm8k_answer("The steps are correct. #### 72") == "72"
    assert extract_gsm8k_answer("#### $18") == "18"
    # 2. Equation fallback
    assert extract_gsm8k_answer("So 48 + 24 = 72") == "72"
    assert extract_gsm8k_answer("This means x = -5.5.") == "-5.5"
    # 3. Textual fallback
    assert extract_gsm8k_answer("The final answer is 1,000.") == "1000"
    assert extract_gsm8k_answer("Therefore, she makes $18 per day.") == "18"
    # 4. Dollar amount format
    assert extract_gsm8k_answer("The total cost is $1,200.") == "1200"
    # 5. Thousands separator enforces 3-digit groups
    assert extract_gsm8k_answer("#### 1,000,000") == "1000000"
    # 6. Last standalone number fallback
    assert extract_gsm8k_answer("Natalia sold 48 clips and then 24. She sold 72.") == "72"
    # 7. None
    assert extract_gsm8k_answer("I don't know the answer to this math problem.") == None

def test_cqa_extractor():
    # 1. Standard Answer: format
    assert extract_cqa_answer("Therefore, Answer: B.") == "B"
    assert extract_cqa_answer("Answer: c") == "C"
    # 2. #### format (model sometimes mimics GSM8K style)
    assert extract_cqa_answer("#### A") == "A"
    # 3. Parentheses/Option fallback
    assert extract_cqa_answer("The correct choice is (A).") == "A"
    assert extract_cqa_answer("I would choose Option E.") == "E"
    # 4. Bracket format [A]
    assert extract_cqa_answer("The answer is [C].") == "C"
    # 5. "A)" format
    assert extract_cqa_answer("B) is the correct one.") == "B"
    # 6. Textual "the answer is A" format
    assert extract_cqa_answer("The correct answer is D based on reasoning.") == "D"
    assert extract_cqa_answer("The best answer is E.") == "E"
    # 7. Last standalone letter fallback
    assert extract_cqa_answer("It is obvious that the answer is D.") == "D"
    assert extract_cqa_answer("The letter C is the right one") == "C"
    # 8. None
    assert extract_cqa_answer("I think it is the supermarket.") == None

if __name__ == "__main__":
    test_gsm8k_extractor()
    test_cqa_extractor()
    print("All extractor tests passed!")
