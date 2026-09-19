from src.experiment_d.extractors import extract_gsm8k_answer, extract_cqa_answer

def test_gsm8k_extractor():
    # 1. Standard
    assert extract_gsm8k_answer("The steps are correct. #### 72") == "72"
    # 2. Equation fallback
    assert extract_gsm8k_answer("So 48 + 24 = 72") == "72"
    assert extract_gsm8k_answer("This means x = -5.5.") == "-5.5"
    # 3. Textual fallback
    assert extract_gsm8k_answer("The final answer is 1,000.") == "1000"
    # 4. Last standalone number fallback
    assert extract_gsm8k_answer("Natalia sold 48 clips and then 24. She sold 72.") == "72"
    # 5. None
    assert extract_gsm8k_answer("I don't know the answer to this math problem.") == None

def test_cqa_extractor():
    # 1. Standard
    assert extract_cqa_answer("Therefore, Answer: B.") == "B"
    assert extract_cqa_answer("Answer: c") == "C"
    # 2. Parentheses/Option fallback
    assert extract_cqa_answer("The correct choice is (A).") == "A"
    assert extract_cqa_answer("I would choose Option E.") == "E"
    # 3. Last standalone letter fallback
    assert extract_cqa_answer("It is obvious that the answer is D.") == "D"
    assert extract_cqa_answer("The letter C is the right one") == "C"
    # 4. None
    assert extract_cqa_answer("I think it is the supermarket.") == None

if __name__ == "__main__":
    test_gsm8k_extractor()
    test_cqa_extractor()
    print("All extractor tests passed!")
