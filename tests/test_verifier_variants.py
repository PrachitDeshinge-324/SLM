import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.verifier_variants import (
    DEFAULTS, VARIANTS, aggregate, backward_messages, backward_score, mask_conditions, parse_backward,
    parse_response, self_correct_messages, two_way_probability, verification_messages,
)

# Every variant has defaults; the single-prompt ones build a user message holding question and solution.
assert set(DEFAULTS) == set(VARIANTS)
for variant in ("mc", "cot_binary", "cot_score", "critic", "judge5"):
    (message,) = verification_messages(variant, "How many?", "It is 7.\n#### 7")
    assert message["role"] == "user" and "How many?" in message["content"] and "#### 7" in message["content"]
try:
    verification_messages("backward", "q", "s")
    raise AssertionError("expected ValueError")
except ValueError:
    pass

# The output formats the prompts ask for are the ones the parsers read; the last match wins.
assert "50 words" in verification_messages("cot_binary", "q", "s")[0]["content"]
assert "Correctness: correct" in verification_messages("cot_binary", "q", "s")[0]["content"]
assert "1 to 10" in verification_messages("cot_score", "q", "s")[0]["content"]
assert "CRITIC RESULT: SOLUTION IS CORRECT" in verification_messages("critic", "q", "s")[0]["content"]
assert "100 words" in verification_messages("judge5", "q", "s")[0]["content"]
assert parse_response("cot_binary", "The sum is wrong.\nCorrectness: incorrect") is False
assert parse_response("cot_binary", "Fine.\nCorrectness: **correct**") is True
assert parse_response("cot_binary", "Correctness: correct\nOn reflection...\nCorrectness: incorrect") is False
assert parse_response("cot_binary", "no verdict here") is None
assert parse_response("cot_binary", "<think>Correctness: correct</think>\nCorrectness: incorrect") is False
assert parse_response("critic", "Step 2 is off.\nCRITIC RESULT: SOLUTION IS INCORRECT") is False
assert parse_response("critic", "All fine. CRITIC RESULT: SOLUTION IS CORRECT") is True
assert parse_response("cot_score", "Good work.\nScore: 9") == 0.9
assert parse_response("cot_score", "Score: 10/10") == 1.0
assert parse_response("cot_score", "Score: 0") is None and parse_response("cot_score", "Score: 11") is None
assert parse_response("judge5", "Score: 4") == 0.8 and parse_response("judge5", "Score: 0") == 0.0
assert parse_response("judge5", "Score: 6") is None and parse_response("judge5", "") is None

# Aggregation: unparsed samples are left out; the critic decides at the threshold on either side.
a = aggregate("cot_binary", [True, True, False, None])
assert a["n"] == 4 and a["parsed"] == 3 and math.isclose(a["score"], 2 / 3) and "decision" not in a
assert aggregate("critic", [True] * 10 + [False] * 6)["decision"] is True       # 0.625 >= 0.6
assert aggregate("critic", [True] * 9 + [False] * 7)["decision"] is None        # 0.5625: neither side reaches 0.6
assert aggregate("critic", [True] * 6 + [False] * 10)["decision"] is False      # incorrect rate 0.625
assert aggregate("critic", [None, None])["score"] is None
assert math.isclose(aggregate("cot_score", [0.9, 0.5])["score"], 0.7)

# P(correct) from two log-probabilities.
assert math.isclose(two_way_probability(-1.0, -1.0), 0.5)
assert math.isclose(two_way_probability(-0.1, -3.0), 1 / (1 + math.exp(-2.9)))
assert two_way_probability(-50.0, -0.0) < 1e-20 and two_way_probability(-0.0, -1000.0) == 1.0

# Backward verification: every distinct number masked in turn, all its occurrences at once.
question = "Tom has 5 apples and buys 12 more. He gives 5 away. How many does he have?"
conditions = mask_conditions(question)
assert [value for value, _ in conditions] == ["5", "12"]
assert conditions[0][1] == "Tom has X apples and buys 12 more. He gives X away. How many does he have?"
assert conditions[1][1] == "Tom has 5 apples and buys X more. He gives 5 away. How many does he have?"
assert [v for v, _ in mask_conditions("A car costs $1,200 and fuel costs 3.5 per litre.")] == ["1200", "3.5"]
assert len(mask_conditions("1 2 3 4 5 6 7 8", max_conditions=3)) == 3
assert mask_conditions("No numbers here.") == []
assert "X" in backward_messages("Tom has X apples.", "42")[0]["content"] and "is 42" in backward_messages("Tom has X apples.", "42")[0]["content"]
assert parse_backward("So X = 5. The answer is 5", "5") is True
assert parse_backward("The answer is 1,200", "1200") is True
assert parse_backward("The answer is 6", "5") is False
assert parse_backward("I am not sure.", "5") is None and parse_backward("", "5") is None
score = backward_score([[True, False, None, True], [True, True, True, True]])
assert score == {"conditions": 2, "samples": 8, "matches": 6, "score": 0.75}
assert backward_score([])["score"] is None

# Self-correction: the review step, then the improve step with the model's own feedback.
base = [{"role": "user", "content": "Q?"}]
first = self_correct_messages(base, "A.\n#### 3")
assert [m["role"] for m in first] == ["user", "assistant", "user"] and "find problems" in first[-1]["content"]
second = self_correct_messages(base, "A.\n#### 3", feedback="No problems.")
assert [m["role"] for m in second] == ["user", "assistant", "user", "assistant", "user"]
assert "\\boxed{answer}" in second[-1]["content"] and second[3]["content"] == "No problems."

# Selecting generator records by question keeps the order of the question list.
from src.experiment_d.question_sets import records_in
records = [{"question": q, "qid": f"id{i}"} for i, q in enumerate(["a", "b", "c"])]
assert [r["qid"] for r in records_in(records, ["c", "a"])] == ["id2", "id0"]
try:
    records_in(records, ["a", "z"])
    raise AssertionError("expected ValueError")
except ValueError:
    pass

print("All verifier variant tests passed!")
