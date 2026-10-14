import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.sampling import closed_boxed_end, truncate_after_first_boxed
from src.experiment_d.vllm_sampling import outputs_to_samples, sample_responses_vllm, stop_at_boxed_answer

STOP = [10, 11]
MAX_NEW = 6


class _Completion:
    def __init__(self, text, token_ids, finish_reason):
        self.text, self.token_ids, self.finish_reason = text, token_ids, finish_reason


class _Tokenizer:
    """One token per character."""

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [[ord(c) for c in t] for t in text]}
        return {"input_ids": [ord(c) for c in text]}


def test_lengths_follow_the_hugging_face_convention():
    texts, lengths = outputs_to_samples([
        _Completion("a", [1, 2, 3, 10], "stop"),          # ended on a stop token: it is not counted
        _Completion("b", [1, 2, 3, 4, 5, 6], "length"),   # ran out of tokens: cut off
        _Completion("c", [1, 2, 3, 4, 5, 11], "stop"),    # stop token as the last allowed token: not cut off
        _Completion("d", [], "stop"),
    ], MAX_NEW, STOP)
    assert texts == ["a", "b", "c", "d"]
    assert lengths == [3, MAX_NEW, 5, 0]
    # The record builder marks cut-off samples as length >= max_new_tokens.
    assert [length >= MAX_NEW for length in lengths] == [False, True, False, False]


def _old_stop_check(text):
    """The StopAfterBoxedAnswer decision as it was written before the scan became a helper."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return False
    depth = 0
    for character in text[start + len("\\boxed{") - 1:]:
        if character == "{" and (depth == 0 or character != "\\"):
            depth += 1
        elif character == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                break
    return depth == 0


def test_boxed_helper_matches_the_original_stop_rule():
    rng = random.Random(0)
    pieces = ["\\boxed{", "{", "}", "12", " ", "\\frac", "x", "\\\\", "}}"]
    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        assert (closed_boxed_end(text) is not None) == _old_stop_check(text), repr(text)
    assert closed_boxed_end("so \\boxed{\\frac{1}{2}} done") == len("so \\boxed{\\frac{1}{2}}")
    assert closed_boxed_end("\\boxed{12") is None and closed_boxed_end("no answer") is None


def test_truncation_after_the_first_closed_boxed_answer():
    assert truncate_after_first_boxed("a \\boxed{3} then \\boxed{4}") == "a \\boxed{3}"
    assert truncate_after_first_boxed("an \\boxed{ open one, then \\boxed{7} end") == "an \\boxed{ open one, then \\boxed{7}"
    assert truncate_after_first_boxed("\\boxed{5") is None and truncate_after_first_boxed("plain") is None
    texts, lengths = stop_at_boxed_answer(["a \\boxed{3} and more", "never closed \\boxed{", "plain"],
                                          [20, 20, 7], _Tokenizer(), max_new_tokens=20)
    assert texts == ["a \\boxed{3}", "never closed \\boxed{", "plain"]
    # The cut text is recounted; a sample with no closed answer keeps its length (and stays cut off).
    assert lengths == [len("a \\boxed{3}"), 20, 7]


def test_seed_count_must_match_prompts():
    try:
        sample_responses_vllm(None, _Tokenizer(), ["a", "b"], [1], 2, 0.7, 50, 0.95, 8, 1.0, STOP)
    except ValueError:
        return
    raise AssertionError("a missing seed was accepted")


def test_resume_never_mixes_engines():
    from scripts.generate_answers import check_resume_is_valid

    base = {"model": "m", "precision": "4bit", "adapter_sha256": None}
    old_header = {"type": "run_config", **base}                        # written before engines existed: hf
    check_resume_is_valid("f", old_header, {"q1"}, {**base, "engine": "hf"}, 64)
    for header, engine in ((old_header, "vllm"), ({**old_header, "engine": "vllm"}, "hf")):
        try:
            check_resume_is_valid("f", header, {"q1"}, {**base, "engine": engine}, 64)
        except RuntimeError as error:
            assert "engine" in str(error)
        else:
            raise AssertionError(f"a run was resumed under another engine ({engine})")
    check_resume_is_valid("f", {**old_header, "engine": "vllm"}, {"q1"}, {**base, "engine": "vllm"}, 64)


def test_vllm_engine_rejects_8bit_and_is_not_imported_eagerly():
    from src.experiment_d.vllm_sampling import load_vllm_engine

    # Importing the module above did not need vllm; the precision is refused before it is imported.
    try:
        load_vllm_engine("m", "8bit")
    except ValueError as error:
        assert "8bit" in str(error)
    else:
        raise AssertionError("8-bit was accepted by the vLLM engine")


if __name__ == "__main__":
    test_lengths_follow_the_hugging_face_convention()
    test_boxed_helper_matches_the_original_stop_rule()
    test_truncation_after_the_first_closed_boxed_answer()
    test_seed_count_must_match_prompts()
    test_resume_never_mixes_engines()
    test_vllm_engine_rejects_8bit_and_is_not_imported_eagerly()
    print("All vLLM sampling tests passed!")
