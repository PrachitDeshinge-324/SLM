import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.benchmarks import load_benchmark, load_gsm8k_train_answers, question_key
from src.experiment_d.lora_training import IGNORE_INDEX, build_training_example
from src.experiment_d.prompts import GSM8K_1SHOT_USER
from src.experiment_d.run_files import (
    adapter_dir,
    generator_run_path,
    is_self_training_file,
    self_training_round_dir,
)
from src.experiment_d.self_consistency import majority_vote, summarize_votes
from src.experiment_d.self_training import assert_disjoint, select_pseudo_labels, shared_questions


def _sample(response, cutoff=False):
    return {"response": response, "cutoff": cutoff, "extracted": "stale", "length": 10}


def _record(qid, responses, **extra):
    samples = [_sample(*r) if isinstance(r, tuple) else _sample(r) for r in responses]
    return {"qid": qid, "question": f"question {qid}", "raw_samples": samples, **extra}


def test_majority_vote_matches_summarize_votes():
    answers = ["18", None, "18.00", "20", "20", "18"]
    answer, key, count, keys = majority_vote(answers)
    assert (answer, count) == ("18", 3)
    assert keys[1] is None and keys[0] == keys[2] == key
    # Ties go to the answer seen first, in the vote and in the summary alike.
    assert majority_vote(["20", "18", "18", "20"])[0] == "20"
    assert summarize_votes(["20", "18", "18", "20"], "18")["majority_answer"] == "20"
    assert majority_vote([None, None]) == (None, None, 0, [None, None])


def test_threshold_is_share_of_all_samples():
    six_of_eight = _record("a", ["x\n#### 5"] * 3 + ["y\n#### 5", "z\n#### 5", "w\n#### 5", "#### 9", "#### 7"])
    five_of_eight = _record("b", ["x\n#### 5", "y\n#### 5", "z\n#### 5", "w\n#### 5", "v\n#### 5", "#### 9", "#### 7", "#### 8"])
    selected = select_pseudo_labels([six_of_eight, five_of_eight], "gsm8k_train", threshold=0.75)
    assert {item["qid"] for item in selected} == {"a"}
    assert all(item["label"] == "5" and item["agreement"] == 0.75 for item in selected)
    # Only solutions that reach the majority answer, and word-for-word repeats once.
    assert [item["trace"] for item in selected] == ["x\n#### 5", "y\n#### 5", "z\n#### 5", "w\n#### 5"]


def test_truncated_samples_never_vote_or_train():
    # Six samples say 5, but two of them were cut off: only four votes out of eight remain.
    cut = _record("a", ["a\n#### 5", "b\n#### 5", "c\n#### 5", "d\n#### 5", ("e\n#### 5", True), ("f\n#### 5", True), "#### 9", "#### 7"])
    assert select_pseudo_labels([cut], "gsm8k_train", threshold=0.75) == []
    # Seven say 5, one of them cut off: six complete votes keep the question, the cut-off text is not used.
    kept = _record("b", ["a\n#### 5", "b\n#### 5", "c\n#### 5", "d\n#### 5", "e\n#### 5", "f\n#### 5", ("g\n#### 5", True), "#### 7"])
    traces = [item["trace"] for item in select_pseudo_labels([kept], "gsm8k_train", threshold=0.75)]
    assert len(traces) == 6 and "g\n#### 5" not in traces


def test_selection_never_reads_reference_answers():
    responses = ["a\n#### 5", "b\n#### 5", "c\n#### 5", "d\n#### 5", "e\n#### 5", "f\n#### 5", "#### 9", "#### 7"]
    plain = _record("a", responses)
    # A wrong reference answer and stale stored metrics must change nothing.
    poisoned = _record("a", responses, ground_truth="9", metrics={"majority_answer": "9", "correctness": True})
    assert select_pseudo_labels([plain], "gsm8k_train") == select_pseudo_labels([poisoned], "gsm8k_train")


def _expect_leak(pool, eval_questions):
    try:
        assert_disjoint(pool, eval_questions)
    except ValueError:
        return
    raise AssertionError("a shared question was not caught")


def test_pool_must_not_share_questions_with_the_test_or_the_prompt():
    pool = ["How many apples are left?", "A train leaves at noon."]
    assert_disjoint(pool, ["Something else entirely?"])
    # The same question with other case and spacing in the test set.
    _expect_leak(pool, ["how many  APPLES are left?"])
    # The prompt's worked example in the pool.
    _expect_leak(pool + [GSM8K_1SHOT_USER], [])
    assert shared_questions(pool, ["A train leaves at noon."]) == ["A train leaves at noon."]


def test_train_pool_has_no_answers_and_is_separate_from_the_test_set():
    pool = load_benchmark("gsm8k_train")
    test = load_benchmark("gsm8k")
    assert len(pool) == 7472  # the 7,473 train questions minus the one shown in the prompt
    assert all(item["ground_truth"] is None and item["raw_answer_str"] is None for item in pool)
    assert all(item["qid"].startswith("gsm8k_train_") and item["dataset"] == "gsm8k_train" for item in pool)
    assert question_key(GSM8K_1SHOT_USER) not in {question_key(item["question"]) for item in pool}
    # The real splits share no question, by text.
    assert_disjoint([item["question"] for item in pool], [item["question"] for item in test])
    assert not {item["qid"] for item in pool} & {item["qid"] for item in test}
    # The answers exist only behind the report-only loader.
    assert load_gsm8k_train_answers()["gsm8k_train_1"] == "10"


def test_round_paths():
    round_dir = self_training_round_dir("results", "pilot", 2)
    assert str(round_dir) == "results/self_training/pilot/round2"
    assert str(adapter_dir("results", "pilot", 2, "Qwen/Qwen3.5-0.8B", "4bit")) == \
        "results/self_training/pilot/round2/adapters/Qwen_Qwen3.5-0.8B_4bit"
    pool_file = generator_run_path(round_dir, "Qwen/Qwen3.5-0.8B", "4bit", "gsm8k_train", 8)
    assert str(pool_file).endswith("round2/gsm8k_train/generator/Qwen_Qwen3.5-0.8B/Qwen_Qwen3.5-0.8B_4bit_gsm8k_train_n8.jsonl")
    assert is_self_training_file(pool_file)
    assert not is_self_training_file("results/gsm8k/generator/Qwen_Qwen3.5-0.8B/Qwen_Qwen3.5-0.8B_4bit_gsm8k_n8.jsonl")


class _FakeTokenizer:
    """One token per character; the chat template writes the question between markers."""
    name_or_path = "fake"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return "<" + messages[-1]["content"] + ">"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


def test_training_example_scores_only_the_solution():
    example = build_training_example(_FakeTokenizer(), "gsm8k_train", "Q", "ab", end_token=0, max_length=10)
    assert example["input_ids"] == [ord("<"), ord("Q"), ord(">"), ord("a"), ord("b"), 0]
    assert example["labels"] == [IGNORE_INDEX] * 3 + [ord("a"), ord("b"), 0]
    # A solution that does not fit is skipped whole, never cut.
    assert build_training_example(_FakeTokenizer(), "gsm8k_train", "Q", "ab", end_token=0, max_length=5) is None


def test_adapter_training_on_a_tiny_model():
    """A fresh adapter changes nothing; training lowers the loss; a saved adapter reloads to the same outputs."""
    import tempfile

    import torch
    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM

    from src.experiment_d.lora_training import attach_lora, train_lora
    from src.experiment_d.run_files import adapter_fingerprint

    def tiny_model():
        torch.manual_seed(0)
        return LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                                            num_hidden_layers=2, num_attention_heads=2)).eval()

    tokenizer = _FakeTokenizer()
    tokenizer.pad_token_id = 1
    examples = [build_training_example(tokenizer, "gsm8k_train", question, trace, end_token=0, max_length=64)
                for question, trace in (("Q1", "one two"), ("Q2", "three four"), ("Q3", "five six"))]
    probe = torch.tensor([examples[0]["input_ids"]])

    base = tiny_model()
    with torch.no_grad():
        base_logits = base(input_ids=probe).logits
    model, target_modules = attach_lora(base, rank=4, alpha=8, dropout=0.0)
    assert target_modules and "lm_head" not in target_modules
    assert all("lora_" in name for name, parameter in model.named_parameters() if parameter.requires_grad)

    assert train_lora(model, tokenizer, examples, steps=0) == []
    with torch.no_grad():
        assert torch.equal(model(input_ids=probe).logits, base_logits)

    losses = train_lora(model, tokenizer, examples, steps=40, batch_size=2, micro_batch_size=1, learning_rate=1e-2, log_every=1000)
    assert len(losses) == 40 and sum(losses[-5:]) < sum(losses[:5])
    assert not model.training
    with torch.no_grad():
        trained_logits = model(input_ids=probe).logits
    assert not torch.equal(trained_logits, base_logits)

    with tempfile.TemporaryDirectory() as folder:
        model.save_pretrained(folder)
        assert len(adapter_fingerprint(folder)) == 64
        reloaded = PeftModel.from_pretrained(tiny_model(), folder).eval()
        with torch.no_grad():
            assert torch.allclose(reloaded(input_ids=probe).logits, trained_logits, atol=1e-6)


def test_trained_run_cannot_resume_into_an_untrained_file():
    from scripts.generate_answers import check_resume_is_valid

    old_header = {"type": "run_config", "model": "m", "precision": "4bit", "batch_size": 64}  # written before adapters existed
    check_resume_is_valid("f", old_header, {"q1"}, {"model": "m", "precision": "4bit", "adapter_sha256": None}, 64)
    try:
        check_resume_is_valid("f", old_header, {"q1"}, {"model": "m", "precision": "4bit", "adapter_sha256": "abc"}, 64)
    except RuntimeError as error:
        assert "adapter_sha256" in str(error)
    else:
        raise AssertionError("an adapter run was allowed to resume an untrained run file")


if __name__ == "__main__":
    test_majority_vote_matches_summarize_votes()
    test_threshold_is_share_of_all_samples()
    test_truncated_samples_never_vote_or_train()
    test_selection_never_reads_reference_answers()
    test_pool_must_not_share_questions_with_the_test_or_the_prompt()
    test_train_pool_has_no_answers_and_is_separate_from_the_test_set()
    test_round_paths()
    test_training_example_scores_only_the_solution()
    test_adapter_training_on_a_tiny_model()
    test_trained_run_cannot_resume_into_an_untrained_file()
    print("All self-training tests passed!")
