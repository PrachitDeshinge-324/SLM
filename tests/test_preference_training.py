import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.experiment_d.lora_training import (
    IGNORE_INDEX,
    _collate,
    attach_lora,
    build_preference_example,
    dpo_loss,
    mean_token_logps,
    reference_logps,
    token_logps,
    train_lora_dpo,
)
from src.experiment_d.self_training import curriculum_pairs, rotate_pairs, select_preference_pairs


def _sample(response, length, cutoff=False):
    return {"response": response, "cutoff": cutoff, "extracted": "stale", "length": length}


def _record(qid, samples, **extra):
    return {"qid": qid, "question": f"question {qid}", "raw_samples": [_sample(*s) for s in samples], **extra}


# Five of eight say 5 (lengths 100..500), three say other answers (lengths 120, 480, 300).
FIVE_OF_EIGHT = [("a\n#### 5", 100), ("b\n#### 5", 200), ("c\n#### 5", 300), ("d\n#### 5", 400), ("e\n#### 5", 500),
                 ("f\n#### 9", 120), ("g\n#### 7", 480), ("h\n#### 8", 290)]


def test_similar_length_pairs_the_closest_lengths():
    pair, = select_preference_pairs([_record("q", FIVE_OF_EIGHT)], "gsm8k_train")
    assert (pair["chosen"], pair["rejected"]) == ("c\n#### 5", "h\n#### 8")
    assert (pair["label"], pair["rejected_answer"], pair["agreement"]) == ("5", "8", 0.625)
    assert (pair["chosen_length"], pair["rejected_length"]) == (300, 290)


def test_longest_shortest_is_the_length_confounded_ablation():
    pair, = select_preference_pairs([_record("q", FIVE_OF_EIGHT)], "gsm8k_train", pairing="longest_shortest")
    assert (pair["chosen"], pair["rejected"]) == ("e\n#### 5", "f\n#### 9")
    try:
        select_preference_pairs([], "gsm8k_train", pairing="longest")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown pairing was accepted")


def test_threshold_is_five_of_eight_of_all_samples():
    four_of_eight = [("a\n#### 5", 1), ("b\n#### 5", 2), ("c\n#### 5", 3), ("d\n#### 5", 4),
                     ("e\n#### 9", 5), ("f\n#### 9", 6), ("g\n#### 7", 7), ("h\n#### 8", 8)]
    assert select_preference_pairs([_record("q", four_of_eight)], "gsm8k_train") == []
    assert len(select_preference_pairs([_record("q", four_of_eight)], "gsm8k_train", threshold=0.5)) == 1


def test_answers_are_compared_after_normalisation():
    # '5.00' is the majority answer written differently: it must not become the rejected solution.
    samples = FIVE_OF_EIGHT[:5] + [("f\n#### 5.00", 120), ("g\n#### 7", 480), ("h\n#### 5.0", 290)]
    pair, = select_preference_pairs([_record("q", samples)], "gsm8k_train")
    assert pair["rejected"] == "g\n#### 7" and pair["agreement"] == 7 / 8


def test_truncated_and_unanimous_questions():
    # Five say 5 but one of them was cut off: four votes, below 5/8.
    cut = FIVE_OF_EIGHT[:4] + [("e\n#### 5", 512, True)] + FIVE_OF_EIGHT[5:]
    assert select_preference_pairs([_record("q", cut)], "gsm8k_train") == []
    # The only minority solutions are cut off: nothing to reject.
    no_minority = FIVE_OF_EIGHT[:5] + [(text, 512, True) for text, _ in FIVE_OF_EIGHT[5:]]
    assert select_preference_pairs([_record("q", no_minority)], "gsm8k_train") == []
    unanimous = [(f"{i}\n#### 5", i) for i in range(8)]
    assert select_preference_pairs([_record("q", unanimous)], "gsm8k_train") == []


def test_pairing_never_reads_reference_answers():
    plain = _record("q", FIVE_OF_EIGHT)
    poisoned = _record("q", FIVE_OF_EIGHT, ground_truth="8", metrics={"majority_answer": "8", "correctness": True})
    for pairing in ("similar_length", "longest_shortest"):
        assert select_preference_pairs([plain], "gsm8k_train", pairing=pairing) == \
            select_preference_pairs([poisoned], "gsm8k_train", pairing=pairing)


class _FakeTokenizer:
    """One token per character; the chat template writes the question between markers."""
    name_or_path = "fake"
    pad_token_id = 1

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return "<" + messages[-1]["content"] + ">"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=2, num_attention_heads=2)).eval()


def test_preference_example_uses_the_generation_prompt():
    tokenizer = _FakeTokenizer()
    pair = build_preference_example(tokenizer, "gsm8k_train", "Q", "ab", "xyz", end_token=0, max_length=7)
    assert pair["chosen"]["labels"] == [IGNORE_INDEX] * 3 + [ord("a"), ord("b"), 0]
    assert pair["rejected"]["input_ids"] == [ord("<"), ord("Q"), ord(">"), ord("x"), ord("y"), ord("z"), 0]
    # Either side too long drops the whole pair.
    assert build_preference_example(tokenizer, "gsm8k_train", "Q", "ab", "xyz", end_token=0, max_length=6) is None


def test_mean_logps_match_a_token_by_token_loop():
    """Masking, the shift and right padding: a short row padded next to a long one scores as if alone."""
    model, tokenizer = _tiny_model(), _FakeTokenizer()
    examples = [build_preference_example(tokenizer, "gsm8k_train", "Q", "a long solution", "no", end_token=0, max_length=64)[side]
                for side in ("chosen", "rejected")]
    input_ids, attention_mask, labels = _collate(examples, tokenizer.pad_token_id, torch.device("cpu"))
    with torch.no_grad():
        batched = mean_token_logps(model, input_ids, attention_mask, labels)
        for row, example in enumerate(examples):
            logps = torch.log_softmax(model(input_ids=torch.tensor([example["input_ids"]])).logits[0].float(), dim=-1)
            scored = [(position, token) for position, token in enumerate(example["labels"]) if token != IGNORE_INDEX]
            manual = sum(logps[position - 1, token].item() for position, token in scored) / len(scored)
            assert math.isclose(batched[row].item(), manual, abs_tol=1e-5), (row, batched[row].item(), manual)


def test_dpo_loss_is_log_two_at_equal_scores():
    zero = torch.zeros(3)
    loss, margin = dpo_loss(zero - 1.0, zero - 2.0, zero - 1.0, zero - 2.0, beta=2.0)
    assert torch.allclose(loss, torch.full((3,), math.log(2))) and torch.equal(margin, zero)
    loss, margin = dpo_loss(torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-1.5]), torch.tensor([-1.5]), beta=2.0)
    assert math.isclose(margin.item(), 1.0) and math.isclose(loss.item(), -math.log(1 / (1 + math.exp(-2.0))), rel_tol=1e-6)


def test_single_pair_overfit():
    """
    End to end on one pair: the base scored as reference gives margin 0 and loss ln 2 once the fresh
    adapter is on; 50 updates on that pair raise the margin and lower the loss.
    """
    model, tokenizer = _tiny_model(), _FakeTokenizer()
    pair = build_preference_example(tokenizer, "gsm8k_train", "Q2", "three four", "seven", end_token=0, max_length=64)
    (pair["ref_chosen"], pair["ref_rejected"]), = reference_logps(model, tokenizer, [pair])
    model, _ = attach_lora(model, rank=4, alpha=8, dropout=0.0)

    start = train_lora_dpo(model, tokenizer, [pair], steps=1, batch_size=1, learning_rate=0.0, log_every=1000)
    assert abs(start["margins"][0]) < 1e-5 and math.isclose(start["losses"][0], math.log(2), abs_tol=1e-5)

    history = train_lora_dpo(model, tokenizer, [pair], steps=50, batch_size=1, learning_rate=1e-2, beta=2.0, log_every=1000)
    assert history["margins"][-1] > 0.5, history["margins"][-1]
    assert history["losses"][-1] < 0.5 * math.log(2), history["losses"][-1]
    assert history["preferred"][-1] == 1.0 and not model.training


def test_several_distinct_pairs_per_question():
    pairs = select_preference_pairs([_record("q", FIVE_OF_EIGHT)], "gsm8k_train", pairs_per_question=3)
    assert len(pairs) == 3
    # The first pair is the single-pair choice; later ones reuse neither solution.
    assert (pairs[0]["chosen"], pairs[0]["rejected"]) == ("c\n#### 5", "h\n#### 8")
    assert len({p["chosen"] for p in pairs}) == 3 and len({p["rejected"] for p in pairs}) == 3
    # Only three minority solutions exist, so a fourth pair is not invented.
    assert len(select_preference_pairs([_record("q", FIVE_OF_EIGHT)], "gsm8k_train", pairs_per_question=9)) == 3
    assert select_preference_pairs([_record("q", FIVE_OF_EIGHT)], "gsm8k_train", pairs_per_question=1) == pairs[:1]
    try:
        select_preference_pairs([], "gsm8k_train", pairing="longest_shortest", pairs_per_question=2)
    except ValueError:
        pass
    else:
        raise AssertionError("several pairs with longest_shortest were accepted")


def test_margin_and_tie_skipping():
    pair, = select_preference_pairs([_record("q", FIVE_OF_EIGHT)], "gsm8k_train")
    assert pair["margin"] == (5 - 1) / 8                       # 5 votes against the runner-up's 1
    tie = [("a\n#### 5", 1), ("b\n#### 5", 2), ("c\n#### 5", 3), ("d\n#### 5", 4),
           ("e\n#### 9", 5), ("f\n#### 9", 6), ("g\n#### 9", 7), ("h\n#### 9", 8)]
    kept, = select_preference_pairs([_record("q", tie)], "gsm8k_train", threshold=0.5)
    assert kept["margin"] == 0.0
    assert select_preference_pairs([_record("q", tie)], "gsm8k_train", threshold=0.5, skip_ties=True) == []
    assert len(select_preference_pairs([_record("q", FIVE_OF_EIGHT)], "gsm8k_train", skip_ties=True)) == 1


def test_rotation_gives_each_epoch_another_pair():
    pairs = [{"qid": "a", "n": 0}, {"qid": "a", "n": 1}, {"qid": "b", "n": 0}]
    schedule = rotate_pairs(pairs, 3)
    assert [(p["qid"], p["n"]) for p in schedule] == [("a", 0), ("b", 0), ("a", 1), ("b", 0), ("a", 0), ("b", 0)]
    assert schedule[1] is schedule[3] is pairs[2]              # shared dicts: scored once as reference


def test_end_tokens_are_left_out_of_the_preference_labels_only():
    tokenizer = _FakeTokenizer()
    pair = build_preference_example(tokenizer, "gsm8k_train", "Q", "ab", "xyz", end_token=[7, 8], max_length=64, mask_end=True)
    assert pair["chosen"]["labels"] == [IGNORE_INDEX] * 3 + [ord("a"), ord("b"), 7, 8]
    assert pair["chosen"]["pref_labels"] == [IGNORE_INDEX] * 3 + [ord("a"), ord("b"), IGNORE_INDEX, IGNORE_INDEX]
    assert pair["rejected"]["pref_labels"][-2:] == [IGNORE_INDEX] * 2 and pair["rejected"]["labels"][-2:] == [7, 8]
    assert "pref_labels" not in build_preference_example(tokenizer, "gsm8k_train", "Q", "ab", "xyz", 0, 64)["chosen"]


def test_chunked_scores_match_full_scores():
    model, tokenizer = _tiny_model(), _FakeTokenizer()
    examples = [build_preference_example(tokenizer, "gsm8k_train", "Q", "a long solution", "no", end_token=0, max_length=64)[side]
                for side in ("chosen", "rejected")]
    input_ids, attention_mask, labels = _collate(examples, tokenizer.pad_token_id, torch.device("cpu"))
    full = token_logps(model, input_ids, attention_mask, labels)
    chunked = token_logps(model, input_ids, attention_mask, labels, chunked=True)
    assert torch.allclose(full, chunked, atol=1e-5) and (full[labels[:, 1:] == IGNORE_INDEX] == 0).all()
    assert torch.allclose(mean_token_logps(model, input_ids, attention_mask, labels),
                          mean_token_logps(model, input_ids, attention_mask, labels, chunked=True), atol=1e-5)


def _pair(tokenizer, model, **kwargs):
    pair = build_preference_example(tokenizer, "gsm8k_train", "Q2", "three four", "seven", end_token=[7, 8], max_length=64, **kwargs)
    (pair["ref_chosen"], pair["ref_rejected"]), = reference_logps(model, tokenizer, [pair])
    return pair


def test_nll_term_weights_and_chunked_loss():
    """
    At the start the DPO loss is ln 2 and the chosen NLL is the base's, so the total is
    weight * (ln 2 + nll_weight * nll); a chunked run takes the same first step as a full one.
    """
    tokenizer = _FakeTokenizer()
    results = {}
    for name, kwargs in (("plain", {}), ("nll_weight", {"nll_weight": 1.0, "weight": 2.0}),
                         ("chunked", {"nll_weight": 1.0, "weight": 2.0, "chunked_loss": True})):
        model = _tiny_model()
        pair = _pair(tokenizer, model, mask_end=True)
        pair["weight"] = kwargs.pop("weight", 1.0)
        model, _ = attach_lora(model, rank=4, alpha=8, dropout=0.0)
        results[name] = train_lora_dpo(model, tokenizer, [pair], steps=1, batch_size=1, learning_rate=0.0, log_every=1000, **kwargs)
    nll = results["plain"]["nll"][0]
    assert nll > 0 and math.isclose(results["plain"]["losses"][0], math.log(2), abs_tol=1e-5)
    assert math.isclose(results["nll_weight"]["losses"][0], 2.0 * (math.log(2) + nll), rel_tol=1e-4)
    assert math.isclose(results["chunked"]["losses"][0], results["nll_weight"]["losses"][0], rel_tol=1e-4)
    assert results["plain"]["end_chosen"][0] < 0 and results["plain"]["end_rejected"][0] < 0   # logged, never trained on
    assert abs(results["plain"]["chosen_shift"][0]) < 1e-5


def test_masked_end_is_not_pushed_down_but_chosen_nll_raises_it():
    """Same pair trained with and without the end in the preference score: the rejected end is the difference."""
    tokenizer = _FakeTokenizer()
    ends = {}
    for mask_end in (False, True):
        model = _tiny_model()
        pair = _pair(tokenizer, model, mask_end=mask_end)
        model, _ = attach_lora(model, rank=4, alpha=8, dropout=0.0)
        history = train_lora_dpo(model, tokenizer, [pair], steps=60, batch_size=1, learning_rate=1e-2, beta=2.0, log_every=1000)
        ends[mask_end] = history["end_rejected"][-1] - history["end_rejected"][0]
    assert ends[False] < ends[True] - 0.05, ends        # with the end scored, the rejected stop falls further


def test_nll_term_raises_the_chosen_solution():
    tokenizer = _FakeTokenizer()
    shifts = {}
    for nll_weight in (0.0, 5.0):
        model = _tiny_model()
        pair = _pair(tokenizer, model, mask_end=True)
        model, _ = attach_lora(model, rank=4, alpha=8, dropout=0.0)
        history = train_lora_dpo(model, tokenizer, [pair], steps=40, batch_size=1, learning_rate=5e-3, beta=2.0,
                                 log_every=1000, nll_weight=nll_weight)
        shifts[nll_weight] = history["chosen_shift"][-1]
    assert shifts[5.0] > shifts[0.0], shifts


def test_curriculum_adds_harder_questions_stage_by_stage():
    pairs = [{"qid": "easy", "n": 0, "agreement": 0.75}, {"qid": "easy", "n": 1, "agreement": 0.75},
             {"qid": "mid", "n": 0, "agreement": 0.625}, {"qid": "hard", "n": 0, "agreement": 0.5},
             {"qid": "hard", "n": 1, "agreement": 0.5}]
    schedule, sizes = curriculum_pairs(pairs, [0.75, 0.625, 0.5])
    assert sizes == [1, 2, 3]
    # A question starts at its first pair when it joins, then takes its next one.
    assert [(p["qid"], p["n"]) for p in schedule] == [("easy", 0), ("easy", 1), ("mid", 0), ("easy", 0), ("mid", 0), ("hard", 0)]
    try:
        curriculum_pairs(pairs, [0.5, 0.75])
    except ValueError:
        pass
    else:
        raise AssertionError("rising thresholds were accepted")


def test_training_meets_the_stages_in_order():
    """Same seed, one pass: stage sizes 2 + 2 keep the first two pairs in the first two steps."""
    tokenizer = _FakeTokenizer()
    model = _tiny_model()
    pairs = [_pair(tokenizer, model, mask_end=True) for _ in range(4)]
    for index, pair in enumerate(pairs):
        pair["ref_chosen"] += 0.0 if index < 2 else 1000.0        # stage 2 shows up as a huge margin
    model, _ = attach_lora(model, rank=4, alpha=8, dropout=0.0)
    history = train_lora_dpo(model, tokenizer, pairs, steps=4, batch_size=1, learning_rate=0.0, log_every=1000, stage_sizes=[2, 2])
    assert all(abs(margin) < 1e-4 for margin in history["margins"][:2]), history["margins"]
    assert all(margin < -500 for margin in history["margins"][2:]), history["margins"]
    try:
        train_lora_dpo(model, tokenizer, pairs, steps=1, batch_size=1, learning_rate=0.0, log_every=1000, stage_sizes=[1, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("stage sizes that do not cover the pairs were accepted")


if __name__ == "__main__":
    test_similar_length_pairs_the_closest_lengths()
    test_longest_shortest_is_the_length_confounded_ablation()
    test_threshold_is_five_of_eight_of_all_samples()
    test_answers_are_compared_after_normalisation()
    test_truncated_and_unanimous_questions()
    test_pairing_never_reads_reference_answers()
    test_preference_example_uses_the_generation_prompt()
    test_mean_logps_match_a_token_by_token_loop()
    test_dpo_loss_is_log_two_at_equal_scores()
    test_single_pair_overfit()
    test_several_distinct_pairs_per_question()
    test_margin_and_tie_skipping()
    test_rotation_gives_each_epoch_another_pair()
    test_end_tokens_are_left_out_of_the_preference_labels_only()
    test_chunked_scores_match_full_scores()
    test_nll_term_weights_and_chunked_loss()
    test_masked_end_is_not_pushed_down_but_chosen_nll_raises_it()
    test_nll_term_raises_the_chosen_solution()
    test_curriculum_adds_harder_questions_stage_by_stage()
    test_training_meets_the_stages_in_order()
    print("All preference training tests passed!")
