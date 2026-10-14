"""
Label-free self-training: turn a model's own sampled answers into training text.

Nothing here reads a reference answer. A question is labelled by the majority vote of its samples
and kept only when enough of them agree; the training text is the model's own solutions that reach
that label.
"""
from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.benchmarks import question_key
from src.experiment_d.prompts import GSM8K_1SHOT_USER
from src.experiment_d.self_consistency import majority_vote


def select_pseudo_labels(records: list, dataset_name: str, threshold: float = 0.75) -> list:
    """
    One {qid, question, label, agreement, trace} per solution worth training on.

    records: per-question records of a generator run; only `qid`, `question` and `raw_samples` are read.
    A question is kept when its majority answer is given by at least `threshold` of ALL its samples.
    Truncated samples are partial scratchpads: they never vote and are never training text.
    Solutions that repeat an earlier one word for word are dropped.
    """
    extractor = get_extractor(dataset_name)
    selected = []
    for record in records:
        samples = record["raw_samples"]
        extracted = [None if sample.get("cutoff", False) else extractor(sample["response"]) for sample in samples]
        label, majority_key, count, keys = majority_vote(extracted)
        if label is None or count < threshold * len(samples) - 1e-9:
            continue
        seen = set()
        for sample, key in zip(samples, keys):
            trace = sample["response"]
            if key != majority_key or trace in seen:
                continue
            seen.add(trace)
            selected.append({
                "qid": record["qid"],
                "question": record["question"],
                "label": label,
                "agreement": count / len(samples),
                "trace": trace,
            })
    return selected


def shared_questions(pool_questions: list, eval_questions: list, prompt_example: str = GSM8K_1SHOT_USER) -> list:
    """Pool question texts that are also an evaluation question or the example shown in the prompt."""
    blocked = {question_key(question) for question in eval_questions} | {question_key(prompt_example)}
    return [question for question in pool_questions if question_key(question) in blocked]


def assert_disjoint(pool_questions: list, eval_questions: list, prompt_example: str = GSM8K_1SHOT_USER):
    """Refuse to train when the pool shares a question with the evaluation set or with the prompt."""
    shared = shared_questions(pool_questions, eval_questions, prompt_example)
    if shared:
        raise ValueError(
            f"{len(shared)} pool question(s) also appear in the evaluation set or the prompt, e.g. {shared[0][:80]!r}. "
            "Training on them would leak the test."
        )


PAIRINGS = ("similar_length", "longest_shortest")


def select_preference_pairs(records: list, dataset_name: str, threshold: float = 0.625,
                            pairing: str = "similar_length") -> list:
    """
    At most one {qid, question, label, rejected_answer, agreement, chosen, rejected, chosen_length,
    rejected_length} per question, for preference training on the model's own votes.

    A question is kept when its majority answer is given by at least `threshold` of ALL its samples
    (0.625 = 5 of 8). Chosen is a complete solution reaching the majority answer, rejected a complete
    solution reaching another answer; answers are compared after normalisation ('18' == '18.00').
    Truncated samples never vote and are never paired. Lengths are the generated token counts.

    pairing: 'similar_length' takes the chosen/rejected pair closest in length, so length does not
    separate the two (ties go to the earliest samples); 'longest_shortest' takes the longest majority
    and the shortest minority solution (the length-confounded ablation).
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    extractor = get_extractor(dataset_name)
    pairs = []
    for record in records:
        samples = record["raw_samples"]
        extracted = [None if sample.get("cutoff", False) else extractor(sample["response"]) for sample in samples]
        label, majority_key, count, keys = majority_vote(extracted)
        if label is None or count < threshold * len(samples) - 1e-9:
            continue
        majority, minority, seen = [], [], set()
        for sample, answer, key in zip(samples, extracted, keys):
            if key is None or sample["response"] in seen:
                continue
            seen.add(sample["response"])
            (majority if key == majority_key else minority).append((sample["response"], sample["length"], answer))
        if not majority or not minority:
            continue
        if pairing == "similar_length":
            chosen, rejected = min(((m, n) for m in majority for n in minority), key=lambda pair: abs(pair[0][1] - pair[1][1]))
        else:
            chosen = max(majority, key=lambda item: item[1])
            rejected = min(minority, key=lambda item: item[1])
        pairs.append({
            "qid": record["qid"],
            "question": record["question"],
            "label": label,
            "rejected_answer": rejected[2],
            "agreement": count / len(samples),
            "chosen": chosen[0],
            "rejected": rejected[0],
            "chosen_length": chosen[1],
            "rejected_length": rejected[1],
        })
    return pairs
