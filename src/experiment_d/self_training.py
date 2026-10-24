"""
Label-free self-training: turn a model's own sampled answers into training text.

Nothing here reads a reference answer. A question is labelled by the majority vote of its samples
and kept only when enough of them agree; the training text is the model's own solutions that reach
that label.
"""
from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import normalize_answer
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


LABEL_RULES = ("vote", "critic")


def _answer_keys(samples: list, extractor) -> list:
    """Normalised answer of each sample; None for cut-off samples and samples with no answer."""
    keys = []
    for sample in samples:
        answer = None if sample.get("cutoff", False) else extractor(sample["response"])
        keys.append(None if answer is None else normalize_answer(answer))
    return keys


def critic_weights(keys: list, judged: dict) -> dict:
    """
    Each answer's weight: its votes times the share of its judged solutions the critic approved. An answer
    with no judged solution keeps all its votes (no evidence either way). judged: {sample index: approved}.
    This is the 'critic + vote' rule of the verifier analyses (va.outcomes, 'approved').
    """
    support, approvals = {}, {}
    for index, key in enumerate(keys):
        if key is None:
            continue
        support[key] = support.get(key, 0) + 1
        if index in judged:
            approvals.setdefault(key, []).append(bool(judged[index]))
    return {key: count * (sum(approvals[key]) / len(approvals[key]) if key in approvals else 1.0)
            for key, count in support.items()}


def select_critic_labels(records: list, verdicts: dict, dataset_name: str, threshold: float = 0.75) -> list:
    """
    Like select_pseudo_labels, with the question's label chosen by critic + vote instead of the vote alone.

    verdicts: {(qid, sample index): approved} from a critic run over these records (approved = the critic's
    forced decision is True); solutions it did not judge are absent. Nothing here reads a reference answer.

    A question is kept when the label's weight is at least `threshold` of the total: every answer's weight
    (critic_weights) plus one per sample with no answer. Where no solution was judged (every finished sample
    agrees) this is exactly the vote rule, so the two rules differ only on contested questions, where answers
    the critic rejected stop counting against the label. A tie at the top is skipped, never broken by sample
    order. On a contested question the label must have at least one approved solution. Training text: the
    label's solutions, except those the critic judged and did not approve.
    """
    extractor = get_extractor(dataset_name)
    selected = []
    for record in records:
        samples, qid = record["raw_samples"], record["qid"]
        keys = _answer_keys(samples, extractor)
        judged = {i: verdicts[(qid, i)] for i in range(len(samples)) if (qid, i) in verdicts}
        weights = critic_weights(keys, judged)
        if not weights:
            continue
        top = max(weights.values())
        best = [key for key, weight in weights.items() if abs(weight - top) < 1e-9]
        if len(best) > 1 or top <= 0:
            continue
        label_key = best[0]
        total = sum(weights.values()) + sum(key is None for key in keys)
        if top < threshold * total - 1e-9:
            continue
        contested = len(weights) > 1
        if contested and not any(judged.get(i) for i, key in enumerate(keys) if key == label_key):
            continue
        label = next(extractor(s["response"]) for s, key in zip(samples, keys) if key == label_key)
        seen = set()
        for index, (sample, key) in enumerate(zip(samples, keys)):
            trace = sample["response"]
            if key != label_key or trace in seen or (index in judged and not judged[index]):
                continue
            seen.add(trace)
            selected.append({"qid": qid, "question": record["question"], "label": label,
                             "agreement": top / total, "contested": contested, "trace": trace})
    return selected


def select_critiques(records: list, critic_rows: dict, labels: dict, dataset_name: str) -> list:
    """
    Critiques worth training on, at most one per judged solution: an approving critique of a solution that
    reaches the question's label, or a rejecting one of a solution that reaches another answer. Both kinds
    are kept so the model learns to tell the two apart, not to approve everything.

    critic_rows: {(qid, sample index): critic cell} with `responses` (the critiques, each the continuation of
    the critic's forced start) and `values` (each critique's parsed verdict: True, False or None for one cut
    off before it). labels: {qid: label} of the questions kept for training. Only critiques that reached
    their verdict line are used. Nothing here reads a reference answer.
    """
    extractor = get_extractor(dataset_name)
    chosen = []
    for record in records:
        qid = record["qid"]
        if qid not in labels:
            continue
        label_key = normalize_answer(labels[qid])
        keys = _answer_keys(record["raw_samples"], extractor)
        for index, key in enumerate(keys):
            cell = critic_rows.get((qid, index))
            if cell is None or key is None:
                continue
            wanted = key == label_key
            for response, value in zip(cell.get("responses") or [], cell.get("values") or []):
                if value is wanted and response:
                    chosen.append({"qid": qid, "question": record["question"], "solution_index": index,
                                   "solution": record["raw_samples"][index]["response"],
                                   "critique": response, "verdict": wanted})
                    break
    return chosen
