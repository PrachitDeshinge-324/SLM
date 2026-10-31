"""
Label-free self-training: turn a model's own sampled answers into training text.

Nothing here reads a reference answer. A question is labelled by the majority vote of its samples
and kept only when enough of them agree; the training text is the model's own solutions that reach
that label.
"""
from collections import Counter

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


def _distinct_pairs(majority: list, minority: list, limit: int) -> list:
    """
    Up to `limit` (chosen, rejected) pairs, closest in length first, none reusing a chosen or a rejected
    solution of an earlier pair, so a question's pairs are different texts. The first pair is the one
    min() over all combinations picks (ties go to the earliest samples).
    """
    combos = sorted(((m, n) for m in majority for n in minority), key=lambda pair: abs(pair[0][1] - pair[1][1]))
    taken, used_chosen, used_rejected = [], set(), set()
    for m, n in combos:
        if m[0] in used_chosen or n[0] in used_rejected:
            continue
        taken.append((m, n))
        used_chosen.add(m[0])
        used_rejected.add(n[0])
        if len(taken) == limit:
            break
    return taken


def select_preference_pairs(records: list, dataset_name: str, threshold: float = 0.625,
                            pairing: str = "similar_length", pairs_per_question: int = 1,
                            skip_ties: bool = False) -> list:
    """
    At most `pairs_per_question` {qid, question, label, rejected_answer, agreement, margin, chosen,
    rejected, chosen_length, rejected_length} per question, for preference training on the model's own
    votes.

    A question is kept when its majority answer is given by at least `threshold` of ALL its samples
    (0.625 = 5 of 8). Chosen is a complete solution reaching the majority answer, rejected a complete
    solution reaching another answer; answers are compared after normalisation ('18' == '18.00').
    Truncated samples never vote and are never paired. Lengths are the generated token counts.
    margin: the majority answer's votes minus the runner-up's, over all samples (0 for a tie).

    pairing: 'similar_length' takes the chosen/rejected pair closest in length, so length does not
    separate the two (ties go to the earliest samples); 'longest_shortest' takes the longest majority
    and the shortest minority solution (the length-confounded ablation, one pair per question).
    pairs_per_question: with 'similar_length', further pairs of the same question reuse neither solution.
    skip_ties: drop questions where another answer has as many votes as the majority one; the vote
    then picks between them by sample order, which is not evidence.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    if pairs_per_question < 1 or (pairing == "longest_shortest" and pairs_per_question != 1):
        raise ValueError("pairs_per_question must be at least 1, and 1 for longest_shortest")
    extractor = get_extractor(dataset_name)
    pairs = []
    for record in records:
        samples = record["raw_samples"]
        extracted = [None if sample.get("cutoff", False) else extractor(sample["response"]) for sample in samples]
        label, majority_key, count, keys = majority_vote(extracted)
        if label is None or count < threshold * len(samples) - 1e-9:
            continue
        others = sorted((votes for key, votes in Counter(k for k in keys if k is not None).items()
                         if key != majority_key), reverse=True)
        runner_up = others[0] if others else 0
        if skip_ties and runner_up == count:
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
            chosen_rejected = _distinct_pairs(majority, minority, pairs_per_question)
        else:
            chosen_rejected = [(max(majority, key=lambda item: item[1]), min(minority, key=lambda item: item[1]))]
        for chosen, rejected in chosen_rejected:
            pairs.append({
                "qid": record["qid"],
                "question": record["question"],
                "label": label,
                "rejected_answer": rejected[2],
                "agreement": count / len(samples),
                "margin": (count - runner_up) / len(samples),
                "chosen": chosen[0],
                "rejected": rejected[0],
                "chosen_length": chosen[1],
                "rejected_length": rejected[1],
            })
    return pairs


def rotate_pairs(pairs: list, epochs: int) -> list:
    """
    The training list for `epochs` passes: epoch e uses each question's (e mod n)-th pair, where n is how
    many distinct pairs it has, so a question with fewer pairs than epochs repeats some. Entries are the
    same dict objects as in `pairs`, so anything computed once per pair (reference scores) is shared.
    """
    by_question = {}
    for pair in pairs:
        by_question.setdefault(pair["qid"], []).append(pair)
    return [group[epoch % len(group)] for epoch in range(epochs) for group in by_question.values()]


def curriculum_pairs(pairs: list, thresholds: list) -> tuple:
    """
    (training list, stage sizes) for a curriculum over vote agreement: stage s holds every question whose
    agreement reaches thresholds[s], and each question uses its next pair each time it appears. With rising
    difficulty (thresholds falling) the first stage is the high-agreement questions alone and the last is all
    of them. Agreement comes from the vote, so no reference answer is involved. Entries are the same dict
    objects as in `pairs`; each pair dict must carry its own `agreement`.
    """
    if sorted(thresholds, reverse=True) != list(thresholds):
        raise ValueError(f"curriculum thresholds must not rise, got {thresholds}")
    by_question = {}
    for pair in pairs:
        by_question.setdefault(pair["qid"], []).append(pair)
    schedule, sizes, seen = [], [], {}
    for threshold in thresholds:
        stage = [qid for qid, group in by_question.items() if group[0]["agreement"] >= threshold - 1e-9]
        for qid in stage:
            schedule.append(by_question[qid][seen.get(qid, 0) % len(by_question[qid])])
            seen[qid] = seen.get(qid, 0) + 1
        sizes.append(len(stage))
    return schedule, sizes


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
        seen, rejected_traces = set(), set()
        for index, (sample, key) in enumerate(zip(samples, keys)):
            trace = sample["response"]
            if index in judged and not judged[index]:
                rejected_traces.add(trace)
            if key != label_key or trace in seen or trace in rejected_traces:
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
                if value is not None and value == wanted and response:
                    chosen.append({"qid": qid, "question": record["question"], "solution_index": index,
                                   "solution": record["raw_samples"][index]["response"],
                                   "critique": response, "verdict": wanted})
                    break
    return chosen
