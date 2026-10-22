"""
Did self-training change the model's own critic? The same test solutions judged by the untrained model and by
the model with each arm's adapter.

Inputs: round 0's test answers (results/self_training/<base>/round0/gsm8k/...) and the signals files of
scripts/run_signals.py over them: one without an adapter (the untrained critic) and one per adapter
(file tag <tag>_by_<base>_<arm>[_s<k>]_r1). Every judge sees the same solutions (the pilot selection depends
only on the stored answers).

Per judge, on the solutions every judge judged:
  verdict reached    share of critiques that reached their closing line within the token cap
  within-q AUROC     forced P(correct) ranking right above wrong solutions inside a question (questions with both)
  approve right / approve wrong   share of right and of wrong solutions it approves (forced P(correct) >= 0.6)
  critic + vote      accuracy of the critic + vote answer over all questions, ties split; and its paired
                     difference with the untrained critic, 95% interval over questions
The test reference answers are read here to score; this is evaluation only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.evaluation_stats import auroc, bootstrap_interval
from src.experiment_d.run_files import generator_run_path, parse_n_samples, read_question_records, self_training_round_dir, signals_path
from src.experiment_d.self_training import critic_weights


def parse_args():
    parser = argparse.ArgumentParser(description="The untrained critic against the critics of self-trained adapters")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--precision", type=str, default="4bit")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--n_samples", type=str, default="8")
    parser.add_argument("--base", type=str, required=True, help="Experiment holding round 0")
    parser.add_argument("--tag", type=str, default="critic_forced_v2_t704_pilot", help="Signals tag of the untrained critic")
    parser.add_argument("--critic", type=str, default="critic_forced")
    args = parser.parse_args()
    args.n_samples = parse_n_samples(args.n_samples)
    return args


def read_cells(path: Path, critic: str) -> dict:
    cells = {}
    for line in path.open():
        row = json.loads(line) if line.strip() else {}
        if "qid" in row and row.get(critic) is not None:
            cells[(row["qid"], row["trace_index"])] = row[critic]
    return cells


def top_share(right: dict, weights: dict) -> float:
    """Expected correctness of the top-weighted answer, a tie broken at random."""
    if not weights:
        return 0.0
    top = max(weights.values())
    return float(np.mean([right[k] for k, w in weights.items() if abs(w - top) < 1e-9]))


def main():
    args = parse_args()
    round0 = self_training_round_dir(args.results_dir, args.base, 0)
    run = generator_run_path(round0, args.model, args.precision, args.dataset, args.n_samples)
    records = read_question_records(run)
    untrained = signals_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples, args.base, 0, tag=args.tag)
    judges = {"untrained": untrained} if untrained.exists() else {}
    for path in sorted(untrained.parent.glob(f"signals_{args.tag}_by_*_{run.name}")):
        judges[path.name[len(f"signals_{args.tag}_by_"):-len(f"_{run.name}")]] = path
    if "untrained" not in judges:
        raise FileNotFoundError(f"No untrained critic run at {untrained}")
    cells = {name: read_cells(path, args.critic) for name, path in judges.items()}

    extractor = get_extractor(args.dataset)
    keys, right = {}, {}
    for r in records:
        ks = []
        for s in r["raw_samples"]:
            a = None if s.get("cutoff", False) else extractor(s["response"])
            ks.append(None if a is None else normalize_answer(a))
        keys[r["qid"]] = ks
        right[r["qid"]] = {k: answers_match(k, r["ground_truth"]) for k in set(ks) if k is not None}
    shared = set.intersection(*(set(c) for c in cells.values()))
    done = {q for q, _ in shared}
    qids = [r["qid"] for r in records if r["qid"] in done or len({k for k in keys[r["qid"]] if k}) < 2]

    lines = [f"Critic ability: {args.model} {args.precision}, round 0 test answers of {args.base}, critic {args.tag}",
             f"{len(qids)} questions ({len(done)} with judged solutions), {len(shared)} solutions judged by every judge.", "",
             "judge | verdict reached | within-q AUROC (n q) | approve right | approve wrong | vote, ties split | critic + vote, ties split"]
    per_question = {}
    for name, judged in cells.items():
        values = [v for key in shared for v in judged[key]["values"]]
        aurocs, approve_right, approve_wrong, vote_q, critic_q = [], [], [], [], []
        for q in qids:
            ks = keys[q]
            idx = [i for i in range(len(ks)) if (q, i) in shared]
            labels = [right[q][ks[i]] for i in idx]
            scores = [judged[(q, i)].get("forced_score") for i in idx]
            pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
            if pairs and any(l for _, l in pairs) and not all(l for _, l in pairs):
                aurocs.append(auroc([s for s, _ in pairs], [l for _, l in pairs]))
            for i, l in zip(idx, labels):
                (approve_right if l else approve_wrong).append(judged[(q, i)].get("forced_decision") is True)
            support = critic_weights(ks, {})
            vote_q.append(top_share(right[q], support))
            approvals = {i: judged[(q, i)].get("forced_decision") is True for i in idx}
            weights = critic_weights(ks, approvals) if any(approvals.values()) else support
            critic_q.append(top_share(right[q], weights))
        per_question[name] = np.array(critic_q)
        lines.append(f"{name} | {np.mean([v is not None for v in values]):.2f} | {np.nanmean(aurocs):.3f} ({len(aurocs)}) | "
                     f"{np.mean(approve_right):.2f} | {np.mean(approve_wrong):.2f} | {100 * np.mean(vote_q):.1f} | "
                     f"{100 * np.mean(critic_q):.1f}")
    lines += ["", "critic + vote, ties split: difference with the untrained critic in points [95%, questions resampled]"]
    base = per_question["untrained"]
    for name, values in per_question.items():
        if name == "untrained":
            continue
        d = values - base
        low, high = bootstrap_interval(lambda index: d[index].mean(), len(d))
        lines.append(f"{name} - untrained | {100 * d.mean():+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]")
    text = "\n".join(lines)
    print(text)
    out = round0.parent / f"report_critic_ability_{args.model.replace('/', '_')}_{args.precision}.txt"
    out.write_text(text + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
