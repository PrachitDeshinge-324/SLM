"""
Does one round of self-training help 4-bit more than 16-bit? (Qwen3.5-0.8B and 2B, critic self-training runs.)

Section 1: generation settings of every test run (GPU, batch size, seed, token limit). Round 0 and the arms were
  not all generated on the same GPU or batch size; the pilot (paper_selftrain_2026-10-13) found that batch size
  alone moved the untrained 4-bit model (batch 64 minus 256: per sample +3.4 [+0.1, +6.8], voted +11 [+3, +20]).
Section 2: precision x training interaction. Per arm, seed and round, on the test questions every run answered:
  (4-bit run - 4-bit round 0) - (16-bit run - 16-bit round 0), with a 95% interval from resampling questions
  (paired over both precisions). Metrics as in report_critic_self_training.py: single sample (cut-off samples
  count as wrong), voted (cut-off samples do not vote), cut off, and right when finished (ratio of sums).
Section 3: re-run noise floor. Two independent 8-sample sets of the same untrained model on all 1,319 GSM8K test
  questions (the n8 run and the first 8 samples of the n16 run). The spread of their difference, scaled to 100
  questions, is how far two runs of the same model drift apart on a 100-question test set.
Reference answers are read only to score.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_self_training import score_run  # noqa: E402

from src.experiment_d.evaluation_stats import bootstrap_interval
from src.experiment_d.rescoring import rescore_record
from src.experiment_d.run_files import (generator_run_path, read_question_records, read_run_config,
                                        self_training_round_dir)

RESULTS = ROOT / "results"
MODELS = {"Qwen/Qwen3.5-0.8B": "q0.8b", "Qwen/Qwen3.5-2B": "q2b"}
PRECISIONS = ("16bit", "4bit")
ARMS = ("vote", "vote_s1", "critic", "critic_s1", "critic_reason", "critic_reason_s1")
ROUNDS = (1, 2, 3)
THRESHOLD = 0.75
METRICS = ("single", "voted", "cutoff", "finished")


def run_path(model: str, precision: str, arm: str, round_index: int) -> Path:
    base = f"{MODELS[model]}_{precision}_critic_p500"
    experiment = base if round_index == 0 else f"{base}_{arm}"
    return generator_run_path(self_training_round_dir(RESULTS, experiment, round_index), model, precision, "gsm8k", 8)


def settings_section():
    print("=== 1. Generation settings of the test runs")
    print("model | precision | run | GPU | batch size | seed | max_new_tokens | questions")
    for model in MODELS:
        for precision in PRECISIONS:
            runs = [("round 0", run_path(model, precision, None, 0))]
            runs += [(f"{arm} r{k}", run_path(model, precision, arm, k)) for arm in ARMS for k in ROUNDS]
            for name, path in runs:
                if not path.exists():
                    continue
                config = read_run_config(path)
                print(f"{model} | {precision} | {name} | {config.get('gpu')} | {config.get('batch_size')} | "
                      f"{config.get('seed')} | {config.get('max_new_tokens')} | {len(read_question_records(path))}")


def values(scored: dict, qids: list, key: str) -> np.ndarray:
    if key == "finished":   # numerator and denominator of the ratio of sums
        return np.array([[scored[q]["single"], 1 - scored[q]["cutoff"]] for q in qids], float)
    return np.array([scored[q][key] for q in qids], float)


def change(after: np.ndarray, before: np.ndarray, index) -> float:
    if after.ndim == 2:
        return after[index, 0].sum() / after[index, 1].sum() - before[index, 0].sum() / before[index, 1].sum()
    return (after[index] - before[index]).mean()


def interaction_section():
    print("\n=== 2. Precision x training: (4-bit run - 4-bit round 0) - (16-bit run - 16-bit round 0), points [95%]")
    print("model | run | questions | metric | 16-bit change | 4-bit change | interaction")
    for model in MODELS:
        round0 = {p: score_run(run_path(model, p, None, 0), "gsm8k", THRESHOLD) for p in PRECISIONS}
        for arm in ARMS:
            for k in ROUNDS:
                paths = {p: run_path(model, p, arm, k) for p in PRECISIONS}
                if not all(path.exists() for path in paths.values()):
                    continue
                trained = {p: score_run(paths[p], "gsm8k", THRESHOLD) for p in PRECISIONS}
                unfinished = [p for p in PRECISIONS if not set(round0[p]) <= set(trained[p])]
                if unfinished:
                    print(f"{model} | {arm} r{k} | left out, test run not finished at {', '.join(unfinished)}")
                    continue
                qids = sorted(set.intersection(*(set(s) for s in [*round0.values(), *trained.values()])))
                for key in METRICS:
                    a = {p: values(trained[p], qids, key) for p in PRECISIONS}
                    b = {p: values(round0[p], qids, key) for p in PRECISIONS}
                    full = np.arange(len(qids))
                    did = lambda i: change(a["4bit"], b["4bit"], i) - change(a["16bit"], b["16bit"], i)  # noqa: E731
                    low, high = bootstrap_interval(did, len(qids))
                    print(f"{model} | {arm} r{k} | {len(qids)} | {key} | "
                          f"{100 * change(a['16bit'], b['16bit'], full):+.1f} | "
                          f"{100 * change(a['4bit'], b['4bit'], full):+.1f} | "
                          f"{100 * did(full):+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]")


def noise_floor_section():
    print("\n=== 3. Re-run noise floor: two independent 8-sample sets of the same untrained Qwen3.5-0.8B, GSM8K test")
    print("precision | questions | voted: right in only one set | 95% band of a vote difference on 100 questions "
          "| 95% band of a per-sample difference on 100 questions")
    model = "Qwen/Qwen3.5-0.8B"
    for precision in PRECISIONS:
        first = {r["qid"]: r for r in read_question_records(generator_run_path(RESULTS, model, precision, "gsm8k", 8))}
        second = {r["qid"]: r for r in read_question_records(generator_run_path(RESULTS, model, precision, "gsm8k", 16))}
        qids = sorted(set(first) & set(second))
        voted, single = [], []
        for q in qids:
            m1 = rescore_record(first[q], "gsm8k")
            m2 = rescore_record({**second[q], "raw_samples": second[q]["raw_samples"][:8]}, "gsm8k")
            voted.append(float(bool(m1["correctness"])) - float(bool(m2["correctness"])))
            single.append(m1["sample_accuracy"] - m2["sample_accuracy"])
        voted, single = np.array(voted), np.array(single)
        print(f"{precision} | {len(qids)} | {100 * np.mean(voted != 0):.1f}% | "
              f"+/-{100 * 1.96 * voted.std() / 10:.1f} | +/-{100 * 1.96 * single.std() / 10:.1f}")


def main():
    settings_section()
    interaction_section()
    noise_floor_section()


if __name__ == "__main__":
    main()
