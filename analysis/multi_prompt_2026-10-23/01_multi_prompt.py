"""
Multi-prompt test (Qwen3.5-0.8B, GSM8K test): does a vote over answers from different prompts beat a vote over
the same number of answers from one prompt, and does it help most where 4-bit settles on a wrong answer?

Answers per question and precision:
  default   the held-out 8 samples of the main runs (last 8 of the n16 run): not used to classify the questions
  plan, check, example2   the variant runs of run_multi_prompt_test.sh (4 samples each)
Damage classes come from analysis/damage_types_2026-10-24/outputs/01_question_types.csv, classified on the
first 16 samples of the default prompt, so every answer compared here is independent of the classification.

Compared at the same budget of 8 answers:
  one prompt   the 8 default answers
  mixed        2 default answers + 2 from each of the 3 variants (averaged over 200 random draws per question)
  mixed, no check   2 default + 3 plan + 3 example2: the check prompt runs past the token limit far more often,
                    so this mix tells prompt diversity apart from the cost of one long prompt
Per class and overall: accuracy of one answer per prompt, the vote (ties split), and how often the right answer
is among the 8 (pass@8); mixed minus one prompt with a 95% interval from resampling questions.
Reference answers are read only to score.
"""
import argparse
import csv
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "damage_types_2026-10-24"))

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.evaluation_stats import bootstrap_interval
from src.experiment_d.run_files import generator_run_path, prompt_variant_dir, read_question_records, read_run_config, token_limit

types = import_module("01_damage_types")

MODEL = "Qwen/Qwen3.5-0.8B"
VARIANTS = ("plan", "check", "example2")
CLASSES = ("preserved", "random", "systematic", "helped", "both fail")
DRAWS = 200
# name -> answers per prompt in a mix of 8 ('default' draws from its 8 held-out answers)
MIXES = {"mixed": {"default": 2, "plan": 2, "check": 2, "example2": 2},
         "mixed, no check": {"default": 2, "plan": 3, "example2": 3}}


def load(path, extractor) -> dict:
    return {r["qid"]: types.scored_samples(r, extractor) for r in read_question_records(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(ROOT / "results"))
    parser.add_argument("--n_samples", type=int, default=4, help="Samples per variant run")
    args = parser.parse_args()
    extractor = get_extractor("gsm8k")
    classes = {r["qid"]: r["type"] for r in csv.DictReader((ROOT / "analysis/damage_types_2026-10-24/outputs/01_question_types.csv").open())
               if r["dataset"] == "gsm8k"}
    rng = np.random.default_rng(0)
    out_lines = []

    def say(line=""):
        print(line)
        out_lines.append(line)

    for precision in ("16bit", "4bit"):
        main16 = generator_run_path(args.results_dir, MODEL, precision, "gsm8k", 16)
        default = {q: s[8:] for q, s in load(main16, extractor).items()}
        runs, limits = {}, {token_limit(read_run_config(main16), "gsm8k")}
        for v in VARIANTS:
            path = generator_run_path(prompt_variant_dir(args.results_dir, v), MODEL, precision, "gsm8k", args.n_samples)
            if path.exists():
                runs[v] = load(path, extractor)
                limits.add(token_limit(read_run_config(path), "gsm8k"))
        if len(runs) < len(VARIANTS):
            say(f"\n=== {precision}: variant runs missing ({sorted(set(VARIANTS) - set(runs))}); skipped")
            continue
        if len(limits) != 1:
            raise RuntimeError(f"{precision}: token limits differ {limits}")
        qids = [q for q in default if all(q in r for r in runs.values()) and q in classes]
        say(f"\n=== {MODEL} {precision}: {len(qids)} questions answered with every prompt, token limit {limits.pop()}")

        say("one answer, by prompt (% right): " + " | ".join(
            [f"default {100 * np.mean([ok for q in qids for _, ok in default[q]]):.1f}"] +
            [f"{v} {100 * np.mean([ok for q in qids for _, ok in runs[v][q]]):.1f}" for v in VARIANTS]))
        cut = {v: np.mean([k is None for q in qids for k, _ in runs[v][q]]) for v in VARIANTS}
        say("no answer or cut off, by prompt (%): default "
            f"{100 * np.mean([k is None for q in qids for k, _ in default[q]]):.1f} | "
            + " | ".join(f"{v} {100 * cut[v]:.1f}" for v in VARIANTS))

        one_vote = {q: types.summary(default[q])[1] for q in qids}
        one_pass = {q: float(any(ok for _, ok in default[q])) for q in qids}
        sources = {"default": default, **runs}
        for mix_name, counts in MIXES.items():
            mix_vote, mix_pass = {}, {}
            for q in qids:
                votes, passes = [], []
                for _ in range(DRAWS):
                    pick = []
                    for prompt, k in counts.items():
                        pool = sources[prompt][q]
                        pick += [pool[i] for i in rng.choice(len(pool), k, replace=False)]
                    votes.append(types.summary(pick)[1])
                    passes.append(any(ok for _, ok in pick))
                mix_vote[q], mix_pass[q] = float(np.mean(votes)), float(np.mean(passes))

            say(f"\n{mix_name} ({', '.join(f'{k} {p}' for p, k in counts.items())}) against one prompt (8 default)")
            say("class | n | vote, one prompt | vote, mix | mix - one [95%] | pass@8 one | pass@8 mix")
            for c in CLASSES + ("all",):
                qs = qids if c == "all" else [q for q in qids if classes[q] == c]
                if not qs:
                    continue
                a = np.array([mix_vote[q] for q in qs])
                b = np.array([one_vote[q] for q in qs])
                d = a - b
                low, high = bootstrap_interval(lambda index: d[index].mean(), len(d))
                say(f"{c} | {len(qs)} | {100 * b.mean():.1f} | {100 * a.mean():.1f} | {100 * d.mean():+.1f} "
                    f"[{100 * low:+.1f}, {100 * high:+.1f}] | {100 * np.mean([one_pass[q] for q in qs]):.1f} | "
                    f"{100 * np.mean([mix_pass[q] for q in qs]):.1f}")

    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    (out / "01_multi_prompt.txt").write_text("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
