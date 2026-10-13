"""
How much room is there for label-free self-improvement at each precision, and does the 4-bit gap survive
at the ceiling a model's own vote sets? GSM8K, 8-sample runs, questions shared by all precisions of a model.

Two idealised ceilings (upper bounds, not predictions of what training achieves):
  one round  the model answers every question it labelled confidently (agreement >= 0.75) with that label,
             and is unchanged elsewhere
  full       the model always gives its own voted answer (per-sample accuracy = voted accuracy)
"""
import collections
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.evaluation_stats import bootstrap_interval

EX = get_extractor("gsm8k")
PREC = ["16bit", "8bit", "4bit"]


def short(m):
    return (m.replace("meta-llama_", "").replace("deepseek-ai_", "").replace("google_", "").replace("Qwen_", "")
             .replace("-Instruct", "").replace("DeepSeek-R1-Distill-Qwen", "DS-R1").replace("Llama-3.2", "Llama"))


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "raw_samples" not in r:
            continue
        S = []
        for s in r["raw_samples"]:
            a = EX(s["response"])
            S.append((None if a is None else normalize_answer(a), bool(a is not None and answers_match(a, r["ground_truth"])), bool(s["cutoff"])))
        valid = [(a, c) for a, c, cut in S if a is not None and not cut]
        cnt = collections.Counter(a for a, _ in valid)
        key = cnt.most_common(1)[0][0] if cnt else None
        out[r["qid"]] = {
            "single": np.mean([c and not cut for _, c, cut in S]),
            "maj": bool(key is not None and any(c for a, c in valid if a == key)),
            "conf": (cnt[key] / len(S)) if key is not None else 0.0,
            "pass": any(c and not cut for _, c, cut in S),
        }
    return out


runs = collections.defaultdict(dict)
for gp in sorted(glob.glob(os.path.join(REPO, "results", "gsm8k", "generator", "*", "*_n8.jsonl"))):
    if "_backup" in gp:
        continue
    model = gp.split(os.sep)[-2]
    runs[model][re.search(r"_(16bit|8bit|4bit)_", os.path.basename(gp)).group(1)] = load(gp)

print("GSM8K, 8 samples, questions shared by the listed precisions. All values are accuracies unless marked.")
print("model | prec | nQ | single now | labelled >=0.75 (label accuracy) | ceiling after one ideal round | full ceiling (= voted) | best possible with a perfect selector (pass@8)")
SUMMARY = []
for model, P in runs.items():
    if not all(p in P for p in PREC):
        continue
    common = sorted(set.intersection(*[set(P[p]) for p in PREC]))
    N = len(common)
    A = {p: {k: np.array([P[p][q][k] for q in common], float) for k in ("single", "maj", "conf", "pass")} for p in PREC}
    for p in PREC:
        a = A[p]
        keep = a["conf"] >= 0.75
        one = np.where(keep, a["maj"], a["single"])
        a["one"] = one
        print(f"{short(model):13s} | {p:5s} | {N:4d} | {a['single'].mean():.3f} | {keep.mean():.3f} ({a['maj'][keep].mean():.3f}) | {one.mean():.3f} | {a['maj'].mean():.3f} | {a['pass'].mean():.3f}")
    a4, a16 = A["4bit"], A["16bit"]
    gap = lambda k: (a4[k] - a16[k])
    rows = []
    for name, k in (("now (single)", "single"), ("after one ideal round", "one"), ("at the full ceiling (voted)", "maj")):
        d = gap(k)
        lo, hi = bootstrap_interval(lambda ix, d=d: d[ix].mean(), N, resamples=1000)
        rows.append(f"{name}: {100*d.mean():+.1f} [{100*lo:+.1f},{100*hi:+.1f}]")
    # 4-bit model at its own ceiling vs the untrained 16-bit model
    cross = a4["maj"] - a16["single"]
    lo, hi = bootstrap_interval(lambda ix: cross[ix].mean(), N, resamples=1000)
    print(f"   4-bit minus 16-bit gap -> " + " | ".join(rows))
    print(f"   4-bit at its full ceiling minus UNTRAINED 16-bit single: {100*cross.mean():+.1f} [{100*lo:+.1f},{100*hi:+.1f}]\n")
