"""Numbers for the course paper, with every comparison restricted to the same 100 questions.

For each model x dataset cell the questions are the first 100, in the order of the 16-bit run,
that every comparable precision of the cell answered (the rule of the Phase 2 web page).
Reads the cache of analysis/reassessment_2026-10-09 (rebuild it there after new run files).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reassessment_2026-10-09"))
from common import *

QUESTIONS = 100
THRESHOLD = 0.75
G = groups()
RNG = np.random.default_rng(0)
NAMES = ["per-sample accuracy", "majority accuracy", "coverage", "masking (majority - per-sample)",
         "yield at agreement>=0.75", "label precision at >=0.75"]


def cell_questions(P):
    """(precisions kept, the 100 questions) of one cell, or (kept, None) with fewer than 100 shared."""
    kept = [p for p in ("8bit", "4bit") if p in P and comparable(P, p)]
    shared = set(P["16bit"]["Q"])
    for p in kept:
        shared &= set(P[p]["Q"])
    questions = [q for q in P["16bit"]["Q"] if q in shared][:QUESTIONS]
    return kept, (questions if len(questions) == QUESTIONS else None)


def effects(P, p, questions, B=2000):
    a, b = question_arrays(P, p, questions), question_arrays(P, "16bit", questions)
    keep_a, keep_b = a["conf_all"] >= THRESHOLD, b["conf_all"] >= THRESHOLD

    def stats(ix):
        ya, yb, ka, kb = a["correct"][ix], b["correct"][ix], keep_a[ix], keep_b[ix]
        single = (a["single"][ix] - b["single"][ix]).mean()
        return np.array([
            single, (ya - yb).mean(), (a["pass"][ix] - b["pass"][ix]).mean(), (ya - yb).mean() - single,
            ka.mean() - kb.mean(),
            (ya[ka].mean() if ka.any() else np.nan) - (yb[kb].mean() if kb.any() else np.nan),
        ])
    n = len(questions)
    boots = np.array([stats(RNG.integers(0, n, n)) for _ in range(B)])
    levels = {k: (b[k].mean(), a[k].mean()) for k in ("single", "correct", "pass")}
    levels["yield"] = (keep_b.mean(), keep_a.mean())
    levels["precision"] = (b["correct"][keep_b].mean() if keep_b.any() else np.nan, a["correct"][keep_a].mean() if keep_a.any() else np.nan)
    levels["kept"] = (int(keep_b.sum()), int(keep_a.sum()))
    return stats(np.arange(n)), np.nanstd(boots, axis=0), np.nanpercentile(boots, [2.5, 97.5], axis=0), levels


cells, chosen = {}, {}
print(f"=== Cells: every comparison on the same {QUESTIONS} questions")
for (d, m, n), P in sorted(G.items()):
    if "16bit" not in P:
        continue
    if n == 16 and (d, m, 8) in G and "16bit" in G[(d, m, 8)]:
        continue
    kept, questions = cell_questions(P)
    if questions is None:
        print(f"left out: {d} {short(m)} n{n}: fewer than {QUESTIONS} shared questions or no comparable precision ({kept})")
        continue
    chosen[(d, short(m), n)] = (P, questions)
    cut = {p: np.mean([s["cut"] for q in questions for s in P[p]["Q"][q]["S"]]) for p in ["16bit"] + kept}
    print(f"kept: {d} {short(m)} n{n}: precisions {kept}; truncated samples " + ", ".join(f"{p} {v:.3f}" for p, v in cut.items()))
    for p in kept:
        cells[(d, short(m), p)] = effects(P, p, questions)

for p in ("4bit", "8bit"):
    sel = {k: v for k, v in cells.items() if k[2] == p}
    print(f"\n=== {p} vs 16-bit: {len(sel)} cells, {QUESTIONS} questions each. Points.")
    print("--- per cell: change [95% bootstrap interval]; levels 16-bit -> quantized")
    for k, (est, se, ci, lv) in sel.items():
        print(f"{k[0][:7]:7s} {k[1]:14s} | " + " | ".join(f"{NAMES[i].split(' (')[0].split(' at')[0]} {est[i]*100:+5.1f} [{ci[0][i]*100:+5.1f},{ci[1][i]*100:+5.1f}]" for i in (0, 1, 4, 5, 2)))
        print(f"{'':22s} |   per-sample {lv['single'][0]:.3f}->{lv['single'][1]:.3f}  majority {lv['correct'][0]:.2f}->{lv['correct'][1]:.2f}  yield {lv['yield'][0]:.2f}->{lv['yield'][1]:.2f}  precision {lv['precision'][0]:.3f}->{lv['precision'][1]:.3f} (labels kept {lv['kept'][0]}->{lv['kept'][1]})  coverage {lv['pass'][0]:.2f}->{lv['pass'][1]:.2f}")
    print("--- pooled (random effects): mean [95% CI] | tau | cells with the pooled sign | range | pooled without Gemma")
    for i, name in enumerate(NAMES):
        y = np.array([v[0][i] for v in sel.values()]) * 100
        se = np.array([v[1][i] for v in sel.values()]) * 100
        k, mean, sem, tau, i2 = pool_random_effects(y, se)
        ng = [j for j, key in enumerate(sel) if "gemma" not in key[1]]
        _, mean_ng, sem_ng, _, _ = pool_random_effects(y[ng], se[ng])
        fin = np.isfinite(y)
        print(f"{name:34s} | {mean:+6.2f} [{mean-1.96*sem:+6.2f},{mean+1.96*sem:+6.2f}] | {tau:4.2f} | {int((np.sign(y[fin]) == np.sign(mean)).sum())}/{int(fin.sum())} | {np.nanmin(y):+6.1f}..{np.nanmax(y):+6.1f} | {mean_ng:+6.2f} [{mean_ng-1.96*sem_ng:+6.2f},{mean_ng+1.96*sem_ng:+6.2f}]")

print(f"\n=== Sample count: Qwen3.5-0.8B GSM8K, 16-sample runs on the {QUESTIONS} questions of the 8-sample cell")
P8, questions = chosen[("gsm8k", "Qwen3.5-0.8B", 8)]
P16 = G[("gsm8k", "Qwen_Qwen3.5-0.8B", 16)]
for label, P in (("n8 ", P8), ("n16", P16)):
    if all(q in P["16bit"]["Q"] and q in P["4bit"]["Q"] for q in questions):
        est, se, ci, lv = effects(P, "4bit", questions)
        print(f"{label}: 4-bit per-sample {est[0]*100:+5.1f} [{ci[0][0]*100:+5.1f},{ci[1][0]*100:+5.1f}]  majority {est[1]*100:+5.1f}  yield {est[4]*100:+5.1f}  precision {est[5]*100:+5.1f}")

print(f"\n=== Gemma-3-1B GSM8K answer format on its {QUESTIONS} questions")
P, questions = chosen[("gsm8k", "gemma-3-1b-it", 8)]
for p in ("16bit", "8bit", "4bit"):
    S = [s for q in questions for s in P[p]["Q"][q]["S"] if not s["cut"]]
    boxed = [s for s in S if s["path"] == "boxed"]
    other = [s for s in S if s["path"] != "boxed"]
    gaps = []
    for q in questions:
        qs = [s for s in P[p]["Q"][q]["S"] if not s["cut"]]
        bx, ot = [s["correct"] for s in qs if s["path"] == "boxed"], [s["correct"] for s in qs if s["path"] != "boxed"]
        if bx and ot:
            gaps.append(np.mean(bx) - np.mean(ot))
    print(f"{p}: complete samples {len(S)}; boxed share {len(boxed)/len(S):.3f}; accuracy boxed {np.mean([s['correct'] for s in boxed]):.3f}, other {np.mean([s['correct'] for s in other]):.3f}; within-question gap {np.mean(gaps):+.3f} over {len(gaps)} questions")
