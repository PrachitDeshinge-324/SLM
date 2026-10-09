"""How general are the findings? Pooling across cells, small-subset reliability, held-out models,
and whether the same questions are fragile for different models."""
import numpy as np
from common import *

G = groups()
RNG = np.random.default_rng(0)


def cell_effects(P, p, B=1000):
    """Paired effects vs 16-bit on common questions, with bootstrap standard errors."""
    common = sorted(set(P["16bit"]["Q"]) & set(P[p]["Q"]))
    a, b = question_arrays(P, p, common), question_arrays(P, "16bit", common)
    N = len(common)
    keep_a, keep_b = a["conf_all"] >= 0.75, b["conf_all"] >= 0.75

    def stats(ix):
        ya, yb = a["correct"][ix], b["correct"][ix]
        ka, kb = keep_a[ix], keep_b[ix]
        prec = (ya[ka].mean() if ka.any() else np.nan) - (yb[kb].mean() if kb.any() else np.nan)
        return np.array([
            (a["single"][ix] - b["single"][ix]).mean(),
            (ya - yb).mean(),
            (a["pass"][ix] - b["pass"][ix]).mean(),
            (ya - yb).mean() - (a["single"][ix] - b["single"][ix]).mean(),      # damage hidden by the vote
            ka.mean() - kb.mean(),
            prec,
            auroc(a["conf_all"][ix], ya.astype(bool)) - auroc(b["conf_all"][ix], yb.astype(bool)),
            (a["conf_all"][ix] - b["conf_all"][ix]).mean() - (a["single"][ix] - b["single"][ix]).mean(),  # estimator error
        ])
    est = stats(np.arange(N))
    boots = np.array([stats(RNG.integers(0, N, N)) for _ in range(B)])
    return N, est, np.nanstd(boots, axis=0)


NAMES = ["per-sample accuracy", "majority accuracy", "coverage (pass@n)", "masking (majority - per-sample)",
         "yield at agreement>=0.75", "label precision at >=0.75", "AUROC of agreement", "estimator error (d agreement - d per-sample)"]


# ---------------------------------------------------------------- (a) pooling
print("=== (a) Random-effects pooling over model x dataset cells (8-sample runs, token limits matched).")
print("    mean = pooled effect in points; tau = between-cell SD (how much a new cell is expected to differ); I2 = share of variation that is real rather than sampling error")
cells = {}
for (d, m, n), P in sorted(G.items()):
    if "16bit" not in P:
        continue
    # one run size per cell: n=8, except cells that only exist at n=16
    if n == 16 and (d, m, 8) in G and "16bit" in G[(d, m, 8)]:
        continue
    for p in ("8bit", "4bit"):
        if p not in P or not comparable(P, p):
            continue
        if len(set(P["16bit"]["Q"]) & set(P[p]["Q"])) < 100:
            continue
        cells[(d, short(m), p)] = cell_effects(P, p)

for p in ("4bit", "8bit"):
    sel = {k: v for k, v in cells.items() if k[2] == p}
    print(f"\n--- {p}: {len(sel)} cells: " + ", ".join(f"{k[0][:6]}/{k[1]}" for k in sel))
    print(f"{'effect':46s} | pooled mean [95% CI] | tau  | I2   | 95% prediction interval for a new cell | same sign | range over cells | pooled without Gemma")
    for i, name in enumerate(NAMES):
        y = np.array([v[1][i] for v in sel.values()]) * 100
        se = np.array([v[2][i] for v in sel.values()]) * 100
        k, mean, sem, tau, i2 = pool_random_effects(y, se)
        pi = 1.96 * np.sqrt(tau ** 2 + sem ** 2)
        ng = [j for j, key in enumerate(sel) if "gemma" not in key[1]]
        _, mean_ng, sem_ng, tau_ng, _ = pool_random_effects(y[ng], se[ng])
        fin = np.isfinite(y)
        print(f"{name:46s} | {mean:+6.2f} [{mean-1.96*sem:+6.2f},{mean+1.96*sem:+6.2f}] | {tau:4.2f} | {i2:4.2f} | [{mean-pi:+6.2f},{mean+pi:+6.2f}] | {int((np.sign(y[fin]) == np.sign(mean)).sum())}/{int(fin.sum())} | {np.nanmin(y):+6.1f}..{np.nanmax(y):+6.1f} | {mean_ng:+6.2f} (tau {tau_ng:4.2f})")

# ---------------------------------------------------------------- (b) small subsets
print("\n=== (b) Would a 100- or 200-question study have reached the same conclusion? Full-set cells, 4-bit vs 16-bit.")
print("cell | full set: d per-sample, d majority | first 100 | first 200 | questions 201+ | 95% range of d majority over 2000 random 100-question subsets | share of those subsets with |d majority| < 3 pts")
for d, m, n in [("gsm8k", "Qwen_Qwen3.5-0.8B", 8), ("gsm8k", "Qwen_Qwen3.5-0.8B", 16), ("gsm8k", "meta-llama_Llama-3.2-1B-Instruct", 8), ("gsm8k", "meta-llama_Llama-3.2-3B-Instruct", 8), ("cqa", "Qwen_Qwen3.5-0.8B", 8)]:
    P = G[(d, m, n)]
    common = sorted(set(P["16bit"]["Q"]) & set(P["4bit"]["Q"]), key=lambda q: int(q.rsplit("_", 1)[1]))
    a, b = question_arrays(P, "4bit", common), question_arrays(P, "16bit", common)
    ds, dm = a["single"] - b["single"], a["correct"] - b["correct"]
    N = len(common)
    sub = np.array([dm[RNG.choice(N, 100, replace=False)].mean() for _ in range(2000)]) * 100
    f = lambda sl: f"{100*ds[sl].mean():+5.1f}, {100*dm[sl].mean():+5.1f}"
    print(f"{d[:5]} {short(m):12s} n{n:<2d} | {f(slice(None))} | {f(slice(0,100))} | {f(slice(0,200))} | {f(slice(200,None))} | [{np.percentile(sub,2.5):+5.1f},{np.percentile(sub,97.5):+5.1f}] | {np.mean(np.abs(sub)<3):.2f}")
    if d == "gsm8k":
        print(f"      16-bit majority accuracy: first 100 = {b['correct'][:100].mean():.3f}, first 200 = {b['correct'][:200].mean():.3f}, rest = {b['correct'][200:].mean():.3f}")

# ---------------------------------------------------------------- (c) held-out models for the estimator
print("\n=== (c) Agreement-shift estimator, leave one model out. Zero-parameter rule: d per-sample = d agreement. Fitted rule: line fitted on the other models' cells.")
rows = []
for (d, m, n), P in sorted(G.items()):
    if "16bit" not in P:
        continue
    for p in ("8bit", "4bit"):
        if p not in P or not comparable(P, p):
            continue
        common = sorted(set(P["16bit"]["Q"]) & set(P[p]["Q"]))
        if len(common) < 100:
            continue
        a, b = question_arrays(P, p, common), question_arrays(P, "16bit", common)
        rows.append((short(m), d, n, p, 100 * (a["single"] - b["single"]).mean(), 100 * (a["conf_all"] - b["conf_all"]).mean()))
models = sorted(set(r[0] for r in rows))
print("held-out model | cells | zero-parameter error | fitted-line error (slope, intercept from the rest) | 'assume no change' error")
tot = {"zero": [], "fit": [], "none": []}
for mo in models:
    te = [r for r in rows if r[0] == mo]; tr = [r for r in rows if r[0] != mo]
    slope, icpt = np.polyfit([r[5] for r in tr], [r[4] for r in tr], 1)
    z = [abs(r[4] - r[5]) for r in te]; fi = [abs(r[4] - (slope * r[5] + icpt)) for r in te]; no = [abs(r[4]) for r in te]
    tot["zero"] += z; tot["fit"] += fi; tot["none"] += no
    print(f"{mo:14s} | {len(te):2d} | {np.mean(z):4.2f} | {np.mean(fi):4.2f} (slope {slope:.2f}, intercept {icpt:+.2f}) | {np.mean(no):4.2f}")
print(f"{'all, held out':14s} | {len(rows):2d} | {np.mean(tot['zero']):4.2f} | {np.mean(tot['fit']):4.2f} | {np.mean(tot['none']):4.2f}")

# ---------------------------------------------------------------- (d) are the same questions fragile for different models?
print("\n=== (d) Per-question change in per-sample accuracy at 4-bit: do models agree on WHICH questions break? (GSM8K)")
def dq(d, m, n):
    P = G[(d, m, n)]
    common = sorted(set(P["16bit"]["Q"]) & set(P["4bit"]["Q"]))
    a, b = question_arrays(P, "4bit", common), question_arrays(P, "16bit", common)
    return dict(zip(common, a["single"] - b["single"])), dict(zip(common, b["single"]))
D = {}
for label, key in [("Qwen-0.8B n8", ("gsm8k", "Qwen_Qwen3.5-0.8B", 8)), ("Qwen-0.8B n16", ("gsm8k", "Qwen_Qwen3.5-0.8B", 16)),
                   ("Llama-1B", ("gsm8k", "meta-llama_Llama-3.2-1B-Instruct", 8)), ("Llama-3B", ("gsm8k", "meta-llama_Llama-3.2-3B-Instruct", 8)),
                   ("Gemma-1B", ("gsm8k", "google_gemma-3-1b-it", 8)), ("DS-R1-1.5B", ("gsm8k", "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B", 8))]:
    D[label] = dq(*key)
labels = list(D)
print("correlation of per-question 4-bit change (upper) and of 16-bit per-question accuracy (lower); n questions in brackets")
print(" " * 14 + "".join(f"{l:>16s}" for l in labels))
for i, a in enumerate(labels):
    row = f"{a:14s}"
    for j, b in enumerate(labels):
        qs = sorted(set(D[a][0]) & set(D[b][0]))
        if i == j:
            row += f"{'-':>16s}"
        else:
            which = 0 if j > i else 1
            r = np.corrcoef([D[a][which][q] for q in qs], [D[b][which][q] for q in qs])[0, 1]
            row += f"{r:>+9.2f} ({len(qs):4d})"
    print(row)
print("Qwen-0.8B n8 vs n16 is the same model on independent runs: it is the ceiling a cross-model correlation could reach.")
