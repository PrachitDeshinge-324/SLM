"""Outliers at three levels: cells (which model/dataset departs from the pattern and does any
conclusion depend on it), questions (unanimous wrong answers, catastrophic 4-bit drops) and samples."""
import collections, os, re, sys
import numpy as np
import pandas as pd
from fractions import Fraction
from common import *

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
G = groups()
RNG = np.random.default_rng(0)
GS = pd.read_parquet(os.path.join(REPO, "data", "gsm8k", "main", "test-00000-of-00001.parquet"))


# ------------------------------------------------------------------ (a) cell-level outliers
print("=== (a) Cell-level outliers at 4-bit: change versus 16-bit in points, one row per model x dataset cell")
rows = []
for (d, m, n), P in sorted(G.items()):
    if "16bit" not in P or "4bit" not in P or not comparable(P, "4bit"):
        continue
    if n == 16 and (d, m, 8) in G and "16bit" in G[(d, m, 8)]:
        continue
    common = sorted(set(P["16bit"]["Q"]) & set(P["4bit"]["Q"]))
    if len(common) < 100:
        continue
    a, b = question_arrays(P, "4bit", common), question_arrays(P, "16bit", common)
    ka, kb = a["conf_all"] >= 0.75, b["conf_all"] >= 0.75
    rows.append({
        "cell": f"{d[:6]}/{short(m)}",
        "per-sample": 100 * (a["single"] - b["single"]).mean(),
        "majority": 100 * (a["correct"] - b["correct"]).mean(),
        "coverage": 100 * (a["pass"] - b["pass"]).mean(),
        "yield": 100 * (ka.mean() - kb.mean()),
        "label precision": 100 * (a["correct"][ka].mean() - b["correct"][kb].mean()),
        "agreement": 100 * (a["conf_all"] - b["conf_all"]).mean(),
    })
T = pd.DataFrame(rows).set_index("cell")
T["estimator residual"] = T["agreement"] - T["per-sample"]
print(T.round(1).to_string())
print("\nrobust z-score of each cell = (value - median) / (1.4826 * MAD); |z| > 3.5 flagged")
Z = (T - T.median()) / (1.4826 * (T - T.median()).abs().median())
flag = Z.abs() > 3.5
for col in T.columns:
    out = [f"{c} (value {T.loc[c, col]:+.1f}, z {Z.loc[c, col]:+.1f})" for c in T.index if flag.loc[c, col]]
    print(f"  {col:20s}: " + ("; ".join(out) if out else "none"))

print("\n=== (a2) Do the headline numbers depend on single cells? Estimator (all 8-bit and 4-bit cells): leave one cell out")
E = []
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
        E.append((f"{d[:6]}/{short(m)}/n{n}/{p}", 100 * (a["single"] - b["single"]).mean(), 100 * (a["conf_all"] - b["conf_all"]).mean()))
y = np.array([e[1] for e in E]); x = np.array([e[2] for e in E])
def spearman(u, v):
    ru, rv = pd.Series(u).rank().values, pd.Series(v).rank().values
    return np.corrcoef(ru, rv)[0, 1]
print(f"all {len(E)} cells: Pearson r = {np.corrcoef(x, y)[0,1]:.3f}, Spearman = {spearman(x, y):.3f}, mean abs error = {np.abs(y-x).mean():.2f}")
loo = [(E[i][0], np.corrcoef(np.delete(x, i), np.delete(y, i))[0, 1], np.abs(np.delete(y, i) - np.delete(x, i)).mean()) for i in range(len(E))]
lo = min(loo, key=lambda t: t[1]); hi = max(loo, key=lambda t: t[2])
print(f"leave-one-out: r ranges {min(t[1] for t in loo):.3f}..{max(t[1] for t in loo):.3f} (lowest when dropping {lo[0]}); error ranges {min(t[2] for t in loo):.2f}..{max(t[2] for t in loo):.2f}")
big = np.argsort(np.abs(y))[-3:]
keep = np.setdiff1d(np.arange(len(E)), big)
print(f"without the three largest-change cells ({', '.join(E[i][0] for i in big)}): r = {np.corrcoef(x[keep], y[keep])[0,1]:.3f}, error = {np.abs(y[keep]-x[keep]).mean():.2f}, 'no change' error = {np.abs(y[keep]).mean():.2f}")
res = y - x
print("largest residuals (d per-sample - d agreement): " + "; ".join(f"{E[i][0]} {res[i]:+.1f}" for i in np.argsort(np.abs(res))[-4:][::-1]))

# ------------------------------------------------------------------ (b) question-level outliers
print("\n=== (b1) Unanimous but wrong: questions where all complete samples agree (agreement = 1.0 over all 8) on a wrong answer. GSM8K, 8-sample runs")
runs16 = {}
for (d, m, n), P in sorted(G.items()):
    if d != "gsm8k" or n != 8:
        continue
    row = f"  {short(m):13s}"
    for p in PREC:
        if p not in P:
            continue
        st = {q: qstats(P[p]["Q"][q]) for q in P[p]["Q"]}
        uw = [q for q, s in st.items() if s["conf_all"] == 1.0 and not s["correct"]]
        un = [q for q, s in st.items() if s["conf_all"] == 1.0]
        row += f" | {p}: {len(uw)}/{len(un)} unanimous are wrong ({100*len(uw)/max(len(un),1):.1f}%)"
        if p == "16bit":
            runs16[short(m)] = st
    print(row)

print("\n=== (b2) The same wrong answer from several models: 16-bit runs, majority wrong with agreement >= 0.75. Candidates for a faulty gold label or an ambiguous question")
votes = collections.defaultdict(list)
for model, st in runs16.items():
    for q, s in st.items():
        if not s["correct"] and s["conf_all"] >= 0.75 and s["maj"] is not None:
            votes[q].append((model, s["maj"]))
n_models_with_q = collections.Counter(q for st in runs16.values() for q in st)
shared = []
for q, v in votes.items():
    top, cnt = collections.Counter(a for _, a in v).most_common(1)[0]
    if cnt >= 3:
        shared.append((q, top, cnt, n_models_with_q[q]))
print(f"questions where >=3 models confidently give the SAME wrong answer: {len(shared)} of {len(n_models_with_q)} questions seen")
for q, top, cnt, tot in sorted(shared, key=lambda t: -t[2])[:8]:
    idx = int(q.rsplit("_", 1)[1])
    gold = GS.iloc[idx]["answer"].split("####")[-1].strip()
    try:
        shown = str(float(Fraction(top))).rstrip("0").rstrip(".")
    except Exception:
        shown = top
    print(f"  {q}: gold {gold}, {cnt} of {tot} models answer {shown} | {GS.iloc[idx]['question'][:150].strip()}...")
conf_wrong_any = {q for q, v in votes.items()}
print(f"questions with a confident wrong majority in at least one model: {len(conf_wrong_any)}; in exactly one model: {sum(len(v)==1 for v in votes.values())}")

print("\n=== (b3) Catastrophic 4-bit drops for Qwen-0.8B (24 samples per precision, n8+n16 pooled): what distinguishes these questions?")
A, B = G[("gsm8k", "Qwen_Qwen3.5-0.8B", 8)], G[("gsm8k", "Qwen_Qwen3.5-0.8B", 16)]
common = sorted(set.intersection(*[set(v["Q"]) for v in list(A.values()) + list(B.values())]))
def pool(p):
    return np.array([np.mean([s["correct"] and not s["cut"] for s in A[p]["Q"][q]["S"] + B[p]["Q"][q]["S"]]) for q in common])
def poolcut(p):
    return np.array([np.mean([s["cut"] for s in A[p]["Q"][q]["S"] + B[p]["Q"][q]["S"]]) for q in common])
p16, p4 = pool("16bit"), pool("4bit")
d = p4 - p16
idx = np.array([int(q.rsplit("_", 1)[1]) for q in common])
feat = pd.DataFrame({
    "question length (chars)": GS.iloc[idx]["question"].str.len().values,
    "gold solution steps": GS.iloc[idx]["answer"].str.count("<<").values,
    "numbers in question": GS.iloc[idx]["question"].str.count(r"\d+(?:[.,]\d+)?").values,
    "gold answer >= 1000 (share)": [abs(float(a.split("####")[-1].strip().replace(",", ""))) >= 1000 for a in GS.iloc[idx]["answer"]],
    "question has % or decimal (share)": GS.iloc[idx]["question"].str.contains(r"%|\d\.\d").values,
    "16-bit per-sample accuracy": p16,
    "16-bit truncation share": poolcut("16bit"),
    "4-bit truncation share": poolcut("4bit"),
})
groups_ = {"drop >= 0.5": d <= -0.5, "drop 0.25-0.5": (d > -0.5) & (d <= -0.25), "within +-0.25": np.abs(d) < 0.25, "gain >= 0.25": d >= 0.25}
print("group | n | " + " | ".join(feat.columns))
for name, mask in groups_.items():
    print(f"{name:14s} | {int(mask.sum()):4d} | " + " | ".join(f"{feat.loc[mask, c].astype(float).mean():.2f}" for c in feat.columns))
solv = p16 >= 0.5
print(f"among questions the 16-bit model usually solves (p>=0.5, n={int(solv.sum())}): share dropping by >=0.5 = {np.mean(d[solv] <= -0.5):.3f}; correlation of drop with steps {np.corrcoef(d[solv], feat['gold solution steps'][solv])[0,1]:+.2f}, with question length {np.corrcoef(d[solv], feat['question length (chars)'][solv])[0,1]:+.2f}, with 4-bit minus 16-bit truncation {np.corrcoef(d[solv], (feat['4-bit truncation share']-feat['16-bit truncation share'])[solv])[0,1]:+.2f}")

# ------------------------------------------------------------------ (c) sample-level outliers
print("\n=== (c) Sample-level outliers, GSM8K 8-sample runs: share of samples that are truncated / extreme / duplicated")
print("model | prec | truncated | complete samples with answer magnitude > 1e6 | complete samples with no answer | exact duplicate of another sample of the same question")
for (dd, m, n), P in sorted(G.items()):
    if dd != "gsm8k" or n != 8:
        continue
    for p in PREC:
        if p not in P:
            continue
        S = [(q, s) for q, v in P[p]["Q"].items() for s in v["S"]]
        cut = np.mean([s["cut"] for _, s in S])
        comp = [s for _, s in S if not s["cut"]]
        def big(s):
            try:
                return s["ans"] is not None and abs(Fraction(s["ans"])) > 10**6
            except Exception:
                return False
        dup = 0
        for q, v in P[p]["Q"].items():
            c = collections.Counter(s["h"] for s in v["S"])
            dup += sum(k - 1 for k in c.values() if k > 1)
        print(f"  {short(m):13s} {p:5s} | {cut:.3f} | {np.mean([big(s) for s in comp]):.4f} | {np.mean([s['ans'] is None for s in comp]):.4f} | {dup/len(S):.4f}")
