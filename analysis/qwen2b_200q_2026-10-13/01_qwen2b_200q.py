"""
Qwen3.5-2B on GSM8K at 200 questions, 8 samples, 16/8/4-bit: generator and self-verifier, from the files
currently in results/. Everything is re-scored here with the current extractor (nothing stored is trusted).

Sections
  (a) generator: per-sample and voted accuracy, coverage, agreement, labels kept at >= 0.75, truncation,
      paired change against 16-bit with 95% bootstrap intervals over questions; all 200 and each half
  (b) correctness flips against 16-bit
  (c) verifier: signal detection (d', c) on complete traces, approval of truncated traces
  (d) trace level: what a training set would contain under a vote filter, a verifier filter, and both
"""
import collections
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.evaluation_stats import auroc, bootstrap_interval, sensitivity_and_criterion
from src.experiment_d.verifier import parse_verdict

MODEL, DS, PREC = "Qwen_Qwen3.5-2B", "gsm8k", ["16bit", "8bit", "4bit"]
EX = get_extractor("gsm8k")
RES = os.path.join(REPO, "results", DS)
B = 1000


def qnum(qid):
    return int(qid.rsplit("_", 1)[1])


# ------------------------------------------------------------------ load generator files
G = {}
for p in PREC:
    G[p] = {}
    path = os.path.join(RES, "generator", MODEL, f"{MODEL}_{p}_{DS}_n8.jsonl")
    for line in open(path):
        r = json.loads(line)
        if "raw_samples" not in r:
            continue
        S = []
        for s in r["raw_samples"]:
            a = EX(s["response"])
            S.append({"resp": s["response"], "cut": bool(s["cutoff"]),
                      "ans": None if a is None else normalize_answer(a),
                      "correct": bool(a is not None and answers_match(a, r["ground_truth"]))})
        G[p][r["qid"]] = {"S": S, "question": r["question"]}


def qstat(S):
    """Same definitions as the 9 Oct analysis: truncated samples never vote and never count as correct."""
    valid = [s for s in S if s["ans"] is not None and not s["cut"]]
    if valid:
        key, c = collections.Counter(s["ans"] for s in valid).most_common(1)[0]
    else:
        key, c = None, 0
    return {
        "single": np.mean([s["correct"] and not s["cut"] for s in S]),
        "maj": bool(key is not None and any(s["correct"] for s in valid if s["ans"] == key)),
        "pass": any(s["correct"] for s in valid),
        "conf": c / len(S), "key": key, "cut": np.mean([s["cut"] for s in S]),
    }


def arrays(p, qids):
    st = [qstat(G[p][q]["S"]) for q in qids]
    return {k: np.array([s[k] for s in st], float) for k in ("single", "maj", "pass", "conf", "cut")}


common = sorted(set.intersection(*[set(G[p]) for p in PREC]), key=qnum)
halves = {"all 200": common, "questions 1-100": common[:100], "questions 101-200": common[100:]}
print(f"Generator files: " + ", ".join(f"{p}: {len(G[p])} questions" for p in PREC) + f"; shared: {len(common)}")


def ci(f, N):
    lo, hi = bootstrap_interval(f, N, resamples=B)
    return lo, hi


def fmt(x, lo, hi, s=100):
    star = "*" if lo > 0 or hi < 0 else " "
    return f"{s*x:+5.1f} [{s*lo:+5.1f},{s*hi:+5.1f}]{star}"


# ------------------------------------------------------------------ (a) generator
print("\n=== (a) Generator. Levels per precision, then paired change against 16-bit (95% bootstrap over questions; * = interval excludes 0)")
for name, qids in halves.items():
    A = {p: arrays(p, qids) for p in PREC}
    N = len(qids)
    print(f"\n--- {name} (n = {N} questions)")
    print("prec | per-sample | voted | pass@8 | mean agreement | kept at >=0.75 | label precision (kept) | AUROC agreement | truncated samples")
    for p in PREC:
        a = A[p]
        keep = a["conf"] >= 0.75
        print(f"{p:5s} | {a['single'].mean():.3f} | {a['maj'].mean():.3f} | {a['pass'].mean():.3f} | {a['conf'].mean():.3f} | "
              f"{keep.mean():.3f} | {a['maj'][keep].mean():.3f} (n={int(keep.sum())}) | {auroc(a['conf'], a['maj'].astype(bool)):.3f} | {a['cut'].mean():.3f}")
    print("change vs 16-bit: prec | per-sample | voted | pass@8 | mean agreement | kept at >=0.75 | label precision")
    for p in ("8bit", "4bit"):
        a, b = A[p], A["16bit"]
        ka, kb = a["conf"] >= 0.75, b["conf"] >= 0.75
        rows = []
        for k in ("single", "maj", "pass", "conf"):
            d = a[k] - b[k]
            rows.append(fmt(d.mean(), *ci(lambda ix, d=d: d[ix].mean(), N)))
        dk = ka.astype(float) - kb.astype(float)
        rows.append(fmt(dk.mean(), *ci(lambda ix: dk[ix].mean(), N)))
        prec = lambda ix: (a["maj"][ix][ka[ix]].mean() if ka[ix].any() else np.nan) - (b["maj"][ix][kb[ix]].mean() if kb[ix].any() else np.nan)
        rows.append(fmt(prec(np.arange(N)), *ci(prec, N)))
        print(f"{p:5s} | " + " | ".join(rows))

print("\nAUROC of agreement for correctness of the voted answer, change vs 16-bit (all 200), 95% bootstrap")
Aall = {p: arrays(p, common) for p in PREC}
for p in ("8bit", "4bit"):
    f = lambda ix, p=p: auroc(Aall[p]["conf"][ix], Aall[p]["maj"][ix].astype(bool)) - auroc(Aall["16bit"]["conf"][ix], Aall["16bit"]["maj"][ix].astype(bool))
    est = f(np.arange(len(common)))
    lo, hi = ci(f, len(common))
    print(f"{p}: {100*est:+.1f} [{100*lo:+.1f},{100*hi:+.1f}]")

# ------------------------------------------------------------------ (b) flips
print("\n=== (b) Voted-answer correctness flips against 16-bit (all 200). Re-running the SAME model gives ~15% on Qwen-0.8B (different model; no replicate exists for Qwen-2B)")
A = {p: arrays(p, common) for p in PREC}
for p in ("8bit", "4bit"):
    a, b = A[p]["maj"], A["16bit"]["maj"]
    print(f"{p}: flips {np.mean(a != b):.3f} (correct->wrong {np.mean((b == 1) & (a == 0)):.3f}, wrong->correct {np.mean((b == 0) & (a == 1)):.3f})"
          f" | majority answer differs {np.mean([qstat(G[p][q]['S'])['key'] != qstat(G['16bit'][q]['S'])['key'] for q in common]):.3f}")

# ------------------------------------------------------------------ verifier
V = {}
for p in PREC:
    path = os.path.join(RES, "verifier_output", MODEL, f"verifier_output_{MODEL}_{p}_{DS}_n8.jsonl")
    byq = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        a = EX(r["student_solution"])
        byq[r["question"]][r["trace_index"]] = {
            "correct": bool(a is not None and answers_match(a, r["gt_answer"])),
            "verdict": parse_verdict(r.get("verifier_raw_response", "")),
            "text": r["student_solution"],
        }
    V[p] = byq
q_of = {p: {v["question"]: q for q, v in G[p].items()} for p in PREC}
full = {p: {q for q, t in V[p].items() if len(t) == 8} for p in PREC}
vq = sorted(set.intersection(*[{q_of[p][s] for s in full[p] if s in q_of[p]} for p in PREC]), key=qnum)
print(f"\nVerifier files: " + ", ".join(f"{p}: {len(full[p])} questions fully judged" for p in PREC) + f"; shared: {len(vq)}")

T = {p: [] for p in PREC}   # per question: list of trace dicts with generator-side fields
for p in PREC:
    for q in vq:
        S = G[p][q]["S"]
        recs = V[p][G[p][q]["question"]]
        row = []
        for i in range(8):
            v = recs[i]
            joined = S[i]["resp"] == v["text"]
            row.append({"correct": v["correct"], "verdict": v["verdict"], "cut": S[i]["cut"] if joined else None,
                        "joined": joined, "ans": S[i]["ans"]})
        T[p].append(row)


def sdt(rows):
    pos = [t for t in rows if t["correct"]]
    neg = [t for t in rows if not t["correct"]]
    if not pos or not neg:
        return (np.nan,) * 4
    tpr = np.mean([t["verdict"] for t in pos])
    fpr = np.mean([t["verdict"] for t in neg])
    return (tpr, fpr) + sensitivity_and_criterion(tpr, fpr, len(pos), len(neg))


print(f"\n=== (c) Verifier signal detection on complete, parsed traces ({len(vq)} shared questions). Each precision judged its OWN traces.")
print("prec | traces | complete | verdict parsed | approval | TPR | FPR | d' [95% CI] | c [95% CI] (negative = lenient) | approves truncated traces")
for p in PREC:
    allt = [t for g in T[p] for t in g]
    comp = [[t for t in g if t["cut"] is False and t["verdict"] is not None] for g in T[p]]
    flat = [t for g in comp for t in g]
    tpr, fpr, dp, c = sdt(flat)
    N = len(comp)
    cd = ci(lambda ix: sdt([t for i in ix for t in comp[i]])[2], N)
    cc = ci(lambda ix: sdt([t for i in ix for t in comp[i]])[3], N)
    tr = [t for t in allt if t["cut"] is True and t["verdict"] is not None]
    unj = sum(not t["joined"] for t in allt)
    print(f"{p:5s} | {len(allt)} | {sum(t['cut'] is False for t in allt)} | {np.mean([t['verdict'] is not None for t in allt if t['cut'] is False]):.3f} | "
          f"{np.mean([t['verdict'] for t in flat]):.3f} | {tpr:.3f} | {fpr:.3f} | {dp:+.2f} [{cd[0]:+.2f},{cd[1]:+.2f}] | {c:+.2f} [{cc[0]:+.2f},{cc[1]:+.2f}] | "
          f"{np.mean([t['verdict'] for t in tr]) if tr else float('nan'):.3f} (n={len(tr)})" + (f"  [unjoined traces: {unj}]" if unj else ""))

# within-question discrimination
print("\nWithin-question discrimination (questions with both correct and wrong complete traces): approval(correct) - approval(wrong)")
for p in PREC:
    diffs = []
    for g in T[p]:
        rows = [t for t in g if t["cut"] is False and t["verdict"] is not None]
        pos = [t["verdict"] for t in rows if t["correct"]]
        neg = [t["verdict"] for t in rows if not t["correct"]]
        if pos and neg:
            diffs.append(np.mean(pos) - np.mean(neg))
    d = np.array(diffs)
    lo, hi = ci(lambda ix: d[ix].mean(), len(d))
    print(f"{p:5s} | mixed questions {len(d):3d} | {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")

# ------------------------------------------------------------------ (d) trace level
print(f"\n=== (d) Trace level: what a self-training set would contain ({len(vq)} shared questions, 8 traces each = {8*len(vq)} traces)")
print("Share of ALL traces that are correct is the baseline. 'kept' = share of all traces that pass the filter; 'correct' = share of kept traces whose answer is right.")
print("prec | all traces correct | complete traces only (no filter): kept / correct | vote filter (agreement>=0.75, trace = majority answer): kept / correct | verifier filter (complete + approved): kept / correct | both: kept / correct | vote-filtered questions with a wrong label")
for p in PREC:
    n_all = 8 * len(vq)
    corr_all = np.mean([t["correct"] and t["cut"] is False for g in T[p] for t in g])
    vote, ver, both = [], [], []
    comp = [t["correct"] for g in T[p] for t in g if t["cut"] is False]
    wrong_q = 0
    for q, g in zip(vq, T[p]):
        st = qstat(G[p][q]["S"])
        for i, t in enumerate(g):
            ok_c = t["correct"] and t["cut"] is False
            in_vote = st["conf"] >= 0.75 and t["ans"] is not None and t["ans"] == st["key"] and t["cut"] is False
            in_ver = t["cut"] is False and t["verdict"] is True
            if in_vote: vote.append(ok_c)
            if in_ver: ver.append(ok_c)
            if in_vote and in_ver: both.append(ok_c)
        wrong_q += (st["conf"] >= 0.75) and not st["maj"]
    f = lambda L: f"{len(L)/n_all:.3f} / {np.mean(L):.3f}" if L else " - "
    print(f"{p:5s} | {corr_all:.3f} | {f(comp)} | {f(vote)} | {f(ver)} | {f(both)} | {wrong_q}")
