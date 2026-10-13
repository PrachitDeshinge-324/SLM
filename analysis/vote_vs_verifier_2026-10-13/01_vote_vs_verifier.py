"""
Does the vote work everywhere, and does adding the model's own verifier beat the vote?
All 8-sample runs currently in results/, re-scored with the current extractor.

Table 1  every generator run: per-sample vs voted accuracy, headroom for self-training, label quality of the filter
Table 2  every run that has verifier verdicts: vote vs verifier vs vote+verifier, as selectors of which questions
         and which traces to train on
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
from src.experiment_d.answer_extraction import _last_boxed, get_extractor
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.evaluation_stats import auroc
from src.experiment_d.verifier import parse_verdict

RNG = np.random.default_rng(0)
PREC = ["16bit", "8bit", "4bit"]


def short(m):
    return (m.replace("meta-llama_", "").replace("deepseek-ai_", "").replace("google_", "").replace("Qwen_", "")
             .replace("-Instruct", "").replace("DeepSeek-R1-Distill-Qwen", "DS-R1").replace("Llama-3.2", "Llama"))


def extractor(ds):
    return get_extractor("ChilleD/SVAMP" if ds == "ChilleD_SVAMP" else ds)


def load_generator(path, ex):
    """question text -> list of samples (response, cut, ans, correct)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "raw_samples" not in r:
            continue
        S = []
        for s in r["raw_samples"]:
            a = ex(s["response"])
            S.append({"resp": s["response"], "cut": bool(s["cutoff"]),
                      "ans": None if a is None else normalize_answer(a),
                      "correct": bool(a is not None and answers_match(a, r["ground_truth"]))})
        out[r["question"]] = S
    return out


def vote(S):
    """Majority over complete, parsed samples; returns key, count, voted-correct."""
    valid = [s for s in S if s["ans"] is not None and not s["cut"]]
    if not valid:
        return None, 0, False
    key, c = collections.Counter(s["ans"] for s in valid).most_common(1)[0]
    return key, c, any(s["correct"] for s in valid if s["ans"] == key)


# ------------------------------------------------------------------ Table 1
print("=== Table 1. Is the vote working? Every 8-sample generator run, re-scored. Truncated samples count as wrong per-sample and never vote.")
print("cell | nQ | per-sample (all) | per-sample (complete only) | voted | voted - complete = sharpening headroom | pass@8 | pass@8 - voted = selection headroom | never solved | truncated | kept >=0.75 | precision of kept | kept >=0.5 | precision of kept")
T1 = []
for gp in sorted(glob.glob(os.path.join(REPO, "results", "*", "generator", "*", "*_n8.jsonl"))):
    if "_backup" in gp:
        continue
    ds, model = gp.split(os.sep)[-4], gp.split(os.sep)[-2]
    prec = re.search(r"_(16bit|8bit|4bit)_", os.path.basename(gp)).group(1)
    G = load_generator(gp, extractor(ds))
    if len(G) < 20:
        continue
    n = len(G)
    allS = [s for S in G.values() for s in S]
    comp = [s for s in allS if not s["cut"]]
    st = [vote(S) for S in G.values()]
    ps_all = np.mean([s["correct"] and not s["cut"] for s in allS])
    ps_comp = np.mean([s["correct"] for s in comp]) if comp else np.nan
    voted = np.mean([c for _, _, c in st])
    pas = np.mean([any(s["correct"] and not s["cut"] for s in S) for S in G.values()])
    conf = np.array([c / len(S) for (_, c, _), S in zip(st, G.values())])
    y = np.array([c for _, _, c in st])
    row = dict(cell=f"{ds[:7]:7s} {short(model):13s} {prec:5s}", n=n, ps_all=ps_all, ps_comp=ps_comp, voted=voted, pas=pas,
               cut=np.mean([s["cut"] for s in allS]))
    k75, k50 = conf >= 0.75, conf >= 0.5
    row.update(k75=k75.mean(), p75=y[k75].mean() if k75.any() else np.nan, k50=k50.mean(), p50=y[k50].mean() if k50.any() else np.nan)
    T1.append(row)
    print(f"{row['cell']} | {n:4d} | {ps_all:.3f} | {ps_comp:.3f} | {voted:.3f} | {voted-ps_comp:+.3f} | {pas:.3f} | {pas-voted:+.3f} | {1-pas:.3f} | {row['cut']:.3f} | "
          f"{row['k75']:.3f} | {row['p75']:.3f} | {row['k50']:.3f} | {row['p50']:.3f}")

gain_all = np.array([r["voted"] - r["ps_all"] for r in T1])
print(f"\nVoted accuracy is above per-sample accuracy counting truncated samples as wrong in {int((gain_all > 0).sum())}/{len(T1)} runs "
      f"(median {np.median(gain_all):+.3f}, range {gain_all.min():+.3f}..{gain_all.max():+.3f}); that gap includes the vote ignoring cut-off samples.")
gain = np.array([r["voted"] - r["ps_comp"] for r in T1])
lab = np.array([r["p75"] - r["voted"] for r in T1 if r["p75"] == r["p75"]])
print(f"\nVote check over {len(T1)} runs: voted accuracy is above per-sample accuracy (complete samples) in {int((gain > 0).sum())}/{len(T1)} runs "
      f"(median {np.median(gain):+.3f}, range {gain.min():+.3f}..{gain.max():+.3f}); "
      f"precision of labels kept at >=0.75 is above overall voted accuracy in {int((lab > 0).sum())}/{len(lab)} runs (median {np.median(lab):+.3f}).")

# ------------------------------------------------------------------ Table 2
print("\n\n=== Table 2. Vote vs verifier vs both. Each precision judged its own traces; only questions with all 8 verdicts.")
print("scores per question: vote = agreement of the majority answer; ver = approval rate among complete traces that carry the majority answer; both = vote x ver")
print("cell | nQ | verdict parsed | approval (complete) | voted | verifier-weighted vote | rescued/harmed | AUROC vote / ver / both | precision at k=#kept by vote>=0.75 (k): vote / ver / both | gate (vote>=0.75 & ver>=0.5): kept / precision vs vote: kept / precision")
print("trace level -> correct share of kept traces: complete (no filter) | vote | verifier | both  (kept share in brackets)")
T2 = []
for vp in sorted(glob.glob(os.path.join(REPO, "results", "*", "verifier_output", "*", "verifier_output_*_n8.jsonl"))):
    if "_backup" in vp:
        continue
    ds, model = vp.split(os.sep)[-4], vp.split(os.sep)[-2]
    prec = re.search(r"_(16bit|8bit|4bit)_", os.path.basename(vp)).group(1)
    ex = extractor(ds)
    gp = os.path.join(REPO, "results", ds, "generator", model, os.path.basename(vp).replace("verifier_output_", ""))
    gen = {}
    if os.path.exists(gp):
        for line in open(gp):
            r = json.loads(line)
            if "raw_samples" in r:
                gen[r["question"]] = r["raw_samples"]
    byq = collections.defaultdict(dict)
    for line in open(vp):
        r = json.loads(line)
        text, i = r["student_solution"], r["trace_index"]
        a = ex(text)
        S = gen.get(r["question"])
        joined = S is not None and i < len(S) and S[i]["response"] == text
        cut = bool(S[i]["cutoff"]) if joined else (_last_boxed(text) is None if ds == "math500" else False)
        byq[r["question"]][i] = {"ans": None if a is None else normalize_answer(a), "correct": bool(a is not None and answers_match(a, r["gt_answer"])),
                                 "cut": cut, "verdict": parse_verdict(r.get("verifier_raw_response", ""))}
    Q = [[t[i] for i in sorted(t)] for t in byq.values() if len(t) == 8]
    if len(Q) < 20:
        continue
    N = len(Q)
    keys, conf, y, s_ver, w_ok, tr = [], [], [], [], [], []
    for traces in Q:
        valid = [t for t in traces if t["ans"] is not None and not t["cut"]]
        c = collections.Counter(t["ans"] for t in valid)
        key = c.most_common(1)[0][0] if c else None
        keys.append(key)
        conf.append(c[key] / 8 if key is not None else 0.0)
        y.append(bool(key is not None and any(t["correct"] for t in valid if t["ans"] == key)))
        v = [t["verdict"] for t in valid if t["ans"] == key and t["verdict"] is not None]
        s_ver.append(np.mean(v) if v else 0.0)
        # verifier-weighted vote: answer with the most approved traces, ties by vote count
        appr = collections.Counter()
        for t in valid:
            appr[t["ans"]] += 1 if t["verdict"] is True else 0
        if c:
            wkey = max(c, key=lambda a: (appr[a], c[a]))
            w_ok.append(any(t["correct"] for t in valid if t["ans"] == wkey))
        else:
            w_ok.append(False)
    conf, y, s_ver, w_ok = np.array(conf), np.array(y), np.array(s_ver), np.array(w_ok)
    s_both = conf * s_ver
    k = int((conf >= 0.75).sum())

    def prec_at_k(score):
        if k == 0:
            return np.nan
        out = []
        for _ in range(100):
            order = np.lexsort((RNG.random(N), -score))
            out.append(y[order[:k]].mean())
        return float(np.mean(out))

    gate = (conf >= 0.75) & (s_ver >= 0.5)
    flat = [t for traces in Q for t in traces]
    comp_t = [t for t in flat if not t["cut"]]
    parsed = np.mean([t["verdict"] is not None for t in comp_t]) if comp_t else np.nan
    appr_rate = np.mean([t["verdict"] is True for t in comp_t if t["verdict"] is not None]) if comp_t else np.nan
    # trace level
    vt, rt, bt = [], [], []
    for traces, key, cf in zip(Q, keys, conf):
        for t in traces:
            ok = t["correct"] and not t["cut"]
            in_v = cf >= 0.75 and t["ans"] is not None and t["ans"] == key and not t["cut"]
            in_r = (not t["cut"]) and t["verdict"] is True
            if in_v: vt.append(ok)
            if in_r: rt.append(ok)
            if in_v and in_r: bt.append(ok)
    n_tr = 8 * N
    cell = f"{ds[:7]:7s} {short(model):13s} {prec:5s}"
    f = lambda L: f"{np.mean(L):.3f} ({len(L)/n_tr:.2f})" if L else "  -  "
    rescued, harmed = int((w_ok & ~y).sum()), int((~w_ok & y).sum())
    T2.append(dict(cell=cell, n=N, approval=appr_rate, parsed=parsed, auc=(auroc(conf, y), auroc(s_ver, y), auroc(s_both, y)), pk=(prec_at_k(conf), prec_at_k(s_ver), prec_at_k(s_both)),
                   rescued=rescued, harmed=harmed, vote_tr=np.mean(vt) if vt else np.nan, ver_tr=np.mean(rt) if rt else np.nan, both_tr=np.mean(bt) if bt else np.nan,
                   comp_tr=np.mean([t["correct"] for t in comp_t]) if comp_t else np.nan))
    print(f"{cell} | {N:3d} | {parsed:.2f} | {appr_rate:.2f} | {y.mean():.3f} | {w_ok.mean():.3f} | +{rescued}/-{harmed} | "
          f"{auroc(conf, y):.3f} / {auroc(s_ver, y):.3f} / {auroc(s_both, y):.3f} | ({k}) {prec_at_k(conf):.3f} / {prec_at_k(s_ver):.3f} / {prec_at_k(s_both):.3f} | "
          f"{gate.mean():.3f} / {(y[gate].mean() if gate.any() else np.nan):.3f} vs {k/N:.3f} / {(y[conf >= 0.75].mean() if k else np.nan):.3f}")
    print(f"      trace level: {np.mean([t['correct'] for t in comp_t]):.3f} ({len(comp_t)/n_tr:.2f}) | {f(vt)} | {f(rt)} | {f(bt)}")

# usable verifiers: most verdicts parsed and not a constant (approval strictly between 5% and 97%)
def usable(r):
    return r["parsed"] >= 0.9 and 0.05 < r["approval"] < 0.97
ok = [r for r in T2 if usable(r)]
qwen = [r for r in ok if "Qwen3.5" in r["cell"]]
print(f"\nSummary over {len(T2)} verifier runs; {len(ok)} usable (>=90% of verdicts parsed, approval between 5% and 97%); {len(qwen)} of those are Qwen:")
for label, S in (("all usable", ok), ("Qwen only", qwen)):
    print(f"  [{label}]")
    for name, i in (("vote", 0), ("verifier", 1), ("both", 2)):
        print(f"    AUROC for 'voted answer is correct', {name:8s}: median {np.median([r['auc'][i] for r in S]):.3f}; precision at the vote's yield: median {np.nanmedian([r['pk'][i] for r in S]):.3f}")
    print(f"    runs where 'both' beats the vote alone: AUROC {sum(r['auc'][2] > r['auc'][0] + 0.01 for r in S)}/{len(S)}, precision at the vote's yield {sum(r['pk'][2] > r['pk'][0] + 0.005 for r in S)}/{len(S)}; "
          f"worse: AUROC {sum(r['auc'][2] < r['auc'][0] - 0.01 for r in S)}/{len(S)}")
    print(f"    verifier-weighted vote: rescued {sum(r['rescued'] for r in S)} questions, harmed {sum(r['harmed'] for r in S)} (of {sum(r['n'] for r in S)} question-runs)")
    print(f"    trace-level correct share, median: complete {np.nanmedian([r['comp_tr'] for r in S]):.3f} | vote {np.nanmedian([r['vote_tr'] for r in S]):.3f} | verifier {np.nanmedian([r['ver_tr'] for r in S]):.3f} | both {np.nanmedian([r['both_tr'] for r in S]):.3f}")
