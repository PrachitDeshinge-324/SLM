"""Verifier deep-dive on stored verdicts (signal detection, plurality comparison, within-question discrimination)."""
import glob, json, os, re, sys, collections, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.verifier import parse_verdict
from common import short, bootstrap_interval, sensitivity_and_criterion

from src.experiment_d.answer_extraction import _last_boxed

# A trace counts as truncated when the generator file says so. The join is by exact trace text:
# when the generator file was later overwritten by a rerun (MATH-500 Qwen-0.8B 16/4-bit), the judged
# traces no longer exist there, so for that model a format heuristic is used at every precision
# (MATH-500 only: "complete" = the trace contains a closed \boxed{...}).
V = collections.defaultdict(dict)
UNJOINED = collections.defaultdict(dict)
for vp in sorted(glob.glob(os.path.join(REPO, "results", "*", "verifier_output", "*", "verifier_output_*_n8.jsonl"))):
    if "_backup" in vp: continue
    ds, model = vp.split(os.sep)[-4], vp.split(os.sep)[-2]
    prec = re.search(r"_(16bit|8bit|4bit)_", os.path.basename(vp)).group(1)
    gp = os.path.join(REPO, "results", ds, "generator", model, os.path.basename(vp).replace("verifier_output_", ""))
    gen = {}
    if os.path.exists(gp):
        for line in open(gp):
            r = json.loads(line)
            if "raw_samples" in r: gen[r["question"]] = r["raw_samples"]
    ex = get_extractor("ChilleD/SVAMP" if ds == "ChilleD_SVAMP" else ds)
    byq = collections.defaultdict(list); miss = tot = 0
    for line in open(vp):
        r = json.loads(line)
        a = ex(r["student_solution"])
        S = gen.get(r["question"]); i = r["trace_index"]
        joined = S is not None and i < len(S) and S[i]["response"] == r["student_solution"]
        tot += 1; miss += not joined
        byq[r["question"]].append({
            "ans": None if a is None else normalize_answer(a),
            "correct": bool(a is not None and answers_match(a, r["gt_answer"])),
            "cut_join": bool(S[i]["cutoff"]) if joined else None,
            "cut_heur": _last_boxed(r["student_solution"]) is None,
            "verdict": parse_verdict(r.get("verifier_raw_response", "")),
        })
    UNJOINED[(ds, model)][prec] = miss / max(tot, 1)
    V[(ds, model)][prec] = byq
for key, P in V.items():
    heur = any(v > 0.5 for v in UNJOINED[key].values())
    if heur:
        print(f"NOTE {key[0]} {short(key[1])}: judged traces are missing from the current generator file at {[p for p, v in UNJOINED[key].items() if v > 0.5]}; truncation decided by the boxed-answer heuristic at every precision")
    for prec, byq in P.items():
        for q, T in byq.items():
            for t in T: t["cut"] = t["cut_heur"] if heur else bool(t["cut_join"])
            c = collections.Counter(t["ans"] for t in T if t["ans"] is not None and not t["cut"])
            top = c.most_common(1)[0][0] if c else None
            for t in T: t["plur"] = t["ans"] is not None and t["ans"] == top

def sdt(T):
    pos = [t for t in T if t["correct"]]; neg = [t for t in T if not t["correct"]]
    if not pos or not neg: return (np.nan,) * 5
    tpr = np.mean([t["verdict"] for t in pos]); fpr = np.mean([t["verdict"] for t in neg])
    d_prime, criterion = sensitivity_and_criterion(tpr, fpr, len(pos), len(neg))
    return tpr, fpr, d_prime, criterion, (tpr + 1 - fpr) / 2

print("=== V1/V4: signal-detection view on COMMON questions, complete + parsed traces. d' = sensitivity, c = criterion (negative = lenient)")
print("cell | prec | nQ | traces | parsed | approval | TPR | FPR | d' [95% CI] | c [95% CI] | approve truncated traces")
for (ds, model), P in sorted(V.items()):
    common = sorted(set.intersection(*[set(b) for b in P.values()]))
    for prec in ["16bit", "8bit", "4bit"]:
        if prec not in P: continue
        per_q = [[t for t in P[prec][q] if not t["cut"] and t["verdict"] is not None] for q in common]
        allc = [t for q in common for t in P[prec][q] if not t["cut"]]
        T = [t for g in per_q for t in g]
        if not T: continue
        tpr, fpr, dp, c, ba = sdt(T)
        N = len(common)
        cd = bootstrap_interval(lambda ix: sdt([t for i in ix for t in per_q[i]])[2], N, resamples=600); cc = bootstrap_interval(lambda ix: sdt([t for i in ix for t in per_q[i]])[3], N, resamples=600)
        tr = [t for q in common for t in P[prec][q] if t["cut"] and t["verdict"] is not None]
        print(f"{ds[:7]:7s} {short(model):13s} | {prec:5s} | {N:3d} | {len(allc):4d} | {len(T)/max(len(allc),1):.2f} | {np.mean([t['verdict'] for t in T]):.2f} | {tpr:.2f} | {fpr:.2f} | {dp:+.2f} [{cd[0]:+.2f},{cd[1]:+.2f}] | {c:+.2f} [{cc[0]:+.2f},{cc[1]:+.2f}] | {np.mean([t['verdict'] for t in tr]) if tr else float('nan'):.2f} (n={len(tr)})")
    print()

print("=== V2: does the verdict add information beyond 'answer equals the plurality answer'?  P(correct | plurality?, verdict)")
print("cell | prec | in-plurality: approve / reject (n) | minority: approve / reject (n)")
for (ds, model), P in sorted(V.items()):
    common = sorted(set.intersection(*[set(b) for b in P.values()]))
    for prec in ["16bit", "8bit", "4bit"]:
        if prec not in P: continue
        T = [t for q in common for t in P[prec][q] if not t["cut"] and t["verdict"] is not None]
        def cell(pl, vd):
            S = [t["correct"] for t in T if t["plur"] == pl and t["verdict"] == vd]
            return f"{np.mean(S):.2f} ({len(S)})" if S else "  -  (0)"
        print(f"{ds[:7]:7s} {short(model):13s} | {prec:5s} | {cell(True, True)} / {cell(True, False)} | {cell(False, True)} / {cell(False, False)}")

print("\n=== V3: within-question discrimination. Only questions that contain BOTH correct and incorrect complete traces; approval(correct) - approval(incorrect) averaged within question")
print("cell | prec | mixed questions | within-q TPR-FPR [95% CI] | pooled TPR-FPR | questions where every trace gets the same verdict")
for (ds, model), P in sorted(V.items()):
    common = sorted(set.intersection(*[set(b) for b in P.values()]))
    for prec in ["16bit", "8bit", "4bit"]:
        if prec not in P: continue
        diffs = []; same = []
        for q in common:
            T = [t for t in P[prec][q] if not t["cut"] and t["verdict"] is not None]
            if len(T) >= 2: same.append(len({t["verdict"] for t in T}) == 1)
            pos = [t["verdict"] for t in T if t["correct"]]; neg = [t["verdict"] for t in T if not t["correct"]]
            if pos and neg: diffs.append(np.mean(pos) - np.mean(neg))
        allT = [t for q in common for t in P[prec][q] if not t["cut"] and t["verdict"] is not None]
        tpr, fpr = sdt(allT)[:2]
        if len(diffs) >= 10:
            d = np.array(diffs); ci = bootstrap_interval(lambda ix: d[ix].mean(), len(d), resamples=1000)
            print(f"{ds[:7]:7s} {short(model):13s} | {prec:5s} | {len(d):3d} | {d.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] | {tpr-fpr:+.3f} | {np.mean(same):.2f}")
