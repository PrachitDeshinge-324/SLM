"""
Single sample vs voted, for the generator and for the verifier, and what connects them.

Generator  single = share of samples that are correct (cut-off counts as wrong); voted = majority answer correct.
Verifier   single = a verdict on ONE trace is right (approve a correct trace, reject a wrong one);
           voted  = the verdicts on all traces that carry the majority answer, averaged (approve if >= 0.5),
                    judged against whether that majority answer is correct.
Because most traces/answers are correct, accuracy alone flatters an approve-everything judge, so each is
shown with its trivial baseline and as balanced accuracy (mean of hit rate and correct-rejection rate).
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
from src.experiment_d.verifier import parse_verdict


def short(m):
    return (m.replace("meta-llama_", "").replace("deepseek-ai_", "").replace("google_", "").replace("Qwen_", "")
             .replace("-Instruct", "").replace("DeepSeek-R1-Distill-Qwen", "DS-R1").replace("Llama-3.2", "Llama"))


def bal(pos_ok, neg_ok):
    """Balanced accuracy from lists of 0/1 outcomes on positives (should approve) and negatives (should reject)."""
    if not pos_ok or not neg_ok:
        return np.nan
    return (np.mean(pos_ok) + np.mean(neg_ok)) / 2


ROWS = []
for vp in sorted(glob.glob(os.path.join(REPO, "results", "*", "verifier_output", "*", "verifier_output_*_n8.jsonl"))):
    if "_backup" in vp:
        continue
    ds, model = vp.split(os.sep)[-4], vp.split(os.sep)[-2]
    prec = re.search(r"_(16bit|8bit|4bit)_", os.path.basename(vp)).group(1)
    ex = get_extractor("ChilleD/SVAMP" if ds == "ChilleD_SVAMP" else ds)
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
        byq[r["question"]][i] = {"ans": None if a is None else normalize_answer(a),
                                 "correct": bool(a is not None and answers_match(a, r["gt_answer"])),
                                 "cut": cut, "verdict": parse_verdict(r.get("verifier_raw_response", ""))}
    Q = [[t[i] for i in sorted(t)] for t in byq.values() if len(t) == 8]
    if len(Q) < 20:
        continue

    # generator side
    single = np.mean([t["correct"] and not t["cut"] for tr in Q for t in tr])
    voted_flags, dec_pos, dec_neg = [], [], []
    for tr in Q:
        valid = [t for t in tr if t["ans"] is not None and not t["cut"]]
        c = collections.Counter(t["ans"] for t in valid)
        key = c.most_common(1)[0][0] if c else None
        y = bool(key is not None and any(t["correct"] for t in valid if t["ans"] == key))
        voted_flags.append(y)
        v = [t["verdict"] for t in valid if t["ans"] == key and t["verdict"] is not None]
        if v:
            approve = np.mean(v) >= 0.5
            (dec_pos if y else dec_neg).append(int(approve) if y else int(not approve))
    voted = np.mean(voted_flags)
    n_dec = len(dec_pos) + len(dec_neg)
    voted_dec_acc = (sum(dec_pos) + sum(dec_neg)) / n_dec if n_dec else np.nan
    voted_dec_base = len(dec_pos) / n_dec if n_dec else np.nan          # approve-everything accuracy
    voted_dec_bal = bal(dec_pos, dec_neg)

    # verifier side, single trace (complete traces with a verdict)
    T = [t for tr in Q for t in tr if not t["cut"] and t["verdict"] is not None]
    parsed = len(T) / max(sum(1 for tr in Q for t in tr if not t["cut"]), 1)
    pos = [int(t["verdict"]) for t in T if t["correct"]]
    neg = [int(not t["verdict"]) for t in T if not t["correct"]]
    tr_acc = (sum(pos) + sum(neg)) / len(T) if T else np.nan
    tr_base = len(pos) / len(T) if T else np.nan
    tr_bal = bal(pos, neg)
    approval = np.mean([t["verdict"] for t in T]) if T else np.nan

    # connection: does the verdict just copy "this trace has the majority answer"?
    match = []
    for tr in Q:
        valid = [t for t in tr if t["ans"] is not None and not t["cut"]]
        c = collections.Counter(t["ans"] for t in valid)
        key = c.most_common(1)[0][0] if c else None
        for t in tr:
            if not t["cut"] and t["verdict"] is not None and t["ans"] is not None:
                match.append((t["ans"] == key, bool(t["verdict"]), t["correct"]))
    m = np.array(match, dtype=bool) if match else np.zeros((0, 3), bool)
    follows_vote = (m[:, 0] == m[:, 1]).mean() if len(m) else np.nan
    vote_alone = (m[:, 0] == m[:, 2]).mean() if len(m) else np.nan
    verdict_alone = (m[:, 1] == m[:, 2]).mean() if len(m) else np.nan
    ROWS.append(dict(cell=f"{ds[:7]:7s} {short(model):13s} {prec:5s}", n=len(Q), parsed=parsed, approval=approval,
                     single=single, voted=voted, tr_acc=tr_acc, tr_base=tr_base, tr_bal=tr_bal,
                     vd_acc=voted_dec_acc, vd_base=voted_dec_base, vd_bal=voted_dec_bal, n_dec=n_dec,
                     follows=follows_vote, vote_alone=vote_alone, verdict_alone=verdict_alone))

usable = lambda r: r["parsed"] >= 0.9 and 0.05 < r["approval"] < 0.97
print("GENERATOR single -> voted | VERIFIER single trace -> voted over the traces of the majority answer")
print("(verifier accuracy / what an approve-everything judge would score / balanced accuracy)\n")
print("cell | nQ | GEN single | GEN voted | gain | VER one trace: acc / base / bal | VER voted: acc / base / bal | bal gain | verdict follows 'has majority answer' | 'has majority answer' alone right | verdict alone right | usable verifier")
for r in ROWS:
    print(f"{r['cell']} | {r['n']:3d} | {r['single']:.3f} | {r['voted']:.3f} | {r['voted']-r['single']:+.3f} | "
          f"{r['tr_acc']:.3f} / {r['tr_base']:.3f} / {r['tr_bal']:.3f} | {r['vd_acc']:.3f} / {r['vd_base']:.3f} / {r['vd_bal']:.3f} | {r['vd_bal']-r['tr_bal']:+.3f} | "
          f"{r['follows']:.3f} | {r['vote_alone']:.3f} | {r['verdict_alone']:.3f} | {'yes' if usable(r) else 'NO'}")

U = [r for r in ROWS if usable(r)]
print(f"\nUsable verifier runs: {len(U)} of {len(ROWS)}")
g = lambda k: np.array([r[k] for r in U])
print(f"  generator: single {np.median(g('single')):.3f} -> voted {np.median(g('voted')):.3f} (median); voting gain median {np.median(g('voted')-g('single')):+.3f}")
print(f"  verifier : one trace balanced {np.nanmedian(g('tr_bal')):.3f} (acc {np.median(g('tr_acc')):.3f}, approve-all baseline {np.median(g('tr_base')):.3f}) "
      f"-> voted balanced {np.nanmedian(g('vd_bal')):.3f} (acc {np.nanmedian(g('vd_acc')):.3f}, baseline {np.nanmedian(g('vd_base')):.3f}); balanced gain median {np.nanmedian(g('vd_bal')-g('tr_bal')):+.3f}")
print(f"  verdict equals 'trace has the majority answer' in {np.median(g('follows')):.3f} of traces (median); that membership alone is right {np.median(g('vote_alone')):.3f}, the verdict alone {np.median(g('verdict_alone')):.3f}")


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 3 else np.nan


print("\nAcross usable runs (Pearson r, n=%d; cells share models, so treat as descriptive):" % len(U))
print(f"  generator single accuracy  vs verifier one-trace balanced accuracy : {corr(g('single'), g('tr_bal')):+.2f}")
print(f"  generator voted accuracy   vs verifier voted balanced accuracy     : {corr(g('voted'), g('vd_bal')):+.2f}")
print(f"  generator voting gain      vs verifier balanced gain               : {corr(g('voted')-g('single'), g('vd_bal')-g('tr_bal')):+.2f}")
print(f"  generator single accuracy  vs how often the verdict follows the vote: {corr(g('single'), g('follows')):+.2f}")
print(f"  generator voted accuracy   vs verifier approval rate               : {corr(g('voted'), g('approval')):+.2f}")
