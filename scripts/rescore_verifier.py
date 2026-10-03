"""
Offline re-scoring of verifier runs -- no GPU needed.

Fixes applied to the stored verifier outputs:
  * is_correct is recomputed from the stored student trace with the current extractor/matcher
    (stored labels came from older extractors).
  * Each trace is joined to its generator cutoff flag; truncated traces are excluded because
    their "answer" is a partial scratchpad.
  * Verdicts are re-parsed strictly: only an explicit Correct/Incorrect statement counts.
    The model re-solving the problem and having its last number compared to the student's
    (parser fallback 6) is NOT a verdict, and the prompt's own echoed instruction line
    ("Final Conclusion: Correct (or Final Conclusion: Incorrect)") is stripped first.

Reports per (dataset, model, precision) on questions common to all precisions:
  parse coverage, TPR, FPR, balanced accuracy, approval rate, and a free baseline that
  approves a trace iff its answer equals the question's plurality answer. Paired cluster
  bootstrap (over questions) gives CIs for the 8/4-bit deltas vs 16-bit.

Caveat that cannot be fixed offline: each precision verifies its own generator's traces, so
the trace pool differs across precisions. FPR/TPR are rates within the correct/incorrect
classes, which removes most of the prevalence shift, but a crossed design is still needed.

Usage: python scripts/rescore_verifier.py [--results results] [--out results/rescored] [--boot 2000]
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.extractors import get_extractor
from src.experiment_d.metrics import compute_metrics  # noqa: F401  (kept for parity of imports)
from src.experiment_d.utils import answers_match, normalize_answer
from src.experiment_d.verifier import parse_verdict

PRECISIONS = ["16bit", "8bit", "4bit"]


def parse_verdict_strict(response):
    return parse_verdict(response)


def load_cutoffs(path):
    """question -> list of cutoff flags per trace_index."""
    out = {}
    for line in open(path):
        rec = json.loads(line)
        if "raw_samples" in rec:
            out[rec["question"]] = [bool(s["cutoff"]) for s in rec["raw_samples"]]
    return out


def tidy_file(vpath, gpath, dataset):
    extractor = get_extractor("ChilleD/SVAMP" if dataset == "ChilleD_SVAMP" else dataset)
    cutoffs = load_cutoffs(gpath)
    byq = collections.defaultdict(list)
    for line in open(vpath):
        r = json.loads(line)
        flags = cutoffs.get(r["question"])
        if flags is None or r["trace_index"] >= len(flags):
            continue
        ans = extractor(r["student_solution"])
        byq[r["question"]].append({
            "ans": None if ans is None else normalize_answer(ans),
            "correct": bool(ans is not None and answers_match(ans, r["gt_answer"])),
            "complete": not flags[r["trace_index"]],
            "verdict": parse_verdict_strict(r.get("verifier_raw_response", "")),
        })
    return byq


def counts(byq, qids):
    """Per-question [tp, fn, fp, tn, parsed, total, base_tp, base_fn, base_fp, base_tn] over complete traces."""
    rows = []
    for q in qids:
        traces = byq[q]
        plurality = collections.Counter(t["ans"] for t in traces if t["ans"] is not None)
        top = plurality.most_common(1)[0][0] if plurality else None
        c = np.zeros(10)
        for t in traces:
            if not t["complete"]:
                continue
            c[5] += 1
            base = t["ans"] is not None and t["ans"] == top
            if t["correct"]:
                c[6] += base
                c[7] += not base
            else:
                c[8] += base
                c[9] += not base
            if t["verdict"] is None:
                continue
            c[4] += 1
            if t["correct"]:
                c[0] += t["verdict"]
                c[1] += not t["verdict"]
            else:
                c[2] += t["verdict"]
                c[3] += not t["verdict"]
        rows.append(c)
    return np.array(rows)


def rates(c):
    tp, fn, fp, tn = c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum(), c[:, 3].sum()
    tpr = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    btp, bfn, bfp, btn = c[:, 6].sum(), c[:, 7].sum(), c[:, 8].sum(), c[:, 9].sum()
    btpr = btp / (btp + bfn) if btp + bfn else np.nan
    bfpr = bfp / (bfp + btn) if bfp + btn else np.nan
    return {
        "tpr": tpr, "fpr": fpr, "balanced_acc": (tpr + 1 - fpr) / 2,
        "approval": (tp + fp) / (tp + fp + fn + tn) if tp + fp + fn + tn else np.nan,
        "parse_coverage": c[:, 4].sum() / c[:, 5].sum() if c[:, 5].sum() else np.nan,
        "n_complete_traces": int(c[:, 5].sum()),
        "plurality_balanced_acc": (btpr + 1 - bfpr) / 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/rescored")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(args.out, exist_ok=True)

    groups = collections.defaultdict(dict)  # (dataset, model) -> prec -> byq
    for vpath in sorted(glob.glob(os.path.join(args.results, "*", "verifier_output", "*", "verifier_output_*_n8.jsonl"))):
        parts = vpath.split(os.sep)
        dataset, model = parts[-4], parts[-2]
        m = re.match(r"verifier_output_(.+)_(16bit|8bit|4bit)_.*_n8\.jsonl", os.path.basename(vpath))
        if not m:
            continue
        prec = m.group(2)
        gpath = os.path.join(args.results, dataset, "generator", model,
                             os.path.basename(vpath).replace("verifier_output_", ""))
        if not os.path.exists(gpath):
            print(f"skip {vpath}: generator file missing", file=sys.stderr)
            continue
        groups[(dataset, model)][prec] = tidy_file(vpath, gpath, dataset)
        print(f"scored {os.path.basename(vpath)}", file=sys.stderr)

    out_rows = []
    for (dataset, model), P in sorted(groups.items()):
        common = sorted(set.intersection(*[set(b) for b in P.values()]))
        if len(common) < 20:
            continue
        C = {p: counts(P[p], common) for p in P}
        nq = len(common)
        for prec in PRECISIONS:
            if prec not in C:
                continue
            row = {"dataset": dataset, "model": model, "precision": prec, "n_questions": nq,
                   **rates(C[prec])}
            if prec != "16bit" and "16bit" in C:
                for key in ("fpr", "tpr", "balanced_acc"):
                    deltas = []
                    for _ in range(args.boot):
                        ix = rng.integers(0, nq, nq)
                        deltas.append(rates(C[prec][ix])[key] - rates(C["16bit"][ix])[key])
                    row[f"d_{key}"] = row[key] - rates(C["16bit"])[key]
                    row[f"d_{key}_lo"] = float(np.nanpercentile(deltas, 2.5))
                    row[f"d_{key}_hi"] = float(np.nanpercentile(deltas, 97.5))
            out_rows.append(row)

    fields = list(dict.fromkeys(k for r in out_rows for k in r))
    path = os.path.join(args.out, "paired_verifier_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {path} ({len(out_rows)} rows)")
    for r in out_rows:
        d = ""
        if "d_fpr" in r:
            d = (f" dFPR={r['d_fpr']:+.3f} [{r['d_fpr_lo']:+.3f},{r['d_fpr_hi']:+.3f}]"
                 f" dTPR={r['d_tpr']:+.3f} [{r['d_tpr_lo']:+.3f},{r['d_tpr_hi']:+.3f}]")
        print(f"{r['dataset']:13s}{r['model'][-20:]:20s}{r['precision']:6s}nQ={r['n_questions']:4d} "
              f"traces={r['n_complete_traces']:5d} parse={r['parse_coverage']:.2f} "
              f"TPR={r['tpr']:.2f} FPR={r['fpr']:.2f} balAcc={r['balanced_acc']:.3f} "
              f"pluralityBalAcc={r['plurality_balanced_acc']:.3f}{d}")


if __name__ == "__main__":
    main()
