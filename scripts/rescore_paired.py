"""
Offline, paired re-scoring of every generator run -- no GPU needed.

Why: the stored per-run metrics were computed (a) with older extractors, (b) over truncated
generations, and (c) over different question sets per precision. The raw responses and
cutoff flags are all stored, so we rebuild the metrics from them with the current
extractor/matcher and compare precisions on the *same* questions with bootstrap CIs.

Views reported per (dataset, model, n_samples, precision):
  all      every sample votes (legacy behaviour, but with the fixed extractor/normalised vote)
  strict   truncated samples (hit max_new_tokens) do not vote
  nt       strict, restricted to questions with zero truncated samples at EVERY precision
           (removes the truncation-differs-by-precision confound; coverage is reported)

Usage: python scripts/rescore_paired.py [--results results] [--out results/rescored] [--boot 2000]
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.extractors import get_extractor
from src.experiment_d.metrics import compute_metrics

PRECISIONS = ["16bit", "8bit", "4bit"]


def auroc(conf, y):
    conf, y = np.asarray(conf, float), np.asarray(y, bool)
    pos, neg = conf[y], conf[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def ece(conf, y, bins=10):
    conf, y = np.asarray(conf, float), np.asarray(y, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def score_file(path, dataset):
    extractor = get_extractor(dataset)
    out = {}
    for line in open(path):
        rec = json.loads(line)
        if rec.get("type") == "run_config" or "raw_samples" not in rec:
            continue
        samples = rec["raw_samples"]
        extracted = [extractor(s["response"]) for s in samples]
        cutoffs = [bool(s["cutoff"]) for s in samples]
        gt = rec["ground_truth"]
        out[rec["qid"]] = {
            "all": compute_metrics(extracted, gt),
            "strict": compute_metrics(extracted, gt, cutoffs=cutoffs, drop_truncated=True),
            "cut": float(np.mean(cutoffs)),
            "n_cut": int(sum(cutoffs)),
        }
    return out


def load_runs(results_dir):
    runs = collections.defaultdict(dict)  # (dataset, model, n) -> precision -> {qid: scores}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "generator", "*", "*.jsonl"))):
        dataset = path.split(os.sep)[-4]
        m = re.match(r"(.+)_(16bit|8bit|4bit)_.*_n(\d+)\.jsonl", os.path.basename(path))
        if not m:
            continue
        model, prec, n = m.group(1), m.group(2), int(m.group(3))
        ds_key = "ChilleD/SVAMP" if dataset == "ChilleD_SVAMP" else dataset
        runs[(dataset, model, n)][prec] = score_file(path, ds_key)
        print(f"scored {os.path.basename(path)}", file=sys.stderr)
    return runs


def arrays(Q, qids, view):
    y = np.array([Q[q][view]["correctness"] for q in qids], bool)
    c = np.array([Q[q][view]["confidence_all"] for q in qids], float)
    p = np.array([Q[q][view]["oracle_correctness"] for q in qids], bool)
    return y, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/rescored")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for (dataset, model, n), P in sorted(load_runs(args.results).items()):
        common = sorted(set.intersection(*[set(q) for q in P.values()]))
        notrunc = [q for q in common if all(P[p][q]["n_cut"] == 0 for p in P)]
        for view, qids in (("all", common), ("strict", common), ("nt", notrunc)):
            if len(qids) < 20:
                continue
            vname = "strict" if view == "nt" else view
            nq = len(qids)
            base = arrays(P["16bit"], qids, vname) if "16bit" in P else None
            for prec in PRECISIONS:
                if prec not in P:
                    continue
                y, c, orc = arrays(P[prec], qids, vname)
                row = {
                    "dataset": dataset, "model": model, "n_samples": n, "precision": prec,
                    "view": view, "n_questions": nq, "coverage_of_common": nq / len(common),
                    "accuracy": y.mean(), "pass_at_k": orc.mean(),
                    "mean_conf_all": c.mean(), "auroc": auroc(c, y), "ece": ece(c, y),
                    "cutoff_rate": float(np.mean([P[prec][q]["cut"] for q in qids])),
                }
                if base is not None and prec != "16bit":
                    yb, cb, _ = base
                    d_acc, d_auc = [], []
                    for _ in range(args.boot):
                        ix = rng.integers(0, nq, nq)  # paired: same resampled questions for both
                        d_acc.append(y[ix].mean() - yb[ix].mean())
                        d_auc.append(auroc(c[ix], y[ix]) - auroc(cb[ix], yb[ix]))
                    row.update({
                        "d_acc": y.mean() - yb.mean(),
                        "d_acc_lo": np.percentile(d_acc, 2.5), "d_acc_hi": np.percentile(d_acc, 97.5),
                        "d_auroc": auroc(c, y) - auroc(cb, yb),
                        "d_auroc_lo": np.nanpercentile(d_auc, 2.5), "d_auroc_hi": np.nanpercentile(d_auc, 97.5),
                    })
                rows.append(row)

    import csv
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path = os.path.join(args.out, "paired_generator_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    for r in rows:
        if r["view"] == "strict":
            d = (f" dAcc={r['d_acc']*100:+.1f} [{r['d_acc_lo']*100:+.1f},{r['d_acc_hi']*100:+.1f}]"
                 f" dAUROC={r['d_auroc']:+.3f} [{r['d_auroc_lo']:+.3f},{r['d_auroc_hi']:+.3f}]") if "d_acc" in r else ""
            print(f"{r['dataset']:14s} {r['model'][-22:]:22s} n{r['n_samples']:<2d} {r['precision']:5s} "
                  f"nQ={r['n_questions']:4d} acc={r['accuracy']*100:5.1f} auroc={r['auroc']:.3f} "
                  f"cut={r['cutoff_rate']:.2f}{d}")


if __name__ == "__main__":
    main()
