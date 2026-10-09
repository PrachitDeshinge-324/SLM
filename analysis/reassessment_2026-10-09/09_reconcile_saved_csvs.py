"""Reconcile the saved metric CSVs (results/*/generator/metrics/csvs, verifier_output/metrics/csvs,
standardized_verifier_metrics) with a rescoring of the raw generations.

Until 9 Oct 2026 the generator CSVs were written (by scripts/analyze_results.py, now
scripts/report_generator_metrics.py) from each record's stored
`metrics.*` fields, i.e. whatever extractor and vote rule existed when the run was made.
This script recomputes, for the same files:
  stored   mean of the stored metrics.correctness            (should equal the CSV)
  all      current extractor, every sample votes
  strict   current extractor, truncated samples do not vote  (the rule used in the dossier)
and, for verifier files, how many stored trace labels change under the current extractor.
"""
import collections, csv, glob, json, os, re, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.self_consistency import summarize_votes
from src.experiment_d.answer_matching import answers_match
from src.experiment_d.verifier import parse_verdict
from common import short

RES = os.path.join(REPO, "results")


def gen_files():
    for path in sorted(glob.glob(os.path.join(RES, "*", "generator", "*", "*.jsonl"))):
        base = os.path.basename(path)
        m = re.match(r"(.+)_(16bit|8bit|4bit)_(.*)_n(\d+)\.jsonl$", base)
        if not m or "_backup" in base:
            continue
        yield path, path.split(os.sep)[-4], m.group(1), m.group(2), int(m.group(4))


def score(path, dataset):
    ex = get_extractor("ChilleD/SVAMP" if dataset == "ChilleD_SVAMP" else dataset)
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "raw_samples" not in r:
            continue
        S = r["raw_samples"]
        extracted = [ex(s["response"]) for s in S]
        cut = [bool(s["cutoff"]) for s in S]
        stored_changed = np.mean([(a is None) != (s.get("extracted") is None) or
                                  (a is not None and s.get("extracted") is not None and not answers_match(a, s["extracted"]))
                                  for a, s in zip(extracted, S)])
        out[r["qid"]] = {
            "stored": bool(r.get("metrics", {}).get("correctness", False)),
            "stored_conf": float(r.get("metrics", {}).get("confidence", 0.0)),
            "all": summarize_votes(extracted, r["ground_truth"])["correctness"],
            "strict": summarize_votes(extracted, r["ground_truth"], cutoffs=cut, drop_truncated=True)["correctness"],
            "changed": stored_changed,
        }
    return out


print("=== A. Generator CSVs versus rescoring (each file's own question set, as in the CSV)")
print("dataset | model | n | prec | nQ | CSV/stored acc | all-vote acc | strict acc | stored - strict | samples whose extracted answer changed")
runs = collections.defaultdict(dict)
for path, ds, model, prec, n in gen_files():
    Q = score(path, ds)
    runs[(ds, model, n)][prec] = Q
    f = lambda k: 100 * np.mean([q[k] for q in Q.values()])
    print(f"{ds[:7]:7s} {short(model):13s} n{n:<2d} {prec:5s} {len(Q):4d} | {f('stored'):5.1f} | {f('all'):5.1f} | {f('strict'):5.1f} | {f('stored')-f('strict'):+5.1f} | {100*np.mean([q['changed'] for q in Q.values()]):4.1f}%")

print("\n=== B. Deltas versus 16-bit: what the CSV reports (unpaired, stored metrics) against paired strict rescoring on common questions")
print("dataset | model | n | prec | CSV delta | stored, common q | all-vote, common q | strict, common q | n common")
csv_delta = {}
for p in glob.glob(os.path.join(RES, "*", "generator", "metrics", "csvs", "results_*.csv")):
    ds = p.split(os.sep)[-5]
    for row in csv.DictReader(open(p)):
        csv_delta[(ds, row["model"].replace("/", "_"), int(row["n_samples"]), row["precision"])] = row["Acc_Delta_vs_16bit_%"]
for (ds, model, n), P in sorted(runs.items()):
    if "16bit" not in P:
        continue
    for prec in ("8bit", "4bit"):
        if prec not in P:
            continue
        common = sorted(set(P["16bit"]) & set(P[prec]))
        d = lambda k: 100 * (np.mean([P[prec][q][k] for q in common]) - np.mean([P["16bit"][q][k] for q in common]))
        cd = csv_delta.get((ds, model, n, prec), "")
        print(f"{ds[:7]:7s} {short(model):13s} n{n:<2d} {prec:4s} | {cd if cd != '' else '  (blank)':>9s} | {d('stored'):+6.1f} | {d('all'):+6.1f} | {d('strict'):+6.1f} | {len(common)}")

print("\n=== C. Verifier CSVs: stored trace labels versus labels from the current extractor; share of traces that are truncated")
print("dataset | model | prec | traces | stored-correct | relabelled-correct | labels changed | no verdict parsed | truncated (joined to generator file) | traces not found in current generator file")
for vp in sorted(glob.glob(os.path.join(RES, "*", "verifier_output", "*", "verifier_output_*_n8.jsonl"))):
    if "_backup" in vp:
        continue
    ds, model = vp.split(os.sep)[-4], vp.split(os.sep)[-2]
    prec = re.search(r"_(16bit|8bit|4bit)_", os.path.basename(vp)).group(1)
    ex = get_extractor("ChilleD/SVAMP" if ds == "ChilleD_SVAMP" else ds)
    gp = os.path.join(RES, ds, "generator", model, os.path.basename(vp).replace("verifier_output_", ""))
    gen = {}
    if os.path.exists(gp):
        for line in open(gp):
            r = json.loads(line)
            if "raw_samples" in r:
                gen[r["question"]] = r["raw_samples"]
    n = st = re_ = ch = nov = cut = missing = 0
    for line in open(vp):
        r = json.loads(line)
        a = ex(r["student_solution"])
        new = bool(a is not None and answers_match(a, r["gt_answer"]))
        n += 1; st += bool(r["is_correct"]); re_ += new; ch += (new != bool(r["is_correct"]))
        nov += parse_verdict(r.get("verifier_raw_response", "")) is None
        S = gen.get(r["question"])
        if S is None or r["trace_index"] >= len(S) or S[r["trace_index"]]["response"] != r["student_solution"]:
            missing += 1
        else:
            cut += bool(S[r["trace_index"]]["cutoff"])
    print(f"{ds[:7]:7s} {short(model):13s} {prec:5s} {n:5d} | {st/n:.3f} | {re_/n:.3f} | {100*ch/n:4.1f}% | {100*nov/n:4.1f}% | {100*cut/max(n-missing,1):4.1f}% | {100*missing/n:5.1f}%")
