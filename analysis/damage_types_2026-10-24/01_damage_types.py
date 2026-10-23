"""
Two kinds of quantization damage, question by question (Qwen3.5-0.8B, 16-bit against 4-bit).

Every precision has two independent runs of the same questions (n8: 8 samples; n16: 16 samples), so each
question has three independent sets of 8 samples per precision: S1 = the n8 run, S2 = the first 8 samples of
the n16 run, S3 = the last 8. S1 + S2 (16 samples) classify; S3 is held out to check the classes.

Per question, from S1 + S2: p = share of the 16 samples that are right (a cut-off sample counts as wrong) and
the plurality, scored with ties split (the share of the tied top answers that are right; > 0.5 counts as right).

  both fail    16-bit plurality wrong, 4-bit plurality wrong
  helped       16-bit plurality wrong, 4-bit plurality right
  systematic   16-bit plurality right, 4-bit plurality wrong: 4-bit settles on another answer
  random       both pluralities right, but 4-bit gets at least 2 of 16 fewer samples right: it scatters more
  preserved    both pluralities right, 4-bit within 1 of 16 samples of 16-bit

Section 1: counts per dataset (GSM8K, CommonsenseQA).
Section 2: held-out check on S3: per class, per-sample accuracy at each precision and how often each precision's
  plurality is right. If the classes are real, held-out 4-bit pluralities are mostly wrong on 'systematic' and
  mostly right on 'random', with a per-sample drop on both.
Section 3: chance level. The same rules applied to two 16-bit sets of the same size (no quantization) and to
  16-bit against 4-bit, with 8 samples per side (S1, S2) and with 12 (all 24 samples split in two halves:
  A = S1 + first 4 of S2, B = last 4 of S2 + S3). The excess of the 16-vs-4 counts over the 16-vs-16
  counts is what quantization adds; the rest is sampling noise.
Writes outputs/01_question_types.csv (one row per question) for 02_*.
Reference answers are read only to score.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.run_files import generator_run_path, read_question_records, read_run_config, token_limit

MODEL = "Qwen/Qwen3.5-0.8B"
DATASETS = ("gsm8k", "cqa")
CLASSES = ("preserved", "random", "systematic", "helped", "both fail")
DROP = 2 / 16          # 'random' needs at least 2 of 16 fewer right samples at 4-bit
OUT = HERE / "outputs"


def scored_samples(record, extractor) -> list:
    """(answer key or None, right?) per sample; cut-off samples have no answer and are wrong."""
    out = []
    for s in record["raw_samples"]:
        a = None if s.get("cutoff", False) else extractor(s["response"])
        key = None if a is None else normalize_answer(a)
        out.append((key, a is not None and answers_match(a, record["ground_truth"])))
    return out


def summary(samples: list) -> tuple:
    """(share of samples right, plurality right with ties split) of a list of (key, right) samples."""
    p = float(np.mean([ok for _, ok in samples])) if samples else 0.0
    counts = Counter(k for k, _ in samples if k is not None)
    if not counts:
        return p, 0.0
    right = {k: ok for k, ok in samples if k is not None}
    top = max(counts.values())
    return p, float(np.mean([right[k] for k, c in counts.items() if c == top]))


def classify(p16, m16, p4, m4) -> str:
    if m16 <= 0.5:
        return "helped" if m4 > 0.5 else "both fail"
    if m4 <= 0.5:
        return "systematic"
    return "random" if p16 - p4 >= DROP - 1e-9 else "preserved"


def load(dataset: str, precision: str, n: int):
    path = generator_run_path(ROOT / "results", MODEL, precision, dataset, n)
    records = {r["qid"]: r for r in read_question_records(path)}
    limit = token_limit(read_run_config(path), dataset)
    if limit is None:   # oldest headers: the length every cut-off sample stopped at
        cut = {int(s["length"]) for r in records.values() for s in r["raw_samples"] if s.get("cutoff")}
        limit = cut.pop() if len(cut) == 1 else None
    return records, limit


def sets_for(dataset: str):
    """qid -> {precision: (S1, S2, S3)} of scored samples, on questions every run answered."""
    extractor = get_extractor(dataset)
    runs, limits = {}, set()
    for precision in ("16bit", "4bit"):
        for n in (8, 16):
            runs[(precision, n)], limit = load(dataset, precision, n)
            limits.add(limit)
    if len(limits) != 1:
        raise RuntimeError(f"{dataset}: runs have different token limits {limits}")
    qids = sorted(set.intersection(*(set(r) for r in runs.values())), key=lambda q: int(q.rsplit("_", 1)[1]))
    out = {}
    for q in qids:
        out[q] = {}
        for precision in ("16bit", "4bit"):
            a = scored_samples(runs[(precision, 8)][q], extractor)
            b = scored_samples(runs[(precision, 16)][q], extractor)
            out[q][precision] = (a, b[:8], b[8:])
    return out, limits.pop()


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for dataset in DATASETS:
        sets, limit = sets_for(dataset)
        print(f"\n=== {dataset}: {len(sets)} questions, token limit {limit}, 16 samples per precision to classify, 8 held out")
        table = {}
        for q, s in sets.items():
            p16, m16 = summary(s["16bit"][0] + s["16bit"][1])
            p4, m4 = summary(s["4bit"][0] + s["4bit"][1])
            kind = classify(p16, m16, p4, m4)
            h16, hm16 = summary(s["16bit"][2])
            h4, hm4 = summary(s["4bit"][2])
            table[q] = (kind, h16, hm16, h4, hm4)
            rows.append({"dataset": dataset, "qid": q, "type": kind, "p16": round(p16, 4), "p4": round(p4, 4),
                         "plurality16": round(m16, 3), "plurality4": round(m4, 3)})

        print("\n1. Classes (from S1 + S2)")
        counts = Counter(v[0] for v in table.values())
        damaged = counts["random"] + counts["systematic"]
        print(" | ".join(f"{c} {counts[c]} ({100 * counts[c] / len(table):.1f}%)" for c in CLASSES))
        if damaged:
            print(f"of the damaged questions: random {100 * counts['random'] / damaged:.0f}%, "
                  f"systematic {100 * counts['systematic'] / damaged:.0f}%")

        print("\n2. Held-out check on S3 (8 samples per precision, not used to classify)")
        print("class | n | per sample 16-bit | per sample 4-bit | drop | plurality right 16-bit | plurality right 4-bit")
        for c in CLASSES:
            v = [x for x in table.values() if x[0] == c]
            if not v:
                continue
            h16, h4 = np.mean([x[1] for x in v]), np.mean([x[3] for x in v])
            print(f"{c} | {len(v)} | {100 * h16:.1f} | {100 * h4:.1f} | {100 * (h16 - h4):+.1f} | "
                  f"{100 * np.mean([x[2] for x in v]):.1f} | {100 * np.mean([x[4] for x in v]):.1f}")

        print("\n3. Chance level: the same rules with 8 samples per side")
        print("comparison | " + " | ".join(CLASSES))
        for label, left, right in (("16-bit S1 vs 16-bit S2 (no quantization)", ("16bit", 0), ("16bit", 1)),
                                   ("16-bit S1 vs 4-bit S1", ("16bit", 0), ("4bit", 0)),
                                   ("16-bit S2 vs 4-bit S2", ("16bit", 1), ("4bit", 1))):
            c = Counter()
            for s in sets.values():
                pa, ma = summary(s[left[0]][left[1]])
                pb, mb = summary(s[right[0]][right[1]])
                c[classify(pa, ma, pb, mb)] += 1
            print(f"{label} | " + " | ".join(str(c[k]) for k in CLASSES))
        half = {q: {pr: (v[pr][0] + v[pr][1][:4], v[pr][1][4:] + v[pr][2]) for pr in v} for q, v in sets.items()}
        for label, left, right in (("12 samples: 16-bit A vs 16-bit B (no quantization)", ("16bit", 0), ("16bit", 1)),
                                   ("12 samples: 16-bit A vs 4-bit A", ("16bit", 0), ("4bit", 0)),
                                   ("12 samples: 16-bit B vs 4-bit B", ("16bit", 1), ("4bit", 1))):
            c = Counter()
            for v in half.values():
                pa, ma = summary(v[left[0]][left[1]])
                pb, mb = summary(v[right[0]][right[1]])
                c[classify(pa, ma, pb, mb)] += 1
            print(f"{label} | " + " | ".join(str(c[k]) for k in CLASSES))

    with (OUT / "01_question_types.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT / '01_question_types.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
