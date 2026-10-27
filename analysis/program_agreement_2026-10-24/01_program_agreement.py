"""
B1. Prose vs program agreement: can a Python interpreter, the only outside information allowed under the
4-bit on-device rule, pick the correct answer where every internal signal failed (A4)?

Plan: Docs/research_notes/CS329A_SELF_IMPROVEMENT_IDEAS.md section 9. Everything below the line
"PRE-REGISTERED" was fixed on 24 Oct 2026, before any program run existed; do not change it after seeing data.

Inputs (all GSM8K test, 512 tokens, temperature 0.7):
  prose   results/gsm8k/generator/<model>/<model>_<precision>_gsm8k_n8.jsonl   (the main 8-sample run)
  prose16 results/gsm8k/generator/<model>/<model>_<precision>_gsm8k_n16.jsonl  (independent 16-sample run)
  program <program_root>/gsm8k/generator/<model>/<model>_<precision>_gsm8k_n8.jsonl
          (generate_answers.py --prompt_variant program, default root results/prompt_variants/program)
Program answers come only from running the code (program_execution.py); the run file's stored `extracted`
field is ignored. Prose answers are re-extracted with current code. Reference answers are read only to score.

PRE-REGISTERED
  V  Validity: at least 70% of programs return a number (status ok). Otherwise the run is INVALID (a
     prompt or extraction problem), not a negative result, and nothing below is interpreted.
  P  Primary selector, cross-modal vote: plurality over the 8 prose answers and the 8 program answers
     (failed programs and unanswered prose samples do not vote; ties go to the first answer seen, prose
     samples first).
     Baseline: the prose vote over the 16 samples of the independent n16 run, so both use 16 generations
     (A5: 8 -> 16 prose samples alone add about +3.6 points at 0.8B 4-bit).
     Success: paired bootstrap (2,000 resamples of questions) 95% interval of P minus baseline above zero
     on all shared questions. Also reported, not gating: P minus the 8-sample prose vote, overall and on
     the A4 pilot questions (0.8B 4-bit: 108 contested questions where the best internal signal gave +2.8
     [-0.9, +7.4] over the 8-sample vote).
  S  Secondary selectors (reported, never gating): S1 agreement-weighted prose vote, score(a) =
     n_prose(a) * (1 + n_program(a)); S2 program-only vote; S3 prose vote with programs breaking ties;
     S4 cross-modal vote only where the prose plurality share is below 0.5, prose vote elsewhere.
  M  Mechanism, the scientific question: on questions whose 8-sample prose plurality is wrong, the share
     of working programs that return that same wrong answer ("translates the misreading") and the share
     that return the correct answer, against the same shares for independent prose samples (n16 run).
     Translation is supported if programs repeat the wrong plurality at least as often as independent
     prose samples do; different binding is supported if programs reach the correct answer more often.
     Reported overall, by damage type, and for questions whose prose plurality is right (does code break
     answers the prose gets right?).

Usage (repo root):
  python analysis/program_agreement_2026-10-24/01_program_agreement.py --model Qwen/Qwen3.5-0.8B --precision 4bit
  [--results_dir results] [--program_root results/prompt_variants/program]
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.program_execution import STATUSES, execute_many
from src.experiment_d.run_files import PROMPT_VARIANTS_DIR, safe_name

HERE = Path(__file__).parent
B = 2000
VALIDITY_MIN_OK = 0.70
PILOT_SIGNALS = "results/gsm8k/signals/Qwen_Qwen3.5-0.8B/signals_critic_short_v4_t400_pilot_Qwen_Qwen3.5-0.8B_4bit_gsm8k_n8.jsonl"
TYPES_CSV = "analysis/damage_types_2026-10-24/outputs/01_question_types.csv"
EXTRACT = get_extractor("gsm8k")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--precision", default="4bit")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--program_root", default=None)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def run_path(root, model, precision, n):
    name = safe_name(model)
    return Path(root) / "gsm8k" / "generator" / name / f"{name}_{precision}_gsm8k_n{n}.jsonl"


def load(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        if "raw_samples" in d:
            out[d["qid"]] = d
    return out


def prose_keys(record):
    keys = []
    for s in record["raw_samples"]:
        a = None if s.get("cutoff") else EXTRACT(s["response"])
        keys.append(None if a is None else normalize_answer(a))
    return keys


def program_keys(records, tag, workers):
    """Execute every program once; results cached by response hash in outputs/."""
    cache_path = HERE / "outputs" / f"exec_cache_{tag}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = {}
    for d in records.values():
        for s in d["raw_samples"]:
            h = hashlib.sha256(s["response"].encode()).hexdigest()
            if h not in cache:
                todo[h] = s["response"]
    if todo:
        print(f"executing {len(todo)} programs ...", flush=True)
        for h, r in zip(todo, execute_many(list(todo.values()), workers=workers)):
            cache[h] = r
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    keys, statuses = {}, Counter()
    for q, d in records.items():
        ks = []
        for s in d["raw_samples"]:
            r = cache[hashlib.sha256(s["response"].encode()).hexdigest()]
            statuses[r["status"]] += 1
            ks.append(normalize_answer(r["answer"]) if r["status"] == "ok" else None)
        keys[q] = ks
    return keys, statuses


def vote(keys):
    counts, first = Counter(), {}
    for k in keys:
        if k is not None:
            counts[k] += 1
            first.setdefault(k, len(first))
    if not counts:
        return None
    top = max(counts.values())
    return min((k for k, c in counts.items() if c == top), key=first.get)


def share(keys):
    ks = [k for k in keys if k is not None]
    return Counter(ks).most_common(1)[0][1] / len(ks) if ks else 0.0


def weighted(prose, prog):
    cp, cg = Counter(k for k in prose if k), Counter(k for k in prog if k)
    if not cp:
        return vote(prog)
    first = {k: i for i, k in reversed(list(enumerate(prose))) if k}
    return max(cp, key=lambda k: (cp[k] * (1 + cg[k]), -first[k]))


def tiebreak(prose, prog):
    cp, cg = Counter(k for k in prose if k), Counter(k for k in prog if k)
    if not cp:
        return vote(prog)
    top = max(cp.values())
    tied = [k for k in cp if cp[k] == top]
    if len(tied) == 1:
        return tied[0]
    best = max(cg[k] for k in tied)
    return vote([k for k in prose if k in tied and cg[k] == best])


def boot(base, new):
    d = np.asarray(new, float) - np.asarray(base, float)
    idx = np.random.default_rng(0).integers(0, len(d), (B, len(d)))
    s = d[idx].mean(1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    args = parse_args()
    program_root = args.program_root or str(Path(args.results_dir) / PROMPT_VARIANTS_DIR / "program")
    paths = {"prose": run_path(args.results_dir, args.model, args.precision, 8),
             "prose16": run_path(args.results_dir, args.model, args.precision, 16),
             "program": run_path(program_root, args.model, args.precision, 8)}
    for name, path in paths.items():
        if not path.exists():
            print(f"missing {name} run: {path}")
            if name == "program":
                print("generate it with: python scripts/generate_answers.py --prompt_variant program --dataset gsm8k "
                      f"--model {args.model} --precision {args.precision} --n_samples 8 --max_new_tokens 512 "
                      f"--repetition_penalty 1.0 --seed 1100 --results_dir {program_root}")
            if name != "prose16":
                return
    prose = load(paths["prose"])
    prose16 = load(paths["prose16"]) if paths["prose16"].exists() else {}
    program = load(paths["program"])
    qids = sorted(set(prose) & set(program) & (set(prose16) if prose16 else set(program)))
    tag = f"{safe_name(args.model)}_{args.precision}"
    print(f"{args.model} {args.precision}: {len(qids)} shared questions "
          f"(prose {len(prose)}, prose16 {len(prose16)}, program {len(program)})")

    gt = {q: prose[q]["ground_truth"] for q in qids}
    right = lambda q, k: k is not None and answers_match(k, gt[q])
    P = {q: prose_keys(prose[q]) for q in qids}
    P16 = {q: prose_keys(prose16[q]) for q in qids} if prose16 else {}
    G_all, statuses = program_keys({q: program[q] for q in qids}, tag, args.workers)
    G = {q: G_all[q] for q in qids}

    # ---- V ----
    total = sum(statuses.values())
    ok_rate = statuses["ok"] / total
    print("\nV  program status: " + ", ".join(f"{s} {statuses[s] / total:.3f}" for s in STATUSES))
    cut = sum(s.get("cutoff", False) for q in qids for s in program[q]["raw_samples"]) / total
    print(f"   program cut-off rate {cut:.3f}; validity threshold ok >= {VALIDITY_MIN_OK}: "
          f"{'VALID' if ok_rate >= VALIDITY_MIN_OK else 'INVALID: stop here, fix the prompt or extraction'}")
    acc_prose = np.mean([right(q, k) for q in qids for k in P[q]])
    acc_prog = np.mean([right(q, k) for q in qids for k in G[q]])
    acc_prog_ok = np.mean([right(q, k) for q in qids for k in G[q] if k is not None])
    print(f"   per-sample accuracy: prose {acc_prose:.3f}, program {acc_prog:.3f} (failures count as wrong; "
          f"{acc_prog_ok:.3f} among working programs)")
    if ok_rate < VALIDITY_MIN_OK:
        return

    # ---- P and S ----
    sel = {
        "prose vote (8)": lambda q: vote(P[q]),
        "P cross-modal vote (8 prose + 8 program)": lambda q: vote(P[q] + G[q]),
        "S1 agreement-weighted prose vote": lambda q: weighted(P[q], G[q]),
        "S2 program-only vote (8)": lambda q: vote(G[q]),
        "S3 prose vote, programs break ties": lambda q: tiebreak(P[q], G[q]),
        "S4 cross-modal only if prose share < 0.5": lambda q: vote(P[q] + G[q]) if share(P[q]) < 0.5 else vote(P[q]),
    }
    if P16:
        sel["prose vote (16, independent run)"] = lambda q: vote(P16[q])
    res = {name: [right(q, f(q)) for q in qids] for name, f in sel.items()}
    pilot = set()
    if "0.8B" in args.model and args.precision == "4bit" and Path(PILOT_SIGNALS).exists():
        pilot = {json.loads(l)["qid"] for l in list(open(PILOT_SIGNALS))[1:]}
    pilot_idx = [i for i, q in enumerate(qids) if q in pilot]
    oracle = np.mean([any(right(q, k) for k in P[q] + G[q]) for q in qids])
    print(f"\nP/S selection accuracy on {len(qids)} questions (oracle over the 16 answers {oracle:.3f})")
    base16 = res.get("prose vote (16, independent run)")
    for name, r in res.items():
        line = f"  {name:44s} {np.mean(r):.3f}"
        if base16 is not None and name != "prose vote (16, independent run)":
            d, lo, hi = boot(base16, r)
            line += f" | vs prose 16: {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        if name != "prose vote (8)":
            d, lo, hi = boot(res["prose vote (8)"], r)
            line += f" | vs prose 8: {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        if pilot_idx and name != "prose vote (8)":
            d, lo, hi = boot([res["prose vote (8)"][i] for i in pilot_idx], [r[i] for i in pilot_idx])
            line += f" | pilot ({len(pilot_idx)} q) vs prose 8: {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        print(line)
    if base16 is not None:
        d, lo, hi = boot(base16, res["P cross-modal vote (8 prose + 8 program)"])
        print(f"\nPRIMARY RESULT: cross-modal vote minus 16-sample prose vote {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] "
              f"-> {'PASS' if lo > 0 else 'FAIL'} (pre-registered: lower bound above zero)")
    else:
        print("\nPRIMARY RESULT not computable: no independent 16-sample prose run for this model and precision")

    # ---- M ----
    types = {r["qid"]: r["type"] for r in csv.DictReader(open(TYPES_CSV)) if r["dataset"] == "gsm8k"} \
        if "0.8B" in args.model and Path(TYPES_CSV).exists() else {}
    print("\nM  questions whose 8-sample prose plurality is WRONG: what do working programs return?")
    print("   (baseline: independent prose samples from the n16 run, same questions)")

    def mech(subset, label):
        prog_same = prog_right = prog_n = base_same = base_right = base_n = 0
        for q in subset:
            wrong = vote(P[q])
            for k in G[q]:
                if k is not None:
                    prog_n += 1
                    prog_same += k == wrong
                    prog_right += right(q, k)
            for k in P16.get(q, []):
                if k is not None:
                    base_n += 1
                    base_same += k == wrong
                    base_right += right(q, k)
        if not prog_n:
            return
        line = (f"   {label:34s} q={len(subset):4d} | programs: same wrong answer {prog_same / prog_n:.3f}, "
                f"correct {prog_right / prog_n:.3f} (n={prog_n})")
        if base_n:
            line += f" | prose16: same wrong {base_same / base_n:.3f}, correct {base_right / base_n:.3f} (n={base_n})"
        print(line)

    wrong_q = [q for q in qids if vote(P[q]) is not None and not right(q, vote(P[q]))]
    mech(wrong_q, "all prose-wrong questions")
    for t in ("systematic", "random", "both fail", "helped", "preserved"):
        if types:
            mech([q for q in wrong_q if types.get(q) == t], f"  damage type {t}")
    print("\n   questions whose prose plurality is RIGHT: share of working programs agreeing with it")
    right_q = [q for q in qids if right(q, vote(P[q]))]
    agree = [k == vote(P[q]) for q in right_q for k in G[q] if k is not None]
    print(f"   {len(right_q)} questions, programs agree {np.mean(agree):.3f} (n={len(agree)})")
    pp = np.array([np.mean([right(q, k) for k in P[q]]) for q in qids])
    pg = np.array([np.mean([right(q, k) for k in G[q]]) for q in qids])
    print(f"   per-question accuracy correlation prose vs program: {np.corrcoef(pp, pg)[0, 1]:.2f}; questions where "
          f"programs beat prose by >= 0.25: {np.mean(pg - pp >= 0.25):.3f}, prose beats programs by >= 0.25: {np.mean(pp - pg >= 0.25):.3f}")


if __name__ == "__main__":
    main()
