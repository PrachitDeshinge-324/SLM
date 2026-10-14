"""Numbers for the course paper from the Qwen3.5-0.8B self-training pilot (GSM8K, 100 test questions).

Every run is re-scored from its raw samples. The 4-bit rounds are compared with the untrained 4-bit
run made at the same batch size (256) in the batch check, not with round 0 (batch 64): untrained
4-bit accuracy moves with batch size. The target is the untrained 16-bit run (round 0 of the
16-bit pilot, batch 256).
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.experiment_d.rescoring import rescore_record

ROOT = os.path.join(REPO, "results", "self_training")
FILE = "gsm8k/generator/Qwen_Qwen3.5-0.8B/Qwen_Qwen3.5-0.8B_{p}_gsm8k_n8.jsonl"
THRESHOLD = 0.75
RNG = np.random.default_rng(0)

RUNS = {
    "16bit untrained (target)": ("qwen08_16bit_pilot/round0", "16bit"),
    "16bit round 1": ("qwen08_16bit_pilot/round1", "16bit"),
    "16bit round 2": ("qwen08_16bit_pilot/round2", "16bit"),
    "16bit round 3": ("qwen08_16bit_pilot/round3", "16bit"),
    "4bit untrained b256 (control)": ("qwen08_4bit_batch_check/b256", "4bit"),
    "4bit untrained b128": ("qwen08_4bit_batch_check/b128", "4bit"),
    "4bit untrained b64 (round 0)": ("qwen08_4bit_pilot/round0", "4bit"),
    "4bit round 1": ("qwen08_4bit_pilot/round1", "4bit"),
    "4bit round 2": ("qwen08_4bit_pilot/round2", "4bit"),
    "4bit round 3": ("qwen08_4bit_pilot/round3", "4bit"),
    "4bit untrained, 1024 tokens": ("qwen08_4bit_tokens1024/untrained", "4bit"),
    "4bit round 1, 1024 tokens": ("qwen08_4bit_tokens1024/round1", "4bit"),
    "4bit round 2, 1024 tokens": ("qwen08_4bit_tokens1024/round2", "4bit"),
    "4bit round 3, 1024 tokens": ("qwen08_4bit_tokens1024/round3", "4bit"),
}


def load(folder, precision):
    config, rows = {}, {}
    with open(os.path.join(ROOT, folder, FILE.format(p=precision))) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "run_config":
                config = record
            elif "raw_samples" in record:
                m = rescore_record(record, "gsm8k")
                n = len(record["raw_samples"])
                complete = n * (1 - m["cutoff_rate"])
                rows[record["qid"]] = {
                    "single": m["sample_accuracy"], "cut": m["cutoff_rate"], "voted": float(m["correctness"]),
                    "confident": float(m["confidence_all"] >= THRESHOLD), "complete": complete,
                    "right": m["sample_accuracy"] * n,
                    "tokens": np.mean([s.get("length", 0) for s in record["raw_samples"]]),
                }
    return config, rows


data = {name: load(*spec) for name, spec in RUNS.items()}
# Runs that are not finished (fewer than 100 questions) are listed and then left out.
unfinished = [name for name, (_, rows) in data.items() if len(rows) < 100]
shared = set.intersection(*[set(rows) for name, (_, rows) in data.items() if name not in unfinished])
qids = [q for q in data["16bit untrained (target)"][1] if q in shared]
print(f"Questions answered in every finished run: {len(qids)}; unfinished runs left out: {unfinished}")
for name, (config, rows) in data.items():
    print(f"  {name:32s} questions {len(rows):4d} | batch {config.get('batch_size')} | max_new_tokens {config.get('max_new_tokens')} | adapter {'yes' if config.get('adapter') else 'no'}")


def arrays(name):
    rows = data[name][1]
    return {k: np.array([rows[q][k] for q in qids]) for k in ("single", "cut", "voted", "confident", "complete", "right", "tokens")}


def summary(a, ix=None):
    ix = np.arange(len(qids)) if ix is None else ix
    conf = a["confident"][ix] > 0
    return np.array([
        a["single"][ix].mean(), a["cut"][ix].mean(), a["right"][ix].sum() / a["complete"][ix].sum(),
        a["voted"][ix].mean(), conf.mean(), a["voted"][ix][conf].mean() if conf.any() else np.nan,
    ])


COLS = ["single answer", "cut off", "right among finished", "voted", "confident", "confident and right"]
RUNS = {name: spec for name, spec in RUNS.items() if name not in unfinished}
A = {name: arrays(name) for name in RUNS}
print("\n=== Levels on the shared questions")
print(f"{'run':32s} | " + " | ".join(COLS) + " | confident n | tokens")
for name in RUNS:
    s = summary(A[name])
    print(f"{name:32s} | " + " | ".join(f"{v:.3f}" for v in s) + f" | {int(A[name]['confident'].sum())} | {A[name]['tokens'].mean():.0f}")


def difference(after, before, B=4000):
    est = summary(A[after]) - summary(A[before])
    boots = np.array([summary(A[after], ix) - summary(A[before], ix) for ix in (RNG.integers(0, len(qids), len(qids)) for _ in range(B))])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return " | ".join(f"{COLS[i]} {est[i]*100:+5.1f} [{lo[i]*100:+5.1f},{hi[i]*100:+5.1f}]" for i in (0, 1, 2, 3))


print("\n=== 4-bit rounds minus the matched untrained 4-bit control (batch 256). Points.")
for k in (1, 2, 3):
    print(f"round {k}: " + difference(f"4bit round {k}", "4bit untrained b256 (control)"))
print("\n=== 16-bit rounds minus untrained 16-bit. Points.")
for k in (1, 2, 3):
    print(f"round {k}: " + difference(f"16bit round {k}", "16bit untrained (target)"))
print("\n=== Gap to the target: untrained 16-bit minus each 4-bit run. Points.")
for name in ("4bit untrained b256 (control)", "4bit round 1", "4bit round 2", "4bit round 3"):
    print(f"{name:32s}: " + difference("16bit untrained (target)", name))
print("\n=== Untrained 4-bit at other batch sizes minus the batch-256 control. Points.")
for name in ("4bit untrained b128", "4bit untrained b64 (round 0)"):
    print(f"{name:32s}: " + difference(name, "4bit untrained b256 (control)"))
print("\n=== 1,024-token limit: trained 4-bit rounds minus untrained 4-bit, both at 1,024. Points.")
for k in (1, 2, 3):
    if f"4bit round {k}, 1024 tokens" in RUNS:
        print(f"round {k}: " + difference(f"4bit round {k}, 1024 tokens", "4bit untrained, 1024 tokens"))
print("untrained 1,024 minus untrained 512 control: " + difference("4bit untrained, 1024 tokens", "4bit untrained b256 (control)"))
print("gap to target at 1,024 (16-bit untrained at 512 minus 4-bit round k at 1,024):")
for k in (1, 2, 3):
    if f"4bit round {k}, 1024 tokens" in RUNS:
        print(f"  round {k}: " + difference("16bit untrained (target)", f"4bit round {k}, 1024 tokens"))
