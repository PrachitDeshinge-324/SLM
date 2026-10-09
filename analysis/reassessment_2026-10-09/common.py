"""Shared loading and per-question helpers for the analysis scripts in this folder."""
import collections
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.experiment_d.evaluation_stats import auroc, bootstrap_interval, pool_random_effects, sensitivity_and_criterion

R = pickle.load(open(os.path.join(os.path.dirname(__file__), "cache.pkl"), "rb"))
PREC = ["16bit", "8bit", "4bit"]

def groups():
    g = collections.defaultdict(dict)
    for (d, m, n, p), v in R.items():
        g[(d, m, n)][p] = v
    return g

def vote(S, idx=None, strict=True):
    """majority key, count, n_valid, n_total over sample subset idx."""
    S = S if idx is None else [S[i] for i in idx]
    v = [s["ans"] for s in S if s["ans"] is not None and not (strict and s["cut"])]
    if not v:
        return None, 0, 0, len(S)
    k, c = collections.Counter(v).most_common(1)[0]
    return k, c, len(v), len(S)

def qstats(q, idx=None, strict=True):
    S = q["S"] if idx is None else [q["S"][i] for i in idx]
    k, c, nv, nt = vote(S, strict=strict)
    valid = [s for s in S if s["ans"] is not None and not (strict and s["cut"])]
    maj_correct = bool(k is not None and any(s["correct"] for s in valid if s["ans"] == k))
    return {
        "maj": k, "correct": maj_correct, "conf_all": c / nt if nt else 0.0,
        "conf_valid": c / nv if nv else 0.0, "nv": nv,
        "pass": any(s["correct"] for s in valid),
        "single": np.mean([s["correct"] and not (strict and s["cut"]) for s in S]),
        "ncut": sum(s["cut"] for s in S),
    }

def short(m):
    return (m.replace("meta-llama_", "").replace("deepseek-ai_", "").replace("google_", "").replace("Qwen_", "")
             .replace("-Instruct", "").replace("DeepSeek-R1-Distill-Qwen", "DS-R1").replace("Llama-3.2", "Llama"))


def comparable(P, p, base="16bit"):
    """False when both runs record a token limit and the limits differ (not a valid precision contrast)."""
    a, b = (P[base].get("cfg") or {}), (P[p].get("cfg") or {})
    ma, mb = a.get("max_new_tokens"), b.get("max_new_tokens")
    return ma is None or mb is None or ma == mb


def question_arrays(P, p, common):
    """Per-question arrays (per-sample accuracy, majority correct, coverage, agreement) for precision p."""
    st = [qstats(P[p]["Q"][q]) for q in common]
    return {k: np.array([s[k] for s in st], float) for k in ("single", "correct", "pass", "conf_all")}
