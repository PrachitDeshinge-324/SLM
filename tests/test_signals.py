"""Tests for signals: vote support per sample and the verdict-rate gate of the one-pass signal run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.signals import gate_failures, verdict_rates, vote_support


def s(text, cut=False):
    return {"response": text, "cutoff": cut}


extract = get_extractor("gsm8k")
samples = [s("so #### 5"), s("#### 5.0"), s("#### 7"), s("#### 9", cut=True), s("no answer")]
support = vote_support(samples, extract)
assert support[0] == support[1] and support[0][1] == 2 / 5          # "5" and "5.0" are one answer; share of ALL samples
assert support[2][1] == 1 / 5
assert support[3] == (None, None) and support[4] == (None, None)    # cut off, or no answer: no support

rows = [{"critic_short": {"values": [True, None, False, True]}},
        {"critic_short": {"values": [None, None, True, True]}},
        {"mc": {"score": 0.3}}]                                        # a row the critic did not judge
rates = verdict_rates(rows, ["critic_short", "critic_forced"])
assert rates == {"critic_short": (5, 8), "critic_forced": (0, 0)}
assert gate_failures(rates, 0.85) == ["critic_short"]                # 62.5% < 85%
assert gate_failures(rates, 0.6) == []
assert gate_failures({"critic_short": (17, 20)}, 0.85) == []         # exactly 85% passes

print("All signals tests passed!")


# One chunk of the runner, with the model-side functions replaced by stand-ins: every row gets its fields.
import argparse
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("run_signals", ROOT / "scripts/run_signals.py")
run_signals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_signals)

record = {"qid": "q1", "question": "Q?", "raw_samples": [s("#### 5"), s("#### 7"), s("#### 5"), s("#### 8", cut=True)]}


def fake_sampled(model, tokenizer, chunk, ns):
    assert ns.variant == "critic_short" and ns.skip_cutoff and ns.max_per_answer == 2
    return [{"qid": "q1", "trace_index": i, "responses": ["x"], "values": [None], "readouts": [0.7], "score": None,
             "forced_score": 0.7, "forced_decision": True, "n": 1, "parsed": 0} for i in (0, 1, 2)]


def fake_mc(model, tokenizer, chunk, ns):
    assert not ns.skip_cutoff                          # mc judges every solution, cut-off ones too
    return [{"qid": "q1", "trace_index": i, "score": 0.1 * i, "verdict_mass": 0.9} for i in range(4)]


run_signals.verifiers.sampled_rows, run_signals.verifiers.mc_rows = fake_sampled, fake_mc
run_signals.tokens.build_sequence = lambda tok, ids, sample, end: {"flags": []}
run_signals.tokens.score = lambda model, tok, seq, settings: {"logp_mean": -0.5}
run_signals.build_prompt = lambda tok, messages: "P"


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 2]}


args = argparse.Namespace(extractor=extract, dataset="gsm8k", skip_cutoff=True, contested_only=True, max_per_answer=2)
critic_args = {"critic_short": run_signals.runner_args(args, "critic_short", selected=True)}
mc_args = run_signals.runner_args(args, "mc", selected=False)
rows = run_signals.score_chunk(None, _Tok(), [record], args, critic_args, mc_args, ({}, 0))
assert [r["trace_index"] for r in rows] == [0, 1, 2, 3]
assert rows[0]["vote_support"] == 0.5 and rows[3]["answer"] is None and rows[3]["finished"] is False
assert rows[0]["critic_short"]["forced_score"] == 0.7 and "critic_short" not in rows[3]
assert rows[3]["mc"]["score"] == 0.30000000000000004 or abs(rows[3]["mc"]["score"] - 0.3) < 1e-12
assert all(r["token"]["logp_mean"] == -0.5 for r in rows)

print("All run_signals wiring tests passed!")
