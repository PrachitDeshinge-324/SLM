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


# File names: each run's verifier settings in its own file; untagged files from before move to their tag.
import json as _json
import tempfile

from src.experiment_d.run_files import signals_path, signals_paths
from src.experiment_d.signals import signals_tag, tag_of_header

assert signals_tag(["critic_forced"], 4, 704, True) == "critic_forced_v4_t704_all"
assert signals_tag(["critic_short"], 4, 400, False) == "critic_short_v4_t400_pilot"
assert signals_tag([], 4, 400, False) == "no_critic"
assert tag_of_header({"critics": ["critic_forced"], "verdicts": 4, "max_new_tokens": 704, "skip_cutoff": False}) \
    == "critic_forced_v4_t704_all"

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    untagged = signals_path(root, "Qwen/Qwen3.5-2B", "4bit", "gsm8k", 8)
    forced = signals_path(root, "Qwen/Qwen3.5-2B", "4bit", "gsm8k", 8, tag="critic_forced_v4_t704_all")
    short = signals_path(root, "Qwen/Qwen3.5-2B", "4bit", "gsm8k", 8, tag="critic_short_v4_t400_all")
    assert untagged.name == "signals_Qwen_Qwen3.5-2B_4bit_gsm8k_n8.jsonl"
    assert forced.name == "signals_critic_forced_v4_t704_all_Qwen_Qwen3.5-2B_4bit_gsm8k_n8.jsonl"
    assert signals_paths(root, "Qwen/Qwen3.5-2B", "4bit", "gsm8k", 8) == []
    untagged.parent.mkdir(parents=True)
    header = {"type": "run_config", "model": "Qwen/Qwen3.5-2B", "precision": "4bit", "dataset": "gsm8k",
              "generator_n_samples": 8, "critics": ["critic_forced"], "verdicts": 4, "max_new_tokens": 704,
              "skip_cutoff": False, "question_ids": ["q1"]}
    untagged.write_text(_json.dumps(header) + "\n")
    short.write_text("{}\n")
    run_signals.move_untagged(untagged)                       # the runner moves an untagged file to its own tag
    assert not untagged.exists() and forced.exists()
    assert signals_paths(root, "Qwen/Qwen3.5-2B", "4bit", "gsm8k", 8) == [forced, short]
    untagged.write_text(_json.dumps(header) + "\n")          # a second untagged file for a taken name stays put
    run_signals.move_untagged(untagged)
    assert untagged.exists() and forced.read_text() == _json.dumps(header) + "\n"

    spec_m = importlib.util.spec_from_file_location("migrate", ROOT / "scripts/migrate_signals_files.py")
    migrate = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(migrate)
    steps = migrate.plan(root)                                # the migration never targets an existing file
    assert (untagged, None) in [(s[0], s[1]) for s in steps]           # its target is taken: left alone
    assert all(target is None or not target.exists() for _, target, _ in steps)

print("All signals file-name tests passed!")

# A critic run by the model with an adapter is named after that adapter, so it never shares the base model's file.
from src.experiment_d.signals import adapter_label, with_judge

arm = "/content/drive/MyDrive/results/self_training/q2b_4bit_critic_p500_critic_reason/round1/adapters/Qwen_Qwen3.5-2B_4bit"
assert adapter_label(arm) == "q2b_4bit_critic_p500_critic_reason_r1"
assert adapter_label("some/folder/my adapter.v2") == "my_adapter_v2"
assert with_judge("critic_forced_v2_t704_pilot", None) == "critic_forced_v2_t704_pilot"
assert with_judge("critic_forced_v2_t704_pilot", arm) == "critic_forced_v2_t704_pilot_by_q2b_4bit_critic_p500_critic_reason_r1"
assert tag_of_header({"critics": ["critic_forced"], "verdicts": 2, "max_new_tokens": 704, "skip_cutoff": True,
                      "adapter": arm}) == "critic_forced_v2_t704_pilot_by_q2b_4bit_critic_p500_critic_reason_r1"
print("All signals adapter-name tests passed!")
