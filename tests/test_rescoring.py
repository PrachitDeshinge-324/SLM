import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.rescoring import relabel_trace, rescore_record
from src.experiment_d.run_files import (
    generator_path_for_verifier_output,
    generator_run_path,
    is_legacy_file,
    parse_run_name,
    verifier_dataset_path,
    verifier_output_path,
)


def _sample(response, cutoff=False):
    return {"response": response, "cutoff": cutoff, "extracted": "stale", "length": 10}


def test_rescore_record_ignores_stored_metrics():
    record = {
        "qid": "gsm8k_0",
        "dataset": "gsm8k",
        "ground_truth": "18",
        # what an older extractor wrote on the day of the run
        "metrics": {"correctness": False, "confidence": 0.25, "majority_answer": "99"},
        "raw_samples": [
            _sample("She makes 9 * 2 = 18.\n#### **Answer:** \\$18"),
            _sample("#### 18.00"),
            _sample("#### 20"),
            _sample("First, 9 * 11 = 99", cutoff=True),   # partial scratchpad: must not vote
        ],
    }
    metrics = rescore_record(record)
    assert metrics["correctness"] is True
    assert metrics["majority_answer"] == "18"
    assert metrics["confidence"] == 2 / 3          # among the three complete samples
    assert metrics["confidence_all"] == 2 / 4      # among all four
    assert metrics["cutoff_rate"] == 0.25
    assert metrics["failure_rate"] == 0.0          # truncation is not an extraction failure
    assert metrics["sample_accuracy"] == 0.5
    assert metrics["oracle_correctness"] is True


def test_rescore_record_all_truncated():
    record = {"dataset": "gsm8k", "ground_truth": "5",
              "raw_samples": [_sample("so 2 + 3 = 5", cutoff=True), _sample("so 2 + 3", cutoff=True)]}
    metrics = rescore_record(record)
    assert metrics["correctness"] is False
    assert metrics["sample_accuracy"] == 0.0
    assert metrics["cutoff_rate"] == 1.0


def test_relabel_trace_recomputes_label_and_truncation():
    extractor = get_extractor("gsm8k")
    record = {
        "question": "Q1", "trace_index": 1, "gt_answer": "18",
        "student_solution": "#### **Answer:** \\$18",
        "is_correct": False,                         # stale label
        "cutoff": True,                              # verifier truncation in older files: must be ignored
        "verifier_raw_response": "Final Conclusion: Correct",
    }
    generator = {"Q1": [{"response": "other", "cutoff": True},
                        {"response": "#### **Answer:** \\$18", "cutoff": False}]}
    trace = relabel_trace(record, extractor, generator)
    assert trace == {"answer": "18", "correct": True, "truncated": False, "approved": True}

    # generator run overwritten since: the trace text no longer matches, truncation is unknown
    stale = {"Q1": [{"response": "x", "cutoff": False}, {"response": "a newer sample", "cutoff": True}]}
    assert relabel_trace(record, extractor, stale)["truncated"] is None
    assert relabel_trace(record, extractor, None)["truncated"] is None

    # newer verifier outputs carry the trace's own flag
    assert relabel_trace({**record, "trace_cutoff": True}, extractor, None)["truncated"] is True

    # the prompt's echoed instruction line is not a verdict
    echoed = {**record, "verifier_raw_response": "Final Conclusion: Correct (or Final Conclusion: Incorrect)"}
    assert relabel_trace(echoed, extractor, generator)["approved"] is None


def test_run_file_names_and_paths():
    generator = Path("results/gsm8k/generator/Qwen_Qwen3.5-0.8B/Qwen_Qwen3.5-0.8B_4bit_gsm8k_n8.jsonl")
    verifier_out = Path("results/gsm8k/verifier_output/Qwen_Qwen3.5-0.8B/verifier_output_Qwen_Qwen3.5-0.8B_4bit_gsm8k_n8.jsonl")
    # the older 'n8' spelling and the plain number give the same path
    assert generator_run_path("results", "Qwen/Qwen3.5-0.8B", "4bit", "gsm8k", "n8") == generator
    assert generator_run_path("results", "Qwen/Qwen3.5-0.8B", "4bit", "gsm8k", 8) == generator
    assert verifier_output_path("results", "Qwen/Qwen3.5-0.8B", "4bit", "gsm8k", 8) == verifier_out
    assert verifier_dataset_path("results", "ChilleD/SVAMP", "16bit", "ChilleD/SVAMP", 8).name == "verifier_ChilleD_SVAMP_16bit_ChilleD_SVAMP_n8.jsonl"
    assert generator_path_for_verifier_output(verifier_out) == generator

    # model and dataset names both contain underscores
    run = parse_run_name("verifier_output_meta-llama_Llama-3.2-1B-Instruct_8bit_ChilleD_SVAMP_n16.jsonl")
    assert run == {"model": "meta-llama_Llama-3.2-1B-Instruct", "precision": "8bit", "dataset": "ChilleD_SVAMP", "n_samples": 16}
    assert parse_run_name("questions_Qwen_gsm8k_n8_shared100.jsonl") is None

    assert is_legacy_file("results/math500/generator/m/m_16bit_math500_n8_backup.jsonl")
    assert not is_legacy_file(verifier_out)


def test_paired_deltas_use_common_questions():
    import pandas as pd
    from scripts.report_generator_metrics import compute_quantization_deltas

    def run(correct, limit=None):
        frame = pd.DataFrame({"metrics.correctness": correct}, dtype=float)
        frame["metrics.oracle_correctness"] = frame["metrics.correctness"]
        frame["metrics.sample_accuracy"] = frame["metrics.correctness"]
        return {"per_question": frame, "max_new_tokens": limit}

    key = ("gsm8k", "m", 8)
    runs = {
        key + ("16bit",): run({"q1": 1, "q2": 1, "q3": 0, "q4": 1}),
        key + ("4bit",): run({"q1": 1, "q2": 0}),                     # only two questions finished
        key + ("8bit",): run({"q1": 1, "q2": 1, "q3": 0, "q4": 1}, limit=2048),
    }
    runs[key + ("16bit",)]["max_new_tokens"] = 4096
    summary = pd.DataFrame([
        {"dataset": "gsm8k", "model": "m", "n_samples": 8, "precision": precision}
        for precision in ("16bit", "8bit", "4bit")
    ])
    out = compute_quantization_deltas(summary, runs).set_index("precision")
    assert out.loc["4bit", "acc_delta_vs_16bit"] == -0.5          # on q1, q2 only: 0.5 - 1.0
    assert out.loc["4bit", "delta_n_questions"] == 2
    assert pd.isna(out.loc["8bit", "acc_delta_vs_16bit"])         # token limits differ: no delta
    assert out.loc["16bit", "acc_delta_vs_16bit"] == 0.0


if __name__ == "__main__":
    test_rescore_record_ignores_stored_metrics()
    test_rescore_record_all_truncated()
    test_relabel_trace_recomputes_label_and_truncation()
    test_run_file_names_and_paths()
    test_paired_deltas_use_common_questions()
    print("All rescoring tests passed!")
