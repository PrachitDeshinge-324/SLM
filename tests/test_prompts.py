import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.extractors import extract_gsm8k_answer
from src.experiment_d.prompts import build_prompt, get_messages
from src.experiment_d.verifier import parse_verdict

MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "google/gemma-3-1b-it",
]


def test_prompts_render_uniformly():
    from transformers import AutoTokenizer
    seen = 0
    for model_id in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception as error:  # offline / gated: skip rather than fail
            print(f"skip {model_id}: {str(error)[:60]}")
            continue
        seen += 1
        prompt = build_prompt(tok, get_messages("gsm8k", "Q?"))
        # Same instruction text reaches every model, system role or not
        assert "end with the final answer in the format: #### <number>" in prompt, model_id
        assert "Q?" in prompt, model_id
        # No run-date leakage (Llama prints the current date unless pinned)
        assert "Today Date: 01 Jan 2026" in prompt or "Today Date" not in prompt, model_id
        # DeepSeek: instructions live in the user turn, not before it
        if "deepseek" in model_id.lower():
            assert prompt.index("<｜User｜>") < prompt.index("math tutor"), model_id
        # SVAMP shares the GSM8K prompt
        assert build_prompt(tok, get_messages("ChilleD/SVAMP", "Q?")) == prompt, model_id
        # Tokenising with add_special_tokens=False must not produce two BOS tokens
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if tok.bos_token_id is not None and len(ids) > 1:
            assert not (ids[0] == tok.bos_token_id and ids[1] == tok.bos_token_id), model_id
    assert seen > 0, "no tokenizer could be loaded"


def test_verifier_parser():
    assert parse_verdict("Final Conclusion: Correct") is True
    assert parse_verdict("**Final Conclusion: Incorrect**") is False
    # prompt echo is not a verdict
    assert parse_verdict("Final Conclusion: Correct (or Final Conclusion: Incorrect)") is None
    # re-solving is not a verdict
    assert parse_verdict("So the answer is 18.") is None
    # thinking is ignored; last verdict wins
    assert parse_verdict("<think>Final Conclusion: Correct</think>\nFinal Conclusion: Incorrect") is False


def test_boxed_in_gsm8k():
    assert extract_gsm8k_answer("so \\boxed{18}") == "18"
    assert extract_gsm8k_answer("#### 5\n\\boxed{7}") == "5"  # official marker still wins


if __name__ == "__main__":
    test_verifier_parser()
    test_boxed_in_gsm8k()
    test_prompts_render_uniformly()
    print("All prompt tests passed!")
