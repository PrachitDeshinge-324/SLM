import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.answer_extraction import extract_gsm8k_answer
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


def test_prompt_variants():
    from src.experiment_d.answer_matching import answers_match
    from src.experiment_d.prompts import GSM8K_PROMPT_VARIANTS, get_gsm8k_messages, get_gsm8k_variant_messages
    from src.experiment_d.run_files import is_self_training_file, prompt_variant_dir

    # The default variant is the unchanged prompt, so its runs stay comparable with every earlier run.
    assert get_messages("gsm8k", "Q") == get_gsm8k_messages("Q") == get_messages("gsm8k", "Q", variant="default")
    assert get_gsm8k_variant_messages("Q", "default") == get_gsm8k_messages("Q")
    # Every variant ends its worked example with a '#### <number>' line the extractor reads, and asks the real question last.
    for name in GSM8K_PROMPT_VARIANTS:
        messages = get_messages("gsm8k", "What is 2 + 2?", variant=name)
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[-1]["content"] == "What is 2 + 2?"
        assert "#### <number>" in messages[0]["content"]
        expected = "14" if name == "example2" else "72"
        assert answers_match(extract_gsm8k_answer(messages[2]["content"]), expected), name
    assert len({str(get_messages("gsm8k", "Q", variant=v)) for v in GSM8K_PROMPT_VARIANTS}) == len(GSM8K_PROMPT_VARIANTS)
    for bad in (lambda: get_messages("gsm8k", "Q", variant="nope"), lambda: get_messages("cqa", "Q", [], variant="plan")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown variant or a variant outside GSM8K was accepted")
    # Variant runs live in their own tree, which the main reports skip.
    path = prompt_variant_dir("results", "plan") / "gsm8k" / "generator" / "m" / "m_4bit_gsm8k_n4.jsonl"
    assert is_self_training_file(path) and not is_self_training_file("results/gsm8k/generator/m/m_4bit_gsm8k_n8.jsonl")


def test_variant_resume_check():
    from scripts.generate_answers import check_resume_is_valid

    old = {"type": "run_config", "model": "m", "precision": "4bit", "batch_size": 64}   # written before variants existed
    check_resume_is_valid("f", old, {"q1"}, {"model": "m", "precision": "4bit", "prompt_variant": "default"}, 64)
    try:
        check_resume_is_valid("f", old, {"q1"}, {"model": "m", "precision": "4bit", "prompt_variant": "plan"}, 64)
    except RuntimeError as error:
        assert "prompt_variant" in str(error)
    else:
        raise AssertionError("a variant run was allowed to resume a default-prompt run file")


if __name__ == "__main__":
    test_verifier_parser()
    test_boxed_in_gsm8k()
    test_prompts_render_uniformly()
    test_prompt_variants()
    test_variant_resume_check()
    print("All prompt tests passed!")
