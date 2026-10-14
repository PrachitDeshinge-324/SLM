"""
Sample N answers per prompt with vLLM instead of Hugging Face `generate`.

Same contract as `sampling.sample_responses`, so run files keep the same record format:
  - `length` is the number of generated tokens before the stop token, and a sample that ran into
    max_new_tokens has length == max_new_tokens (so `cutoff = length >= max_new_tokens` still holds);
  - responses are decoded without special tokens;
  - prompts are the exact token ids the Hugging Face path would feed (rendered by the same chat
    template, tokenized without special tokens), never re-tokenized by vLLM.

vLLM is imported only when an engine is loaded, so the rest of the library works without it.
Needs a CUDA GPU. Results are not bit-identical to the Hugging Face path (different kernels and
random streams), so a run file records `engine` and runs of different engines must not be mixed.
"""
import json
import time
from pathlib import Path

from src.experiment_d.sampling import truncate_after_first_boxed

VLLM_PRECISIONS = ("16bit", "4bit")
LORA_ADAPTER_NAME = "self_training_adapter"


def stop_token_ids(model_id: str, tokenizer) -> list:
    """The ids that end an answer: the model's generation-config eos ids, else the tokenizer's eos."""
    from transformers import GenerationConfig

    try:
        eos = GenerationConfig.from_pretrained(model_id).eos_token_id
    except Exception:
        eos = None
    ids = [eos] if isinstance(eos, int) else list(eos or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


def adapter_rank(adapter_path) -> int:
    """LoRA rank of a saved peft adapter."""
    return int(json.loads((Path(adapter_path) / "adapter_config.json").read_text())["r"])


def load_vllm_engine(model_id: str, precision: str, adapter_path=None, max_model_len: int = 4096,
                     gpu_memory_utilization: float = 0.85, seed: int = 42):
    """
    (llm, lora_request or None, info) for a model at 16-bit or 4-bit (bitsandbytes NF4, quantized
    while loading). The adapter, if any, is applied on top of the base and not merged.
    """
    if precision not in VLLM_PRECISIONS:
        raise ValueError(f"The vLLM engine supports {VLLM_PRECISIONS}, not {precision!r}; use --engine hf.")

    import torch
    import vllm
    from transformers import AutoConfig

    if not torch.cuda.is_available():
        raise RuntimeError("The vLLM engine needs a CUDA GPU; use --engine hf on this machine.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {
        "model": model_id,
        "dtype": str(dtype).replace("torch.", ""),
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "seed": seed,
        "trust_remote_code": True,
        # Text only: no image or video inputs, so no memory is set aside for them.
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    if precision == "4bit":
        kwargs.update(quantization="bitsandbytes", load_format="bitsandbytes")
    lora_request = None
    if adapter_path:
        from vllm.lora.request import LoRARequest

        kwargs.update(enable_lora=True, max_lora_rank=adapter_rank(adapter_path), max_loras=1)
        lora_request = LoRARequest(LORA_ADAPTER_NAME, 1, str(Path(adapter_path).resolve()))

    print(f"Loading {model_id} with vLLM {vllm.__version__} ({precision}"
          f"{', adapter ' + str(adapter_path) if adapter_path else ''})...")
    llm = vllm.LLM(**kwargs)
    info = {
        "model_revision": getattr(AutoConfig.from_pretrained(model_id), "_commit_hash", None),
        "model_dtype": str(dtype),
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "vllm_version": vllm.__version__,
    }
    return llm, lora_request, info


def outputs_to_samples(completions: list, max_new_tokens: int, stop_ids: list):
    """
    (texts, lengths) for the completions of one request, in the Hugging Face path's convention:
    a sample that stopped on a stop token is counted without it, one that ran out of tokens has
    length max_new_tokens.
    """
    texts, lengths = [], []
    stop = set(stop_ids)
    for completion in completions:
        ids = list(completion.token_ids)
        if completion.finish_reason == "length":
            length = max_new_tokens
        else:
            while ids and ids[-1] in stop:
                ids.pop()
            length = min(len(ids), max_new_tokens - 1)
        texts.append(completion.text)
        lengths.append(length)
    return texts, lengths


def stop_at_boxed_answer(texts: list, lengths: list, tokenizer, max_new_tokens: int):
    """
    The Hugging Face path stops MATH-500 samples as soon as a \\boxed{...} closes; vLLM runs on to
    its stop token, so cut the text there afterwards and recount its tokens. A sample with no closed
    boxed answer keeps its text and length (and stays cut off if it ran out of tokens).
    """
    cut_texts, cut_lengths = [], []
    for text, length in zip(texts, lengths):
        head = truncate_after_first_boxed(text)
        if head is None:
            cut_texts.append(text)
            cut_lengths.append(length)
        else:
            cut_texts.append(head)
            cut_lengths.append(min(len(tokenizer(head, add_special_tokens=False)["input_ids"]), max_new_tokens - 1))
    return cut_texts, cut_lengths


def sample_responses_vllm(llm, tokenizer, prompts: list, seeds: list, n_samples: int, temperature: float,
                          top_k: int, top_p: float, max_new_tokens: int, repetition_penalty: float,
                          stop_ids: list, lora_request=None, stop_on_boxed: bool = False):
    """
    Generates n_samples for every prompt.

    seeds: one per prompt. A prompt's samples depend on its own seed only, not on which other
    prompts share the call, so an interrupted run can resume chunk by chunk.

    Returns (grouped_responses, tokens_per_sec, latency, grouped_lengths) like sampling.sample_responses.
    """
    if len(seeds) != len(prompts):
        raise ValueError("one seed per prompt is required")
    from vllm import SamplingParams

    prompt_ids = [{"prompt_token_ids": ids} for ids in
                  tokenizer(prompts, add_special_tokens=False)["input_ids"]]
    params = [SamplingParams(
        n=n_samples, temperature=temperature, top_p=top_p, top_k=top_k if top_k > 0 else -1,
        repetition_penalty=repetition_penalty, max_tokens=max_new_tokens, seed=seed,
        stop_token_ids=stop_ids, skip_special_tokens=True,
    ) for seed in seeds]

    started = time.time()
    outputs = llm.generate(prompt_ids, params, lora_request=lora_request, use_tqdm=False)
    latency = time.time() - started

    grouped_responses, grouped_lengths, generated = [], [], 0
    for output in outputs:
        texts, lengths = outputs_to_samples(output.outputs, max_new_tokens, stop_ids)
        generated += sum(len(completion.token_ids) for completion in output.outputs)
        if stop_on_boxed:
            texts, lengths = stop_at_boxed_answer(texts, lengths, tokenizer, max_new_tokens)
        grouped_responses.append(texts)
        grouped_lengths.append(lengths)
    return grouped_responses, (generated / latency if latency > 0 else 0.0), latency, grouped_lengths
