"""
Does the vLLM engine compute the same model as Hugging Face? Run this once per model/precision/adapter
on the GPU before trusting `--engine vllm` runs (CUDA only).

Both engines score the SAME token sequences (prompt + a greedy continuation written by the Hugging Face
model), so the comparison is per-token log-probabilities, not sampled text:
  1. base parity       |vLLM base - HF base|                 should be small
  2. adapter applied   |vLLM adapter - HF adapter|           should be small, and clearly smaller than
                       |vLLM base - HF adapter|               the distance the adapter moves the model
If vLLM ignored the adapter (or its linear-attention layers), check 2 fails: it would otherwise
make every trained round look like the untrained model.

Usage:
  python scripts/check_vllm_parity.py --model Qwen/Qwen3.5-0.8B --precision 4bit \
      [--adapter results/self_training/<experiment>/round1/adapters/Qwen_Qwen3.5-0.8B_4bit]
"""
import argparse
import gc
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.warning_filters import silence_harmless_warnings

silence_harmless_warnings()
load_dotenv()

import numpy as np
import torch

from src.experiment_d.benchmarks import load_benchmark
from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer
from src.experiment_d.prompts import build_prompt, get_messages
from src.experiment_d.vllm_sampling import load_vllm_engine


def parse_args():
    parser = argparse.ArgumentParser(description="Compare per-token log-probabilities of the vLLM and Hugging Face engines")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=("16bit", "4bit"), default="16bit")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="gsm8k_train", help="Unlabelled questions to score")
    parser.add_argument("--n_prompts", type=int, default=16)
    parser.add_argument("--new_tokens", type=int, default=48)
    parser.add_argument("--tolerance", type=float, default=0.10, help="Largest acceptable mean |log-prob difference| per token (nats)")
    return parser.parse_args()


@torch.no_grad()
def hf_token_logps(model, prompt_ids, continuation_ids):
    ids = torch.tensor([prompt_ids + continuation_ids], device=next(model.parameters()).device)
    logps = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
    positions = torch.arange(len(prompt_ids) - 1, len(prompt_ids) + len(continuation_ids) - 1)
    return logps[positions, torch.tensor(continuation_ids)].cpu().numpy()


def hf_side(args):
    """Greedy continuations and their HF log-probs (adapter, and base when an adapter is given)."""
    model, tokenizer = load_model_and_tokenizer(args.model, args.precision, adapter_path=args.adapter)
    questions = load_benchmark(args.dataset)[:args.n_prompts]
    prompts = [tokenizer(build_prompt(tokenizer, get_messages(args.dataset, q["question"])), add_special_tokens=False)["input_ids"]
               for q in questions]
    device = next(model.parameters()).device
    rows = []
    for prompt_ids in prompts:
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([prompt_ids], device=device), max_new_tokens=args.new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        continuation = out[0, len(prompt_ids):].tolist()
        row = {"prompt": prompt_ids, "continuation": continuation, "hf": hf_token_logps(model, prompt_ids, continuation)}
        if args.adapter:
            with model.disable_adapter():
                row["hf_base"] = hf_token_logps(model, prompt_ids, continuation)
        rows.append(row)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def vllm_token_logps(llm, rows, lora_request):
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outputs = llm.generate([{"prompt_token_ids": r["prompt"] + r["continuation"]} for r in rows], params,
                           lora_request=lora_request, use_tqdm=False)
    scored = []
    for row, output in zip(rows, outputs):
        start = len(row["prompt"])
        entries = output.prompt_logprobs[start:start + len(row["continuation"])]
        token_ids = row["continuation"][:len(entries)]
        scored.append(np.array([entry[token].logprob for entry, token in zip(entries, token_ids)]))
    return scored


def mean_abs(a, b):
    return float(np.mean(np.concatenate([np.abs(x[:min(len(x), len(y))] - y[:min(len(x), len(y))]) for x, y in zip(a, b)])))


def main():
    args = parse_args()
    rows = hf_side(args)
    print(f"Scored {len(rows)} continuations of {args.new_tokens} tokens with Hugging Face.")
    llm, lora_request, info = load_vllm_engine(args.model, args.precision, args.adapter, max_model_len=2048,
                                               gpu_memory_utilization=0.5)
    print(f"vLLM {info['vllm_version']} loaded.")
    vllm_main = vllm_token_logps(llm, rows, lora_request)
    hf_main = [r["hf"] for r in rows]

    checks = {}
    if not args.adapter:
        diff = mean_abs(vllm_main, hf_main)
        print(f"base: mean |vLLM - HF| = {diff:.4f} nats/token")
        checks[f"base parity (< {args.tolerance})"] = diff < args.tolerance
    else:
        vllm_base = vllm_token_logps(llm, rows, None)
        hf_base = [r["hf_base"] for r in rows]
        applied = mean_abs(vllm_main, hf_main)
        without = mean_abs(vllm_base, hf_main)
        moved = mean_abs(hf_main, hf_base)
        base_diff = mean_abs(vllm_base, hf_base)
        print(f"adapter moves the HF model by      {moved:.4f} nats/token")
        print(f"base:    |vLLM base - HF base|      {base_diff:.4f}")
        print(f"adapter: |vLLM adapter - HF adapter| {applied:.4f}")
        print(f"         |vLLM base - HF adapter|    {without:.4f}   (what an ignored adapter would give)")
        checks[f"base parity (< {args.tolerance})"] = base_diff < args.tolerance
        checks[f"adapter parity (< {args.tolerance})"] = applied < args.tolerance
        checks["adapter is applied (closer to HF adapter than the base is)"] = applied < 0.5 * without
        checks["adapter changes the model enough to tell (> 0.02)"] = moved > 0.02
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        sys.exit(1)
    print("vLLM parity check passed.")


if __name__ == "__main__":
    main()
