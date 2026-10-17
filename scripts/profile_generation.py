"""
How fast does this GPU decode? Measure the time per decode step of Hugging Face `generate` for a few batch
sizes, after a warm-up, so that the cost of a long run can be predicted instead of guessed.

For each batch size the same prompt is repeated B times (no padding) and generated twice: the first call
pays one-time costs (kernel compilation and autotuning), the second is timed. A call with one new token gives
the prefill time; (time of N tokens - time of 1 token) / (N - 1) is the time per decode step.

Output per batch size: prefill seconds, milliseconds per decode step, generated tokens per second, peak
memory, or "out of memory". Nothing is written to disk. Run it before a long job, once per model, precision and dtype:

  python scripts/profile_generation.py --model Qwen/Qwen3.5-0.8B --precision 16bit --dtype float16
"""
import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure the repository root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.stdout.reconfigure(line_buffering=True)

from src.experiment_d.warning_filters import silence_harmless_warnings

silence_harmless_warnings()
load_dotenv()

import torch

from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer, model_device


def parse_args():
    parser = argparse.ArgumentParser(description="Time per decode step of Hugging Face generate, by batch size")
    parser.add_argument("--model", "--model_id", dest="model", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=PRECISIONS, default="16bit")
    parser.add_argument("--dtype", type=str, choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--batch_sizes", type=str, default="1,16,64,128", help="Comma-separated")
    parser.add_argument("--prompt_tokens", type=int, default=400, help="About the length of a verifier prompt")
    parser.add_argument("--new_tokens", type=int, default=48)
    return parser.parse_args()


def synchronise(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def timed_generate(model, input_ids, new_tokens: int, device) -> float:
    synchronise(device)
    start = time.time()
    model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=new_tokens,
                   min_new_tokens=new_tokens, do_sample=True, temperature=0.7, top_p=0.9, top_k=0,
                   pad_token_id=model.generation_config.pad_token_id or 0)
    synchronise(device)
    return time.time() - start


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model, args.precision, dtype=None if args.dtype == "auto" else args.dtype)
    device = model_device(model)
    filler = " ".join(["The quick brown fox jumps over the lazy dog and counts 17 apples, then 25 more."] * 200)
    ids = tokenizer(filler, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :args.prompt_tokens].to(device)
    print(f"{args.model} {args.precision} dtype {model.dtype} on {device}; prompt {ids.shape[1]} tokens, {args.new_tokens} new tokens")
    print("batch | prefill s | ms per decode step | generated tokens/s | peak GB")
    for batch in [int(b) for b in args.batch_sizes.split(",")]:
        batch_ids = ids.repeat(batch, 1)
        try:
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            timed_generate(model, batch_ids, args.new_tokens, device)                 # warm-up: compilation, autotuning
            prefill = timed_generate(model, batch_ids, 1, device)
            total = timed_generate(model, batch_ids, args.new_tokens, device)
            step = max(total - prefill, 1e-9) / (args.new_tokens - 1)
            peak = torch.cuda.max_memory_allocated() / 1e9 if str(device).startswith("cuda") else float("nan")
            print(f"{batch:5d} | {prefill:9.2f} | {step * 1000:18.0f} | {batch / step:18.0f} | {peak:7.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:5d} | out of memory")
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
