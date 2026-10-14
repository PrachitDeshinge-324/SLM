"""
End-to-end check of the length-normalised DPO loss on the real model: overfit one pair.

Takes one (majority, minority) pair from an unlabelled pool run, scores it with the frozen base as
the reference, attaches a fresh adapter and makes `--steps` updates on that pair alone. Passes when
the margin starts at 0 (loss ln 2) and ends clearly positive with the loss well below ln 2, which
exercises the prompt, the masks, the reference scores and the backward pass together.
Nothing is saved.

Usage (locally at 16-bit on CPU, or on Colab at 4-bit):
  python scripts/check_dpo_overfit.py --model Qwen/Qwen3.5-0.8B --precision 16bit \
      --pool_file results/self_training/qwen08_4bit_pilot/round0/gsm8k_train/generator/Qwen_Qwen3.5-0.8B/Qwen_Qwen3.5-0.8B_4bit_gsm8k_train_n8.jsonl
"""
import argparse
import math
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.warning_filters import silence_harmless_warnings

silence_harmless_warnings()
load_dotenv()

from src.experiment_d.lora_training import (
    attach_lora,
    build_preference_example,
    end_token_id,
    reference_logps,
    train_lora_dpo,
)
from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer
from src.experiment_d.run_files import read_question_records
from src.experiment_d.self_training import PAIRINGS, select_preference_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Overfit one preference pair to check the DPO loss end to end")
    parser.add_argument("--model", "--model_id", dest="model", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=PRECISIONS, default="16bit")
    parser.add_argument("--pool_file", type=str, required=True, help="Generator run file of an unlabelled pool")
    parser.add_argument("--dataset", type=str, default="gsm8k_train")
    parser.add_argument("--pairing", type=str, choices=PAIRINGS, default="similar_length")
    parser.add_argument("--threshold", type=float, default=0.625)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--max_length", type=int, default=768)
    return parser.parse_args()


def main():
    args = parse_args()
    selected = select_preference_pairs(read_question_records(args.pool_file), args.dataset, args.threshold, args.pairing)
    if not selected:
        raise RuntimeError(f"No pair in {args.pool_file}.")
    # The shortest pair, so the check is quick on a CPU.
    item = min(selected, key=lambda pair: pair["chosen_length"] + pair["rejected_length"])
    print(f"Pair from {item['qid']}: chosen answer {item['label']} ({item['chosen_length']} tokens), "
          f"rejected answer {item['rejected_answer']} ({item['rejected_length']} tokens)")

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)
    compute_dtype = model.dtype
    pair = build_preference_example(tokenizer, args.dataset, item["question"], item["chosen"], item["rejected"],
                                    end_token_id(model, tokenizer), args.max_length)
    if pair is None:
        raise RuntimeError(f"The pair does not fit in --max_length {args.max_length}.")
    (pair["ref_chosen"], pair["ref_rejected"]), = reference_logps(model, tokenizer, [pair], compute_dtype=compute_dtype)
    print(f"Reference mean log-prob per token: chosen {pair['ref_chosen']:.4f}, rejected {pair['ref_rejected']:.4f}")

    model, target_modules = attach_lora(model)
    print(f"Adapter on {len(target_modules)} kinds of linear layer: {', '.join(target_modules)}")
    started = time.time()
    history = train_lora_dpo(model, tokenizer, [pair], steps=args.steps, batch_size=1, learning_rate=args.learning_rate,
                             beta=args.beta, compute_dtype=compute_dtype, log_every=5)
    print(f"{args.steps} steps in {time.time() - started:.0f} s")

    first_margin, first_loss = history["margins"][0], history["losses"][0]
    last_margin, last_loss = history["margins"][-1], history["losses"][-1]
    checks = {
        f"step 1 margin is 0 (|{first_margin:.2e}| < 1e-3)": abs(first_margin) < 1e-3,
        f"step 1 loss is ln 2 ({first_loss:.4f})": abs(first_loss - math.log(2)) < 1e-3,
        f"margin rose ({first_margin:+.4f} -> {last_margin:+.4f})": last_margin > first_margin + 0.1,
        f"loss fell below 0.5 ({first_loss:.4f} -> {last_loss:.4f})": last_loss < 0.5,
    }
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        sys.exit(1)
    print("DPO overfit check passed.")


if __name__ == "__main__":
    main()
