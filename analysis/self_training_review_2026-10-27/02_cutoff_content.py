"""
What is inside the samples that hit the token limit? (critic self-training runs, GSM8K test, 512 tokens.)

Self-training lowers the cut-off rate in every arm, and cut-off samples count as wrong, so what they contain
decides how to read the per-sample gain. Three readings:
  answered     the sample already wrote a '#### N' answer line and kept going (a stop-on-answer rule would fix it)
  repeating    the sample is stuck in a loop: a 40-character chunk from its last quarter occurs 3+ times in it
  unfinished   neither: the solution was still being written when the limit hit
Rows: round 0 and the vote arm (round 1, train seed 0) of each model and precision.
The last section prints the tails of a few random cut-off samples from 0.8B 4-bit round 0.
Reference answers are read only to score the answer lines.
"""
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_d.answer_matching import answers_match
from src.experiment_d.run_files import generator_run_path, read_question_records, self_training_round_dir

RESULTS = ROOT / "results"
MODELS = {"Qwen/Qwen3.5-0.8B": "q0.8b", "Qwen/Qwen3.5-2B": "q2b"}
PRECISIONS = ("16bit", "4bit")
ANSWER_LINE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
CHUNK = 40


def repeating(text: str) -> bool:
    tail = text[-len(text) // 4:]
    chunks = [tail[i:i + CHUNK] for i in range(0, max(1, len(tail) - CHUNK), CHUNK)]
    return any(text.count(c) >= 3 for c in chunks if c.strip())


def run_path(model: str, precision: str, round_index: int) -> Path:
    base = f"{MODELS[model]}_{precision}_critic_p500"
    experiment = base if round_index == 0 else f"{base}_vote"
    return generator_run_path(self_training_round_dir(RESULTS, experiment, round_index), model, precision, "gsm8k", 8)


def main():
    print("model | precision | run | samples | cut off | answered (answer right) | repeating | unfinished "
          "| median tokens of finished samples")
    for model in MODELS:
        for precision in PRECISIONS:
            for round_index, name in ((0, "round 0"), (1, "vote r1")):
                path = run_path(model, precision, round_index)
                if not path.exists():
                    continue
                n = cut = answered = answered_right = loops = 0
                finished_lengths = []
                for record in read_question_records(path):
                    for sample in record["raw_samples"]:
                        n += 1
                        if not sample["cutoff"]:
                            finished_lengths.append(sample["length"])
                            continue
                        cut += 1
                        text = sample["response"]
                        lines = ANSWER_LINE.findall(text)
                        if lines:
                            answered += 1
                            answered_right += answers_match(lines[0].replace(",", ""), record["ground_truth"])
                        elif repeating(text):
                            loops += 1
                print(f"{model} | {precision} | {name} | {n} | {cut} ({100 * cut / n:.1f}%) | "
                      f"{answered} ({answered_right}) | {loops} | {cut - answered - loops} | "
                      f"{np.median(finished_lengths):.0f}")

    print("\n=== Tails of 4 random cut-off samples, Qwen3.5-0.8B 4-bit round 0 (last 300 characters)")
    path = run_path("Qwen/Qwen3.5-0.8B", "4bit", 0)
    tails = [s["response"] for r in read_question_records(path) for s in r["raw_samples"] if s["cutoff"]]
    for text in random.Random(1).sample(tails, 4):
        print("---", repr(text[-300:]))


if __name__ == "__main__":
    main()
