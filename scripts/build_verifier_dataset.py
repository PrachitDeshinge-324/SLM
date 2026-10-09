"""
Pipeline stage 2: turn a generator run into the set of solutions the verifier will judge.

Takes a fixed random sample of questions (the same questions for every precision of a model)
and writes one record per generated solution, labelled correct or incorrect by its final answer.
Output: results/<dataset>/verifier_dataset/<model>/verifier_<run name>.jsonl
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

# Ensure the repository root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match
from src.experiment_d.run_files import generator_run_path, verifier_dataset_path

QUESTION_SAMPLE_SEED = 42
MAX_QUESTIONS = 200


def parse_args():
    parser = argparse.ArgumentParser(description="Build the verifier dataset from a generator run (Experiment D, stage 2)")
    parser.add_argument("--results_dir", "--directory", "--output_dir", dest="results_dir", type=str, default="results", help="Root directory of the run files")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--precision", type=str, required=True, help="Precision of the generator run (16bit, 8bit, 4bit)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. gsm8k, cqa)")
    parser.add_argument("--n_samples", type=str, required=True, help="Samples per question of the generator run (8, or n8)")
    return parser.parse_args()


def load_question_records(generator_file) -> list:
    """Per-question records of a generator run (the run_config header line is skipped)."""
    with open(generator_file, 'r') as f:
        records = [json.loads(line) for line in f]
    return [record for record in records if 'question' in record]


def sample_questions(records: list) -> list:
    """The same MAX_QUESTIONS questions for every run of the same question set."""
    # Sort by question so random sampling is perfectly deterministic across precisions
    records = sorted(records, key=lambda record: record['question'])
    random.seed(QUESTION_SAMPLE_SEED)
    return random.sample(records, min(MAX_QUESTIONS, len(records)))


def build_trace_records(question_records: list, extractor, precision: str) -> list:
    traces = []
    for item in question_records:
        ground_truth = item.get('ground_truth', '')
        gt_answer = ground_truth.split('####')[-1].strip()

        for trace_idx, sample in enumerate(item['raw_samples']):
            student_solution = sample['response']
            # Re-extract with the current extractor: stored 'extracted' came from older code.
            generated_answer = extractor(student_solution)

            traces.append({
                'question': item['question'],
                'ground_truth': ground_truth,
                'gt_answer': gt_answer,
                'trace_index': trace_idx,
                'generated_answer': generated_answer,
                'student_solution': student_solution,
                'is_correct': answers_match(generated_answer, gt_answer),
                'trace_cutoff': bool(sample.get('cutoff', False)),
                'label_basis': 'final_answer_match',
                'precision': precision
            })
    return traces


def main():
    args = parse_args()

    generator_file = generator_run_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples)
    if not generator_file.exists():
        print(f"Error: generator run not found: {generator_file}")
        sys.exit(1)

    records = load_question_records(generator_file)
    if not records:
        raise RuntimeError(f"No question records found in {generator_file}")
    sampled = sample_questions(records)
    traces = build_trace_records(sampled, get_extractor(args.dataset), args.precision)

    output_file = verifier_dataset_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples)
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, 'w') as f:
        for trace in traces:
            f.write(json.dumps(trace) + '\n')

    print(f"[{args.precision}] Extracted {len(traces)} traces ({len(sampled)} questions) to {output_file}")


if __name__ == '__main__':
    main()
