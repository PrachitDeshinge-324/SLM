"""
Verifier designs from the literature, run on the stored solutions of a generator run.

Variants (see src/experiment_d/verifier_variants.py): mc, cot_binary, cot_score, critic, judge5, backward,
self_correct. The model judges its own stored solutions; reference answers are not read here (they are
used only afterwards, by the analysis). The generator run is never modified.

Input : the generator run (results/<dataset>/generator/... or a self-training round's)
Output: results/<dataset>/verifier_variants/<model>/<variant>_<run name>.jsonl, one record per stored
        solution; interrupted runs resume.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure the repository root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Progress must show up as it happens, also when the output goes through a pipe or a notebook's %%bash cell.
sys.stdout.reconfigure(line_buffering=True)

from src.experiment_d.warning_filters import silence_harmless_warnings

silence_harmless_warnings()
load_dotenv()

import torch

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import normalize_answer
from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer, model_device
from src.experiment_d.prompts import build_prompt, get_messages
from src.experiment_d.question_sets import phase2_questions, records_in
from src.experiment_d.run_files import (
    generator_run_path,
    parse_n_samples,
    read_question_records,
    read_run_config,
    self_training_round_dir,
    verifier_variants_path,
)
from src.experiment_d.sampling import sample_responses
from src.experiment_d.verifier_variants import (
    CRITIC_THRESHOLD,
    DEFAULTS,
    SAMPLING,
    VARIANTS,
    aggregate,
    backward_messages,
    backward_score,
    mask_conditions,
    parse_backward,
    parse_response,
    self_correct_messages,
    two_way_probability,
    verification_messages,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a verifier design from the literature on stored solutions")
    parser.add_argument("--results_dir", "--output_dir", dest="results_dir", type=str, default="results")
    parser.add_argument("--model", "--model_id", dest="model", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=PRECISIONS, default="16bit")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--n_samples", type=str, default="8", help="N of the generator run whose solutions are judged")
    parser.add_argument("--variant", type=str, choices=VARIANTS, required=True)
    parser.add_argument("--experiment", type=str, default=None, help="Self-training experiment whose round holds the run")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--question_set", type=str, choices=("phase2", "first"), default="phase2",
                        help="phase2: the questions the Phase 2 page scores for this model (default, so every variant is "
                             "compared on the same questions as the current verifier); first: the first N of the generator run")
    parser.add_argument("--questions", type=int, default=100, help="Judge at most this many questions of the set")
    parser.add_argument("--dry_run", action="store_true", help="No model: load the tokenizer, print a prompt and count the generations")
    parser.add_argument("--verdicts", type=int, default=None, help="Samples per solution (per condition for backward); default per variant")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128, help="Generations per forward batch")
    parser.add_argument("--chunk_questions", type=int, default=4, help="Questions judged between writes to the output")
    parser.add_argument("--temperature", type=float, default=SAMPLING["temperature"])
    parser.add_argument("--top_p", type=float, default=SAMPLING["top_p"])
    parser.add_argument("--top_k", type=int, default=SAMPLING["top_k"], help="0 turns top-k off")
    parser.add_argument("--max_conditions", type=int, default=6, help="backward: most numbers masked per question")
    parser.add_argument("--tau", type=float, default=CRITIC_THRESHOLD, help="critic: decision threshold on the correct rate")
    args = parser.parse_args()
    args.n_samples = parse_n_samples(args.n_samples)
    defaults = DEFAULTS[args.variant]
    if args.verdicts is None:
        args.verdicts = defaults["n"]
    if args.max_new_tokens is None:
        args.max_new_tokens = defaults["max_new_tokens"]
    if args.experiment is None and args.round != 0:
        parser.error("--round needs --experiment")
    return args


def solutions_of(record: dict) -> list:
    """(trace index, sample) of every stored solution of a question, truncated ones included."""
    return list(enumerate(record["raw_samples"]))


def base_row(record: dict, index: int, sample: dict, variant: str) -> dict:
    return {"qid": record["qid"], "trace_index": index, "variant": variant, "finished": not sample.get("cutoff", False)}


def generate(model, tokenizer, prompts: list, args) -> list:
    """`verdicts` sampled responses per prompt, as the verifier papers draw them."""
    grouped, tokens_per_second, _, _ = sample_responses(
        model, tokenizer, prompts, n_samples=args.verdicts, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    print(f"    {tokens_per_second:.0f} generated tokens/s", flush=True)
    return grouped


@torch.inference_mode()
def generate_greedy(model, tokenizer, prompts: list, max_new_tokens: int, batch_size: int) -> list:
    """One greedy response per prompt (left-padded batches)."""
    device = model_device(model)
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out = []
    try:
        for start in range(0, len(prompts), batch_size):
            inputs = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                               add_special_tokens=False).to(device)
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
            out += tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        tokenizer.padding_side = side
    return out


@torch.inference_mode()
def mc_rows(model, tokenizer, records: list, args) -> list:
    """Mind the Gap's multiple-choice verifier, scored exactly from the first-token probabilities of the two verdicts."""
    device = model_device(model)
    correct, incorrect = (tokenizer(word, add_special_tokens=False)["input_ids"][0] for word in ("Correct", "Incorrect"))
    if correct == incorrect:
        raise RuntimeError("'Correct' and 'Incorrect' start with the same token; the two-way probability is undefined")
    rows = []
    for record in records:
        for index, sample in solutions_of(record):
            prompt = build_prompt(tokenizer, verification_messages("mc", record["question"], sample["response"]))
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            logp = torch.log_softmax(model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float(), -1)
            rows.append({**base_row(record, index, sample, "mc"),
                         "score": two_way_probability(float(logp[correct]), float(logp[incorrect])),
                         "verdict_mass": float(logp[correct].exp() + logp[incorrect].exp())})
    return rows


def sampled_rows(model, tokenizer, records: list, args) -> list:
    """cot_binary, cot_score, critic, judge5: `verdicts` sampled responses per solution, parsed and aggregated."""
    items = [(record, index, sample) for record in records for index, sample in solutions_of(record)]
    prompts = [build_prompt(tokenizer, verification_messages(args.variant, record["question"], sample["response"]))
               for record, _, sample in items]
    grouped = generate(model, tokenizer, prompts, args)
    rows = []
    for (record, index, sample), responses in zip(items, grouped):
        values = [parse_response(args.variant, response) for response in responses]
        rows.append({**base_row(record, index, sample, args.variant), "responses": responses, "values": values,
                     **aggregate(args.variant, values, args.tau)})
    return rows


def backward_rows(model, tokenizer, records: list, args) -> list:
    """Self-verification: each number masked in turn, the candidate answer given, the masked value to recover."""
    extractor = get_extractor(args.dataset)
    units = []     # (qid, answer key, answer text, masked value, prompt)
    for record in records:
        seen = set()
        for _, sample in solutions_of(record):
            answer = None if sample.get("cutoff") else extractor(sample["response"])
            if answer is None or normalize_answer(answer) in seen:
                continue
            seen.add(normalize_answer(answer))
            for value, masked in mask_conditions(record["question"], args.max_conditions):
                units.append((record["qid"], normalize_answer(answer), answer, value,
                              build_prompt(tokenizer, backward_messages(masked, answer))))
    grouped = generate(model, tokenizer, [unit[4] for unit in units], args) if units else []
    per_answer = {}
    for (qid, key, _, value, _), responses in zip(units, grouped):
        per_answer.setdefault((qid, key), []).append([parse_backward(response, value) for response in responses])
    rows = []
    for record in records:
        for index, sample in solutions_of(record):
            answer = None if sample.get("cutoff") else extractor(sample["response"])
            key = None if answer is None else (record["qid"], normalize_answer(answer))
            scored = backward_score(per_answer[key]) if key in per_answer else {"conditions": 0, "samples": 0, "matches": 0, "score": None}
            rows.append({**base_row(record, index, sample, "backward"), "answer": answer, **scored})
    return rows


def self_correct_rows(model, tokenizer, records: list, args) -> list:
    """Intrinsic self-correction: review, then revise, both greedy; an unchanged answer is read as an approval."""
    extractor = get_extractor(args.dataset)
    items = [(record, index, sample) for record in records for index, sample in solutions_of(record)]
    base = lambda record: get_messages(args.dataset, record["question"])
    review = [build_prompt(tokenizer, self_correct_messages(base(r), s["response"])) for r, _, s in items]
    feedback = generate_greedy(model, tokenizer, review, args.max_new_tokens, args.batch_size)
    improve = [build_prompt(tokenizer, self_correct_messages(base(r), s["response"], f)) for (r, _, s), f in zip(items, feedback)]
    revised = generate_greedy(model, tokenizer, improve, args.max_new_tokens, args.batch_size)
    rows = []
    for (record, index, sample), note, final in zip(items, feedback, revised):
        before = None if sample.get("cutoff") else extractor(sample["response"])
        after = extractor(final)
        unchanged = None if before is None or after is None else normalize_answer(before) == normalize_answer(after)
        rows.append({**base_row(record, index, sample, "self_correct"), "feedback": note, "revised": final,
                     "answer": before, "revised_answer": after, "unchanged": unchanged,
                     "score": None if unchanged is None else float(unchanged)})
    return rows


RUNNERS = {"mc": mc_rows, "cot_binary": sampled_rows, "cot_score": sampled_rows, "critic": sampled_rows,
           "judge5": sampled_rows, "backward": backward_rows, "self_correct": self_correct_rows}


def dry_run(args, records: list):
    """Print what a run would send to the model, without loading it: the first prompt and the generation counts."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    first = records[0]
    sample = first["raw_samples"][0]
    solutions = sum(len(record["raw_samples"]) for record in records)
    if args.variant in ("mc", "cot_binary", "cot_score", "critic", "judge5"):
        prompt = build_prompt(tokenizer, verification_messages(args.variant, first["question"], sample["response"]))
        generations = 0 if args.variant == "mc" else solutions * args.verdicts
        note = "one forward pass each (no generation)" if args.variant == "mc" else f"{solutions} solutions x {args.verdicts} verdicts"
    elif args.variant == "backward":
        extractor = get_extractor(args.dataset)
        units = 0
        for record in records:
            answers = {normalize_answer(a) for a in (None if s.get("cutoff") else extractor(s["response"]) for s in record["raw_samples"]) if a}
            units += len(answers) * len(mask_conditions(record["question"], args.max_conditions))
        masked = mask_conditions(first["question"], args.max_conditions)
        prompt = build_prompt(tokenizer, backward_messages(masked[0][1], "ANSWER")) if masked else "(the first question has no numbers)"
        generations = units * args.verdicts
        note = f"{units} (distinct answer, masked number) pairs x {args.verdicts} samples"
    else:
        prompt = build_prompt(tokenizer, self_correct_messages(get_messages(args.dataset, first["question"]), sample["response"]))
        generations = solutions * 2
        note = f"{solutions} solutions x 2 greedy steps (review, revise)"
    print(f"--- first prompt of {args.variant} ({len(tokenizer(prompt, add_special_tokens=False)['input_ids'])} tokens) ---\n{prompt}\n---")
    print(f"{len(records)} questions, {solutions} stored solutions. Generations: {generations} ({note}); "
          f"up to {args.max_new_tokens} new tokens each.")


def main():
    args = parse_args()
    root = self_training_round_dir(args.results_dir, args.experiment, args.round) if args.experiment else Path(args.results_dir)
    run_file = generator_run_path(root, args.model, args.precision, args.dataset, args.n_samples)
    if not run_file.exists():
        raise FileNotFoundError(f"No generator run at {run_file}")
    records = read_question_records(run_file)
    if args.question_set == "phase2":
        questions = phase2_questions(args.results_dir, args.model, args.dataset, args.n_samples, args.questions)
        if not questions:
            raise RuntimeError("No Phase 2 question set for this model and dataset (its verifier runs share too few questions); "
                               "use --question_set first")
        records = records_in(records, questions)
    records = records[:args.questions]
    print(f"Question set '{args.question_set}': {len(records)} questions, first {records[0]['qid']}, last {records[-1]['qid']}.")
    output = verifier_variants_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples,
                                    args.variant, args.experiment, args.round if args.experiment else None)

    if args.dry_run:
        dry_run(args, records)
        return

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)
    done = set()
    if output.exists():
        with output.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if "qid" in row:
                        done.add((row["qid"], row["trace_index"]))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        header = {"type": "run_config", "variant": args.variant, "generator_run": str(run_file), "model": args.model,
                  "precision": args.precision, "dataset": args.dataset, "generator_n_samples": args.n_samples,
                  "questions": len(records), "question_ids": [r["qid"] for r in records],
                  "question_set": args.question_set, "verdicts": args.verdicts, "max_new_tokens": args.max_new_tokens,
                  "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "tau": args.tau,
                  "max_conditions": args.max_conditions, "dtype": str(model.dtype), "device": str(model_device(model)),
                  "torch_version": torch.__version__}
        output.write_text(json.dumps(header) + "\n")

    todo = [record for record in records
            if any((record["qid"], i) not in done for i, _ in solutions_of(record))]
    print(f"{args.variant}: {len(records)} questions, {len(records) - len(todo)} already done, {len(todo)} to do; "
          f"verdicts {args.verdicts}, max_new_tokens {args.max_new_tokens}.")
    start = time.time()
    for first in range(0, len(todo), args.chunk_questions):
        chunk = todo[first:first + args.chunk_questions]
        rows = RUNNERS[args.variant](model, tokenizer, chunk, args)
        with output.open("a") as handle:
            for row in rows:
                if (row["qid"], row["trace_index"]) not in done:
                    handle.write(json.dumps(row) + "\n")
        elapsed = time.time() - start
        count = first + len(chunk)
        print(f"  {count}/{len(todo)} questions, {elapsed / count:.1f}s per question, "
              f"about {elapsed / count * (len(todo) - count) / 60:.0f} min left", flush=True)
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
