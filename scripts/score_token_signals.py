"""
Token-level confidence signals of stored solutions: one teacher-forced pass, no sampling, no training.

Reads a generator run, feeds each stored solution back to the same model after its own prompt, and
writes per solution the sequence log-likelihood, token entropy (also by third and over the final
answer), self-certainty, the sampling-temperature versions, the end-token probability, and how many
of the written tokens lie inside the sampler's top-k and nucleus (every sampled token must: the check
that the reconstruction is exact). Reference answers are not read. See src/experiment_d/token_signals.py.

Input : the generator run (results/<dataset>/generator/... or a self-training round's)
Output: results/.../token_signals/token_signals_<run name>.jsonl, one record per solution; interrupted
        runs resume. The generator run is never modified.
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure the repository root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.warning_filters import silence_harmless_warnings

silence_harmless_warnings()
load_dotenv()

import torch

from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer, model_device
from src.experiment_d.prompts import build_prompt, get_messages
from src.experiment_d.run_files import (
    generator_run_path,
    parse_n_samples,
    read_question_records,
    read_run_config,
    self_training_round_dir,
    token_signals_path,
)
from src.experiment_d.token_signals import aggregate, position_statistics, target_ids

ANSWER_MARKER = "####"


def parse_args():
    parser = argparse.ArgumentParser(description="Token-level confidence signals of stored solutions (one forward pass each)")
    parser.add_argument("--results_dir", "--output_dir", dest="results_dir", type=str, default="results")
    parser.add_argument("--model", "--model_id", dest="model", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=PRECISIONS, default="16bit")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--n_samples", type=str, default="8")
    parser.add_argument("--experiment", type=str, default=None, help="Self-training experiment whose round holds the run")
    parser.add_argument("--round", type=int, default=0, help="Round of the experiment (0 = the untrained model)")
    parser.add_argument("--limit", type=int, default=None, help="Score only this many solutions (a probe)")
    parser.add_argument("--include_cutoff", type=int, default=0, help="With --limit: how many of them are cut-off solutions")
    parser.add_argument("--invariance_check", action="store_true", help="Also score two solutions as one padded batch and compare")
    args = parser.parse_args()
    args.n_samples = parse_n_samples(args.n_samples)
    if args.experiment is None and args.round != 0:
        parser.error("--round needs --experiment")
    return args


def select_solutions(records: list, limit, include_cutoff: int) -> list:
    """(record, trace index) in file order; with a limit, the first finished ones plus `include_cutoff` cut-off ones."""
    every = [(record, i) for record in records for i in range(len(record["raw_samples"]))]
    if limit is None:
        return every
    cut = [item for item in every if item[0]["raw_samples"][item[1]].get("cutoff")][:include_cutoff]
    finished = [item for item in every if not item[0]["raw_samples"][item[1]].get("cutoff")][:max(limit - len(cut), 0)]
    chosen = set(map(lambda item: (item[0]["qid"], item[1]), finished + cut))
    return [item for item in every if (item[0]["qid"], item[1]) in chosen]


def answer_start(tokenizer, response: str, n_tokens: int):
    """Index of the first token after the last answer marker (####) in the response, None without one."""
    position = response.rfind(ANSWER_MARKER)
    if position < 0:
        return None
    offsets = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    for index, (start, _) in enumerate(offsets[:n_tokens]):
        if start >= position + len(ANSWER_MARKER):
            return index
    return None


def build_sequence(tokenizer, prompt_ids: list, sample: dict, end_token: int) -> dict:
    """The ids fed to the model and the written tokens to score, for one stored solution."""
    finished = not sample.get("cutoff", False)
    ids = tokenizer(sample["response"], add_special_tokens=False)["input_ids"]
    written, flags = target_ids(ids, tokenizer.decode(ids[-1:]) if ids else "", finished, sample["length"], end_token)
    return {"input_ids": prompt_ids + written[:-1], "targets": written, "finished": finished, "flags": flags,
            "answer_start": answer_start(tokenizer, sample["response"], len(written) - (1 if finished else 0))}


@torch.inference_mode()
def score(model, tokenizer, sequence: dict, settings: dict) -> dict:
    """Scalars of one solution. Row t of the kept logits scores the token written at position t."""
    device = model_device(model)
    n = len(sequence["targets"])
    input_ids = torch.tensor([sequence["input_ids"]], device=device)
    try:
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=n).logits[0]
    except TypeError:       # a model class without logits_to_keep: full logits, last n rows
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -n:]
    stats = position_statistics(logits, torch.tensor(sequence["targets"], device=device), settings["temperature"],
                                settings["top_k"], settings["top_p"])
    stats = {name: values.cpu() for name, values in stats.items()}
    return aggregate(stats, sequence["finished"], sequence["answer_start"])


@torch.inference_mode()
def invariance_check(model, tokenizer, sequences: list, settings: dict) -> dict:
    """Two solutions scored alone and as one right-padded batch: the largest difference in log-likelihood."""
    device = model_device(model)
    pair = sorted(sequences, key=lambda s: len(s["input_ids"]))[:2]
    alone = [score(model, tokenizer, s, settings)["logp_sum"] for s in pair]
    width = max(len(s["input_ids"]) for s in pair)
    ids = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((2, width), dtype=torch.long)
    for row, s in enumerate(pair):
        ids[row, :len(s["input_ids"])] = torch.tensor(s["input_ids"])
        mask[row, :len(s["input_ids"])] = 1
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).logits
    batched = []
    for row, s in enumerate(pair):
        n, end = len(s["targets"]), len(s["input_ids"])
        rows = logits[row, end - n:end]
        stats = position_statistics(rows, torch.tensor(s["targets"], device=device), settings["temperature"],
                                    settings["top_k"], settings["top_p"])
        stats = {name: values.cpu() for name, values in stats.items()}
        batched.append(aggregate(stats, s["finished"], s["answer_start"])["logp_sum"])
    return {"alone": alone, "batched": batched, "max_abs_difference": max(abs(a - b) for a, b in zip(alone, batched))}


def peak_memory_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9      # bytes on macOS


def main():
    args = parse_args()
    root = self_training_round_dir(args.results_dir, args.experiment, args.round) if args.experiment else Path(args.results_dir)
    run_file = generator_run_path(root, args.model, args.precision, args.dataset, args.n_samples)
    if not run_file.exists():
        raise FileNotFoundError(f"No generator run at {run_file}")
    config, records = read_run_config(run_file), read_question_records(run_file)
    settings = {"temperature": config["temperature"], "top_k": config["top_k"], "top_p": config["top_p"]}
    output = token_signals_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples,
                                args.experiment, args.round if args.experiment else None)

    load_start = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)
    load_seconds = time.time() - load_start
    device = model_device(model)

    # The prompt must be the one generation saw.
    probe = hashlib.sha256(build_prompt(tokenizer, get_messages(args.dataset, "PROBE QUESTION")).encode("utf-8")).hexdigest()
    if config.get("prompt_sha256") and probe != config["prompt_sha256"]:
        raise RuntimeError("The rebuilt prompt differs from the one the run was generated with; refusing to score.")
    # The end token is the model's own end-of-turn token, the one `length` in the run file is counted up to
    # (sampling.py stops counting at tokenizer.eos_token_id). It is NOT lora_training.end_token_id, which
    # returns the config's <|endoftext|> that the base model emits only after "<|im_end|>\n".
    end_token = tokenizer.eos_token_id

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
        header = {"type": "run_config", "generator_run": str(run_file), "model": args.model, "precision": args.precision,
                  "dataset": args.dataset, "n_samples": args.n_samples, "dtype": str(model.dtype), "device": str(device),
                  "torch_version": torch.__version__, "prompt_sha256": config.get("prompt_sha256"),
                  "end_token": tokenizer.convert_ids_to_tokens(end_token), **settings}
        output.write_text(json.dumps(header) + "\n")

    chosen = select_solutions(records, args.limit, args.include_cutoff)
    todo = [item for item in chosen if (item[0]["qid"], item[1]) not in done]
    print(f"{len(chosen)} solutions selected, {len(done)} already scored, {len(todo)} to do. "
          f"Loaded in {load_seconds:.0f}s on {device} ({model.dtype}).")

    prompt_cache, sequences, rows = {}, [], []
    start, scored_tokens = time.time(), 0
    for count, (record, index) in enumerate(todo, 1):
        if record["qid"] not in prompt_cache:
            prompt = build_prompt(tokenizer, get_messages(args.dataset, record["question"]))
            prompt_cache[record["qid"]] = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        sequence = build_sequence(tokenizer, prompt_cache[record["qid"]], record["raw_samples"][index], end_token)
        values = score(model, tokenizer, sequence, settings)
        row = {"qid": record["qid"], "trace_index": index, "finished": sequence["finished"],
               "stored_length": record["raw_samples"][index]["length"], "flags": sequence["flags"], **values}
        with output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        rows.append(row)
        sequences.append(sequence)
        scored_tokens += len(sequence["input_ids"]) + 1
        if count % 4 == 0 or count == len(todo):
            elapsed = time.time() - start
            print(f"  {count}/{len(todo)}  {elapsed / count:.1f}s per solution, {scored_tokens / elapsed:.0f} tokens/s, "
                  f"peak memory {peak_memory_gb():.1f} GB", flush=True)

    if rows:
        elapsed = time.time() - start
        topk = [r["support_topk"] for r in rows if "support_topk" in r]
        nucleus = [r["support_nucleus"] for r in rows if "support_nucleus" in r]
        summary = {
            "solutions": len(rows), "seconds": round(elapsed, 1), "seconds_per_solution": round(elapsed / len(rows), 2),
            "tokens_per_second": round(scored_tokens / elapsed), "peak_memory_gb": round(peak_memory_gb(), 2),
            "extrapolated_minutes_for_1600": round(elapsed / len(rows) * 1600 / 60, 1),
            "length_flags": sum(1 for r in rows if r["flags"]),
            "support_topk": {"min": min(topk), "mean": sum(topk) / len(topk), "below_0.99": sum(1 for v in topk if v < 0.99)},
            "support_nucleus": {"min": min(nucleus), "mean": sum(nucleus) / len(nucleus), "below_0.99": sum(1 for v in nucleus if v < 0.99)},
        }
        print(json.dumps(summary, indent=1))
        if args.invariance_check:
            finished_only = [s for s in sequences if s["finished"]]
            if len(finished_only) >= 2:
                print("Batch invariance:", json.dumps(invariance_check(model, tokenizer, finished_only, settings)))
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
