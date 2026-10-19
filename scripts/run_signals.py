"""
Every per-solution signal of a generator run in one pass, so nothing has to be computed twice.

For each stored solution of the question set, one row with:
  answer, vote_support   from the stored answers alone (no model)
  token                  token signals of one teacher-forced pass: sequence log-likelihood, token entropy,
                         self-certainty, ... (src/experiment_d/token_signals.py)
  mc                     Mind the Gap's multiple-choice verifier: P(Correct) from one forward pass
  <critic>               for the solutions the pilot selects: sampled verdicts of each critic named in --critics
                         (default critic_short), with the forced verdict readout for critiques cut off before
                         their closing line (src/experiment_d/verifier_variants.py)

Gate: after --gate_questions questions with critic verdicts, the run stops (exit code 2) if fewer than
--min_verdict_rate of a critic's critiques reached their closing line. Set --min_verdict_rate 0 to turn it off.

Reference answers are not read here; they are used only afterwards, by the analysis.

Input : the generator run (results/<dataset>/generator/... or a self-training round's)
Output: results/<dataset>/signals/<model>/signals_<tag>_<run name>.jsonl, where the tag names the critics, verdicts,
        token cap and selection (e.g. critic_forced_v4_t704_all), so different runs never share a file; interrupted
        runs resume by question. A file from before tags is moved to its tagged name on the next run.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# The verifier and token-signal runners hold the model-side code; this script runs them together.
import run_verifier_variants as verifiers      # noqa: E402  (sets up warnings, .env and line-buffered output)
import score_token_signals as tokens           # noqa: E402
import torch                                    # noqa: E402

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer, model_device
from src.experiment_d.prompts import build_prompt, get_messages
from src.experiment_d.question_sets import phase2_questions, records_in
from src.experiment_d.run_files import (
    generator_run_path,
    parse_n_samples,
    read_question_records,
    read_run_config,
    self_training_round_dir,
    signals_path,
)
from src.experiment_d.signals import gate_failures, signals_tag, tag_of_header, verdict_rates, vote_support
from src.experiment_d.verifier_variants import (
    CRITIC_THRESHOLD,
    CRITICS,
    SAMPLING,
    answer_prefix,
    select_traces,
    verdict_cue,
)

# A resumed run must have been started with the same settings.
SETTINGS = ("question_ids", "critics", "verdicts", "max_new_tokens", "temperature", "top_p", "top_k", "tau",
            "skip_cutoff", "contested_only", "max_per_answer", "mc", "token_signals", "dtype")


def parse_args():
    parser = argparse.ArgumentParser(description="Every per-solution signal of a generator run in one pass")
    parser.add_argument("--results_dir", "--output_dir", dest="results_dir", type=str, default="results")
    parser.add_argument("--model", "--model_id", dest="model", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=PRECISIONS, default="16bit")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--n_samples", type=str, default="8")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--question_set", type=str, choices=("phase2", "first"), default="phase2")
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--dtype", type=str, choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--critics", type=str, default="critic_short",
                        help=f"Comma-separated sampled critics to run ({', '.join(CRITICS)}); empty for none")
    parser.add_argument("--verdicts", type=int, default=4, help="Sampled verdicts per solution and critic")
    parser.add_argument("--max_new_tokens", type=int, default=400, help="Longest critique")
    parser.add_argument("--all_solutions", action="store_true",
                        help="Critics judge every solution; default: the pilot selection (no cut-off solutions, "
                             "no uncontested questions, at most 2 solutions per answer)")
    parser.add_argument("--no_mc", action="store_true", help="Skip the mc verifier")
    parser.add_argument("--no_token_signals", action="store_true", help="Skip the teacher-forced token signals")
    parser.add_argument("--min_verdict_rate", type=float, default=0.85,
                        help="Stop when fewer than this share of a critic's critiques reach their closing line")
    parser.add_argument("--gate_questions", type=int, default=16, help="Questions with critic verdicts before the gate is checked")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--readout_batch", type=int, default=16)
    parser.add_argument("--chunk_questions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=SAMPLING["temperature"])
    parser.add_argument("--top_p", type=float, default=SAMPLING["top_p"])
    parser.add_argument("--top_k", type=int, default=SAMPLING["top_k"])
    parser.add_argument("--tau", type=float, default=CRITIC_THRESHOLD)
    parser.add_argument("--dry_run", action="store_true", help="No model: print the counts and the first critic prompt")
    args = parser.parse_args()
    args.n_samples = parse_n_samples(args.n_samples)
    args.critics = [c for c in args.critics.split(",") if c]
    unknown = [c for c in args.critics if c not in CRITICS]
    if unknown:
        parser.error(f"unknown critics {unknown}; known: {list(CRITICS)}")
    if args.experiment is None and args.round != 0:
        parser.error("--round needs --experiment")
    args.extractor = get_extractor(args.dataset)
    pilot = not args.all_solutions
    args.skip_cutoff, args.contested_only, args.max_per_answer = pilot, pilot, (2 if pilot else None)
    return args


def runner_args(args, variant: str, selected: bool) -> argparse.Namespace:
    """The settings the verifier runner's functions read, for one variant; selected=False judges every solution."""
    ns = argparse.Namespace(**vars(args))
    ns.variant = variant
    if not selected:
        ns.skip_cutoff, ns.contested_only, ns.max_per_answer = False, False, None
    return ns


def critic_fields(row: dict) -> dict:
    keep = ("responses", "values", "readouts", "readout_mass", "n", "parsed", "score", "decision",
            "forced_score", "forced_decision")
    return {k: row[k] for k in keep if k in row}


def score_chunk(model, tokenizer, chunk: list, args, critic_args: dict, mc_args, token_setup) -> list:
    """All rows of a chunk of questions, every signal filled in."""
    rows = {}
    for record in chunk:
        for index, (key, support) in enumerate(vote_support(record["raw_samples"], args.extractor)):
            sample = record["raw_samples"][index]
            rows[(record["qid"], index)] = {"qid": record["qid"], "trace_index": index,
                                            "finished": not sample.get("cutoff", False),
                                            "answer": key, "vote_support": support}
    for critic, ns in critic_args.items():
        for row in verifiers.sampled_rows(model, tokenizer, chunk, ns):
            rows[(row["qid"], row["trace_index"])][critic] = critic_fields(row)
    if mc_args is not None:
        for row in verifiers.mc_rows(model, tokenizer, chunk, mc_args):
            rows[(row["qid"], row["trace_index"])]["mc"] = {"score": row["score"], "verdict_mass": row["verdict_mass"]}
    if token_setup is not None:
        settings, end_token = token_setup
        for record in chunk:
            prompt_ids = tokenizer(build_prompt(tokenizer, get_messages(args.dataset, record["question"])),
                                   add_special_tokens=False)["input_ids"]
            for index, sample in enumerate(record["raw_samples"]):
                sequence = tokens.build_sequence(tokenizer, prompt_ids, sample, end_token)
                values = tokens.score(model, tokenizer, sequence, settings)
                rows[(record["qid"], index)]["token"] = {"flags": sequence["flags"], **values}
    return [rows[key] for key in sorted(rows, key=lambda k: ([r["qid"] for r in chunk].index(k[0]), k[1]))]


def move_untagged(untagged: Path) -> None:
    """
    A signals file written before names carried tags is moved to the tagged name its own header gives, so
    that no run writes into or reads another run's file. Never overwrites: if the tagged file already exists,
    the untagged one is left where it is and reported.
    """
    if not untagged.exists():
        return
    with untagged.open() as handle:
        header = json.loads(handle.readline())
    target = untagged.with_name(f"signals_{tag_of_header(header)}_{untagged.name[len('signals_'):]}")
    if target.exists():
        print(f"WARNING: {untagged.name} has no tag and {target.name} already exists; left both as they are.")
        return
    untagged.rename(target)
    print(f"Moved the untagged {untagged.name} (critics {header.get('critics')}) to {target.name}.")


def dry_run(args, records: list):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stored = sum(len(r["raw_samples"]) for r in records)
    judged = sum(len(select_traces(r["raw_samples"], args.extractor, args.skip_cutoff, args.contested_only,
                                   args.max_per_answer)) for r in records)
    print(f"{len(records)} questions, {stored} stored solutions. Per solution: vote support"
          + ("" if args.no_token_signals else ", token signals (one pass)") + ("" if args.no_mc else ", mc (one pass)") + ".")
    for critic in args.critics:
        print(f"{critic}: {judged} solutions x {args.verdicts} verdicts = {judged * args.verdicts} generations, "
              f"up to {args.max_new_tokens} tokens each; gate: stop if under {args.min_verdict_rate:.0%} of critiques "
              f"reach a verdict after {args.gate_questions} questions.")
    if args.critics:
        first = next(r for r in records if select_traces(r["raw_samples"], args.extractor, args.skip_cutoff,
                                                          args.contested_only, args.max_per_answer))
        index = select_traces(first["raw_samples"], args.extractor, args.skip_cutoff, args.contested_only, args.max_per_answer)[0]
        prompt = verifiers.critic_prompt(tokenizer, first, first["raw_samples"][index], args.critics[0])
        print(f"--- first prompt of {args.critics[0]} ---\n{prompt}\n---")


def main():
    args = parse_args()
    root = self_training_round_dir(args.results_dir, args.experiment, args.round) if args.experiment else Path(args.results_dir)
    run_file = generator_run_path(root, args.model, args.precision, args.dataset, args.n_samples)
    if not run_file.exists():
        raise FileNotFoundError(f"No generator run at {run_file}")
    config, records = read_run_config(run_file), read_question_records(run_file)
    if args.question_set == "phase2":
        questions = phase2_questions(args.results_dir, args.model, args.dataset, args.n_samples, args.questions)
        if not questions:
            raise RuntimeError("No Phase 2 question set for this model and dataset; use --question_set first")
        records = records_in(records, questions)
    records = records[:args.questions]
    print(f"Question set '{args.question_set}': {len(records)} questions, first {records[0]['qid']}, last {records[-1]['qid']}.")
    round_index = args.round if args.experiment else None
    tag = signals_tag(args.critics, args.verdicts, args.max_new_tokens, args.all_solutions)
    output = signals_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples,
                          args.experiment, round_index, tag=tag)
    if not args.dry_run:
        move_untagged(signals_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples,
                                   args.experiment, round_index))
    print(f"Output: {output}")
    if args.dry_run:
        dry_run(args, records)
        return

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision, dtype=None if args.dtype == "auto" else args.dtype)
    token_setup = None
    if not args.no_token_signals:
        probe = hashlib.sha256(build_prompt(tokenizer, get_messages(args.dataset, "PROBE QUESTION")).encode("utf-8")).hexdigest()
        if config.get("prompt_sha256") and probe != config["prompt_sha256"]:
            raise RuntimeError("The rebuilt generator prompt differs from the one the run was generated with; "
                               "token signals would be wrong. Use --no_token_signals or fix the prompt.")
        # The model's own end-of-turn token, the one the stored lengths count up to (see score_token_signals.py).
        token_setup = ({"temperature": config["temperature"], "top_k": config["top_k"], "top_p": config["top_p"]},
                       tokenizer.eos_token_id)

    header = {"type": "run_config", "generator_run": str(run_file), "model": args.model, "precision": args.precision,
              "dataset": args.dataset, "generator_n_samples": args.n_samples, "question_set": args.question_set,
              "question_ids": [r["qid"] for r in records], "critics": args.critics, "verdicts": args.verdicts,
              "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p,
              "top_k": args.top_k, "tau": args.tau, "skip_cutoff": args.skip_cutoff,
              "contested_only": args.contested_only, "max_per_answer": args.max_per_answer, "mc": not args.no_mc,
              "token_signals": not args.no_token_signals,
              "answer_prefix": {c: answer_prefix(c) for c in args.critics},
              "verdict_cue": {c: verdict_cue(c) for c in args.critics}, "dtype": str(model.dtype),
              "device": str(model_device(model)), "torch_version": torch.__version__,
              "generator_sampling": token_setup[0] if token_setup else None,
              "end_token": tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)}

    done_rows = []
    if output.exists():
        for line in output.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "run_config":
                changed = [k for k in SETTINGS if row.get(k, "<missing>") != header[k]]
                if changed:
                    raise RuntimeError(f"{output} holds a run with other settings ({', '.join(changed)}). "
                                       "Delete it or use another --results_dir.")
            elif "qid" in row:
                done_rows.append(row)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(header) + "\n")
    done = {row["qid"] for row in done_rows}

    critic_args = {c: runner_args(args, c, selected=True) for c in args.critics}
    mc_args = None if args.no_mc else runner_args(args, "mc", selected=False)
    todo = [r for r in records if r["qid"] not in done]
    print(f"{len(records)} questions, {len(done)} already done, {len(todo)} to do. Critics: {args.critics or 'none'}; "
          f"mc: {not args.no_mc}; token signals: {not args.no_token_signals}.")

    gate_checked = False
    all_rows = list(done_rows)
    start = time.time()
    for first in range(0, len(todo), args.chunk_questions):
        chunk = todo[first:first + args.chunk_questions]
        rows = score_chunk(model, tokenizer, chunk, args, critic_args, mc_args, token_setup)
        with output.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        all_rows += rows
        elapsed, count = time.time() - start, first + len(chunk)
        rates = verdict_rates(all_rows, args.critics)
        shown = "".join(f", {c} verdict reached {reached / total:.0%} of {total}" for c, (reached, total) in rates.items() if total)
        print(f"  {count}/{len(todo)} questions, {elapsed / count:.1f}s per question, "
              f"about {elapsed / count * (len(todo) - count) / 60:.0f} min left{shown}", flush=True)

        judged_questions = len({row["qid"] for row in all_rows if any(c in row for c in args.critics)})
        if args.critics and not gate_checked and judged_questions >= args.gate_questions:
            gate_checked = True
            failed = gate_failures(rates, args.min_verdict_rate)
            if failed:
                print(f"STOPPED by the gate: under {args.min_verdict_rate:.0%} of critiques reached a verdict "
                      f"after {judged_questions} questions ({', '.join(failed)}). The rows so far are kept in {output}. "
                      "Fix the prompt or the token limit before a full run; rerun with --min_verdict_rate 0 to go on anyway.")
                sys.exit(2)
            print(f"  gate passed after {judged_questions} questions.")
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
