"""
Backward-check (FOBAR) diagnostic for 4-bit SLMs on GSM8K.

Tests whether the 4-bit model's backward consistency score can rank the minority
correct answer above the majority wrong answer on settled-wrong ('systematic') questions.

Usage in Colab:
  python analysis/backward_check_diagnostic.py --model Qwen/Qwen3.5-0.8B --precision 4bit

Dry-run locally (no GPU needed):
  python analysis/backward_check_diagnostic.py --dry_run
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.run_files import generator_run_path, read_question_records
from src.experiment_d.verifier_variants import mask_conditions, backward_messages, parse_backward


def parse_args():
    parser = argparse.ArgumentParser(description="Backward check (FOBAR) diagnostic on stored 4-bit runs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B", help="Model ID")
    parser.add_argument("--precision", type=str, default="4bit", help="Model precision")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--n_samples", type=int, default=16, help="Generator run n_samples (16 or 8)")
    parser.add_argument("--results_dir", type=str, default=str(ROOT / "results"), help="Results directory")
    parser.add_argument("--classes_file", type=str,
                        default=str(ROOT / "analysis/damage_types_2026-10-24/outputs/01_question_types.csv"),
                        help="Question classification CSV")
    parser.add_argument("--class_filter", type=str, default="systematic",
                        choices=["systematic", "random", "preserved", "all"],
                        help="Which damage class to diagnose")
    parser.add_argument("--limit", type=int, default=0, help="Max questions to evaluate (0 for all)")
    parser.add_argument("--max_conditions", type=int, default=3, help="Max numbers to mask per question")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens for backward check")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (0.0 for greedy)")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "analysis/backward_diagnostic_outputs"),
                        help="Directory to save diagnostic reports")
    parser.add_argument("--dry_run", action="store_true", help="Print pairs and prompts without running generation")
    return parser.parse_args()


def load_target_questions(args):
    """
    Finds questions matching class_filter that have both a wrong majority answer
    and at least one correct minority answer (or vice versa for non-systematic).
    """
    extractor = get_extractor(args.dataset)
    classes = {}
    if Path(args.classes_file).exists():
        with open(args.classes_file, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") == args.dataset:
                    classes[row["qid"]] = row["type"]

    run_path = generator_run_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples)
    if not run_path.exists():
        # Fall back to n8 if n16 not found
        alt_path = generator_run_path(args.results_dir, args.model, args.precision, args.dataset, 8 if args.n_samples == 16 else 16)
        if alt_path.exists():
            print(f"Notice: {run_path.name} not found; falling back to {alt_path.name}")
            run_path = alt_path
        else:
            raise FileNotFoundError(f"No generator run found at {run_path} or {alt_path}")

    records = read_question_records(run_path)
    print(f"Loaded {len(records)} records from {run_path.name}")

    targets = []
    for r in records:
        qid = r["qid"]
        q_type = classes.get(qid, "unknown")
        if args.class_filter != "all" and q_type != args.class_filter:
            continue

        gt = r.get("ground_truth")
        if not gt:
            continue

        samples = r.get("raw_samples", [])
        answers = []
        for s in samples:
            ans = None if s.get("cutoff", False) else extractor(s["response"])
            if ans is not None:
                answers.append(ans)

        if not answers:
            continue

        # Count frequencies
        norm_counts = {}
        first_rep = {}
        for ans in answers:
            k = normalize_answer(ans)
            norm_counts[k] = norm_counts.get(k, 0) + 1
            if k not in first_rep:
                first_rep[k] = ans

        # Majority answer
        maj_key = max(norm_counts, key=norm_counts.get)
        maj_ans = first_rep[maj_key]
        maj_is_right = answers_match(maj_ans, gt)

        # Check for presence of right and wrong answers
        right_ans = None
        wrong_ans = None

        if maj_is_right:
            right_ans = maj_ans
            # Find most frequent wrong answer
            wrong_keys = [k for k in norm_counts if not answers_match(first_rep[k], gt)]
            if wrong_keys:
                best_wrong_key = max(wrong_keys, key=norm_counts.get)
                wrong_ans = first_rep[best_wrong_key]
        else:
            wrong_ans = maj_ans
            # Find any right answer
            right_keys = [k for k in norm_counts if answers_match(first_rep[k], gt)]
            if right_keys:
                best_right_key = max(right_keys, key=norm_counts.get)
                right_ans = first_rep[best_right_key]

        if right_ans is not None and wrong_ans is not None:
            conditions = mask_conditions(r["question"], max_conditions=args.max_conditions)
            if conditions:
                targets.append({
                    "qid": qid,
                    "type": q_type,
                    "question": r["question"],
                    "ground_truth": gt,
                    "majority_answer": maj_ans,
                    "majority_is_right": maj_is_right,
                    "right_answer": right_ans,
                    "wrong_answer": wrong_ans,
                    "right_votes": norm_counts.get(normalize_answer(right_ans), 0),
                    "wrong_votes": norm_counts.get(normalize_answer(wrong_ans), 0),
                    "total_samples": len(answers),
                    "conditions": conditions,
                })

    if args.limit > 0:
        targets = targets[:args.limit]

    return targets, run_path


def main():
    args = parse_args()
    targets, run_path = load_target_questions(args)
    print(f"Found {len(targets)} eligible questions (filter: '{args.class_filter}') with both a right and wrong candidate.")

    if not targets:
        print("No eligible questions found. Exiting.")
        return

    # Prepare all backward verification prompts
    # unit: (qid, is_right_cand, cand_ans, target_value, prompt_dict)
    units = []
    for item in targets:
        for is_right, cand in [(True, item["right_answer"]), (False, item["wrong_answer"])]:
            for target_val, masked_q in item["conditions"]:
                msgs = backward_messages(masked_q, cand)
                units.append({
                    "qid": item["qid"],
                    "is_right": is_right,
                    "candidate_answer": cand,
                    "target_value": target_val,
                    "masked_question": masked_q,
                    "messages": msgs,
                })

    print(f"Total backward checks to perform: {len(units)} across {len(targets)} questions.")

    if args.dry_run:
        print("\n--- Dry Run Sample Question ---")
        sample = targets[0]
        print(f"QID: {sample['qid']} (Class: {sample['type']})")
        print(f"Question: {sample['question']}")
        print(f"Ground Truth: {sample['ground_truth']}")
        print(f"Majority Answer: {sample['majority_answer']} (Right? {sample['majority_is_right']})")
        print(f"Right Candidate (Votes {sample['right_votes']}): {sample['right_answer']}")
        print(f"Wrong Candidate (Votes {sample['wrong_votes']}): {sample['wrong_answer']}")
        print(f"Number of masked conditions: {len(sample['conditions'])}")
        print("\nSample Backward Prompt (for Candidate = {}):".format(sample['right_answer']))
        print(sample['conditions'][0][1])
        print("\n--- End Dry Run ---")
        return

    # CUDA execution
    import torch
    from src.experiment_d.model_loading import load_model_and_tokenizer
    from src.experiment_d.prompts import build_prompt
    from src.experiment_d.sampling import sample_responses

    print(f"\nLoading model {args.model} ({args.precision})...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)

    prompts = [build_prompt(tokenizer, u["messages"]) for u in units]

    print(f"Running generation on {len(prompts)} prompts (batch_size={args.batch_size}, tokens={args.max_new_tokens})...")
    start_t = time.time()
    responses, tps, latency, lengths = sample_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        n_samples=1,
        temperature=args.temperature if args.temperature > 0 else 0.01,
        top_k=1 if args.temperature == 0 else 50,
        top_p=1.0 if args.temperature == 0 else 0.95,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - start_t
    print(f"Completed in {elapsed:.1f}s ({tps:.1f} tokens/s).")

    # Score each check
    results_by_qid = {}
    for item in targets:
        results_by_qid[item["qid"]] = {
            "meta": item,
            "right_checks": [],
            "wrong_checks": [],
        }

    for u, resp_group in zip(units, responses):
        resp = resp_group[0]
        match = parse_backward(resp, u["target_value"])
        check_res = {
            "target_value": u["target_value"],
            "response": resp,
            "match": match is True,
        }
        qid = u["qid"]
        if u["is_right"]:
            results_by_qid[qid]["right_checks"].append(check_res)
        else:
            results_by_qid[qid]["wrong_checks"].append(check_res)

    # Compute per-question scores
    wins, ties, losses = 0, 0, 0
    right_recovers, wrong_recovers = 0, 0
    total_right_conds, total_wrong_conds = 0, 0

    per_q_summary = []
    for qid, res in results_by_qid.items():
        meta = res["meta"]
        r_matches = sum(c["match"] for c in res["right_checks"])
        r_total = len(res["right_checks"])
        w_matches = sum(c["match"] for c in res["wrong_checks"])
        w_total = len(res["wrong_checks"])

        r_score = r_matches / r_total if r_total > 0 else 0.0
        w_score = w_matches / w_total if w_total > 0 else 0.0

        right_recovers += r_matches
        total_right_conds += r_total
        wrong_recovers += w_matches
        total_wrong_conds += w_total

        if r_score > w_score:
            outcome = "WIN (Right > Wrong)"
            wins += 1
        elif r_score == w_score:
            outcome = "TIE"
            ties += 1
        else:
            outcome = "LOSS (Wrong > Right)"
            losses += 1

        per_q_summary.append({
            "qid": qid,
            "type": meta["type"],
            "right_score": f"{r_matches}/{r_total} ({r_score:.2f})",
            "wrong_score": f"{w_matches}/{w_total} ({w_score:.2f})",
            "outcome": outcome,
        })

    total_pairs = wins + ties + losses
    pairwise_auroc = (wins + 0.5 * ties) / total_pairs if total_pairs > 0 else 0.5
    right_cond_acc = (right_recovers / total_right_conds) * 100 if total_right_conds > 0 else 0.0
    wrong_cond_acc = (wrong_recovers / total_wrong_conds) * 100 if total_wrong_conds > 0 else 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_model = args.model.replace("/", "_")
    report_file = output_dir / f"backward_diagnostic_{clean_model}_{args.precision}_{args.class_filter}.txt"
    json_file = output_dir / f"backward_diagnostic_{clean_model}_{args.precision}_{args.class_filter}.json"

    # Markdown output
    report_lines = [
        f"# Backward-Check (FOBAR) Diagnostic Report",
        f"- Model: `{args.model}` ({args.precision})",
        f"- Class Filter: `{args.class_filter}`",
        f"- Total Evaluated Questions: {total_pairs}",
        f"- Total Backward Conditions Tested: {total_right_conds + total_wrong_conds}",
        "",
        "## Overall Verdict",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| **Condition Recovery Rate for Right Answer ($A_{{\\text{{right}}}}$)** | **{right_cond_acc:.1f}%** ({right_recovers}/{total_right_conds}) |",
        f"| **Condition Recovery Rate for Wrong Answer ($A_{{\\text{{wrong}}}}$)** | **{wrong_cond_acc:.1f}%** ({wrong_recovers}/{total_wrong_conds}) |",
        f"| **Pairwise Win Rate (Right > Wrong)** | **{wins}/{total_pairs} ({(wins/total_pairs)*100:.1f}%)** |",
        f"| **Pairwise Tie Rate** | **{ties}/{total_pairs} ({(ties/total_pairs)*100:.1f}%)** |",
        f"| **Pairwise Loss Rate (Wrong > Right)** | **{losses}/{total_pairs} ({(losses/total_pairs)*100:.1f}%)** |",
        f"| **Within-Question AUROC** | **{pairwise_auroc:.3f}** |",
        "",
        "## Decision Gate Verdict",
    ]

    if pairwise_auroc > 0.77:
        report_lines.append(f"> [!IMPORTANT]\n> **PASSED (AUROC = {pairwise_auroc:.3f} > 0.77)**: The backward check successfully breaks the majority mode bias on {args.class_filter} questions and reliably scores minority correct answers above majority wrong answers.")
    else:
        report_lines.append(f"> [!WARNING]\n> **FAILED (AUROC = {pairwise_auroc:.3f} <= 0.77)**: The backward check fails to reliably separate right from wrong answers on {args.class_filter} questions (suffers from symmetrical misreadings). Do NOT use as a training filter.")

    report_lines.append("\n## Per-Question Breakdown (First 20)")
    report_lines.append("| QID | Class | Right Score | Wrong Score | Outcome |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for s in per_q_summary[:20]:
        report_lines.append(f"| {s['qid']} | {s['type']} | {s['right_score']} | {s['wrong_score']} | {s['outcome']} |")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_file.write_text(report_text, encoding="utf-8")
    with open(json_file, mode="w", encoding="utf-8") as jf:
        json.dump({
            "model": args.model,
            "precision": args.precision,
            "class_filter": args.class_filter,
            "pairwise_auroc": pairwise_auroc,
            "right_cond_acc": right_cond_acc,
            "wrong_cond_acc": wrong_cond_acc,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "details": results_by_qid,
        }, jf, indent=2)

    print(f"\nSaved report to {report_file}")
    print(f"Saved JSON data to {json_file}")


if __name__ == "__main__":
    main()
