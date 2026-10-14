import json
import re
import math
from collections import Counter

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_LOG_PATH = "results/gsm8k/generator/Qwen_Qwen3.5-0.8B/Qwen_Qwen3.5-0.8B_4bit_gsm8k_n8.jsonl"
OUTPUT_DPO_PAIRS = "dpo_pairs.jsonl"
OUTPUT_ANCHORS = "anchors.jsonl"
MIN_AGREEMENT_RATIO = 0.5  # At least 4 of 8 must agree

def get_distinct_n_ratio(texts, n=3):
    """Calculates distinct-n ratio over all generated tokens."""
    all_ngrams = []
    for t in texts:
        tokens = re.findall(r"\w+", t.lower())
        if len(tokens) >= n:
            all_ngrams.extend([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def compute_point_biserial(chosen_lens, rejected_lens):
    """Computes correlation between token length and chosen (1) vs rejected (0)."""
    n1, n0 = len(chosen_lens), len(rejected_lens)
    if n1 == 0 or n0 == 0:
        return 0.0
    n = n1 + n0
    mean1 = sum(chosen_lens) / n1
    mean0 = sum(rejected_lens) / n0
    all_lens = chosen_lens + rejected_lens
    mean_all = sum(all_lens) / n
    var = sum((x - mean_all) ** 2 for x in all_lens) / n
    if var == 0:
        return 0.0
    s = math.sqrt(var)
    return ((mean1 - mean0) / s) * math.sqrt((n1 * n0) / (n ** 2))

def process_logs():
    print(f"Loading logs from: {INPUT_LOG_PATH}...")
    records = []

    with open(INPUT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Skip the run_config metadata line
            if data.get("type") == "run_config":
                continue
            records.append(data)

    total_prompts = len(records)
    print(f"Loaded {total_prompts} test questions.")

    valid_dpo_pairs = []
    anchors = []
    discarded_low_agreement = 0
    discarded_no_complete_minority = 0

    for item in records:
        prompt = item.get("question", "")
        samples = item.get("raw_samples", [])

        if not prompt or not samples:
            continue

        # Parse each sample using YOUR EXACT log schema
        parsed_samples = []
        for s in samples:
            text = s.get("response", "")
            # Use pre-extracted answer from log
            ans = s.get("extracted")
            if ans is not None:
                ans = str(ans).strip()
            cutoff = s.get("cutoff", False)
            length = s.get("length", len(text.split()))

            # Complete trace = not cutoff AND has extracted answer
            is_complete = (not cutoff) and (ans is not None) and (ans != "") and (ans != "None")

            parsed_samples.append({
                "text": text,
                "answer": ans,
                "complete": is_complete,
                "length": length
            })

        # Count votes among PARSED answers
        valid_answers = [p["answer"] for p in parsed_samples if p["answer"] is not None]
        if not valid_answers:
            continue

        counts = Counter(valid_answers)
        maj_answer, maj_count = counts.most_common(1)[0]
        n_total = len(parsed_samples)
        agreement = maj_count / n_total

        # Filter: Agreement must be >= 0.5 (at least 4 of 8 agree)
        if agreement < MIN_AGREEMENT_RATIO:
            discarded_low_agreement += 1
            continue

        # Case A: Unanimous agreement (8/8) -> Anchor Buffer
        if maj_count == n_total:
            complete_maj = [p for p in parsed_samples if p["complete"]]
            if complete_maj:
                complete_maj.sort(key=lambda x: x["length"])
                med_trace = complete_maj[len(complete_maj) // 2]
                anchors.append({
                    "prompt": prompt,
                    "target": med_trace["text"],
                    "length": med_trace["length"],
                    "answer": maj_answer
                })
            continue

        # Case B: DPO Pair Construction
        # 1. Chosen: LONGEST Complete Majority
        complete_maj = [p for p in parsed_samples if p["answer"] == maj_answer and p["complete"]]
        if not complete_maj:
            continue
        complete_maj.sort(key=lambda x: x["length"], reverse=True)
        chosen = complete_maj[0]

        # 2. Rejected: SHORTEST Complete Minority
        complete_min = [p for p in parsed_samples if p["answer"] != maj_answer and p["complete"]]
        if not complete_min:
            discarded_no_complete_minority += 1
            continue
        complete_min.sort(key=lambda x: x["length"])
        rejected = complete_min[0]

        valid_dpo_pairs.append({
            "prompt": prompt,
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "chosen_len": chosen["length"],
            "rejected_len": rejected["length"],
            "chosen_ans": maj_answer,
            "rejected_ans": rejected["answer"],
            "agreement": agreement
        })

    # ==============================================================================
    # PRINT STAGE 1 SANITY METRICS
    # ==============================================================================
    print("\n" + "="*55)
    print("STAGE 1 SANITY AUDIT RESULTS (FROM EXISTING LOGS)")
    print("="*55)
    yield_pct = (len(valid_dpo_pairs) / total_prompts) * 100 if total_prompts else 0
    print(f"Total Questions Evaluated:    {total_prompts}")
    print(f"Valid DPO Pairs Created:      {len(valid_dpo_pairs)} ({yield_pct:.1f}% yield)")
    print(f"Unanimous (8/8) Anchors:      {len(anchors)} ({(len(anchors)/total_prompts)*100:.1f}%)")
    print(f"Discarded (Agreement < 0.5):  {discarded_low_agreement}")
    print(f"Discarded (No Complete Min.): {discarded_no_complete_minority}")

    if valid_dpo_pairs:
        chosen_lens = [p["chosen_len"] for p in valid_dpo_pairs]
        rejected_lens = [p["rejected_len"] for p in valid_dpo_pairs]
        mean_chosen = sum(chosen_lens) / len(chosen_lens)
        mean_rejected = sum(rejected_lens) / len(rejected_lens)

        print("-" * 55)
        print(f"Mean Chosen Token Length:     {mean_chosen:.1f} tokens")
        print(f"Mean Rejected Token Length:   {mean_rejected:.1f} tokens")

        r_pb = compute_point_biserial(chosen_lens, rejected_lens)
        print(f"Length Correlation (r_pb):    {r_pb:+.3f}")
        if r_pb > 0.4:
            print("  [!] WARNING: Chosen traces are significantly longer.")
            print("      Length-normalized DPO loss is strictly required.")
        else:
            print("  [OK] Length distribution between chosen and rejected is balanced.")

        dist_3 = get_distinct_n_ratio([p["chosen"] for p in valid_dpo_pairs], n=3)
        print(f"Distinct-3 Ratio (Chosen):    {dist_3:.3f}")

        # Save files
        with open(OUTPUT_DPO_PAIRS, "w", encoding="utf-8") as f:
            for pair in valid_dpo_pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"\nSaved {len(valid_dpo_pairs)} DPO pairs to: {OUTPUT_DPO_PAIRS}")

        with open(OUTPUT_ANCHORS, "w", encoding="utf-8") as f:
            for anc in anchors:
                f.write(json.dumps(anc) + "\n")
        print(f"Saved {len(anchors)} anchor traces to: {OUTPUT_ANCHORS}")

if __name__ == "__main__":
    process_logs()
