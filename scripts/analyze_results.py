import argparse
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def analyze_file(filepath: str):
    """
    Reads a JSONL result file. Extracts the run_config header (if present)
    and all per-question records.
    """
    records = []
    run_config = None

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "run_config":
                    run_config = obj
                else:
                    records.append(obj)
            except Exception:
                pass

    if not records:
        return None

    df = pd.json_normalize(records)

    avg_confidence = df['metrics.confidence'].mean() if 'metrics.confidence' in df else 0.0
    accuracy = df['metrics.correctness'].mean() if 'metrics.correctness' in df else 0.0
    oracle_accuracy = df['metrics.oracle_correctness'].mean() if 'metrics.oracle_correctness' in df else 0.0
    avg_failure = df['metrics.failure_rate'].mean() if 'metrics.failure_rate' in df else 0.0
    avg_diversity = df['metrics.diversity'].mean() if 'metrics.diversity' in df else 0.0
    avg_cutoff = df['metrics.cutoff_rate'].mean() if 'metrics.cutoff_rate' in df else 0.0
    avg_tps = df['performance.tokens_per_sec'].mean() if 'performance.tokens_per_sec' in df else 0.0
    avg_latency = df['performance.latency_sec'].mean() if 'performance.latency_sec' in df else 0.0

    return {
        "run_config": run_config,
        "count": len(df),
        "accuracy": accuracy,
        "oracle_accuracy": oracle_accuracy,
        "avg_confidence": avg_confidence,
        "avg_failure_rate": avg_failure,
        "avg_diversity": avg_diversity,
        "avg_cutoff_rate": avg_cutoff,
        "avg_tps": avg_tps,
        "avg_latency": avg_latency,
        "raw_df": df,
    }


def _extract_metadata(result: dict, filepath: str):
    """
    Extract model/precision/dataset from run_config if available,
    otherwise fall back to filename parsing.
    """
    cfg = result.get("run_config")
    if cfg:
        return cfg.get("model", "unknown"), cfg.get("precision", "unknown"), cfg.get("dataset", "unknown")

    # Fallback: parse filename (fragile, but better than nothing)
    basename = os.path.basename(filepath).replace('.jsonl', '')
    parts = basename.split('_')
    dataset = parts[-1]
    precision = parts[-2]
    model = "_".join(parts[:-2])
    return model, precision, dataset


def main():
    parser = argparse.ArgumentParser(description="Analyze results from Experiment D")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with JSONL result files")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.results_dir, "*.jsonl"))
    if not files:
        print(f"No jsonl files found in {args.results_dir}")
        return

    summary_stats = []

    for fpath in files:
        res = analyze_file(fpath)
        if not res:
            continue

        model, precision, dataset = _extract_metadata(res, fpath)

        summary_stats.append({
            "model": model,
            "precision": precision,
            "dataset": dataset,
            "accuracy": res['accuracy'],
            "pass@k": res['oracle_accuracy'],
            "confidence": res['avg_confidence'],
            "failure_rate": res['avg_failure_rate'],
            "diversity": res['avg_diversity'],
            "cutoff_rate": res['avg_cutoff_rate'],
            "tps": res['avg_tps'],
            "n_questions": res['count'],
        })

        # ── Calibration plot ──────────────────────────────────────────────
        df = res['raw_df']
        if 'metrics.confidence' in df and 'metrics.correctness' in df:
            df['conf_bin'] = pd.cut(
                df['metrics.confidence'], bins=10, labels=False, right=True
            ) / 10.0
            calibration = (
                df.groupby('conf_bin')['metrics.correctness'].mean().reset_index()
            )

            short_model = model.split("/")[-1] if "/" in model else model

            plt.figure(figsize=(6, 5))
            sns.lineplot(data=calibration, x='conf_bin', y='metrics.correctness', marker='o')
            plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
            plt.title(f"Calibration: {short_model} ({precision}) on {dataset}")
            plt.xlabel("Confidence (Agreement Fraction)")
            plt.ylabel("Actual Accuracy")
            plt.legend()

            plot_name = f"{short_model}_{precision}_{dataset}_calibration.png"
            plot_path = os.path.join(args.results_dir, plot_name)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {plot_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_stats)
    print("\n=== Summary Stats ===")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(os.path.join(args.results_dir, "summary_stats.csv"), index=False)
    print(f"\nSaved plots and summary to {args.results_dir}")


if __name__ == "__main__":
    main()
