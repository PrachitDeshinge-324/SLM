import argparse
import json
import glob
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
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
            line_str = line.strip()
            if not line_str:
                continue
            try:
                obj = json.loads(line_str)
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
    total_time_sec = df['performance.latency_sec'].sum() if 'performance.latency_sec' in df else 0.0

    # Format total seconds into human-readable string (e.g. 5h 30m 7s)
    total_sec_int = int(round(total_time_sec))
    h = total_sec_int // 3600
    m = (total_sec_int % 3600) // 60
    s = total_sec_int % 60
    if h > 0:
        total_time_str = f"{h}h {m}m {s}s"
    elif m > 0:
        total_time_str = f"{m}m {s}s"
    else:
        total_time_str = f"{s}s"

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
        "total_time_sec": total_time_sec,
        "total_time_formatted": total_time_str,
        "raw_df": df,
    }


def _extract_metadata(result: dict, filepath: str):
    """
    Extract model/precision/dataset/n_samples from run_config if available,
    otherwise fall back to filename parsing.
    """
    cfg = result.get("run_config")
    n_samples = 8
    if cfg:
        model = cfg.get("model", "unknown")
        precision = cfg.get("precision", "unknown")
        dataset = cfg.get("dataset", "unknown")
        n_samples = int(cfg.get("n_samples", 8))
        return model, precision, dataset, n_samples

    basename = os.path.basename(filepath).replace('.jsonl', '')
    parts = basename.split('_')
    if parts[-1].startswith('n') and parts[-1][1:].isdigit():
        n_samples = int(parts[-1][1:])
        dataset = parts[-2]
        precision = parts[-3]
        model = "_".join(parts[:-3])
    else:
        dataset = parts[-1]
        precision = parts[-2]
        model = "_".join(parts[:-2])
    return model, precision, dataset, n_samples


def compute_calibration_curve(df: pd.DataFrame, num_bins: int = 10, min_bin_count: int = 5):
    """
    Computes binned calibration (Confidence vs. Actual Accuracy).
    Filters out statistically noisy bins with fewer than `min_bin_count` questions.
    Returns DataFrame with columns ['conf_bin', 'accuracy', 'count'].
    """
    if 'metrics.confidence' not in df or 'metrics.correctness' not in df:
        return None

    try:
        bins = [i / float(num_bins) for i in range(num_bins + 1)]
        df_copy = df.copy()
        df_copy['conf_bin'] = pd.cut(
            df_copy['metrics.confidence'],
            bins=bins,
            labels=[bins[i + 1] for i in range(num_bins)],
            include_lowest=True,
            right=True
        )
        df_copy['conf_bin'] = df_copy['conf_bin'].astype(float)

        grouped = df_copy.groupby('conf_bin', observed=False).agg(
            accuracy=('metrics.correctness', 'mean'),
            count=('metrics.correctness', 'count')
        ).reset_index()

        # Filter out sparse, statistically unrepresentative outlier bins (e.g., bins with 1 question)
        grouped = grouped[grouped['count'] >= min_bin_count]
        return grouped
    except Exception as e:
        print(f"Warning: Calibration curve computation failed: {e}")
        return None


def generate_individual_calibration_plot(df: pd.DataFrame, model: str, precision: str, dataset: str, n_samples: int, output_path: str):
    """Generates an individual calibration plot for a single model/precision/dataset/n_samples."""
    calib = compute_calibration_curve(df)
    if calib is None or calib.empty:
        return

    short_model = model.split("/")[-1] if "/" in model else model

    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration", linewidth=1.5, alpha=0.7)
    plt.plot(calib['conf_bin'], calib['accuracy'], marker='o', linewidth=2.5, label=f"{precision} (N={n_samples})")

    if dataset.lower() == 'cqa':
        plt.axhline(y=0.20, color='r', linestyle=':', label="Random Guess (20%)", alpha=0.7)

    plt.title(f"Calibration: {short_model} ({precision}, N={n_samples}) on {dataset.upper()}", fontsize=13, fontweight='bold')
    plt.xlabel("Confidence (Agreement Fraction)", fontsize=11)
    plt.ylabel("Actual Accuracy", fontsize=11)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"  Saved individual plot: {output_path}")


def generate_combined_calibration_plot(dataset_runs: dict, model: str, dataset: str, n_samples: int, output_path: str):
    """
    Overlays all precisions (16-bit, 8-bit, 4-bit) for a specific (model, dataset, n_samples)
    onto a single combined calibration chart in the exact same style as individual plots.
    """
    short_model = model.split("/")[-1] if "/" in model else model

    palette = {
        '16bit': '#2563eb',  # Blue
        '8bit': '#10b981',   # Emerald Green
        '4bit': '#f59e0b',   # Amber
    }

    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration", linewidth=1.5, alpha=0.7)

    if dataset.lower() == 'cqa':
        plt.axhline(y=0.20, color='r', linestyle=':', label="Random Guess (20%)", alpha=0.7)

    precision_order = {'16bit': 0, '8bit': 1, '4bit': 2}
    sorted_precisions = sorted(dataset_runs.keys(), key=lambda p: precision_order.get(p, 99))

    for prec in sorted_precisions:
        run_info = dataset_runs[prec]
        calib = compute_calibration_curve(run_info['df'], min_bin_count=5)
        if calib is None or calib.empty:
            continue

        acc_pct = run_info['accuracy'] * 100
        passk_pct = run_info['pass@k'] * 100
        total_t_str = run_info.get('total_time', '')
        time_tag = f", Time: {total_t_str}" if total_t_str else ""
        color = palette.get(prec, None)

        plt.plot(
            calib['conf_bin'],
            calib['accuracy'],
            marker='o',
            linewidth=2.5,
            color=color,
            label=f"{prec} (Acc: {acc_pct:.1f}%, Pass@{n_samples}: {passk_pct:.1f}%{time_tag})"
        )

    plt.title(f"Calibration: {short_model} (N={n_samples}) on {dataset.upper()}", fontsize=13, fontweight='bold')
    plt.xlabel("Confidence (Agreement Fraction)", fontsize=11)
    plt.ylabel("Actual Accuracy", fontsize=11)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    # Always place in lower right so it never overlaps the curve or upper left region
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"  Saved COMBINED plot: {output_path}")


def generate_accuracy_vs_precision_plot(summary_df: pd.DataFrame, results_dir: str):
    """
    Generates a clean grouped bar chart comparing Accuracy and Pass@k
    across precisions for each (model, dataset, n_samples) group in the exact same visual design.
    """
    models = summary_df['model'].unique()
    precision_order = ['16bit', '8bit', '4bit']

    for model in models:
        m_df = summary_df[summary_df['model'] == model].copy()
        short_model = model.split("/")[-1] if "/" in model else model
        datasets = m_df['dataset'].unique()

        for dataset in datasets:
            d_df = m_df[m_df['dataset'] == dataset].copy()
            n_samples_list = sorted(d_df['n_samples'].unique())

            for n_samples in n_samples_list:
                sub_df = d_df[d_df['n_samples'] == n_samples].copy()

                # Sort by canonical precision order
                sub_df['prec_cat'] = pd.Categorical(sub_df['precision'], categories=precision_order, ordered=True)
                sub_df = sub_df.sort_values('prec_cat')

                precisions = sub_df['precision'].tolist()
                acc_vals = [a * 100 for a in sub_df['accuracy'].tolist()]
                passk_vals = [p * 100 for p in sub_df['pass@k'].tolist()]

                plt.figure(figsize=(6, 5))
                sns.set_style("whitegrid")

                x = list(range(len(precisions)))
                width = 0.32

                # Use consistent palette matching calibration curves
                rects1 = plt.bar([i - width / 2 for i in x], acc_vals, width, label='Majority Accuracy', color='#2563eb', alpha=0.9, edgecolor='none')
                rects2 = plt.bar([i + width / 2 for i in x], passk_vals, width, label=f'Pass@{n_samples} (Oracle Ceiling)', color='#10b981', alpha=0.9, edgecolor='none')

                # Annotate values above bars with ample spacing to avoid overlaps
                for rect in rects1:
                    h = rect.get_height()
                    plt.annotate(f'{h:.1f}%',
                                 xy=(rect.get_x() + rect.get_width() / 2, h),
                                 xytext=(0, 4), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e3a8a')

                for rect in rects2:
                    h = rect.get_height()
                    plt.annotate(f'{h:.1f}%',
                                 xy=(rect.get_x() + rect.get_width() / 2, h),
                                 xytext=(0, 4), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#064e3b')

                # Add random guess line for multiple choice CQA
                if dataset.lower() == 'cqa':
                    plt.axhline(y=20.0, color='r', linestyle=':', label="Random Guess (20%)", alpha=0.7)

                plt.title(f"Accuracy vs. Precision: {short_model} (N={n_samples}) on {dataset.upper()}", fontsize=13, fontweight='bold')
                plt.xlabel("Precision Format", fontsize=11)
                plt.ylabel("Accuracy (%)", fontsize=11)
                plt.xticks(x, precisions, fontsize=10, fontweight='600')
                plt.ylim(0, 105)
                plt.legend(loc="lower right", frameon=True, fontsize=9)
                plt.tight_layout()

                out_name = f"{short_model}_{dataset}_n{n_samples}_accuracy_vs_precision.png"
                out_path = os.path.join(results_dir, out_name)
                plt.savefig(out_path, dpi=180)
                plt.close()
                print(f"  Saved ACCURACY VS PRECISION plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze results from Experiment D")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with JSONL result files")
    parser.add_argument("--model", type=str, default=None, help="Filter analysis to a specific model ID")
    parser.add_argument("--dataset", type=str, default=None, choices=["gsm8k", "cqa"], help="Filter analysis to a specific dataset")
    parser.add_argument("--n_samples", type=int, default=None, help="Filter analysis to a specific sample count (e.g. 8 or 16)")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.results_dir, "*.jsonl"))
    if not files:
        print(f"No jsonl files found in {args.results_dir}")
        return

    summary_stats = []
    grouped_runs = defaultdict(lambda: defaultdict(dict))

    for fpath in files:
        res = analyze_file(fpath)
        if not res:
            continue

        model, precision, dataset, n_samples = _extract_metadata(res, fpath)

        if args.model and args.model.lower() not in model.lower():
            continue
        if args.dataset and args.dataset.lower() != dataset.lower():
            continue
        if args.n_samples and args.n_samples != n_samples:
            continue

        run_stat = {
            "model": model,
            "precision": precision,
            "dataset": dataset,
            "n_samples": n_samples,
            "accuracy": res['accuracy'],
            "pass@k": res['oracle_accuracy'],
            "confidence": res['avg_confidence'],
            "failure_rate": res['avg_failure_rate'],
            "diversity": res['avg_diversity'],
            "cutoff_rate": res['avg_cutoff_rate'],
            "tps": res['avg_tps'],
            "total_time": res['total_time_formatted'],
            "total_time_sec": round(res['total_time_sec'], 2),
            "avg_latency_sec": round(res['avg_latency'], 2),
            "n_questions": res['count'],
        }
        summary_stats.append(run_stat)

        short_model = model.split("/")[-1] if "/" in model else model
        grouped_runs[(short_model, dataset, n_samples)][precision] = {
            "df": res['raw_df'],
            "accuracy": res['accuracy'],
            "pass@k": res['oracle_accuracy'],
            "total_time": res['total_time_formatted'],
            "full_model": model,
        }

        plot_name = f"{short_model}_{precision}_{dataset}_n{n_samples}_calibration.png"
        plot_path = os.path.join(args.results_dir, plot_name)
        generate_individual_calibration_plot(res['raw_df'], model, precision, dataset, n_samples, plot_path)

    if not summary_stats:
        print("No matching records found for the specified filters.")
        return

    print("\nGenerating combined calibration graphs per model, dataset & sample size...")
    for (m_name, d_name, n_s), prec_dict in grouped_runs.items():
        if len(prec_dict) >= 1:
            combined_name = f"{m_name}_{d_name}_n{n_s}_combined_calibration.png"
            combined_path = os.path.join(args.results_dir, combined_name)
            generate_combined_calibration_plot(prec_dict, m_name, d_name, n_s, combined_path)

    summary_df = pd.DataFrame(summary_stats)
    print("\n=== Summary Stats ===")
    print(summary_df.to_string(index=False))

    summary_csv_path = os.path.join(args.results_dir, "summary_stats.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved plots and summary to {args.results_dir}")

    # ── Accuracy vs. Precision Comparison Plot ─────────────────────────
    print("\nGenerating Accuracy vs. Precision bar charts...")
    generate_accuracy_vs_precision_plot(summary_df, args.results_dir)


if __name__ == "__main__":
    main()
