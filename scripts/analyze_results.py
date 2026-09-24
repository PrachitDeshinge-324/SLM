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






def generate_combined_calibration_plot(dataset_runs: dict, model: str, dataset: str, output_path: str):
    """
    Overlays all precisions (16-bit, 8-bit, 4-bit) AND n_samples (e.g., 8, 16)
    for a specific (model, dataset) onto a single combined calibration chart.
    """
    short_model = model.split("/")[-1] if "/" in model else model

    # Colors by precision
    palette = {
        '16bit': '#2563eb',  # Blue
        '8bit': '#10b981',   # Emerald Green
        '4bit': '#f59e0b',   # Amber
    }
    
    # Markers by n_samples
    markers = {
        8: 'o',
        16: 's',
        32: '^'
    }
    # Linestyles by n_samples to help differentiate
    linestyles = {
        8: '-',
        16: '--',
        32: ':'
    }

    plt.figure(figsize=(7, 6))
    sns.set_style("whitegrid")
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration", linewidth=1.5, alpha=0.7)

    if dataset.lower() == 'cqa':
        plt.axhline(y=0.20, color='r', linestyle=':', label="Random Guess (20%)", alpha=0.7)

    # Sort keys by precision, then n_samples
    precision_order = {'16bit': 0, '8bit': 1, '4bit': 2}
    sorted_keys = sorted(dataset_runs.keys(), key=lambda k: (precision_order.get(k[0], 99), k[1]))

    for prec, n_samp in sorted_keys:
        run_info = dataset_runs[(prec, n_samp)]
        calib = compute_calibration_curve(run_info['df'], min_bin_count=5)
        if calib is None or calib.empty:
            continue

        acc_pct = run_info['accuracy'] * 100
        passk_pct = run_info['pass@k'] * 100
        total_t_str = run_info.get('total_time', '')
        time_tag = f", Time: {total_t_str}" if total_t_str else ""
        
        color = palette.get(prec, '#888888')
        marker = markers.get(n_samp, 'd')
        ls = linestyles.get(n_samp, '-')

        plt.plot(
            calib['conf_bin'],
            calib['accuracy'],
            marker=marker,
            linestyle=ls,
            linewidth=2.0,
            markersize=6,
            color=color,
            label=f"{prec} (N={n_samp}): Acc={acc_pct:.1f}%, P@{n_samp}={passk_pct:.1f}%"
        )

    plt.title(f"Calibration: {short_model} on {dataset.upper()}", fontsize=13, fontweight='bold')
    plt.xlabel("Confidence (Agreement Fraction)", fontsize=11)
    plt.ylabel("Actual Accuracy", fontsize=11)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    # Put legend outside the plot if it's too big, or lower right with small font
    plt.legend(loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"  Saved COMBINED plot: {output_path}")

def generate_accuracy_vs_precision_plot(summary_df: pd.DataFrame, results_dir: str):
    """
    Generates a grouped bar chart comparing Accuracy and Pass@k
    across precisions, with n_samples (e.g., 8, 16) grouped together.
    """
    models = summary_df['model'].unique()
    precision_order = ['16bit', '8bit', '4bit']

    for model in models:
        m_df = summary_df[summary_df['model'] == model].copy()
        short_model = model.split("/")[-1] if "/" in model else model
        datasets = m_df['dataset'].unique()

        for dataset in datasets:
            d_df = m_df[m_df['dataset'] == dataset].copy()
            
            d_df['prec_cat'] = pd.Categorical(d_df['precision'], categories=precision_order, ordered=True)
            
            plt.figure(figsize=(9, 6))
            sns.set_style("whitegrid")
            
            # Create a grouped bar chart manually without numpy
            precisions = sorted(d_df['precision'].unique(), key=lambda p: precision_order.index(p) if p in precision_order else 99)
            x_indices = list(range(len(precisions)))
            
            n_samples_list = sorted(d_df['n_samples'].unique())
            num_n = len(n_samples_list)
            
            # Each N has an Acc bar and a Pass@k bar. So 2 bars per N.
            total_bars_per_group = 2 * num_n
            width = 0.8 / total_bars_per_group
            
            acc_colors = {8: '#2563eb', 16: '#60a5fa', 32: '#93c5fd'} # Blues
            passk_colors = {8: '#10b981', 16: '#34d399', 32: '#6ee7b7'} # Greens
            
            for i, n_samp in enumerate(n_samples_list):
                sub_df = d_df[d_df['n_samples'] == n_samp]
                
                acc_vals = []
                passk_vals = []
                for prec in precisions:
                    row = sub_df[sub_df['precision'] == prec]
                    if not row.empty:
                        acc_vals.append(row['accuracy'].values[0] * 100)
                        passk_vals.append(row['pass@k'].values[0] * 100)
                    else:
                        acc_vals.append(0)
                        passk_vals.append(0)
                
                # Offsets
                offset_acc = i * 2 * width - (total_bars_per_group * width) / 2 + width / 2
                offset_passk = offset_acc + width
                
                x_acc = [xi + offset_acc for xi in x_indices]
                x_passk = [xi + offset_passk for xi in x_indices]
                
                rects1 = plt.bar(x_acc, acc_vals, width, label=f'Acc (N={n_samp})', color=acc_colors.get(n_samp, '#2563eb'), edgecolor='white')
                rects2 = plt.bar(x_passk, passk_vals, width, label=f'Pass@{n_samp}', color=passk_colors.get(n_samp, '#10b981'), edgecolor='white')
                
                for rect in rects1:
                    h = rect.get_height()
                    if h > 0:
                        plt.annotate(f'{h:.0f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9, rotation=90)
                for rect in rects2:
                    h = rect.get_height()
                    if h > 0:
                        plt.annotate(f'{h:.0f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9, rotation=90)
            
            if dataset.lower() == 'cqa':
                plt.axhline(y=20.0, color='r', linestyle=':', label="Random Guess (20%)", alpha=0.7)

            plt.title(f"Accuracy vs. Precision: {short_model} on {dataset.upper()}", fontsize=13, fontweight='bold')
            plt.xlabel("Precision Format", fontsize=11)
            plt.ylabel("Accuracy (%)", fontsize=11)
            plt.xticks(x_indices, precisions, fontsize=10, fontweight='600')
            plt.ylim(0, 115)  # give room for vertical annotations
            
            # Put legend at bottom
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=num_n*2, frameon=True, fontsize=9)
            plt.tight_layout()

            out_name = f"{short_model}_{dataset}_accuracy_vs_precision.png"
            out_path = os.path.join(results_dir, out_name)
            plt.savefig(out_path, dpi=180, bbox_inches='tight')
            plt.close()
            print(f"  Saved COMBINED BAR CHART: {out_path}")


def compute_quantization_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes accuracy and pass@k deltas relative to the 16bit baseline
    for each (dataset, model, n_samples) group.
    """
    df = df.copy()
    acc_deltas = []
    passk_deltas = []

    # Map baseline accuracies: (dataset, model, n_samples) -> (16bit_acc, 16bit_passk)
    baselines = {}
    for _, row in df.iterrows():
        if str(row["precision"]).lower() == "16bit":
            key = (row["dataset"], row["model"], row["n_samples"])
            baselines[key] = (row["accuracy"], row["pass@k"])

    for _, row in df.iterrows():
        key = (row["dataset"], row["model"], row["n_samples"])
        if key in baselines:
            base_acc, base_passk = baselines[key]
            acc_deltas.append(round(row["accuracy"] - base_acc, 4))
            passk_deltas.append(round(row["pass@k"] - base_passk, 4))
        else:
            acc_deltas.append(None)
            passk_deltas.append(None)

    df["acc_delta_vs_16bit"] = acc_deltas
    df["pass@k_delta_vs_16bit"] = passk_deltas
    return df
    

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
            "dataset": dataset,
            "model": model,
            "precision": precision,
            "n_samples": n_samples,
            "accuracy": res['accuracy'],
            "pass@k": res['oracle_accuracy'],
            "confidence": res['avg_confidence'],
            "failure_rate": res['avg_failure_rate'],
            "diversity": res['avg_diversity'],
            "cutoff_rate": res['avg_cutoff_rate'],
            "tps": res['avg_tps'],
            "avg_latency_sec": round(res['avg_latency'], 2),
            "total_time": res['total_time_formatted'],
            "total_time_sec": round(res['total_time_sec'], 2),
            "n_questions": res['count'],
        }
        summary_stats.append(run_stat)

        short_model = model.split("/")[-1] if "/" in model else model
        grouped_runs[(short_model, dataset)][(precision, n_samples)] = {
            "df": res['raw_df'],
            "accuracy": res['accuracy'],
            "pass@k": res['oracle_accuracy'],
            "total_time": res['total_time_formatted'],
            "full_model": model,
        }

    if not summary_stats:
        print("No matching records found for the specified filters.")
        return

    print("\nGenerating combined calibration graphs per model & dataset...")
    for (m_name, d_name), prec_dict in grouped_runs.items():
        if len(prec_dict) >= 1:
            combined_name = f"{m_name}_{d_name}_combined_calibration.png"
            combined_path = os.path.join(args.results_dir, combined_name)
            generate_combined_calibration_plot(prec_dict, m_name, d_name, combined_path)

    summary_df = pd.DataFrame(summary_stats)

    # ── Sorting: Dataset -> Model -> Precision (Logical 16bit -> 8bit -> 4bit) -> N ──
    precision_order = {"16bit": 2, "8bit": 1, "4bit": 0}
    summary_df["prec_rank"] = summary_df["precision"].map(lambda p: precision_order.get(str(p).lower(), 99))
    summary_df = summary_df.sort_values(
        by=["dataset","model", "n_samples", "prec_rank"],
        ascending=[True, True, True, True]
    ).drop(columns=["prec_rank"])

    # ── Compute Delta vs 16-bit Baseline ──────────────────────────────
    summary_df = compute_quantization_deltas(summary_df)

    # ── Clean Rounding for Readability ────────────────────────────────
    float_cols_4 = ["accuracy", "pass@k", "confidence", "failure_rate", "diversity", "cutoff_rate"]
    for c in float_cols_4:
        if c in summary_df.columns:
            summary_df[c] = summary_df[c].round(4)
    if "tps" in summary_df.columns:
        summary_df["tps"] = summary_df["tps"].round(2)

    # ── Identity columns shared across both CSVs ──────────────────────
    id_cols = ["dataset", "model", "precision", "n_samples"]

    # ══════════════════════════════════════════════════════════════════
    # PRIMARY CSV — Core experiment results
    # ══════════════════════════════════════════════════════════════════
    primary_cols = id_cols + [
        c for c in [
            "accuracy", "pass@k",
            "acc_delta_vs_16bit", "pass@k_delta_vs_16bit",
            "confidence", "n_questions",
        ] if c in summary_df.columns
    ]
    primary_df = summary_df[primary_cols].copy()

    # Convert fractions → readable percentages
    for c in ["accuracy", "pass@k", "acc_delta_vs_16bit", "pass@k_delta_vs_16bit", "confidence"]:
        if c in primary_df.columns:
            primary_df[c] = (primary_df[c] * 100).round(2)
    primary_df = primary_df.rename(columns={
        "accuracy": "accuracy_%",
        "pass@k": "pass@k_%",
        "acc_delta_vs_16bit": "acc_delta_vs_16bit_%",
        "pass@k_delta_vs_16bit": "pass@k_delta_vs_16bit_%",
        "confidence": "confidence_%",
    })

    primary_path = os.path.join(args.results_dir, "primary_results.csv")
    primary_df.to_csv(primary_path, index=False)
    print(f"\nSaved PRIMARY: {primary_path}")

    # ══════════════════════════════════════════════════════════════════
    # SECONDARY CSV — Reliability & Performance (grouped columns)
    # ══════════════════════════════════════════════════════════════════
    secondary_cols = id_cols + [
        c for c in [
            # ── Reliability ──
            "failure_rate", "diversity", "cutoff_rate",
            # ── Performance ──
            "tps", "avg_latency_sec", "total_time", "total_time_sec",
        ] if c in summary_df.columns
    ]
    secondary_df = summary_df[secondary_cols].copy()
    secondary_df = secondary_df.rename(columns={
        "failure_rate": "reliability.failure_rate",
        "diversity": "reliability.diversity",
        "cutoff_rate": "reliability.cutoff_rate",
        "tps": "performance.tokens_per_sec",
        "avg_latency_sec": "performance.avg_latency_sec",
        "total_time": "performance.total_time",
        "total_time_sec": "performance.total_time_sec",
    })

    secondary_path = os.path.join(args.results_dir, "secondary_metrics.csv")
    secondary_df.to_csv(secondary_path, index=False)
    print(f"Saved SECONDARY: {secondary_path}")

    # ── Console summary ───────────────────────────────────────────────
    print("\n=== Primary Results ===")
    print(primary_df.to_string(index=False))
    print("\n=== Secondary Metrics (Reliability + Performance) ===")
    print(secondary_df.to_string(index=False))

    # ── Accuracy vs. Precision Comparison Plot ─────────────────────────
    print("\nGenerating Accuracy vs. Precision bar charts...")
    generate_accuracy_vs_precision_plot(summary_df, args.results_dir)


if __name__ == "__main__":
    main()
