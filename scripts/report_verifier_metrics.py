"""
Report: verifier metrics per model, one row per precision, on the questions all precisions share.

Trace labels are recomputed from the stored solutions and truncated traces are dropped
(see src/experiment_d/rescoring.py). Writes
results/<dataset>/verifier_output/metrics/csvs/metrics_<model>_<dataset>_n<N>.csv and confusion-matrix plots.
"""
import argparse
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ensure project root is in sys.path so we can import from src and scripts
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiment_d.rescoring import load_verifier_traces
from src.experiment_d.run_files import is_legacy_file, parse_run_name


def parse_args():
    parser = argparse.ArgumentParser(description="Process verifier JSONL logs and generate metrics.")
    parser.add_argument("--results_dir", "--input_dir", dest="results_dir", type=str, default="results", help="Root directory of the run files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save CSV and plots (default: the results directory).")
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.results_dir
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all relevant jsonl files across subdirectories
    files = glob.glob(os.path.join(args.results_dir, "**", "verifier_output_*.jsonl"), recursive=True)
    files = [f for f in files if not is_legacy_file(f)]
    if not files:
        print(f"No verifier_output_*.jsonl files found in {args.results_dir}")
        return
        
    # Group files by (model_name, dataset_name)
    grouped_files = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        run = parse_run_name(filename)
        if run is None:
            print(f"Warning: Could not parse model/dataset from {filename}. Skipping.")
            continue
        model_name, precision, dataset_name = run["model"], run["precision"], run["dataset"]
        samples_n = f"n{run['n_samples']}"
            
        key = (model_name, dataset_name, samples_n)
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append((filepath, precision))
        
    for (model_name, dataset_name, samples_n), file_list in grouped_files.items():
        print(f"\nProcessing Group: Model='{model_name}', Dataset='{dataset_name}', Samples='{samples_n}'")
        results_list = []
        matrices = {}
        
        # Traces are relabelled from the stored solution text with the current extractor, and the
        # precisions of a model are compared on the questions they all contain.
        traces_by_precision = {
            precision: load_verifier_traces(filepath, dataset_name) for filepath, precision in file_list
        }
        common_questions = set.intersection(*[set(traces) for traces in traces_by_precision.values()])

        for filepath, precision in file_list:
            tp = fp = tn = fn = 0
            total = 0
            truncated_dropped = truncation_unknown = unparsed = 0

            # --- Question-level tracking ---
            from collections import Counter
            questions_data = {}

            for question in common_questions:
                questions_data[question] = []
                for trace in traces_by_precision[precision][question]:
                    # A truncated trace is a partial scratchpad: it is neither a vote nor a fair test of the verifier.
                    if trace["truncated"]:
                        truncated_dropped += 1
                        continue
                    if trace["truncated"] is None:
                        truncation_unknown += 1

                    conc = trace["approved"]
                    is_correct = trace["correct"]
                    if conc is None:
                        unparsed += 1
                    else:
                        total += 1
                        if conc and is_correct: tp += 1
                        elif conc and not is_correct: fp += 1
                        elif not conc and not is_correct: tn += 1
                        elif not conc and is_correct: fn += 1

                    # Every complete trace with an answer votes, whether or not the verdict parsed.
                    if trace["answer"] is not None:
                        questions_data[question].append({
                            "gen_ans": trace["answer"],
                            "is_correct": is_correct,
                            "verifier_says_correct": conc is True
                        })

            if total == 0:
                continue

            # --- Question-level Metrics ---
            gen_maj_correct_count = 0
            ver_maj_correct_count = 0
            total_questions = len(questions_data)

            for q, traces in questions_data.items():
                all_ans = [t["gen_ans"] for t in traces]

                # 1. Generator Majority (Base performance without verifier)
                if all_ans:
                    gen_maj = Counter(all_ans).most_common(1)[0][0]
                    if any(t["is_correct"] for t in traces if t["gen_ans"] == gen_maj):
                        gen_maj_correct_count += 1

                # 2. Verifier-Guided Majority
                ver_approved_ans = [t["gen_ans"] for t in traces if t["verifier_says_correct"]]

                if ver_approved_ans:
                    # Majority vote among traces the verifier approved
                    ver_maj = Counter(ver_approved_ans).most_common(1)[0][0]
                elif all_ans:
                    # Fallback to standard generator majority if verifier rejected EVERYTHING
                    ver_maj = Counter(all_ans).most_common(1)[0][0]
                else:
                    ver_maj = None

                if ver_maj is not None:
                    if any(t["is_correct"] for t in traces if t["gen_ans"] == ver_maj):
                        ver_maj_correct_count += 1

            gen_maj_acc = gen_maj_correct_count / total_questions if total_questions > 0 else 0
            ver_maj_acc = ver_maj_correct_count / total_questions if total_questions > 0 else 0

            # --- Trace-level Metrics ---
            accuracy = (tp + tn) / total
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision_metric = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tpr
            f1 = 2 * (precision_metric * recall) / (precision_metric + recall) if (precision_metric + recall) > 0 else 0
            
            results_list.append({
                "Precision_Level": precision,
                "Total_Traces": total,
                "Total_Questions": total_questions,
                "Gen_Maj_Acc": f"{gen_maj_acc*100:.2f}%",
                "Ver_Maj_Acc": f"{ver_maj_acc*100:.2f}%",
                "Delta (Ver-Gen)": f"{(ver_maj_acc - gen_maj_acc)*100:+.2f}%",
                "Trace_Accuracy": f"{accuracy*100:.2f}%",
                "Trace_Precision": f"{precision_metric*100:.2f}%",
                "Trace_Recall": f"{recall*100:.2f}%",
                "Trace_F1": f"{f1*100:.2f}%",
                "Trace_FPR": f"{fpr*100:.2f}%",
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Unparsed_Verdicts": unparsed,
                "Truncated_Traces_Dropped": truncated_dropped,
                "Truncation_Unknown": truncation_unknown
            })
            
            # Matrix format for Seaborn: [[TP, FP], [FN, TN]]
            matrices[precision] = [[tp, fp], [fn, tn]]
            
        if not results_list:
            continue
            
        # Write CSV
        df = pd.DataFrame(results_list)
        # Sort by precision (e.g. 16bit, 8bit, 4bit)
        def sort_key(val):
            match = re.search(r'\d+', val)
            return int(match.group()) if match else 999
        
        df['SortKey'] = df['Precision_Level'].apply(sort_key)
        df = df.sort_values('SortKey', ascending=False).drop('SortKey', axis=1)
        
        clean_model = model_name.replace("/", "_")
        
        # Create dataset-specific subdirectories
        dataset_metrics_dir = os.path.join(args.output_dir, dataset_name, "verifier_output", "metrics")
        csv_dir = os.path.join(dataset_metrics_dir, "csvs")
        plots_dir = os.path.join(dataset_metrics_dir, "plots")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        csv_path = os.path.join(csv_dir, f"metrics_{clean_model}_{dataset_name}_{samples_n}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  -> Saved metrics CSV to: {csv_path}")
        
        # Plot Confusion Matrices
        num_plots = len(matrices)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
            
        # Sort the matrices to plot in order (16bit -> 8bit -> 4bit)
        sorted_precisions = sorted(matrices.keys(), key=sort_key, reverse=True)
        
        fig.suptitle(f"Verifier Performance: {model_name} on {dataset_name} ({samples_n})", fontsize=16, y=1.05)
        
        for ax, prec in zip(axes, sorted_precisions):
            matrix = matrices[prec]
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Actual Correct", "Actual Incorrect"],
                        yticklabels=["Model Correct", "Model Incorrect"],
                        annot_kws={"size": 16})
            ax.set_title(f"{prec} Confusion Matrix", fontsize=14, pad=15)
            ax.set_xlabel("Ground Truth", fontsize=12)
            ax.set_ylabel("Verifier Prediction", fontsize=12)
            
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"confusion_matrices_{clean_model}_{dataset_name}_{samples_n}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved confusion matrices plot to: {plot_path}")

if __name__ == "__main__":
    main()
