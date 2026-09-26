import argparse
import glob
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Process verifier JSONL logs and generate metrics.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the JSONL logs.")
    parser.add_argument("--output_dir", type=str, default="results/metrics", help="Directory to save CSV and plots.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all relevant jsonl files
    files = glob.glob(os.path.join(args.input_dir, "verifier_output_*.jsonl"))
    if not files:
        print(f"No verifier_output_*.jsonl files found in {args.input_dir}")
        return
        
    # Group files by (model_name, dataset_name)
    grouped_files = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        # Expected format: verifier_output_{model_id}_{precision}_{dataset}_{samples}.jsonl
        match = re.search(r'verifier_output_(.+?)_(\d+bit)_(.+?)_n\d+\.jsonl', filename)
        if match:
            model_name = match.group(1)
            precision = match.group(2)
            dataset_name = match.group(3)
        else:
            print(f"Warning: Could not parse model/dataset from {filename}. Skipping.")
            continue
            
        key = (model_name, dataset_name)
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append((filepath, precision))
        
    for (model_name, dataset_name), file_list in grouped_files.items():
        print(f"\nProcessing Group: Model='{model_name}', Dataset='{dataset_name}'")
        results_list = []
        matrices = {}
        
        for filepath, precision in file_list:
            filename = os.path.basename(filepath)
            tp = fp = tn = fn = 0
            total = 0
            
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    conc = data.get('verifier_final_conclusion')
                    is_correct = data.get('is_correct')
                    
                    if conc is not None:
                        total += 1
                        if conc and is_correct: tp += 1
                        elif conc and not is_correct: fp += 1
                        elif not conc and not is_correct: tn += 1
                        elif not conc and is_correct: fn += 1
                        
            if total == 0:
                continue
                
            accuracy = (tp + tn) / total
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision_metric = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tpr
            f1 = 2 * (precision_metric * recall) / (precision_metric + recall) if (precision_metric + recall) > 0 else 0
            
            results_list.append({
                "Precision_Level": precision,
                "Total_Parsed": total,
                "Accuracy": f"{accuracy*100:.2f}%",
                "Precision": f"{precision_metric*100:.2f}%",
                "Recall (TPR)": f"{recall*100:.2f}%",
                "F1_Score": f"{f1*100:.2f}%",
                "FPR": f"{fpr*100:.2f}%",
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn
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
        csv_path = os.path.join(args.output_dir, f"metrics_{clean_model}_{dataset_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  -> Saved metrics CSV to: {csv_path}")
        
        # Plot Confusion Matrices
        num_plots = len(matrices)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
            
        # Sort the matrices to plot in order (16bit -> 8bit -> 4bit)
        sorted_precisions = sorted(matrices.keys(), key=sort_key, reverse=True)
        
        fig.suptitle(f"Verifier Performance: {model_name} on {dataset_name}", fontsize=16, y=1.05)
        
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
        plot_path = os.path.join(args.output_dir, f"confusion_matrices_{clean_model}_{dataset_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved confusion matrices plot to: {plot_path}")

if __name__ == "__main__":
    main()
