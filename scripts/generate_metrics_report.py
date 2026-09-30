import argparse
import glob
import json
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

def parse_verifier_output(output_text, student_answer=None):
    if not output_text:
        return [], None
    text = output_text
    # Strip <think> blocks completely to avoid parsing internal monologue
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL).strip()

    # 1. Direct "Final Conclusion: Correct / Incorrect"
    m = re.search(r'(?i)(?:\*{1,2})?Final\s+Conclusion(?:\*{1,2})?\s*:\s*(?:\*{1,2})?\s*(Correct|Incorrect)\b', text)
    if m:
        return [], m.group(1).lower() == "correct"

    # 2. XML tags: <final_conclusion>Correct</final_conclusion>
    m = re.search(r'(?i)<final_conclusion>\s*(Correct|Incorrect)\s*</final_conclusion>', text)
    if m:
        return [], m.group(1).lower() == "correct"

    # 3. "Conclusion: The student's solution is Correct/Incorrect"
    m = re.search(r'(?i)(?:\*{1,2})?Conclusion(?:\*{1,2})?\s*:\s*(?:the\s+student\'?s?\s+solution\s+is\s+)?(?:\*{1,2})?\s*(Correct|Incorrect)\b', text)
    if m:
        return [], m.group(1).lower() == "correct"

    # 4. "[student's] solution is correct/incorrect"
    m = re.search(r'(?i)\b(?:the\s+student\'?s?\s+|this\s+)?solution\s+is\s+(?:\*{1,2})?\s*(correct|incorrect)\b', text)
    if m:
        return [], m.group(1).lower() == "correct"

    # 5. Tail check for explicit statement in the last 250 characters
    tail = text[-250:]
    m_inc = re.search(r'(?i)\b(?:is\s+|judged\s+as\s+|marked\s+as\s+)(?:\*{1,2})?incorrect\b', tail)
    if m_inc:
        return [], False
    m_cor = re.search(r'(?i)\b(?:is\s+|judged\s+as\s+|marked\s+as\s+)(?:\*{1,2})?correct\b', tail)
    if m_cor:
        return [], True

    # 6. Fallback for Reasoning/Thinking models (e.g. DeepSeek-R1) that re-solve the problem:
    if student_answer is not None:
        try:
            from src.experiment_d.extractors import extract_gsm8k_answer
            from scripts.extract_verifier_dataset import answers_match
            ver_ans = extract_gsm8k_answer(text)
            if ver_ans:
                return [], answers_match(ver_ans, str(student_answer))
        except Exception:
            pass

    # 7. Ultimate fallback: if completely unparseable, assume None (unparseable)
    return [], None

def parse_args():
    parser = argparse.ArgumentParser(description="Process verifier JSONL logs and generate metrics.")
    parser.add_argument("--input_dir", type=str, default="results/verifier_output", help="Directory containing the JSONL logs.")
    parser.add_argument("--output_dir", type=str, default="results/verifier_output/metrics", help="Directory to save CSV and plots.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all relevant jsonl files across subdirectories
    files = glob.glob(os.path.join(args.input_dir, "**", "verifier_output_*.jsonl"), recursive=True)
    if not files:
        print(f"No verifier_output_*.jsonl files found in {args.input_dir}")
        return
        
    # Group files by (model_name, dataset_name)
    grouped_files = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        # Expected format: verifier_output_{model_id}_{precision}_{dataset}_{samples}.jsonl
        match = re.search(r'verifier_output_(.+?)_(\d+bit)_(.+?)_(n\d+)\.jsonl', filename)
        if match:
            model_name = match.group(1)
            precision = match.group(2)
            dataset_name = match.group(3)
            samples_n = match.group(4)
        else:
            print(f"Warning: Could not parse model/dataset from {filename}. Skipping.")
            continue
            
        key = (model_name, dataset_name, samples_n)
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append((filepath, precision))
        
    for (model_name, dataset_name, samples_n), file_list in grouped_files.items():
        print(f"\nProcessing Group: Model='{model_name}', Dataset='{dataset_name}', Samples='{samples_n}'")
        results_list = []
        matrices = {}
        
        for filepath, precision in file_list:
            filename = os.path.basename(filepath)
            tp = fp = tn = fn = 0
            total = 0
            
            # --- Question-level tracking ---
            from collections import Counter
            questions_data = {}
            
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    gen_ans = str(data.get('generated_answer'))
                    if 'verifier_raw_response' in data:
                        _, conc = parse_verifier_output(data['verifier_raw_response'], student_answer=gen_ans)
                    else:
                        conc = data.get('verifier_final_conclusion')
                    is_correct = data.get('is_correct')
                    question = data.get('question')
                    
                    if conc is not None:
                        total += 1
                        if conc and is_correct: tp += 1
                        elif conc and not is_correct: fp += 1
                        elif not conc and not is_correct: tn += 1
                        elif not conc and is_correct: fn += 1
                        
                        if question not in questions_data:
                            questions_data[question] = []
                            
                        questions_data[question].append({
                            "gen_ans": gen_ans,
                            "is_correct": is_correct,
                            "verifier_says_correct": conc
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
        csv_path = os.path.join(args.output_dir, f"metrics_{clean_model}_{dataset_name}_{samples_n}.csv")
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
        plot_path = os.path.join(args.output_dir, f"confusion_matrices_{clean_model}_{dataset_name}_{samples_n}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved confusion matrices plot to: {plot_path}")

if __name__ == "__main__":
    main()
