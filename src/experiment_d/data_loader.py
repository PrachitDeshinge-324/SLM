import pandas as pd
import numpy as np
import os
import ast

def parse_cqa_choices(choices_str: str):
    """
    Parses the string representation of CQA choices.
    Example string:
    {'label': array(['A', 'B', 'C', 'D', 'E'], dtype=object), 'text': array(['bank', 'library', ...], dtype=object)}
    """
    # Safe eval environment with numpy array and builtins
    env = {
        'array': np.array,
        'object': object
    }
    
    try:
        parsed = eval(choices_str, {"__builtins__": {}}, env)
        labels = parsed['label'].tolist()
        texts = parsed['text'].tolist()
        return [{"label": l, "text": t} for l, t in zip(labels, texts)]
    except Exception as e:
        print(f"Error parsing choices: {e}\nString: {choices_str}")
        return []

def load_cqa_dataset(csv_path: str = "CQA/validation.csv"):
    """
    Loads CommonsenseQA and standardizes format.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Please ensure it is present.")
    
    df = pd.read_csv(csv_path)
    dataset = []
    
    for idx, row in df.iterrows():
        choices = parse_cqa_choices(row['choices'])
        dataset.append({
            "qid": f"cqa_{idx}",
            "dataset": "commonsenseqa",
            "question": row['question'],
            "ground_truth": row['answerKey'],
            "choices": choices
        })
        
    return dataset

def load_gsm8k_dataset(parquet_path: str = "gsm8k/main/test-00000-of-00001.parquet"):
    """
    Loads GSM8K test set and standardizes format.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find {parquet_path}. Please ensure it is present.")
        
    df = pd.read_parquet(parquet_path)
    dataset = []
    
    for idx, row in df.iterrows():
        # Ground truth format usually has "#### <answer>"
        answer_str = row['answer']
        # We can extract the raw numeric ground truth, or just store the full answer string.
        # It's better to isolate the number for the `metrics.py` check, but let's store both.
        # "#### 72" -> "72"
        truth_num = answer_str.split("####")[-1].strip() if "####" in answer_str else answer_str.strip()
        # Normalize: strip commas so "1,000" becomes "1000" (matching extractor output)
        truth_num = truth_num.replace(',', '')
        
        dataset.append({
            "qid": f"gsm8k_{idx}",
            "dataset": "gsm8k",
            "question": row['question'],
            "ground_truth": truth_num,
            "choices": None,
            "raw_answer_str": answer_str
        })
        
    return dataset

def load_datasets():
    """
    Convenience method to load both test sets.
    """
    print("Loading CQA...")
    cqa_data = load_cqa_dataset()
    print(f"Loaded {len(cqa_data)} CQA items.")
    
    print("Loading GSM8K...")
    gsm8k_data = load_gsm8k_dataset()
    print(f"Loaded {len(gsm8k_data)} GSM8K items.")
    
    return cqa_data, gsm8k_data

if __name__ == "__main__":
    c, g = load_datasets()
    print("CQA Sample:")
    print(c[0])
    print("GSM8K Sample:")
    print(g[0])
