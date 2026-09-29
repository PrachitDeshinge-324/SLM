
def load_svamp_dataset(path_or_name: str = "ChilleD/SVAMP"):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the 'datasets' library to load HuggingFace datasets.")
    ds = load_dataset(path_or_name, split="test")
    dataset = []
    for idx, row in enumerate(ds):
        # SVAMP has 'Body', 'Question', 'Answer'
        body = row.get('Body', '')
        q = row.get('Question', '')
        full_question = f"{body} {q}".strip()
        answer = str(row.get('Answer', ''))
        
        dataset.append({
            "qid": f"svamp_{row.get('ID', idx)}",
            "dataset": path_or_name,
            "question": full_question,
            "ground_truth": answer,
            "choices": None,
            "raw_answer_str": answer
        })
    return dataset

import pandas as pd
import numpy as np
import os
import ast

def parse_cqa_choices(choices_str: str):
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
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find {parquet_path}. Please ensure it is present.")
    df = pd.read_parquet(parquet_path)
    dataset = []
    for idx, row in df.iterrows():
        answer_str = row['answer']
        truth_num = answer_str.split("####")[-1].strip() if "####" in answer_str else answer_str.strip()
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

def load_math500_dataset(path_or_name: str = "HuggingFaceH4/MATH-500"):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the 'datasets' library to load HuggingFace datasets.")
    ds = load_dataset(path_or_name, split="test")
    dataset = []
    for idx, row in enumerate(ds):
        answer = str(row.get('answer', ''))
        if not answer and 'solution' in row:
            import re
            match = re.search(r'\\boxed{(.+?)}', row['solution'])
            if match:
                answer = match.group(1)
        dataset.append({
            "qid": f"math500_{idx}",
            "dataset": "math500",
            "question": row['problem'],
            "ground_truth": answer,
            "choices": None,
            "raw_answer_str": row.get('solution', '')
        })
    return dataset

def get_dataset_loader(dataset_name: str):
    name = dataset_name.lower()
    if name == "gsm8k":
        return load_gsm8k_dataset()
    elif name == "cqa":
        return load_cqa_dataset()
    elif name == "math500":
        return load_math500_dataset()
    elif "svamp" in name:
        return load_svamp_dataset(dataset_name)
    else:
        # Fallback to HuggingFace loading attempt
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="test")
            dataset = []
            for idx, row in enumerate(ds):
                # Guess standard column names (case insensitive)
                keys = list(row.keys())
                lower_keys = [k.lower() for k in keys]
                
                q_col = keys[0]
                if 'question' in lower_keys:
                    q_col = keys[lower_keys.index('question')]
                elif 'problem' in lower_keys:
                    q_col = keys[lower_keys.index('problem')]
                    
                a_col = keys[-1]
                if 'answer' in lower_keys:
                    a_col = keys[lower_keys.index('answer')]
                elif 'solution' in lower_keys:
                    a_col = keys[lower_keys.index('solution')]
                elif 'target' in lower_keys:
                    a_col = keys[lower_keys.index('target')]
                dataset.append({
                    "qid": f"{dataset_name.replace('/', '_')}_{idx}",
                    "dataset": dataset_name,
                    "question": row[q_col],
                    "ground_truth": str(row[a_col]),
                    "choices": None,
                    "raw_answer_str": str(row[a_col])
                })
            print(f"Dynamically loaded dataset '{dataset_name}' from HuggingFace.")
            return dataset
        except Exception as e:
            raise ValueError(f"Unknown dataset '{dataset_name}' and could not load from HuggingFace dynamically. Error: {e}")

