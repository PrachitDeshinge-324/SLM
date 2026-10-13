"""
Benchmark loading.

Every loader returns a list of question dicts with the same keys:
    qid, dataset, question, ground_truth, choices (None unless multiple choice), raw_answer_str
"""
import ast
import os
import re

import pandas as pd

from src.experiment_d.prompts import GSM8K_1SHOT_USER

GSM8K_TRAIN_PARQUET = "data/gsm8k/main/train-00000-of-00001.parquet"


def question_key(text: str) -> str:
    """Question text with case and spacing ignored, for telling whether two questions are the same."""
    return " ".join(text.lower().split())


def _gsm8k_reference(answer_str: str) -> str:
    truth_num = answer_str.split("####")[-1].strip() if "####" in answer_str else answer_str.strip()
    return truth_num.replace(',', '')


def load_gsm8k(parquet_path: str = "data/gsm8k/main/test-00000-of-00001.parquet"):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find {parquet_path}. Please ensure it is present.")
    df = pd.read_parquet(parquet_path)
    questions = []
    for idx, row in df.iterrows():
        answer_str = row['answer']
        questions.append({
            "qid": f"gsm8k_{idx}",
            "dataset": "gsm8k",
            "question": row['question'],
            "ground_truth": _gsm8k_reference(answer_str),
            "choices": None,
            "raw_answer_str": answer_str
        })
    return questions


def load_gsm8k_train_pool(parquet_path: str = GSM8K_TRAIN_PARQUET):
    """
    Unlabelled GSM8K train questions, the pool a model self-trains on.

    Only the question column is read, so no reference answer can reach the caller. The question the
    prompt shows as its worked example is left out: its answer is in every prompt.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find {parquet_path}. Please ensure it is present.")
    df = pd.read_parquet(parquet_path, columns=["question"])
    prompt_example = question_key(GSM8K_1SHOT_USER)
    return [
        {
            "qid": f"gsm8k_train_{idx}",
            "dataset": "gsm8k_train",
            "question": row['question'],
            "ground_truth": None,
            "choices": None,
            "raw_answer_str": None,
        }
        for idx, row in df.iterrows()
        if question_key(row['question']) != prompt_example
    ]


def load_gsm8k_train_answers(parquet_path: str = GSM8K_TRAIN_PARQUET) -> dict:
    """
    qid -> reference answer of the GSM8K train questions.

    For reports written after a run only. Nothing that selects labels or trains may call this.
    """
    df = pd.read_parquet(parquet_path, columns=["answer"])
    return {f"gsm8k_train_{idx}": _gsm8k_reference(row['answer']) for idx, row in df.iterrows()}


def load_svamp(path_or_name: str = "ChilleD/SVAMP"):
    load_dataset = _import_hf_load_dataset()
    ds = load_dataset(path_or_name, split="test")
    questions = []
    for idx, row in enumerate(ds):
        # SVAMP has 'Body', 'Question', 'Answer'
        body = row.get('Body', '')
        q = row.get('Question', '')
        full_question = f"{body} {q}".strip()
        answer = str(row.get('Answer', ''))

        questions.append({
            "qid": f"svamp_{row.get('ID', idx)}",
            "dataset": path_or_name,
            "question": full_question,
            "ground_truth": answer,
            "choices": None,
            "raw_answer_str": answer
        })
    return questions


def load_math500(path_or_name: str = "HuggingFaceH4/MATH-500"):
    load_dataset = _import_hf_load_dataset()
    ds = load_dataset(path_or_name, split="test")
    questions = []
    for idx, row in enumerate(ds):
        answer = str(row.get('answer', ''))
        if not answer and 'solution' in row:
            match = re.search(r'\\boxed{(.+?)}', row['solution'])
            if match:
                answer = match.group(1)
        questions.append({
            "qid": f"math500_{idx}",
            "dataset": "math500",
            "question": row['problem'],
            "ground_truth": answer,
            "choices": None,
            "raw_answer_str": row.get('solution', '')
        })
    return questions


def parse_cqa_choices(choices_str: str):
    """Safely parse CQA choices string without eval()."""
    # Strip numpy array(..., dtype=object) wrappers → plain Python lists
    cleaned = re.sub(r'\barray\((\[.*?\])\s*,\s*dtype=\w+\)', r'\1', choices_str, flags=re.DOTALL)
    try:
        parsed = ast.literal_eval(cleaned)
        labels = parsed['label']
        texts = parsed['text']
        return [{"label": l, "text": t} for l, t in zip(labels, texts)]
    except (ValueError, SyntaxError, KeyError) as e:
        print(f"Error parsing choices: {e}\nString: {choices_str}")
        return []


def load_commonsenseqa(csv_path: str = "data/CQA/validation.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Please ensure it is present.")
    df = pd.read_csv(csv_path)
    questions = []
    for idx, row in df.iterrows():
        choices = parse_cqa_choices(row['choices'])
        questions.append({
            "qid": f"cqa_{idx}",
            "dataset": "commonsenseqa",
            "question": row['question'],
            "ground_truth": row['answerKey'],
            "choices": choices
        })
    return questions


def load_from_huggingface(dataset_name: str):
    """Best-effort loader for a dataset this project has no dedicated loader for."""
    try:
        load_dataset = _import_hf_load_dataset()
        ds = load_dataset(dataset_name, split="test")
        questions = []
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
            questions.append({
                "qid": f"{dataset_name.replace('/', '_')}_{idx}",
                "dataset": dataset_name,
                "question": row[q_col],
                "ground_truth": str(row[a_col]),
                "choices": None,
                "raw_answer_str": str(row[a_col])
            })
        print(f"Dynamically loaded dataset '{dataset_name}' from HuggingFace.")
        return questions
    except Exception as e:
        raise ValueError(f"Unknown dataset '{dataset_name}' and could not load from HuggingFace dynamically. Error: {e}")


def load_benchmark(dataset_name: str):
    """
    Questions of the named benchmark: gsm8k, gsm8k_train (unlabelled train questions), cqa, math500,
    any *svamp* id, or a HuggingFace dataset id.
    """
    name = dataset_name.lower()
    if name == "gsm8k":
        return load_gsm8k()
    elif name == "gsm8k_train":
        return load_gsm8k_train_pool()
    elif name == "cqa":
        return load_commonsenseqa()
    elif name == "math500":
        return load_math500()
    elif "svamp" in name:
        return load_svamp(dataset_name)
    else:
        return load_from_huggingface(dataset_name)


def _import_hf_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the 'datasets' library to load HuggingFace datasets.")
    return load_dataset
