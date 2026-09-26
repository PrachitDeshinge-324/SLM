import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import re
from dotenv import load_dotenv

load_dotenv()

VERIFIER_PROMPT_TEMPLATE = """You are an objective math evaluator.
Question: {question}

<student_solution>
{student_solution}
</student_solution>

Read the student's solution carefully. Is this solution correct or not?

Output exactly one line:
Final Conclusion: Correct (or Final Conclusion: Incorrect)"""

def parse_verifier_output(output_text):
    # Extract final conclusion
    final_conclusion = None
    conclusion_match = re.search(
        r'(?im)^\s*Final\s+Conclusion\s*:\s*(Correct|Incorrect)\s*[.!]?\s*$',
        output_text,
    )
    if conclusion_match:
        final_conclusion = conclusion_match.group(1).lower() == "correct"
        
    return [], final_conclusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Directory containing the extracted verifier datasets")
    parser.add_argument("--model_id", type=str, required=True, help="HF Model ID (e.g., Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--precision", type=str, required=True, choices=["16bit", "8bit", "4bit"])
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., gsm8k, cqa)")
    parser.add_argument("--n_samples", type=str, required=True, help="Number of samples (e.g., n16, n8)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the evaluated results")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Automatically find the input file
    import glob
    model_name = args.model_id.split('/')[-1] # Extract just the model name part
    pattern = f'verifier_*_{model_name}_{args.precision}_{args.dataset}_{args.n_samples}.jsonl'
    search_path = os.path.join(args.directory, pattern)
    matching_files = glob.glob(search_path)
    
    if not matching_files:
        print(f"Error: No files matching {pattern} found in {args.directory}")
        import sys
        sys.exit(1)
        
    if len(matching_files) > 1:
        raise RuntimeError(f"Ambiguous input pattern {pattern}: found {len(matching_files)} files. Specify a directory containing exactly one matching dataset.")
    input_file = matching_files[0]
    input_basename = os.path.basename(input_file)
    output_file = os.path.join(args.output_dir, input_basename.replace("verifier_", "verifier_output_"))

    print(f"Loading Tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    # Essential for batched autoregressive generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine best dtype for the hardware (bfloat16 for A100/L4, float16 for T4)
    # T4 GPUs do not support bfloat16 natively and will emulate it, causing massive slowdowns
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        print("Hardware supports bfloat16. Using bfloat16 for optimal precision.")
    else:
        compute_dtype = torch.float16
        print("Hardware does not support bfloat16 (e.g., T4/Mac). Falling back to float16.")

    # Configure Quantization & Precision
    # We use "cuda" instead of "auto" to prevent small models from being needlessly
    # split across multiple GPUs (e.g. Dual T4), which causes massive PCIe overhead.
    model_kwargs = {"device_map": "cuda", "torch_dtype": compute_dtype}
    if args.precision == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.precision == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
    print(f"Loading model {args.model_id} in {args.precision} mode...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    
    print(f"Loading input file: {input_file}")
    with open(input_file, 'r') as f:
        dataset = [json.loads(line) for line in f]
        
    if args.limit is not None:
        dataset = dataset[:args.limit]
        
    # --- Resume Functionality ---
    existing_results = []
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}. Attempting to resume...")
        try:
            with open(output_file, 'r') as f:
                existing_results = [json.loads(line) for line in f]
        except json.JSONDecodeError:
            print("Warning: Output file contains invalid JSON. Starting from scratch.")
            existing_results = []
            
    num_existing = len(existing_results)
    if num_existing > 0:
        if num_existing >= len(dataset):
            print(f"All {len(dataset)} traces have already been evaluated. Exiting.")
            return
        print(f"Resuming from trace {num_existing}... ({len(dataset) - num_existing} remaining)")
        dataset = dataset[num_existing:]
    else:
        # If starting fresh, clear the output file
        open(output_file, 'w').close()
        
    results = []
    
    tp = fp = tn = fn = 0
    
    print(f"Evaluating {len(dataset)} traces in batches of {args.batch_size}...")
    
    # Chunk dataset into batches
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch_items = dataset[i : i + args.batch_size]
        
        batch_texts = []
        for item in batch_items:
            prompt = VERIFIER_PROMPT_TEMPLATE.format(
                question=item['question'],
                student_solution=item['student_solution']
            )
            messages = [
                {"role": "system", "content": "You are a helpful, strict, and precise math teacher."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)
            
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Extract generated tokens (ignoring the prompt)
        prompt_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, prompt_length:]
        decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for idx, item in enumerate(batch_items):
            response_text = decoded_responses[idx]
            _, final_conclusion = parse_verifier_output(response_text)
            is_actually_correct = item['is_correct']
            
            item['verifier_raw_response'] = response_text
            item['verifier_final_conclusion'] = final_conclusion

        # Incremental save (Append only the new batch)
        with open(output_file, 'a') as f:
            for item in batch_items:
                f.write(json.dumps(item) + '\n')
                
    # --- Metrics Calculation ---
    # Reload full dataset to calculate metrics across both resumed and newly processed items
    with open(output_file, 'r') as f:
        full_results = [json.loads(line) for line in f]
        
    for item in full_results:
        final_conclusion = item.get('verifier_final_conclusion')
        is_actually_correct = item['is_correct']
        
        # Calculate ORM metrics
        if final_conclusion is not None:
            if final_conclusion and is_actually_correct:
                tp += 1
            elif final_conclusion and not is_actually_correct:
                fp += 1
            elif not final_conclusion and not is_actually_correct:
                tn += 1
            elif not final_conclusion and is_actually_correct:
                fn += 1
                    
    # Final Metrics Summary
    print("\n" + "="*40)
    print("=== VERIFIER BENCHMARK RESULTS ===")
    print("="*40)
    
    orm_total = tp + fp + tn + fn
    print(f"Final-conclusion parse coverage: {orm_total}/{len(dataset)} ({(orm_total / len(dataset) * 100) if dataset else 0:.2f}%)")
    if orm_total > 0:
        accuracy = (tp + tn) / orm_total
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr  # Recall is mathematically identical to TPR
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall (TPR): {recall*100:.2f}%")
        print(f"F1 Score: {f1*100:.2f}%")
        print(f"False Positive Rate (FPR): {fpr*100:.2f}%")
        
        print("\n--- 2x2 Confusion Matrix ---")
        print(f"                 | Actual Correct | Actual Incorrect |")
        print(f"-----------------|----------------|------------------|")
        print(f" Model Correct   | TP: {tp:<10} | FP: {fp:<14} |")
        print(f" Model Incorrect | FN: {fn:<10} | TN: {tn:<14} |")
        print(f"------------------------------------------------------")
    else:
        print("ORM metrics: Could not parse final conclusions.")
        
if __name__ == "__main__":
    main()
