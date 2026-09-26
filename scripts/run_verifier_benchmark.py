import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import re
from dotenv import load_dotenv

load_dotenv()

VERIFIER_PROMPT_TEMPLATE = """You are an expert math teacher grading a student's solution.
Question: {question}

Student's Solution:
{student_solution}

Please evaluate the solution step-by-step. For each step, determine if the mathematical logic and calculations are correct.
Finally, conclude whether the overall answer is correct.

Format your response exactly as follows:
Step 1: [Brief Analysis] - Score: [1 or 0]
Step 2: [Brief Analysis] - Score: [1 or 0]
...
Final Conclusion: [Correct / Incorrect]"""

def parse_verifier_output(output_text):
    # Extract step scores
    # Look for "Score: 1" or "Score: 0" (case insensitive)
    step_scores = []
    score_pattern = r'Score:\s*([01])'
    matches = re.findall(score_pattern, output_text, re.IGNORECASE)
    for m in matches:
        step_scores.append(int(m))
        
    # Extract final conclusion
    final_conclusion = None
    if re.search(r'Final Conclusion:\s*Correct', output_text, re.IGNORECASE):
        final_conclusion = True
    elif re.search(r'Final Conclusion:\s*Incorrect', output_text, re.IGNORECASE):
        final_conclusion = False
        
    return step_scores, final_conclusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Directory containing the extracted verifier datasets")
    parser.add_argument("--model_id", type=str, required=True, help="HF Model ID (e.g., Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--precision", type=str, required=True, choices=["16bit", "8bit", "4bit"])
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., gsm8k, cqa)")
    parser.add_argument("--n_samples", type=str, required=True, help="Number of samples (e.g., n16, n8)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the evaluated results")
    parser.add_argument("--max_new_tokens", type=int, default=512)
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
        
    input_file = matching_files[0]
    input_basename = os.path.basename(input_file)
    output_file = os.path.join(args.output_dir, input_basename.replace("verifier_", "verifier_output_"))

    print(f"Loading Tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Configure Quantization & Precision
    model_kwargs = {"device_map": "auto"}
    if args.precision == "16bit":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.precision == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.precision == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        
    print(f"Loading model {args.model_id} in {args.precision} mode...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    
    print(f"Loading input file: {input_file}")
    with open(input_file, 'r') as f:
        dataset = [json.loads(line) for line in f]
        
    if args.limit is not None:
        dataset = dataset[:args.limit]
        
    results = []
    
    tp = fp = tn = fn = 0
    prm_catch_rate_hits = prm_catch_rate_total = 0
    prm_false_alarm_hits = prm_false_alarm_total = 0
    
    print(f"Evaluating {len(dataset)} traces...")
    for item in tqdm(dataset):
        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            question=item['question'],
            student_solution=item['student_solution']
        )
        
        # Use Chat Template for instruct models
        messages = [
            {"role": "system", "content": "You are a helpful and precise math teacher."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens,
                temperature=0.0, # Greedy decoding is standard for verifiers
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        step_scores, final_conclusion = parse_verifier_output(response_text)
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
                
        # Calculate PRM metrics
        if len(step_scores) > 0:
            if not is_actually_correct:
                prm_catch_rate_total += 1
                if 0 in step_scores:
                    prm_catch_rate_hits += 1
            else:
                prm_false_alarm_total += 1
                if 0 in step_scores:
                    prm_false_alarm_hits += 1
                    
        item['verifier_raw_response'] = response_text
        item['verifier_step_scores'] = step_scores
        item['verifier_final_conclusion'] = final_conclusion
        results.append(item)
        
        # Incremental save
        with open(output_file, 'w') as f:
            for res in results:
                f.write(json.dumps(res) + '\n')
                
    # Final Metrics Summary
    print("\n" + "="*40)
    print("=== VERIFIER BENCHMARK RESULTS ===")
    print("="*40)
    
    orm_total = tp + fp + tn + fn
    if orm_total > 0:
        accuracy = (tp + tn) / orm_total
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"Total Parses: {orm_total}/{len(dataset)}")
        print(f"ORM Accuracy: {accuracy*100:.2f}%")
        print(f"True Positive Rate (TPR): {tpr*100:.2f}%")
        print(f"False Positive Rate (FPR): {fpr*100:.2f}%")
    else:
        print("ORM metrics: Could not parse final conclusions.")
        
    if prm_catch_rate_total > 0:
        print(f"PRM Catch Rate: {(prm_catch_rate_hits/prm_catch_rate_total)*100:.2f}% ({prm_catch_rate_hits}/{prm_catch_rate_total} incorrect traces flagged)")
    if prm_false_alarm_total > 0:
        print(f"PRM False Alarm Rate: {(prm_false_alarm_hits/prm_false_alarm_total)*100:.2f}% ({prm_false_alarm_hits}/{prm_false_alarm_total} correct traces wrongly flagged)")

if __name__ == "__main__":
    main()
