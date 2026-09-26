import json
import random
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract seeded traces for verifier benchmark.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing the raw generated jsonl files")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Qwen3.5-0.8B)")
    parser.add_argument("--precision", type=str, required=True, help="Precision to extract (e.g., 16bit, 8bit, 4bit)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., gsm8k, cqa)")
    parser.add_argument("--n_samples", type=str, required=True, help="Number of samples (e.g., n16, n8)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted verifier datasets")
    args = parser.parse_args()

    seed = 42
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    import glob
    
    # Use glob to find the input file without needing the exact provider prefix
    pattern = f'*_{args.model}_{args.precision}_{args.dataset}_{args.n_samples}.jsonl'
    search_path = os.path.join(args.directory, pattern)
    matching_files = glob.glob(search_path)
    
    if not matching_files:
        print(f"Error: No files matching {pattern} found in {args.directory}")
        sys.exit(1)
        
    filepath = matching_files[0]
    if len(matching_files) > 1:
        print(f"Warning: Multiple files matched. Using {filepath}")
        
    input_basename = os.path.basename(filepath)
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Parse all lines, skipping the metadata line which doesn't have 'question'
    dataset = []
    for line in lines:
        parsed = json.loads(line)
        if 'question' in parsed:
            dataset.append(parsed)
    
    # Sort dataset by question so random sampling is perfectly deterministic across precisions
    dataset.sort(key=lambda x: x['question'])
    
    # Reset the seed right before sampling so each run gets the exact same 200 items
    random.seed(seed)
    sampled_items = random.sample(dataset, 200)
    
    output_data = []
    for item in sampled_items:
        question = item['question']
        ground_truth = item.get('ground_truth', '')
        
        # Determine ground truth answer
        gt_answer = ground_truth.split('####')[-1].strip()
        
        # Extract ALL generated traces for this question
        for trace_idx, trace in enumerate(item['raw_samples']):
            student_solution = trace['response']
            generated_answer = trace.get('extracted', '')
            
            # Determine if the generated trace is correct by comparing to ground truth
            is_correct = (generated_answer.strip() == gt_answer.strip())
            
            output_data.append({
                'question': question,
                'ground_truth': ground_truth,
                'gt_answer': gt_answer,
                'trace_index': trace_idx,
                'generated_answer': generated_answer,
                'student_solution': student_solution,
                'is_correct': is_correct,
                'precision': args.precision
            })
        
    output_filename = f'verifier_{input_basename}'
    output_filepath = os.path.join(args.output_dir, output_filename)
    
    with open(output_filepath, 'w') as f:
        for out_item in output_data:
            f.write(json.dumps(out_item) + '\n')
            
    print(f"[{args.precision}] Extracted 200 traces to {output_filepath}")

if __name__ == '__main__':
    main()
