#!/usr/bin/env bash
# Multi-prompt test: does asking the same question with different prompts break the shared misreadings
# behind systematic quantization damage, so that the vote over mixed prompts finds the right answer?
#
# Answers the GSM8K test questions with each prompt variant other than 'default' (src/experiment_d/prompts.py:
# plan, check, example2), --n_samples each, into results/prompt_variants/<variant>/. The default prompt's
# answers already exist (the main n8 and n16 runs), so they are not generated again. Inference only, no
# training. Analysis: analysis/multi_prompt_2026-10-23/01_multi_prompt.py. Every run resumes.
#
# Usage:
#   bash run_multi_prompt_test.sh --model Qwen/Qwen3.5-0.8B --precision 4bit \
#        [--n_samples 4] [--limit 0] [--batch_size 64] [--variants "plan check example2"] [--results_dir DIR]
set -e

MODEL="Qwen/Qwen3.5-0.8B"
PRECISION="4bit"
N_SAMPLES=4
LIMIT=0
BATCH_SIZE=64
VARIANTS="plan check example2"
MAX_NEW_TOKENS=512          # the token limit of the main GSM8K runs these are compared with
RESULTS_DIR="/content/drive/MyDrive/results"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --precision) PRECISION="$2"; shift ;;
        --n_samples) N_SAMPLES="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --variants) VARIANTS="$2"; shift ;;
        --results_dir|--output_dir) RESULTS_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
export PYTHONPATH=.

# Each variant gets its own sampling seed (the main runs used 42), so no two prompts share random draws.
SEED=1000
for VARIANT in ${VARIANTS}; do
  SEED=$((SEED + 1))
  echo "======================================================"
  echo " ${MODEL} ${PRECISION}: GSM8K test with the '${VARIANT}' prompt, ${N_SAMPLES} samples per question"
  echo "======================================================"
  ROOT=$(python -c "import sys; from src.experiment_d.run_files import prompt_variant_dir; print(prompt_variant_dir(sys.argv[1], sys.argv[2]))" "${RESULTS_DIR}" "${VARIANT}")
  python scripts/generate_answers.py --model "${MODEL}" --precision "${PRECISION}" --dataset gsm8k \
    --n_samples "${N_SAMPLES}" --batch_size "${BATCH_SIZE}" --limit "${LIMIT}" --max_new_tokens "${MAX_NEW_TOKENS}" \
    --repetition_penalty 1.0 --seed "${SEED}" --prompt_variant "${VARIANT}" --results_dir "${ROOT}"
done

echo "======================================================"
echo " Done. Sync results/prompt_variants/ and run analysis/multi_prompt_2026-10-23/01_multi_prompt.py"
echo "======================================================"
