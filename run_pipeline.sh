#!/usr/bin/env bash
# Runs the three pipeline stages for ONE model at ONE precision on ONE dataset:
#   1. scripts/generate_answers.py        sample N answers per question
#   2. scripts/build_verifier_dataset.py  pick the solutions the verifier will judge
#   3. scripts/run_verifier.py            let the model judge those solutions
#
# Usage:
#   bash run_pipeline.sh --model Qwen/Qwen3.5-0.8B --precision 4bit --dataset gsm8k \
#        [--n_samples 8] [--batch_size 64] [--limit 100] [--results_dir DIR]
#
# Make script stop on first error
set -e

# Default configurations
MODEL="Qwen/Qwen3.5-0.8B"
PRECISION="16bit"
DATASET="math500"
N_SAMPLES=8
BATCH_SIZE=64
LIMIT=100

# Root directory of the run files (Colab-mounted Drive by default)
RESULTS_DIR="/content/drive/MyDrive/Experiment_D_Results"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --precision) PRECISION="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --n_samples) N_SAMPLES="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --results_dir|--output_dir) RESULTS_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "======================================================"
echo " Stage 1 of 3: Generate answers"
echo "======================================================"

if [[ "${MODEL}" == *"DeepSeek-R1"* ]]; then
  MAX_NEW_TOKENS=8192
  REP_PENALTY=1.0
  echo "Detected DeepSeek R1 model: setting MAX_NEW_TOKENS=8192 to prevent CoT truncation."
elif [[ "${DATASET}" == *"math500"* ]]; then
  MAX_NEW_TOKENS=2048
  REP_PENALTY=1.1
  echo "Detected math500: setting MAX_NEW_TOKENS=2048."
else
  MAX_NEW_TOKENS=512
  REP_PENALTY=1.1
  echo "Detected short-answer dataset: setting MAX_NEW_TOKENS=512 and REP_PENALTY=1.1."
fi

PYTHONPATH=. python scripts/generate_answers.py \
  --model "${MODEL}" \
  --precision "${PRECISION}" \
  --dataset "${DATASET}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --results_dir "${RESULTS_DIR}" \
  --limit "${LIMIT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --repetition_penalty "${REP_PENALTY}"

echo "======================================================"
echo " Stage 2 of 3: Build verifier dataset"
echo "======================================================"
PYTHONPATH=. python scripts/build_verifier_dataset.py \
  --results_dir "${RESULTS_DIR}" \
  --model "${MODEL}" \
  --precision "${PRECISION}" \
  --dataset "${DATASET}" \
  --n_samples "${N_SAMPLES}"

echo "======================================================"
echo " Stage 3 of 3: Run verifier"
echo "======================================================"
# Calculate limit for verifier (LIMIT * N_SAMPLES)
if [ "${LIMIT}" -gt 0 ]; then
    VERIFIER_LIMIT=$((LIMIT * N_SAMPLES))
    LIMIT_ARG="--limit ${VERIFIER_LIMIT}"
else
    LIMIT_ARG=""
fi

# Verifier only needs enough tokens to review and conclude (not full 8k scratchpad generation)
if [[ "${MODEL}" == *"DeepSeek-R1"* ]]; then
  VERIFIER_MAX_NEW_TOKENS=1536
  VERIFIER_BATCH_SIZE=16
else
  VERIFIER_MAX_NEW_TOKENS=512
  VERIFIER_BATCH_SIZE="${BATCH_SIZE}"
fi

PYTHONPATH=. python scripts/run_verifier.py \
    --results_dir "${RESULTS_DIR}" \
    --model "${MODEL}" \
    --precision "${PRECISION}" \
    --dataset "${DATASET}" \
    --n_samples "${N_SAMPLES}" \
    --batch_size "${VERIFIER_BATCH_SIZE}" \
    --max_new_tokens "${VERIFIER_MAX_NEW_TOKENS}" \
    --repetition_penalty "${REP_PENALTY}" \
    ${LIMIT_ARG}

echo "======================================================"
echo " Pipeline complete!"
echo "======================================================"
