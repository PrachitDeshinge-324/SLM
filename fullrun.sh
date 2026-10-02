#!/usr/bin/env bash
# Make script stop on first error
set -e

# Default configurations
MODEL="Qwen/Qwen3.5-0.8B"
PRECISION="16bit"
DATASET="math500"
N_SAMPLES=8
BATCH_SIZE=64
LIMIT=100

# Base directory for Colab mounted Drive
BASE_DIR="/content/drive/MyDrive/Experiment_D_Results"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --precision) PRECISION="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --n_samples) N_SAMPLES="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --output_dir) BASE_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "======================================================"
echo " 4. Generate Reasoning Traces"
echo "======================================================"

if [[ "${MODEL}" == *"DeepSeek-R1"* ]]; then
  MAX_NEW_TOKENS=8192
  REP_PENALTY=1.0
  echo "Detected DeepSeek R1 model: setting MAX_NEW_TOKENS=8192 to prevent CoT truncation."
else
  MAX_NEW_TOKENS=2048
  REP_PENALTY=1.1
  echo "Detected standard model: setting REP_PENALTY=1.1 to prevent hallucination loops."
fi

PYTHONPATH=. python scripts/run_experiment_d.py \
  --model "${MODEL}" \
  --precision "${PRECISION}" \
  --dataset "${DATASET}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --output_dir "${BASE_DIR}" \
  --limit "${LIMIT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --repetition_penalty "${REP_PENALTY}"

echo "======================================================"
echo " 5. Extract Verifier Dataset"
echo "======================================================"
PYTHONPATH=. python scripts/extract_verifier_dataset.py \
  --directory "${BASE_DIR}" \
  --model "${MODEL}" \
  --precision "${PRECISION}" \
  --dataset "${DATASET}" \
  --n_samples "n${N_SAMPLES}" \
  --output_dir "${BASE_DIR}"

echo "======================================================"
echo " 7. Evaluate Verifier Accuracy"
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

PYTHONPATH=. python scripts/run_verifier_benchmark.py \
    --directory "${BASE_DIR}" \
    --model_id "${MODEL}" \
    --precision "${PRECISION}" \
    --dataset "${DATASET}" \
    --n_samples "n${N_SAMPLES}" \
    --batch_size "${VERIFIER_BATCH_SIZE}" \
    --output_dir "${BASE_DIR}" \
    --max_new_tokens "${VERIFIER_MAX_NEW_TOKENS}" \
    --repetition_penalty "${REP_PENALTY}" \
    ${LIMIT_ARG}

echo "======================================================"
echo " Full run complete!"
echo "======================================================"