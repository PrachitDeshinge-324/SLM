#!/usr/bin/env bash
# Label-free self-training for ONE model at ONE precision, over several rounds.
#
#   round 0            the untrained model answers the test questions and the unlabelled train pool
#   round k = 1..N     scripts/train_self_labels.py trains a fresh adapter on the confident answers
#                      of round k-1; the model with that adapter answers the test questions and,
#                      unless it is the last round, the pool again
#
# The pool is GSM8K train questions loaded without their answers; the test questions are GSM8K
# test and are only ever evaluated. Every stage resumes, so the same command can be run again
# after an interrupted session.
#
# Usage:
#   bash run_self_training.sh --model Qwen/Qwen3.5-0.8B --precision 4bit --experiment qwen08_4bit_pilot \
#        [--rounds 3] [--pool_limit 200] [--eval_limit 100] [--batch_size 64] [--results_dir DIR]
#
# Make script stop on first error
set -e

# Default configurations
MODEL="Qwen/Qwen3.5-0.8B"
PRECISION="4bit"
EXPERIMENT="pilot"
ROUNDS=3
POOL_LIMIT=200
EVAL_LIMIT=100
N_SAMPLES=8
BATCH_SIZE=64
TRAIN_STEPS=200
THRESHOLD=0.75
# Generation settings of the stored GSM8K runs these rounds are compared with.
MAX_NEW_TOKENS=512
REP_PENALTY=1.0
POOL_DATASET="gsm8k_train"
EVAL_DATASET="gsm8k"

# Root directory of the run files (Colab-mounted Drive by default)
RESULTS_DIR="/content/drive/MyDrive/results"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --precision) PRECISION="$2"; shift ;;
        --experiment) EXPERIMENT="$2"; shift ;;
        --rounds) ROUNDS="$2"; shift ;;
        --pool_limit) POOL_LIMIT="$2"; shift ;;
        --eval_limit) EVAL_LIMIT="$2"; shift ;;
        --n_samples) N_SAMPLES="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --train_steps) TRAIN_STEPS="$2"; shift ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift ;;
        --repetition_penalty) REP_PENALTY="$2"; shift ;;
        --results_dir|--output_dir) RESULTS_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Folder of a round's run files and of its adapter (naming lives in run_files.py).
round_paths() {
  PYTHONPATH=. python -c "
import sys
from src.experiment_d.run_files import adapter_dir, self_training_round_dir
results_dir, experiment, round_index, model, precision = sys.argv[1:]
print(self_training_round_dir(results_dir, experiment, round_index))
print(adapter_dir(results_dir, experiment, round_index, model, precision))
" "${RESULTS_DIR}" "${EXPERIMENT}" "$1" "${MODEL}" "${PRECISION}"
}

# generate ROUND DATASET LIMIT: the round's model answers the first LIMIT questions of DATASET.
generate() {
  local round_dir adapter
  { read -r round_dir; read -r adapter; } < <(round_paths "$1")
  local adapter_args=()
  if [ "$1" -gt 0 ]; then
    adapter_args=(--adapter "${adapter}")
  fi
  PYTHONPATH=. python scripts/generate_answers.py \
    --model "${MODEL}" \
    --precision "${PRECISION}" \
    --dataset "$2" \
    --n_samples "${N_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --results_dir "${round_dir}" \
    --limit "$3" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --repetition_penalty "${REP_PENALTY}" \
    "${adapter_args[@]}"
}

for ROUND in $(seq 0 "${ROUNDS}"); do
  if [ "${ROUND}" -gt 0 ]; then
    echo "======================================================"
    echo " Round ${ROUND} of ${ROUNDS}: train on the confident answers of round $((ROUND - 1))"
    echo "======================================================"
    PYTHONPATH=. python scripts/train_self_labels.py \
      --results_dir "${RESULTS_DIR}" \
      --model "${MODEL}" \
      --precision "${PRECISION}" \
      --dataset "${POOL_DATASET}" \
      --eval_dataset "${EVAL_DATASET}" \
      --n_samples "${N_SAMPLES}" \
      --experiment "${EXPERIMENT}" \
      --round "${ROUND}" \
      --threshold "${THRESHOLD}" \
      --train_steps "${TRAIN_STEPS}"
  fi

  echo "======================================================"
  echo " Round ${ROUND} of ${ROUNDS}: answer the test questions"
  echo "======================================================"
  generate "${ROUND}" "${EVAL_DATASET}" "${EVAL_LIMIT}"

  # The last round's model labels nothing, so it does not answer the pool.
  if [ "${ROUND}" -lt "${ROUNDS}" ]; then
    echo "======================================================"
    echo " Round ${ROUND} of ${ROUNDS}: answer the unlabelled train pool"
    echo "======================================================"
    generate "${ROUND}" "${POOL_DATASET}" "${POOL_LIMIT}"
  fi
done

echo "======================================================"
echo " Report"
echo "======================================================"
PYTHONPATH=. python scripts/report_self_training.py \
  --results_dir "${RESULTS_DIR}" \
  --model "${MODEL}" \
  --precision "${PRECISION}" \
  --dataset "${EVAL_DATASET}" \
  --pool_dataset "${POOL_DATASET}" \
  --n_samples "${N_SAMPLES}" \
  --experiment "${EXPERIMENT}" \
  --threshold "${THRESHOLD}"

echo "======================================================"
echo " Self-training complete!"
echo "======================================================"
