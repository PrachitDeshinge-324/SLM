#!/usr/bin/env bash
# ==============================================================================
# Rerun Qwen3.5-2B 4-bit Program-of-Thought (B1) with a valid 512-token ceiling
# ==============================================================================
# The previous 2B PoT run used an accidental 200-token cap (which caused 27.7%
# truncation and 100% syntax crashes among truncated samples). This script
# runs 2B with the standard 512-token limit used by all other generator runs.
#
# Usage:
#   bash run_2b_program_test.sh [--limit 200] [--batch_size 64] [--results_dir results]
# ==============================================================================
set -e

MODEL="Qwen/Qwen3.5-2B"
PRECISION="4bit"
DATASET="gsm8k"
N_SAMPLES=8
LIMIT=200
BATCH_SIZE=64
MAX_NEW_TOKENS=512
REPETITION_PENALTY="1.1"
SEED=1001
RESULTS_DIR="results"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --results_dir) RESULTS_DIR="$2"; shift ;;
        --n_samples) N_SAMPLES="$2"; shift ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

export PYTHONPATH=.

PROGRAM_ROOT="${RESULTS_DIR}/prompt_variants/program"
OUTPUT_FILE="${PROGRAM_ROOT}/gsm8k/generator/Qwen_Qwen3.5-2B/Qwen_Qwen3.5-2B_4bit_gsm8k_n8.jsonl"

# Check if an existing 200-token file is blocking regeneration
if [[ -f "${OUTPUT_FILE}" ]]; then
    OLD_MAX=$(python3 -c "import json; print(json.loads(open('${OUTPUT_FILE}').readline()).get('max_new_tokens', 0))" 2>/dev/null || echo 0)
    if [[ "${OLD_MAX}" -ne "${MAX_NEW_TOKENS}" ]]; then
        BACKUP="${OUTPUT_FILE%.jsonl}_old${OLD_MAX}t.jsonl"
        echo "Found existing file with max_new_tokens=${OLD_MAX} (target is ${MAX_NEW_TOKENS})."
        echo "Backing up stale file to: ${BACKUP}"
        mv "${OUTPUT_FILE}" "${BACKUP}"
    fi
fi

echo "=============================================================================="
echo " Generating ${MODEL} (${PRECISION}) Program-of-Thought (max_tokens=${MAX_NEW_TOKENS})"
echo "=============================================================================="

python scripts/generate_answers.py \
    --model "${MODEL}" \
    --precision "${PRECISION}" \
    --dataset "${DATASET}" \
    --n_samples "${N_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --limit "${LIMIT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --seed "${SEED}" \
    --prompt_variant "program" \
    --results_dir "${PROGRAM_ROOT}"

echo ""
echo "=============================================================================="
echo " Evaluating Program Agreement & Pre-Registered Validity Gate"
echo "=============================================================================="

python analysis/program_agreement_2026-10-24/01_program_agreement.py \
    --model "${MODEL}" \
    --precision "${PRECISION}" \
    --results_dir "${RESULTS_DIR}" \
    --program_root "${PROGRAM_ROOT}"

echo "=============================================================================="
echo " Completed 2B Program-of-Thought rerun and evaluation."
echo "=============================================================================="
