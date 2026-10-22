#!/usr/bin/env bash
# Critic self-training, one round, several arms sharing one round 0, for ONE model at ONE precision.
#
#   1. round 0 (experiment <base>)  the untrained model answers the test questions and the unlabelled train pool
#   2. critic over the pool         scripts/run_signals.py: forced critic on contested pool questions only
#                                   (at most 2 solutions per answer, --verdicts per solution, no mc, no token signals)
#   3. label report                 how many labels each rule gives and how many are right (checked afterwards)
#   4. one round per arm            (experiment <base>_<arm>) train on round 0's pool, answer the test questions
#        vote           majority-vote labels
#        critic         critic + vote labels
#        critic_reason  critic + vote labels, plus the critiques that agree with them as a second task
#   5. report                       scripts/report_critic_self_training.py
#
# Every arm trains with the chat end format (<|im_end|> \n <|endoftext|>), so the vote arm is the earlier
# self-training rule with the end token fixed. Every stage resumes: rerun the same command after an interruption.
#
# Usage:
#   bash run_critic_self_training.sh --model Qwen/Qwen3.5-2B --precision 4bit --base q2b_4bit_critic_p500 \
#        [--pool_limit 500] [--eval_limit 100] [--verdicts 2] [--critic_tokens 704] [--arms "vote critic critic_reason"] \
#        [--batch_size 64] [--train_steps 200] [--threshold 0.75] [--stop_after_labels] [--results_dir DIR]
#        [--micro_batch_size 2] [--gradient_checkpointing] [--chunked_loss]
#
# --batch_size is the generation batch (sequences at once). Training memory is set by --micro_batch_size
# (solutions per forward pass; the update is always 8 solutions) and --gradient_checkpointing. For 16-bit
# on a 16 GB GPU (T4) use --gradient_checkpointing --chunked_loss: both give the same training (checkpointing
# keeps no activations, the chunked loss never holds the logits of a whole sequence over the ~248k vocabulary).
# If that still runs out of memory, add --micro_batch_size 1 (same update in expectation; LoRA dropout draws
# other masks).
set -e

MODEL="Qwen/Qwen3.5-2B"
PRECISION="4bit"
BASE=""
POOL_LIMIT=500
EVAL_LIMIT=100
N_SAMPLES=8
BATCH_SIZE=64
TRAIN_STEPS=200
MICRO_BATCH_SIZE=2
CHECKPOINTING=()
CHUNKED=()
THRESHOLD=0.75
VERDICTS=2
CRITIC="critic_forced"
CRITIC_TOKENS=704
MIN_VERDICT_RATE=0.85
ARMS="vote critic critic_reason"
STOP_AFTER_LABELS=0
MAX_NEW_TOKENS=512
POOL_DATASET="gsm8k_train"
EVAL_DATASET="gsm8k"
RESULTS_DIR="/content/drive/MyDrive/results"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --precision) PRECISION="$2"; shift ;;
        --base) BASE="$2"; shift ;;
        --pool_limit) POOL_LIMIT="$2"; shift ;;
        --eval_limit) EVAL_LIMIT="$2"; shift ;;
        --n_samples) N_SAMPLES="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --train_steps) TRAIN_STEPS="$2"; shift ;;
        --micro_batch_size) MICRO_BATCH_SIZE="$2"; shift ;;
        --gradient_checkpointing) CHECKPOINTING=(--gradient_checkpointing) ;;
        --chunked_loss) CHUNKED=(--chunked_loss) ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --verdicts) VERDICTS="$2"; shift ;;
        --critic_tokens) CRITIC_TOKENS="$2"; shift ;;
        --min_verdict_rate) MIN_VERDICT_RATE="$2"; shift ;;
        --arms) ARMS="$2"; shift ;;
        --stop_after_labels) STOP_AFTER_LABELS=1 ;;
        --results_dir|--output_dir) RESULTS_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "${BASE}" ]; then echo "--base is required"; exit 1; fi
TAG="${CRITIC}_v${VERDICTS}_t${CRITIC_TOKENS}_pilot"
export PYTHONPATH=.
# Fewer out-of-memory failures from fragmentation (memory reserved but unusable); changes no result.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

banner() { echo "======================================================"; echo " $1"; echo "======================================================"; }

# generate EXPERIMENT ROUND DATASET LIMIT: that round's model (with its adapter after round 0) answers the questions.
generate() {
  local paths round_dir adapter
  paths=$(python -c "
import sys
from src.experiment_d.run_files import adapter_dir, self_training_round_dir
results_dir, experiment, round_index, model, precision = sys.argv[1:]
print(self_training_round_dir(results_dir, experiment, round_index))
print(adapter_dir(results_dir, experiment, round_index, model, precision))
" "${RESULTS_DIR}" "$1" "$2" "${MODEL}" "${PRECISION}")
  round_dir=$(echo "${paths}" | sed -n 1p)
  adapter=$(echo "${paths}" | sed -n 2p)
  local adapter_args=()
  if [ "$2" -gt 0 ]; then adapter_args=(--adapter "${adapter}"); fi
  python scripts/generate_answers.py --model "${MODEL}" --precision "${PRECISION}" --dataset "$3" \
    --n_samples "${N_SAMPLES}" --batch_size "${BATCH_SIZE}" --results_dir "${round_dir}" --limit "$4" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --repetition_penalty 1.0 "${adapter_args[@]}"
}

report() {
  python scripts/report_critic_self_training.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" \
    --precision "${PRECISION}" --dataset "${EVAL_DATASET}" --pool_dataset "${POOL_DATASET}" --n_samples "${N_SAMPLES}" \
    --base "${BASE}" --signals_tag "${TAG}" --critic "${CRITIC}" --threshold "${THRESHOLD}" "$@"
}

banner "Round 0: the untrained model answers the test questions and the pool"
generate "${BASE}" 0 "${EVAL_DATASET}" "${EVAL_LIMIT}"
generate "${BASE}" 0 "${POOL_DATASET}" "${POOL_LIMIT}"

banner "Critic over the pool (${TAG})"
python scripts/run_signals.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" --precision "${PRECISION}" \
  --dataset "${POOL_DATASET}" --n_samples "${N_SAMPLES}" --experiment "${BASE}" --round 0 \
  --question_set first --questions "${POOL_LIMIT}" --critics "${CRITIC}" --verdicts "${VERDICTS}" \
  --max_new_tokens "${CRITIC_TOKENS}" --no_mc --no_token_signals --min_verdict_rate "${MIN_VERDICT_RATE}" \
  --batch_size "${BATCH_SIZE}"

banner "Labels on the pool (reference answers read for this report only)"
report --labels_only
if [ "${STOP_AFTER_LABELS}" = 1 ]; then echo "Stopped after the label report (--stop_after_labels)."; exit 0; fi

for ARM in ${ARMS}; do
  banner "Arm ${ARM}: train on round 0's pool, answer the test questions"
  arm_args=()
  case "${ARM}" in
    vote) ;;
    critic) arm_args=(--labels critic --signals_tag "${TAG}" --critic "${CRITIC}") ;;
    critic_reason) arm_args=(--labels critic --signals_tag "${TAG}" --critic "${CRITIC}" --critiques) ;;
    *) echo "Unknown arm ${ARM}"; exit 1 ;;
  esac
  python scripts/train_self_labels.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" --precision "${PRECISION}" \
    --dataset "${POOL_DATASET}" --eval_dataset "${EVAL_DATASET}" --n_samples "${N_SAMPLES}" \
    --experiment "${BASE}_${ARM}" --pool_experiment "${BASE}" --round 1 --threshold "${THRESHOLD}" \
    --train_steps "${TRAIN_STEPS}" --micro_batch_size "${MICRO_BATCH_SIZE}" --end_format chat \
    "${CHECKPOINTING[@]}" "${CHUNKED[@]}" "${arm_args[@]}"
  generate "${BASE}_${ARM}" 1 "${EVAL_DATASET}" "${EVAL_LIMIT}"
done

banner "Report"
report

banner "Critic self-training complete!"
