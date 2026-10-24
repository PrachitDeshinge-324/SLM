#!/usr/bin/env bash
# Critic self-training, several arms sharing one round 0, for ONE model at ONE precision.
#
#   1. round 0 (experiment <base>)  the untrained model answers the test questions and the unlabelled train pool
#   2. critic over the pool         scripts/run_signals.py: forced critic on contested pool questions only
#                                   (at most 2 solutions per answer, --verdicts per solution, no mc, no token signals)
#   3. label report                 how many labels each rule gives and how many are right (checked afterwards)
#   4. round 1 per arm              (experiment <base>_<arm>) train on round 0's pool, answer the test questions
#        vote           majority-vote labels
#        critic         critic + vote labels
#        critic_reason  critic + vote labels, plus the critiques that agree with them as a second task
#   4b. rounds 2..--rounds per arm  the arm's round k-1 model answers the pool again (and, for the critic arms,
#                                   judges it: --round_critic self, the default, or the untrained model with
#                                   --round_critic base); a fresh adapter is trained on those labels and answers
#                                   the test questions. Labels are re-made every round, so they can drift.
#   5. report                       scripts/report_critic_self_training.py (every round, and the labels of each)
#
# Every arm trains with the chat end format (<|im_end|> \n <|endoftext|>), so the vote arm is the earlier
# self-training rule with the end token fixed. Every stage resumes: rerun the same command after an interruption.
#
# Usage:
#   bash run_critic_self_training.sh --model Qwen/Qwen3.5-2B --precision 4bit --base q2b_4bit_critic_p500 \
#        [--pool_limit 500] [--eval_limit 100] [--verdicts 2] [--critic_tokens 704] [--arms "vote critic critic_reason"] \
#        [--batch_size 64] [--train_steps 200] [--threshold 0.75] [--stop_after_labels] [--results_dir DIR]
#        [--micro_batch_size 2] [--gradient_checkpointing] [--chunked_loss] [--train_seed 0]
#        [--rounds 1] [--round_critic self|base]
#
# --rounds N > 1 continues every arm to round N. Runs that already have round 1 keep it (every stage resumes),
# so the same command with --rounds 3 extends finished one-round experiments.
#
# --train_seed k > 0 trains the arms again with another seed (adapter init, batch order, dropout) under
# experiment <base>_<arm>_s<k>, sharing round 0 and the critic run; the report shows each seed as its own row.
# Two seeds of the same arm measure training noise, the floor any difference between arms must clear.
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
TRAIN_SEED=0
CHECKPOINTING=()
CHUNKED=()
THRESHOLD=0.75
VERDICTS=2
CRITIC="critic_forced"
CRITIC_TOKENS=704
MIN_VERDICT_RATE=0.85
ARMS="vote critic critic_reason"
STOP_AFTER_LABELS=0
ROUNDS=1
ROUND_CRITIC="self"
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
        --train_seed) TRAIN_SEED="$2"; shift ;;
        --gradient_checkpointing) CHECKPOINTING=(--gradient_checkpointing) ;;
        --chunked_loss) CHUNKED=(--chunked_loss) ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --verdicts) VERDICTS="$2"; shift ;;
        --critic_tokens) CRITIC_TOKENS="$2"; shift ;;
        --min_verdict_rate) MIN_VERDICT_RATE="$2"; shift ;;
        --arms) ARMS="$2"; shift ;;
        --stop_after_labels) STOP_AFTER_LABELS=1 ;;
        --rounds) ROUNDS="$2"; shift ;;
        --round_critic) ROUND_CRITIC="$2"; shift ;;
        --results_dir|--output_dir) RESULTS_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "${BASE}" ]; then echo "--base is required"; exit 1; fi
if ! [[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]]; then echo "--rounds must be 1 or more, got ${ROUNDS}"; exit 1; fi
if [[ "${ROUND_CRITIC}" != "self" && "${ROUND_CRITIC}" != "base" ]]; then
    echo "--round_critic must be self or base, got ${ROUND_CRITIC}"; exit 1
fi
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

# adapter_of EXPERIMENT ROUND: where the adapter trained in that round is saved.
adapter_of() {
  python -c "
import sys
from src.experiment_d.run_files import adapter_dir
results_dir, experiment, round_index, model, precision = sys.argv[1:]
print(adapter_dir(results_dir, experiment, int(round_index), model, precision))
" "${RESULTS_DIR}" "$1" "$2" "${MODEL}" "${PRECISION}"
}

# critic_tag EXPERIMENT ROUND: tag of the critic run over that round's pool. After round 0 with --round_critic self
# the round's own model judges, and the tag names its adapter (signals.with_judge), as run_signals.py writes it.
critic_tag() {
  if [ "$2" -eq 0 ] || [ "${ROUND_CRITIC}" = "base" ]; then echo "${TAG}"; return; fi
  python -c "
import sys
from src.experiment_d.signals import with_judge
print(with_judge(sys.argv[1], sys.argv[2]))
" "${TAG}" "$(adapter_of "$1" "$2")"
}

# critic_over_pool EXPERIMENT ROUND: the forced critic judges that round's pool answers (contested questions only).
critic_over_pool() {
  local judge_args=()
  if [ "$2" -gt 0 ] && [ "${ROUND_CRITIC}" = "self" ]; then judge_args=(--adapter "$(adapter_of "$1" "$2")"); fi
  python scripts/run_signals.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" --precision "${PRECISION}" \
    --dataset "${POOL_DATASET}" --n_samples "${N_SAMPLES}" --experiment "$1" --round "$2" \
    --question_set first --questions "${POOL_LIMIT}" --critics "${CRITIC}" --verdicts "${VERDICTS}" \
    --max_new_tokens "${CRITIC_TOKENS}" --no_mc --no_token_signals --min_verdict_rate "${MIN_VERDICT_RATE}" \
    --batch_size "${BATCH_SIZE}" "${judge_args[@]}"
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
critic_over_pool "${BASE}" 0

banner "Labels on the pool (reference answers read for this report only)"
report --labels_only
if [ "${STOP_AFTER_LABELS}" = 1 ]; then echo "Stopped after the label report (--stop_after_labels)."; exit 0; fi

SEED_SUFFIX=""
if [ "${TRAIN_SEED}" != 0 ]; then SEED_SUFFIX="_s${TRAIN_SEED}"; fi
for ARM in ${ARMS}; do
  case "${ARM}" in vote|critic|critic_reason) ;; *) echo "Unknown arm ${ARM}"; exit 1 ;; esac
  EXPERIMENT="${BASE}_${ARM}${SEED_SUFFIX}"
  for ROUND in $(seq 1 "${ROUNDS}"); do
    PREVIOUS=$((ROUND - 1))
    # Round 1 learns from the shared round 0; later rounds from the arm's own previous round.
    if [ "${ROUND}" -eq 1 ]; then POOL_EXPERIMENT="${BASE}"; else POOL_EXPERIMENT="${EXPERIMENT}"; fi
    banner "Arm ${ARM}${SEED_SUFFIX}, round ${ROUND} of ${ROUNDS}: train on round ${PREVIOUS}'s pool, answer the test questions"
    if [ "${ROUND}" -gt 1 ]; then
      generate "${EXPERIMENT}" "${PREVIOUS}" "${POOL_DATASET}" "${POOL_LIMIT}"
      if [ "${ARM}" != vote ]; then critic_over_pool "${EXPERIMENT}" "${PREVIOUS}"; fi
    fi
    arm_args=()
    if [ "${ARM}" != vote ]; then
      arm_args=(--labels critic --signals_tag "$(critic_tag "${POOL_EXPERIMENT}" "${PREVIOUS}")" --critic "${CRITIC}")
      if [ "${ARM}" = critic_reason ]; then arm_args+=(--critiques); fi
    fi
    python scripts/train_self_labels.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" --precision "${PRECISION}" \
      --dataset "${POOL_DATASET}" --eval_dataset "${EVAL_DATASET}" --n_samples "${N_SAMPLES}" \
      --experiment "${EXPERIMENT}" --pool_experiment "${POOL_EXPERIMENT}" --round "${ROUND}" --threshold "${THRESHOLD}" \
      --train_steps "${TRAIN_STEPS}" --micro_batch_size "${MICRO_BATCH_SIZE}" --end_format chat \
      --train_seed "${TRAIN_SEED}" "${CHECKPOINTING[@]}" "${CHUNKED[@]}" "${arm_args[@]}"
    generate "${EXPERIMENT}" "${ROUND}" "${EVAL_DATASET}" "${EVAL_LIMIT}"
  done
done

banner "Report"
report

banner "Critic self-training complete!"
