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
CHUNK_QUESTIONS=32
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
        --chunk_questions) CHUNK_QUESTIONS="$2"; shift ;;
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

# 16-bit LoRA on 2B+ models requires gradient checkpointing and chunked loss to stay within 24GB VRAM
if [ "${PRECISION}" = "16bit" ]; then
  if [ ${#CHECKPOINTING[@]} -eq 0 ]; then
    CHECKPOINTING=(--gradient_checkpointing)
  fi
  if [ ${#CHUNKED[@]} -eq 0 ]; then
    CHUNKED=(--chunked_loss)
  fi
fi

if [ -z "${PYTHON:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif [ -x "/opt/homebrew/Caskroom/miniforge/base/envs/SLM/bin/python" ]; then
    PYTHON="/opt/homebrew/Caskroom/miniforge/base/envs/SLM/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    PYTHON="python"
  fi
fi

banner() { echo "======================================================"; echo " $1"; echo "======================================================"; }

SAFE_MODEL="${MODEL//\//_}"

round_dir_of() {
  local exp="$1" rnd="$2"
  echo "${RESULTS_DIR}/self_training/${exp}/round${rnd}"
}

# adapter_of EXPERIMENT ROUND: where the adapter trained in that round is saved.
adapter_of() {
  local exp="$1" rnd="$2"
  echo "${RESULTS_DIR}/self_training/${exp}/round${rnd}/adapters/${SAFE_MODEL}_${PRECISION}"
}

# critic_tag EXPERIMENT ROUND: tag of the critic run over that round's pool. After round 0 with --round_critic self
# the round's own model judges, and the tag names its adapter (signals.with_judge), as run_signals.py writes it.
critic_tag() {
  local exp="$1" rnd="$2"
  if [ "$rnd" -eq 0 ] || [ "${ROUND_CRITIC}" = "base" ]; then
    echo "${TAG}"
  else
    local safe_exp
    safe_exp="$(echo "${exp}" | sed 's/[^A-Za-z0-9_-]/_/g')"
    echo "${TAG}_by_${safe_exp}_r${rnd}"
  fi
}

# generate EXPERIMENT ROUND DATASET LIMIT: that round's model (with its adapter after round 0) answers the questions.
generate() {
  local exp="$1" rnd="$2" dataset="$3" limit="$4"
  local round_dir
  round_dir=$(round_dir_of "${exp}" "${rnd}")
  local adapter
  adapter=$(adapter_of "${exp}" "${rnd}")
  local gen_file="${round_dir}/${dataset}/generator/${SAFE_MODEL}/${SAFE_MODEL}_${PRECISION}_${dataset}_n${N_SAMPLES}.jsonl"

  # Fast cache check: if output file exists and has >= limit questions
  if [ -f "${gen_file}" ]; then
    local lines
    lines=$(wc -l < "${gen_file}" 2>/dev/null || echo 0)
    if [ "${lines}" -gt "${limit}" ] && [ "${limit}" -gt 0 ]; then
      echo "  [Cached] ${dataset} (${limit} questions) already complete in ${round_dir}; skipping generation."
      return
    fi
  fi

  local adapter_args=()
  if [ "${rnd}" -gt 0 ]; then adapter_args=(--adapter "${adapter}"); fi
  "${PYTHON}" scripts/generate_answers.py --model "${MODEL}" --precision "${PRECISION}" --dataset "${dataset}" \
    --n_samples "${N_SAMPLES}" --batch_size "${BATCH_SIZE}" --results_dir "${round_dir}" --limit "${limit}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --repetition_penalty 1.0 "${adapter_args[@]}"
}

# critic_over_pool EXPERIMENT ROUND: the forced critic judges that round's pool answers (contested questions only).
critic_over_pool() {
  local exp="$1" rnd="$2"
  local ctag
  ctag=$(critic_tag "${exp}" "${rnd}")
  local round_dir
  round_dir=$(round_dir_of "${exp}" "${rnd}")
  local sig_file="${round_dir}/${POOL_DATASET}/signals/${SAFE_MODEL}/signals_${ctag}_${SAFE_MODEL}_${PRECISION}_${POOL_DATASET}_n${N_SAMPLES}.jsonl"

  # Fast cache check: if signals file exists and has >= (POOL_LIMIT * N_SAMPLES) records
  if [ -f "${sig_file}" ]; then
    local lines
    lines=$(wc -l < "${sig_file}" 2>/dev/null || echo 0)
    local expected=$(( POOL_LIMIT * N_SAMPLES ))
    if [ "${lines}" -gt "${expected}" ] && [ "${POOL_LIMIT}" -gt 0 ]; then
      echo "  [Cached] Critic run (${ctag}) already complete over ${POOL_LIMIT} questions; skipping."
      return
    fi
  fi

  local judge_args=()
  if [ "${rnd}" -gt 0 ] && [ "${ROUND_CRITIC}" = "self" ]; then judge_args=(--adapter "$(adapter_of "${exp}" "${rnd}")"); fi
  "${PYTHON}" scripts/run_signals.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" --precision "${PRECISION}" \
    --dataset "${POOL_DATASET}" --n_samples "${N_SAMPLES}" --experiment "${exp}" --round "${rnd}" \
    --question_set first --questions "${POOL_LIMIT}" --critics "${CRITIC}" --verdicts "${VERDICTS}" \
    --max_new_tokens "${CRITIC_TOKENS}" --no_mc --no_token_signals --min_verdict_rate "${MIN_VERDICT_RATE}" \
    --chunk_questions "${CHUNK_QUESTIONS}" --batch_size "${BATCH_SIZE}" "${judge_args[@]}"
}

train_round_adapter() {
  local exp="$1" pool_exp="$2" rnd="$3" arm="$4"
  shift 4
  local log_file
  log_file="$(adapter_of "${exp}" "${rnd}")/training_log.json"
  if [ -f "${log_file}" ]; then
    echo ">>> Step 3/4: [Cached] Round ${rnd} adapter for arm '${arm}' already trained; skipping."
    return
  fi

  echo ">>> Step 3/4: Training Round ${rnd} adapter for arm '${arm}' (${TRAIN_STEPS} steps)..."
  "${PYTHON}" scripts/train_self_labels.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" --precision "${PRECISION}" \
    --dataset "${POOL_DATASET}" --eval_dataset "${EVAL_DATASET}" --n_samples "${N_SAMPLES}" \
    --experiment "${exp}" --pool_experiment "${pool_exp}" --round "${rnd}" --threshold "${THRESHOLD}" \
    --train_steps "${TRAIN_STEPS}" --micro_batch_size "${MICRO_BATCH_SIZE}" --end_format chat \
    --train_seed "${TRAIN_SEED}" "${CHECKPOINTING[@]}" "${CHUNKED[@]}" "$@"
}

score_eval() {
  local exp="$1" rnd="$2"
  "${PYTHON}" scripts/report_critic_self_training.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" \
    --precision "${PRECISION}" --dataset "${EVAL_DATASET}" --base "${BASE}" --threshold "${THRESHOLD}" \
    --score_arm "${exp}" --score_round "${rnd}"
}

report() {
  "${PYTHON}" scripts/report_critic_self_training.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" \
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

for ROUND in $(seq 1 "${ROUNDS}"); do
  PREVIOUS=$((ROUND - 1))
  banner "==================== ROUND ${ROUND} OF ${ROUNDS} ===================="

  for ARM in ${ARMS}; do
    case "${ARM}" in vote|critic|critic_reason) ;; *) echo "Unknown arm ${ARM}"; exit 1 ;; esac
    EXPERIMENT="${BASE}_${ARM}${SEED_SUFFIX}"
    # Round 1 learns from the shared round 0; later rounds from the arm's own previous round.
    if [ "${ROUND}" -eq 1 ]; then POOL_EXPERIMENT="${BASE}"; else POOL_EXPERIMENT="${EXPERIMENT}"; fi

    echo ""
    echo "----------------------------------------------------------------------"
    echo " Arm: ${ARM}${SEED_SUFFIX} | Round: ${ROUND}/${ROUNDS}"
    echo "----------------------------------------------------------------------"

    if [ "${ROUND}" -gt 1 ]; then
      echo ">>> Step 1/4: Generating pool answers (${POOL_LIMIT} questions) with Round ${PREVIOUS} model..."
      generate "${EXPERIMENT}" "${PREVIOUS}" "${POOL_DATASET}" "${POOL_LIMIT}"

      if [ "${ARM}" != vote ]; then
        echo ">>> Step 2/4: Running critic verifier over Round ${PREVIOUS} pool answers ($(critic_tag "${EXPERIMENT}" "${PREVIOUS}"))..."
        critic_over_pool "${EXPERIMENT}" "${PREVIOUS}"
      fi
    fi

    arm_args=()
    if [ "${ARM}" != vote ]; then
      arm_args=(--labels critic --signals_tag "$(critic_tag "${POOL_EXPERIMENT}" "${PREVIOUS}")" --critic "${CRITIC}")
      if [ "${ARM}" = critic_reason ]; then arm_args+=(--critiques); fi
    fi

    train_round_adapter "${EXPERIMENT}" "${POOL_EXPERIMENT}" "${ROUND}" "${ARM}" "${arm_args[@]}"

    echo ">>> Step 4/4: Evaluating Round ${ROUND} on test questions (${EVAL_DATASET}, ${EVAL_LIMIT} questions)..."
    generate "${EXPERIMENT}" "${ROUND}" "${EVAL_DATASET}" "${EVAL_LIMIT}"

    score_eval "${EXPERIMENT}" "${ROUND}"
  done

  banner "Comparative Report after Round ${ROUND} of ${ROUNDS}"
  report
done

banner "Critic self-training complete!"
