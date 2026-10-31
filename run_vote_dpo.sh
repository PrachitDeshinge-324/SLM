#!/usr/bin/env bash
# One round of vote-based preference training (DPO) for ONE model at ONE precision. No critic, no verifier.
#
#   1. train     scripts/train_preference_pairs.py on the pool answers an earlier run already wrote
#                (--pool_experiment, round 0). The vote labels a question when its top answer has >= 4 of 8
#                votes (ties skipped); chosen = a finished solution reaching it, rejected = a finished
#                solution reaching another answer; up to 3 distinct pairs per question, one per epoch.
#                Loss per pair: vote-margin weight * (length-normalised DPO + NLL of the chosen solution),
#                end tokens left out of the DPO score. No reference answer is read anywhere.
#   2. diagnose  training log: does the chosen solution's log-prob fall (displacement), do the end tokens?
#   3. adapter   the adapter answers the first --eval_limit test questions (adapter applied, not merged)
#   4. base      the untrained model answers the same questions with the same settings on the same GPU
#                (skipped when --base names an experiment whose round 0 already holds them)
#   5. report    single sample, majority vote, cut-off rate and accuracy when finished, adapter vs base
#                (reference answers are read here, after the fact, and nowhere else)
#
# Every stage resumes: rerun the same command after an interruption. The pool must already exist at
#   <results_dir>/self_training/<pool_experiment>/round0/gsm8k_train/generator/.../*_n8.jsonl
# (the 500 x 8 answers of the untrained model, e.g. from run_critic_self_training.sh). Nothing is regenerated.
# Changes must be pushed before a Colab run, which clones the repository.
#
# Rough cost on a T4 at Qwen3.5-2B 4-bit, estimated from earlier run files (training time was never logged):
# training ~1 h, 200 x 8 test answers ~1.2-1.5 h for the adapter and the same for the base.
#
# Usage:
#   bash run_vote_dpo.sh [--model Qwen/Qwen3.5-2B] [--precision 4bit] [--pool_experiment q2b_4bit_critic_p500] \
#        [--experiment q2b_4bit_vote_dpo] [--base EXPERIMENT] [--eval_limit 200] [--batch_size 64] \
#        [--train_seed 0] [--epochs 3] [--pairs_per_question 3] [--threshold 0.5] [--learning_rate 1e-5] \
#        [--beta 27] [--nll_weight 1.0] [--weighting margin|none] [--micro_batch_size 2] \
#        [--curriculum] [--stop_after_train] [--dry_run] [--results_dir DIR]
#
# --curriculum trains under <experiment>_curriculum: one pass over the questions with >= 6/8 votes, then >= 5/8,
# then >= 4/8 (shuffled within a stage), 397 pair-epochs at 2B 4-bit instead of 516. It shares round 0 with the
# plain arm and defaults to --weighting none: margin weights on top would leave the 4-5/8 questions 21% of the
# training instead of 31% (47% in the plain arm), and those are the questions the model does not already solve.
# --train_seed k > 0 trains again under <experiment>_s<k>; round 0 (the base answers) is shared, so a seed
# costs training + adapter answers only.
# --base EXPERIMENT compares with that experiment's round 0 instead of generating one here. Use it only if
# its test answers match (same questions, 8 samples, 512 tokens, ideally the same GPU and batch size).
# --beta 27 is a sequence-level beta of about 0.1 for ~270-token solutions (the loss uses per-token means).
# --stop_after_train ends after the training diagnostics, so a bad adapter costs ~1 hour, not the evaluation.
set -e

MODEL="Qwen/Qwen3.5-2B"
PRECISION="4bit"
POOL_EXPERIMENT="q2b_4bit_critic_p500"
EXPERIMENT="q2b_4bit_vote_dpo"
BASE=""
EVAL_LIMIT=200
N_SAMPLES=8
BATCH_SIZE=64
MAX_NEW_TOKENS=512
TRAIN_SEED=0
THRESHOLD=0.5
EPOCHS=3
PAIRS_PER_QUESTION=3
LEARNING_RATE=1e-5
BETA=27
NLL_WEIGHT=1.0
WEIGHTING=""
CURRICULUM=0
CURRICULUM_STAGES="0.75,0.625,0.5"
TRAIN_BATCH_SIZE=8
MICRO_BATCH_SIZE=2
POOL_LIMIT=500
POOL_DATASET="gsm8k_train"
EVAL_DATASET="gsm8k"
STOP_AFTER_TRAIN=0
DRY_RUN=0
RESULTS_DIR="/content/drive/MyDrive/results"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --precision) PRECISION="$2"; shift ;;
        --pool_experiment) POOL_EXPERIMENT="$2"; shift ;;
        --experiment) EXPERIMENT="$2"; shift ;;
        --base) BASE="$2"; shift ;;
        --eval_limit) EVAL_LIMIT="$2"; shift ;;
        --n_samples) N_SAMPLES="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift ;;
        --train_seed) TRAIN_SEED="$2"; shift ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --pairs_per_question) PAIRS_PER_QUESTION="$2"; shift ;;
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        --beta) BETA="$2"; shift ;;
        --nll_weight) NLL_WEIGHT="$2"; shift ;;
        --weighting) WEIGHTING="$2"; shift ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift ;;
        --micro_batch_size) MICRO_BATCH_SIZE="$2"; shift ;;
        --pool_limit) POOL_LIMIT="$2"; shift ;;
        --curriculum) CURRICULUM=1 ;;
        --curriculum_stages) CURRICULUM_STAGES="$2"; shift ;;
        --stop_after_train) STOP_AFTER_TRAIN=1 ;;
        --dry_run) DRY_RUN=1 ;;
        --results_dir|--output_dir) RESULTS_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "${WEIGHTING}" ]; then
  if [ "${CURRICULUM}" = 1 ]; then WEIGHTING="none"; else WEIGHTING="margin"; fi
fi
if [[ "${WEIGHTING}" != "margin" && "${WEIGHTING}" != "none" ]]; then
    echo "--weighting must be margin or none, got ${WEIGHTING}"; exit 1
fi
if ! [[ "${TRAIN_SEED}" =~ ^[0-9]+$ ]]; then echo "--train_seed must be a number, got ${TRAIN_SEED}"; exit 1; fi

export PYTHONPATH=.
# Fewer out-of-memory failures from fragmentation (memory reserved but unusable); changes no result.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -z "${PYTHON:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif [ -x "/opt/homebrew/Caskroom/miniforge/base/envs/SLM/bin/python" ]; then
    PYTHON="/opt/homebrew/Caskroom/miniforge/base/envs/SLM/bin/python"
  else
    PYTHON="python3"
  fi
fi

SAFE_MODEL="${MODEL//\//_}"
SEED_SUFFIX=""
if [ "${TRAIN_SEED}" != 0 ]; then SEED_SUFFIX="_s${TRAIN_SEED}"; fi
CURRICULUM_ARGS=()
ARM_SUFFIX=""
if [ "${CURRICULUM}" = 1 ]; then CURRICULUM_ARGS=(--curriculum "${CURRICULUM_STAGES}"); ARM_SUFFIX="_curriculum"; fi
ARM="${EXPERIMENT}${ARM_SUFFIX}${SEED_SUFFIX}"
GENERATE_BASE=0
if [ -z "${BASE}" ]; then BASE="${EXPERIMENT}"; GENERATE_BASE=1; fi

round_dir_of() { echo "${RESULTS_DIR}/self_training/$1/round$2"; }
ADAPTER="$(round_dir_of "${ARM}" 1)/adapters/${SAFE_MODEL}_${PRECISION}"
POOL_FILE="$(round_dir_of "${POOL_EXPERIMENT}" 0)/${POOL_DATASET}/generator/${SAFE_MODEL}/${SAFE_MODEL}_${PRECISION}_${POOL_DATASET}_n${N_SAMPLES}.jsonl"

banner() { echo "======================================================"; echo " $1"; echo "======================================================"; }
# run COMMAND...: executes it, or only prints it with --dry_run.
run() { if [ "${DRY_RUN}" = 1 ]; then echo "+ $*"; else "$@"; fi; }

# generate EXPERIMENT ROUND [ADAPTER]: that round answers the first EVAL_LIMIT test questions.
generate_eval() {
  local exp="$1" rnd="$2" adapter="$3"
  local round_dir gen_file adapter_args=()
  round_dir=$(round_dir_of "${exp}" "${rnd}")
  gen_file="${round_dir}/${EVAL_DATASET}/generator/${SAFE_MODEL}/${SAFE_MODEL}_${PRECISION}_${EVAL_DATASET}_n${N_SAMPLES}.jsonl"
  if [ -f "${gen_file}" ] && [ "$(wc -l < "${gen_file}" 2>/dev/null || echo 0)" -gt "${EVAL_LIMIT}" ]; then
    echo "  [Cached] ${EVAL_LIMIT} test questions already answered in ${round_dir}; skipping generation."
    return
  fi
  if [ -n "${adapter}" ]; then adapter_args=(--adapter "${adapter}"); fi
  run "${PYTHON}" scripts/generate_answers.py --model "${MODEL}" --precision "${PRECISION}" --dataset "${EVAL_DATASET}" \
    --n_samples "${N_SAMPLES}" --batch_size "${BATCH_SIZE}" --results_dir "${round_dir}" --limit "${EVAL_LIMIT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" --repetition_penalty 1.0 "${adapter_args[@]}"
}

banner "Pool check"
if [ ! -f "${POOL_FILE}" ] || [ "$(wc -l < "${POOL_FILE}")" -le "${POOL_LIMIT}" ]; then
  if [ "${DRY_RUN}" = 1 ]; then
    echo "  (dry run) pool answers not complete at ${POOL_FILE}"
  else
    echo "No complete pool run (${POOL_LIMIT} questions x ${N_SAMPLES} samples) at:"
    echo "  ${POOL_FILE}"
    echo "Point --results_dir / --pool_experiment at the finished round 0, e.g. from run_critic_self_training.sh."
    exit 1
  fi
else
  echo "  Pool: ${POOL_FILE}"
fi

banner "1. Train: DPO + chosen NLL on vote pairs (agreement >= ${THRESHOLD}, ties skipped), arm ${ARM}, seed ${TRAIN_SEED}"
run "${PYTHON}" scripts/train_preference_pairs.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" --precision "${PRECISION}" \
  --dataset "${POOL_DATASET}" --eval_dataset "${EVAL_DATASET}" --n_samples "${N_SAMPLES}" \
  --experiment "${ARM}" --pool_experiment "${POOL_EXPERIMENT}" --round 1 \
  --threshold "${THRESHOLD}" --pairing similar_length --skip_ties --pairs_per_question "${PAIRS_PER_QUESTION}" \
  --epochs "${EPOCHS}" --train_batch_size "${TRAIN_BATCH_SIZE}" --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" --beta "${BETA}" --nll_weight "${NLL_WEIGHT}" --weighting "${WEIGHTING}" \
  --preference_end exclude --end_format chat --gradient_checkpointing --chunked_loss --train_seed "${TRAIN_SEED}" \
  "${CURRICULUM_ARGS[@]}"

banner "2. Training diagnostics"
if [ "${DRY_RUN}" = 1 ]; then
  echo "  (dry run) would read ${ADAPTER}/training_log.json"
else
  "${PYTHON}" - "${ADAPTER}/training_log.json" <<'PY'
import json
import sys

log = json.load(open(sys.argv[1]))
mean = lambda values: sum(values) / len(values)
print(f"questions {log['labelled_questions']}, distinct pairs {log['selected_pairs']}, pair-epochs {log['pair_epochs']}, "
      f"steps {log['train_steps']}, mean weight {log['mean_weight']:.2f}, skipped too long {log['skipped_too_long']}, "
      f"peak GPU memory {log.get('peak_gpu_memory_gb')} GB")
print(f"chosen longer in {100 * log['chosen_longer_share']:.0f}% of pairs; mean lengths chosen {log['mean_chosen_length']:.0f} "
      f"/ rejected {log['mean_rejected_length']:.0f}; agreement counts {log['agreement_counts']}")
print("per-step means, first 5 steps -> last 5 steps:")
for key in ("losses", "margins", "preferred", "nll", "chosen_shift", "rejected_shift", "end_chosen", "end_rejected"):
    values = log[key]
    print(f"  {key:14s} {mean(values[:5]):+.4f} -> {mean(values[-5:]):+.4f}")
warnings = []
if mean(log["chosen_shift"][-5:]) < -0.05:
    warnings.append("chosen log-prob fell below the base's: likelihood displacement")
if mean(log["end_chosen"][-5:]) < mean(log["end_chosen"][:5]) - 0.5:
    warnings.append("end-token log-prob fell: the model is learning to stop later (runaway risk)")
if mean(log["margins"][-5:]) <= 0:
    warnings.append("margin never became positive: the model did not learn the preference")
for warning in warnings:
    print("WARNING:", warning)
if not warnings:
    print("No training warning. The test answers decide whether this helped (cut-off rate first).")
PY
fi
if [ "${STOP_AFTER_TRAIN}" = 1 ]; then echo "Stopped after training (--stop_after_train)."; exit 0; fi

banner "3. The adapter answers the first ${EVAL_LIMIT} test questions"
generate_eval "${ARM}" 1 "${ADAPTER}"

if [ "${GENERATE_BASE}" = 1 ]; then
  banner "4. The untrained model answers the same questions (matched settings, same GPU)"
  generate_eval "${BASE}" 0 ""
else
  banner "4. Base: round 0 of ${BASE} (not regenerated)"
fi

banner "5. Report: adapter vs base on ${EVAL_LIMIT} test questions (reference answers read here only)"
run "${PYTHON}" scripts/report_critic_self_training.py --results_dir "${RESULTS_DIR}" --model "${MODEL}" \
  --precision "${PRECISION}" --dataset "${EVAL_DATASET}" --n_samples "${N_SAMPLES}" --base "${BASE}" \
  --threshold "${THRESHOLD}" --score_arm "${ARM}" --score_round 1
