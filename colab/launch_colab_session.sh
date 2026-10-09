#!/usr/bin/env bash
# Run from your own machine. Orchestrates one full Colab run:
#   new session -> mount Drive -> colab_entrypoint.py (clones the repo and runs
#   run_pipeline.sh there) -> logs/results land on Drive -> always stop the
#   session so credits stop, even if the run fails.
#
# Usage:
#   EXPERIMENT_ARGS="--model Qwen/Qwen3.5-0.8B --precision 4bit --dataset gsm8k --n_samples 8 --limit 0" \
#   GPU=L4 HF_TOKEN=hf_xxx bash colab/launch_colab_session.sh
set -uo pipefail

SESSION="${SESSION:-exp}"
GPU="${GPU:-L4}"
EXPERIMENT_ARGS="${EXPERIMENT_ARGS:---model meta-llama/Llama-3.2-3B-Instruct --precision 4bit --dataset both --n_samples 8 --batch_size 96 --limit 100}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "$HF_TOKEN" ]]; then
  echo "WARNING: HF_TOKEN is not set in your local shell -- gated HF models/datasets will fail remotely."
  echo "         export HF_TOKEN=hf_xxx before running this script if your model needs it."
fi

cleanup() {
  echo "Stopping session '${SESSION}' (this releases the VM and stops billing)..."
  colab stop -s "${SESSION}" || true
}
trap cleanup EXIT

echo "Provisioning ${GPU} session '${SESSION}'..."
colab new -s "${SESSION}" --gpu "${GPU}"

echo "Mounting Google Drive on this session..."
colab drivemount -s "${SESSION}"

# Bypass google.colab.userdata entirely: it expects a notebook frontend to
# grant secret access, which doesn't exist in a CLI session and will raise
# NotebookAccessError immediately. Push HF_TOKEN as a plain env var instead --
# built locally and piped in so it's never echoed to the terminal or logs.
echo "Setting experiment args and HF token on the session..."
python3 - "$EXPERIMENT_ARGS" "$HF_TOKEN" <<'PYEOF' | colab exec -s "${SESSION}"
import sys
exp_args, hf_token = sys.argv[1], sys.argv[2]
print(f'import os; os.environ["EXPERIMENT_ARGS"] = {exp_args!r}')
if hf_token:
    print(f'os.environ["HF_TOKEN"] = {hf_token!r}')
PYEOF

echo "Running colab_entrypoint.py (clones the repo, logs + results go to Drive)..."
colab exec -s "${SESSION}" -f "$(dirname "$0")/colab_entrypoint.py"

echo "Experiment finished."
echo "  Logs and results are on Drive under Experiment_D_Results/"
echo "  You can check them from any device via Google Drive."
