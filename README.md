# Experiment D: does quantization break self-improvement in small language models?

A small model answers each question several times, the answers are put to a vote, and the model is asked to judge its own solutions. This is done at 16-bit, 8-bit and 4-bit weight precision to see what quantization does to those self-judgment signals.

Findings, decisions and open questions: [RESEARCH_DOSSIER.md](RESEARCH_DOSSIER.md). Paper-by-paper literature review: [RESEARCH_DOSSIER_PAPERS.md](RESEARCH_DOSSIER_PAPERS.md).

## Layout

| Path | What it holds |
|---|---|
| `run_pipeline.sh` | Runs the three pipeline stages for one model, one precision, one dataset |
| `run_self_training.sh` | Label-free self-training rounds for one model at one precision (section below) |
| `scripts/` | One file per pipeline stage and per report (table below) |
| `src/experiment_d/` | The library the scripts are built from (table below) |
| `colab/` | Launching a run on Google Colab from your own machine |
| `notebooks/run_pipeline_on_colab.ipynb` | The same pipeline, cell by cell, for a manual Colab session |
| `notebooks/self_training_4bit_pilot.ipynb` | The self-training pilot for a manual Colab session |
| `analysis/` | Dated analyses behind the dossier, with their saved outputs |
| `tests/` | Unit tests |
| `data/` | GSM8K and CommonsenseQA files; MATH-500 and SVAMP are downloaded |
| `results/` | Run files and reports (not in git; synced with Google Drive) |
| `web/` | Project site (static; the result pages are generated from `results/` by `scripts/export_web_data.py`) |
| `Docs/` | Original proposal and mentor documents (not in git) |
| `archive/` | Superseded scripts, documents, notebooks and duplicates (not in git) |

### Pipeline scripts

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1 | `scripts/generate_answers.py` | a benchmark | `results/<dataset>/generator/<model>/<run>.jsonl` |
| 2 | `scripts/build_verifier_dataset.py` | stage 1 file | `results/<dataset>/verifier_dataset/<model>/verifier_<run>.jsonl` |
| 3 | `scripts/run_verifier.py` | stage 2 file | `results/<dataset>/verifier_output/<model>/verifier_output_<run>.jsonl` |

`<run>` is `<model>_<precision>_<dataset>_n<samples>`, for example `Qwen_Qwen3.5-0.8B_4bit_gsm8k_n8`.

### Report scripts

| Script | Output |
|---|---|
| `scripts/report_generator_metrics.py` | `results/<dataset>/generator/metrics/`: one CSV per dataset, calibration and accuracy plots |
| `scripts/report_verifier_metrics.py` | `results/<dataset>/verifier_output/metrics/`: one CSV per model, confusion matrices |
| `scripts/report_verifier_shared_questions.py` | `results/standardized_verifier_metrics/`: all models on the same 100 questions |
| `scripts/export_web_data.py` | `web/assets/data/results.js`: every number on the Phase 1, 2 and 3 web pages, re-scored from `results/`. Verifier rows cover a fixed 100 questions and all their traces (truncated ones kept as wrong), so they differ from the two verifier reports. Phase 3 also reads `results/self_training/` (the experiments listed in `SELF_TRAINING_EXPERIMENTS` in `web_export.py`; missing ones show as not synced) |

Reports never trust the metrics stored inside a run file. They re-score every run from its raw outputs with the current code, drop truncated samples, and compare precisions on the questions they share.

### Self-training

`run_self_training.sh` asks whether a model can improve from its own answers, with no labels:

| Round | What happens | Written under `results/self_training/<experiment>/round<k>/` |
|---|---|---|
| 0 | The untrained model answers the test questions and an unlabelled pool of train questions | `gsm8k/generator/...`, `gsm8k_train/generator/...` |
| k ≥ 1 | `scripts/train_self_labels.py` labels the round k−1 pool answers by majority vote, keeps questions with agreement ≥ 0.75 and trains a fresh LoRA adapter on those solutions; the model with that adapter answers the test questions and the pool again | `adapters/<model>_<precision>/`, then the same two run files |

With `--objective dpo` round k instead runs `scripts/train_preference_pairs.py`: for each pool question with agreement ≥ 5/8 it pairs a complete majority solution (chosen) with a complete minority one (rejected), scores both once with the frozen base as the reference, and trains the adapter with length-normalised DPO (per-token mean log-ratio, β = 2). `--pairing similar_length` (default) takes the pair closest in length so length does not separate chosen from rejected; `--pairing longest_shortest` is the length-confounded ablation. The pairs and their reference scores are saved next to the adapter. `scripts/check_dpo_overfit.py` overfits one pair on the real model as an end-to-end check of the loss.

`scripts/report_self_training.py` compares every round with round 0 on the same test questions and writes `report_<model>_<precision>.txt` next to the rounds.

The pool (`gsm8k_train`) is loaded without its answers and without the question the prompt uses as its worked example, and training refuses to start if a pool question is also a test question. The test questions are only ever evaluated. The report's label-accuracy column is the one place the train answers are read, after the run.

### Library modules (`src/experiment_d/`)

| Module | Responsibility |
|---|---|
| `benchmarks.py` | Load GSM8K, MATH-500, SVAMP, CommonsenseQA into one question format |
| `prompts.py` | Build the chat prompt for each benchmark |
| `model_loading.py` | Load a model at 16-bit, 8-bit or 4-bit |
| `sampling.py` | Sample N answers per prompt |
| `answer_extraction.py` | Pull the final answer out of a generated solution |
| `answer_matching.py` | Decide whether two answers are equal (numbers, fractions, LaTeX, symbolic) |
| `self_consistency.py` | Majority vote and agreement over the N answers |
| `self_training.py` | Pick a model's own vote-filtered solutions as training text, or (majority, minority) preference pairs; check the pool shares no question with the test set |
| `lora_training.py` | Train a LoRA adapter on a frozen base at any precision, by fine-tuning or length-normalised DPO |
| `verifier.py` | The verifier prompt and the parser for its verdict |
| `rescoring.py` | Rebuild metrics and labels from stored raw outputs |
| `verifier_metrics.py` | Verifier metrics (false positive rate, MCC, guided majority) from relabelled traces |
| `web_export.py` | Everything the web result pages show, with paired intervals against 16-bit |
| `run_files.py` | Names and locations of run files |
| `evaluation_stats.py` | AUROC, bootstrap intervals, signal detection, random-effects pooling |
| `warning_filters.py` | Silence two uninformative library warnings |

## Running

All commands are run from the repository root.

```bash
# one model, one precision, one dataset, all three stages (needs a CUDA GPU for 8-bit and 4-bit)
bash run_pipeline.sh --model Qwen/Qwen3.5-0.8B --precision 4bit --dataset gsm8k \
     --n_samples 8 --batch_size 64 --limit 0 --results_dir results
```

```bash
# the same on Google Colab, launched from your machine
EXPERIMENT_ARGS="--model Qwen/Qwen3.5-0.8B --precision 4bit --dataset gsm8k --n_samples 8 --limit 0" \
GPU=L4 HF_TOKEN=hf_xxx bash colab/launch_colab_session.sh
```

```bash
# self-training: 3 rounds at 4-bit on 200 train questions, tested on 100 test questions
bash run_self_training.sh --model Qwen/Qwen3.5-0.8B --precision 4bit --experiment qwen08_4bit_pilot \
     --rounds 3 --pool_limit 200 --eval_limit 100 --results_dir results
python scripts/report_self_training.py --model Qwen/Qwen3.5-0.8B --precision 4bit --experiment qwen08_4bit_pilot

# the same with length-normalised DPO on the model's own (majority, minority) pairs
bash run_self_training.sh --model Qwen/Qwen3.5-0.8B --precision 4bit --experiment qwen08_4bit_dpo_similar \
     --objective dpo --pairing similar_length --rounds 1 --pool_limit 2000 --eval_limit 100 --batch_size 256 --results_dir results
```

```bash
# reports over everything in results/
python scripts/report_generator_metrics.py
python scripts/report_verifier_metrics.py
python scripts/report_verifier_shared_questions.py

# refresh the numbers on the web pages (after any new or changed run file)
python scripts/export_web_data.py
```

```bash
# tests
for test in tests/test_*.py; do python "$test"; done
```

The Colab launcher clones this repository from GitHub, so changes must be pushed before they take effect there.

## Names that changed on 9 Oct 2026

| Before | Now |
|---|---|
| `fullrun.sh` | `run_pipeline.sh` |
| `run_experiment.sh` | `colab/launch_colab_session.sh` |
| `colab_run.py` | `colab/colab_entrypoint.py` |
| `scripts/run_experiment_d.py` | `scripts/generate_answers.py` |
| `scripts/extract_verifier_dataset.py` | `scripts/build_verifier_dataset.py` |
| `scripts/run_verifier_benchmark.py` | `scripts/run_verifier.py` |
| `scripts/analyze_results.py` | `scripts/report_generator_metrics.py` |
| `scripts/generate_metrics_report.py` | `scripts/report_verifier_metrics.py` |
| `scripts/generate_standardized_verifier_report.py` | `scripts/report_verifier_shared_questions.py` |
| `src/experiment_d/data_loader.py` | `benchmarks.py` |
| `src/experiment_d/inference.py` | `model_loading.py` and `sampling.py` |
| `src/experiment_d/extractors.py` | `answer_extraction.py` |
| `src/experiment_d/utils.py` | `answer_matching.py` |
| `src/experiment_d/metrics.py` | `self_consistency.py` |
| `notebooks/SLM.ipynb` | `notebooks/run_pipeline_on_colab.ipynb` |

Every script takes `--results_dir`, `--model`, `--precision`, `--dataset`, `--n_samples`. The older spellings (`--output_dir`, `--directory`, `--model_id`, `n8`) are still accepted. Result files and folders kept their names, so existing runs resume and reports read them unchanged.
