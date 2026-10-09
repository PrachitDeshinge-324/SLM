# Experiment D: does quantization break self-improvement in small language models?

A small model answers each question several times, the answers are put to a vote, and the model is asked to judge its own solutions. This is done at 16-bit, 8-bit and 4-bit weight precision to see what quantization does to those self-judgment signals.

Findings, literature review, decisions and open questions: [RESEARCH_DOSSIER.md](RESEARCH_DOSSIER.md).

## Layout

| Path | What it holds |
|---|---|
| `run_pipeline.sh` | Runs the three pipeline stages for one model, one precision, one dataset |
| `scripts/` | One file per pipeline stage and per report (table below) |
| `src/experiment_d/` | The library the scripts are built from (table below) |
| `colab/` | Launching a run on Google Colab from your own machine |
| `notebooks/run_pipeline_on_colab.ipynb` | The same pipeline, cell by cell, for a manual Colab session |
| `analysis/` | Dated analyses behind the dossier, with their saved outputs |
| `tests/` | Unit tests |
| `data/` | GSM8K and CommonsenseQA files; MATH-500 and SVAMP are downloaded |
| `results/` | Run files and reports (not in git; synced with Google Drive) |
| `web/` | Project site |
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

Reports never trust the metrics stored inside a run file. They re-score every run from its raw outputs with the current code, drop truncated samples, and compare precisions on the questions they share.

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
| `verifier.py` | The verifier prompt and the parser for its verdict |
| `rescoring.py` | Rebuild metrics and labels from stored raw outputs |
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
# reports over everything in results/
python scripts/report_generator_metrics.py
python scripts/report_verifier_metrics.py
python scripts/report_verifier_shared_questions.py
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
