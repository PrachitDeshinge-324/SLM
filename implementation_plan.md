# Experiment D: Quantization Self-Improvement (Pre-flight & Execution)

This implementation plan focuses exclusively on **Experiment D** from the project proposal: *Does compression break the reliability of self-judgment, with NO training involved?* We will evaluate how model shrinking (16-bit vs 8-bit vs 4-bit) impacts a model's ability to judge its own correctness through self-consistency (majority voting) on GSM8K and CommonsenseQA.

## User Review Required

> [!IMPORTANT]
> - **Hardware Strategy**: Since you are testing on an M2 Mac but deploying on Google Colab, the inference scripts will be built to auto-detect the environment (`mps` for Mac, `cuda` for Colab). Note that `bitsandbytes` quantization is designed for CUDA, so local Mac testing will focus on fp16/bf16 and pipeline validation, while the 8-bit/4-bit quantization runs will occur on Colab.
> - **Data Persistence**: To ensure no data is lost during expensive Colab runs, the logger will incrementally write each question's results to a JSONL file immediately after generation. This acts as an automatic checkpoint mechanism.

## Proposed Changes

### 1. Workspace Configuration

#### [MODIFY] [implementation_plan.md](file:///Users/prachitdeshinge/BITS/SEM%201/Agentic%20AI/Experiment_D/implementation_plan.md)
Replace the outdated implementation plan in the workspace with this version to act as the true project reference.

### 2. Data & Prompts

#### [NEW] [src/experiment_d/data_loader.py](file:///Users/prachitdeshinge/BITS/SEM%201/Agentic%20AI/Experiment_D/src/experiment_d/data_loader.py)
Load datasets into a unified format using the existing local files to avoid unnecessary downloads.
- **GSM8K**: Load from the local `gsm8k/` directory.
- **CommonsenseQA**: Load from the local `CQA/` directory.

#### [NEW] [src/experiment_d/prompts.py](file:///Users/prachitdeshinge/BITS/SEM%201/Agentic%20AI/Experiment_D/src/experiment_d/prompts.py)
Define robust 1-shot and few-shot prompt templates. Small models (0.8B-3B) are "chatty" and struggle with exact formatting, so these templates will provide strict examples of the expected output.

### 3. Inference & Quantization Core

#### [NEW] [src/experiment_d/inference.py](file:///Users/prachitdeshinge/BITS/SEM%201/Agentic%20AI/Experiment_D/src/experiment_d/inference.py)
Implements the core generation logic for Experiment D:
- **Models**: `Llama-3.2-1B`, `Llama-3.2-3B`, `Qwen3.5-0.8B`, `Qwen3.5-4B`.
- **Environment Aware**: Uses `mps` for local Mac testing and `cuda` for Colab.
- **Quantization**: Support loading models in 16-bit (bfloat16/fp16), 8-bit (`bitsandbytes`), and 4-bit (`bitsandbytes` NF4) on Colab.
- **Sampling**: For each prompt, generate N (e.g., 16) independent outputs using a non-zero temperature (e.g., T=0.7).

#### [NEW] [src/experiment_d/extractors.py](file:///Users/prachitdeshinge/BITS/SEM%201/Agentic%20AI/Experiment_D/src/experiment_d/extractors.py)
Robust answer extraction designed specifically for non-compliant SLMs. We will implement multi-stage fallback regexes:
- **GSM8K**: 
  1. Look for explicit `#### N` or `= N`.
  2. Fallback: Look for "The answer is N".
  3. Fallback: Extract the absolute last standalone number in the response.
- **CommonsenseQA**: 
  1. Look for `Answer: A` or `(A)`.
  2. Fallback: Look for "Option A" or the last standalone capital letter A-E.

#### [NEW] [src/experiment_d/metrics.py](file:///Users/prachitdeshinge/BITS/SEM%201/Agentic%20AI/Experiment_D/src/experiment_d/metrics.py)
Calculate an expanded suite of Experiment D metrics:
- **Majority-vote answer**: The most frequent extracted answer among the N samples.
- **Confidence (Agreement)**: The fraction of valid samples that agree with the majority vote.
- **Correctness**: Whether the majority-vote answer matches the ground truth.
- **Extraction Failure Rate**: Percentage of the N samples where the model's output could not be parsed into a valid answer.
- **Performance Metrics**: Tokens per second and total generation time (important for evaluating quantization overhead).

### 4. Execution & Logging

#### [NEW] [scripts/run_experiment_d.py](file:///Users/prachitdeshinge/BITS/SEM%201/Agentic%20AI/Experiment_D/scripts/run_experiment_d.py)
The main script to run the configuration matrix:
- **Comprehensive Incremental Logging**: Writes a JSON object for every processed question to a `results.jsonl` file. This log will contain the question ID, the ground truth, the aggregated metrics, AND **every single one of the N raw generated samples** along with what the extractor pulled from each. This ensures maximum visibility into model behavior and extraction failures.
- **Resume Capability**: On startup, checks the `results.jsonl` and skips already-processed question IDs.

#### [NEW] [scripts/analyze_results.py](file:///Users/prachitdeshinge/BITS/SEM%201/Agentic%20AI/Experiment_D/scripts/analyze_results.py)
Reads the JSONL logs and generates plots showing the relationship between Confidence (x-axis) and Correctness (y-axis) across precision levels, while also plotting extraction failure rates.

## Verification Plan

### Automated Tests (Rigorous)
- **Extractor Unit Tests**: Create `tests/test_extractors.py` with dozens of mock outputs simulating "chatty" or poorly formatted SLM responses to guarantee the multi-stage fallback regexes work flawlessly.
- **Metric Tests**: `tests/test_metrics.py` to verify majority voting correctly handles ties and ignores un-extractable samples.

### Local Mac Verification
- Run a small pilot on 5 questions using `Qwen3.5-0.8B` in 16-bit precision (`mps` backend).
- Validate that the incremental JSONL logging saves state properly (including all raw samples) and that the extraction logic handles real generations successfully.
