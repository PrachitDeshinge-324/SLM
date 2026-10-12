# Does Compressing a Small AI Model Break Its Ability to Learn From Itself?

*Experiment D research dossier. Living document, last full revision 9 Oct 2026. Code map: `README.md`.*

**How this document is organised.** Part A is written for anyone: an abstract, an introduction, the findings in plain words, and what happens next. No statistics background is needed. Part B is the full technical record for readers who want to check or build on the work: every number, every paper read, every mistake found, and the exact commands to reproduce it. Each Part A statement points to the Part B section that backs it up.

| If you are… | Read |
|---|---|
| In a hurry (2 minutes) | A.1 Abstract only |
| A supervisor, examiner or non-specialist | All of Part A (about 10 minutes) |
| A researcher who wants the evidence | Part B, sections 1 and 2, then 4 |
| Checking how this fits the literature | Part B, sections 3 and 4 |
| Planning the next experiments | Part B, sections 5, 6 and 10 |
| Reproducing the numbers | Part B, section 8 and `README.md` |

---

# Part A — Summary for everyone

## A.1 Abstract

Small language models, the kind that fit on a laptop or phone, are often *compressed* (quantized) by storing their internal numbers with fewer bits. Compression makes them cheaper to run, but it can make them slightly worse. A separate line of research asks whether a model can improve itself without human help: it answers each question several times, trusts the answer most of its attempts agree on, and learns from those trusted answers. We asked what compression does to that self-checking signal.

We tested six small models on four question sets (mostly grade-school maths) at full precision, 8-bit and 4-bit. Our main result is that **compression makes a model less often right and less often sure, but when it is sure it is still just as trustworthy.** At 4-bit, a model's single attempts were correct about 4.5 percentage points less often in every one of ten model-and-dataset comparisons, and it produced about 6 points fewer "confident" answers. The confident answers that remained were as accurate as before (a change of +0.3 points, indistinguishable from zero). Taking a vote across attempts hides roughly 1.4 points of the damage. 8-bit compression left voted answers essentially unchanged.

We cannot yet say *how much* damage to expect for a new model: it ranged from 1 to nearly 20 points depending on the model. Our results come from one compression method, small question sets in places, and no actual self-training; the next step is a larger, cleaner confirmation run.

## A.2 Introduction: why this question matters

**The problem.** Getting a language model to improve normally needs people to write or check many correct answers, which is slow and expensive. One cheaper idea is *self-improvement*: ask the model the same question many times, and if most attempts reach the same answer, treat that answer as probably correct and train on it. Nobody has to supply the right answer. This works best when the model's agreement is a reliable hint that it is right.

**The twist.** Small models are rarely used at full precision. To fit them on cheap hardware, their weights are *quantized*: each stored number is rounded to 8 or 4 bits instead of 16. This usually costs a little accuracy. What nobody had checked carefully is whether rounding also damages the *agreement signal* that self-improvement depends on. If it did, a compressed model could teach itself wrong answers with great confidence.

**What was already known** (details in Part B, section 3):

- Compression lowers the accuracy of small reasoning models, and the effect is larger below 4 bits. This is established by earlier work, including a paper on the same DeepSeek model we use ("Quantization Hurts Reasoning?"), so we do not claim it as new.
- Models can improve themselves from majority-vote signals, and one earlier study did this on a 4-bit model. But it did not compare different precisions. So "nobody has combined compression and self-improvement" would be an overstatement, and we avoid it.
- What we found no one reporting is how compression changes the *quality of the self-generated labels* (how many there are, how correct they are, and how well they can be trusted).

**Our question.** Does compressing a small model break the machinery it would use to improve itself?

## A.3 What we did

| Step | In plain words |
|---|---|
| Models | Six small models from four families, 0.8 to 3 billion parameters |
| Compression levels | Full 16-bit, 8-bit, and 4-bit (using the common `bitsandbytes` tool) |
| Questions | Grade-school maths (GSM8K, SVAMP), harder competition maths (MATH-500), and common-sense multiple choice |
| Procedure | Each question answered 8 times by each model; the majority answer and the level of agreement were recorded |
| A second check | The same model was asked to judge whether its own solutions were right ("verifier") |
| Comparison | Always the *same questions* at different precisions, with uncertainty ranges from resampling the questions |

**An important honest step.** While analysing, we found that the scores in our first saved result tables had been computed with old versions of the answer-reading code, and some were off by up to 21 points. We rebuilt every number from the raw model outputs and fixed the report scripts so this cannot recur. Everything below uses the corrected numbers (Part B, section 2.1).

## A.4 What we found

| # | Finding | Confidence | Where |
|---|---|---|---|
| 1 | **At 4-bit, single attempts are right less often.** On average 4.5 points lower, and lower in all 10 comparisons. | High for direction, low for size | B 2.10 |
| 2 | **At 4-bit, the model is "confident" less often.** About 6 points fewer questions reach a strong majority (at least 6 of 8 attempts agree), also in all 10 comparisons. | High for direction | B 2.10 |
| 3 | **When it is confident, it is still just as right.** The accuracy of confident answers changed by +0.3 points (range −0.8 to +1.3). | Medium to high | B 2.6, 2.10 |
| 4 | **Voting hides damage.** The majority answer looks about 1.4 points better than single attempts suggest, so reporting only voted accuracy understates the harm. | High | B 2.10 |
| 5 | **The drop in agreement estimates the drop in accuracy, with no answer key needed.** Predicting a held-out model's loss this way was off by 0.85 points on average, against 3.4 for guessing "no change". | Medium | B 2.7, 2.10 |
| 6 | **8-bit is nearly harmless to voted answers** (−0.4 points, indistinguishable from zero), with a small loss in single attempts. | Medium (a 16-bit vs 8-bit number-format difference is mixed in) | B 2.10 |
| 7 | **The size of the damage depends on the model, and we cannot predict it.** From about 1 point (Llama-3B, DeepSeek) to 19.5 (Gemma-1B). The questions that break in one model are unrelated to those that break in another. | High | B 2.10, 2.11 |
| 8 | **One outlier, with an explanation.** Gemma-1B at 4-bit mostly changed *how it writes answers* (it stopped using the boxed format our reader looks for), so part of its drop is a formatting change, not lost reasoning. Removing it changes sizes but no conclusion. | Medium | B 2.4, 2.11 |
| 9 | **The model-judging-itself check is not yet usable.** Some models approve everything, reject everything, or do not answer in a parseable way, so this part reflects the checker as much as compression. | Low | B 2.8 |

**Plain-language reading of finding 3.** Think of a student who, after being slightly "scrambled", gets fewer questions right and is less often certain, but whose certain answers are as reliable as before. Compression reduces the *amount* of trustworthy material, not its trustworthiness.

## A.5 What this means

- **For the research question.** The step that turns a model's attempts into training labels *survives* 4-bit compression. The expectation, not yet tested, is that a compressed model would self-improve **more slowly and plateau lower**, rather than being poisoned by bad labels.
- **For anyone compressing a small model.** Do not trust voted accuracy alone, because it hides some damage. Check the model itself, because damage is not predictable from other models. The drop in agreement is a cheap, label-free way to estimate the loss.
- **For the project.** Our measurement choices (which answer reader, how many questions, how long a model may write) changed results as much as compression did. Fixing and recording these choices is the first priority.

## A.6 What we do not know (limits)

1. **One compression tool.** All results use `bitsandbytes`. Other common methods (AWQ, GPTQ, GGUF) and anything below 4 bits are untested.
2. **Small models and small question sets.** Models are 0.8 to 3B parameters. Several comparisons use only 100 to 200 questions, which can mislead by many points: the first 100 GSM8K questions showed a 16-point drop where the full set shows 7.
3. **No actual self-training yet.** We measured the ingredients (label quantity, accuracy, trust), not a training loop. The prediction in A.5 is a hypothesis.
4. **Length limit.** Models were limited to 512 tokens (word pieces) on most tests, which cut off a large share of some models' answers.
5. **Prompts and answer reading.** One prompt per task, and a rule-based answer reader.
6. **Literature.** I (the AI assistant) read 13 papers in full and 9 through summaries; the document labels which is which, and any quoted number from a summary should be rechecked before citing.

## A.7 What happens next

**Next step: a clean confirmation run** (Part B, section 6, phase 1):

1. Use the **full GSM8K test set** (1,319 questions), not 100 to 200.
2. Set the **length limit from a short pilot** so almost nothing is cut off.
3. Add **one more compression method** (AWQ or GPTQ) at 4-bit, and **one 3-bit** setting.
4. Run **five models**, the fragile and the robust ones.
5. Fix the measures in advance and log the full configuration with every run.

**If it holds**, a second phase runs one real self-training round at 16-bit and 4-bit to test the prediction in A.5. **Possible venues**: a workshop paper first, then a fuller journal-style version (Part B, section 9).

**Decisions that are yours** (Part B, section 10): your compute and time budget, the target venue, whether the mentor's PDFs in `Docs/` add constraints (they are image-only and could not be read), (the web pages now follow `results/` automatically; see section 10).

## A.8 Glossary

| Term | Meaning |
|---|---|
| Quantization | Storing a model's numbers with fewer bits (16 → 8 → 4) to make it smaller and cheaper, at some cost in accuracy |
| Self-consistency / majority vote | Answer the same question several times and pick the most common answer |
| Agreement (confidence) | The share of attempts that gave the majority answer |
| Per-sample accuracy | How often a single attempt is correct |
| Majority accuracy | How often the voted answer is correct |
| Pseudo-label | A model-made answer used as if it were a correct label for training |
| Label precision / yield | Of the pseudo-labels kept, how many are correct / how many questions produce one |
| Coverage | Whether at least one of the attempts is correct (the room available for improvement) |
| Verifier | The model asked to judge whether a solution is right |
| Flip | A question whose voted answer changes between two runs |
| Replicate noise floor | How much answers change just from re-running the same model, so that real changes can be told apart |
| Truncation | An answer cut off by the length limit before the model finished |
| Paired comparison / bootstrap | Compare the same questions under both settings, and resample the questions to get an uncertainty range |
| Percentage points (pts) | Differences between two percentages, e.g. 60% → 55.5% is −4.5 pts |
| τ (tau) | How much a true effect differs from one model or dataset to another |

---

# Part B — Technical record

*Living document. Last full revision: 9 Oct 2026 (night). Supersedes the direction section of the 3 Oct audit (now `archive/2026-10-03_audit/AUDIT_AND_DIRECTION.md`) and my plan of 3 Oct. Code map: `README.md`.*

*Revision note, 9 Oct night: the codebase was reorganised (new file names in `README.md`); script names in this document were updated. Outputs before and after the reorganisation are identical (section 8).*

*Revision note, 9 Oct late: the three report scripts now rescore from raw outputs and the CSVs were regenerated (section 2.1); sections 2.10, 2.11 and 4.5 answer the mentor's questions on generality, outliers and meaning.*

*Revision note, 9 Oct evening: every number in section 2 was re-computed in the `SLM` conda environment (Python 3.13, `sympy` 1.14), on the current `results/` folder, with MATH-500 included and `_backup` files excluded. The CSVs saved on 4 Oct were reconciled against the raw generations; see section 2.1. Raw outputs of each script are in `analysis/reassessment_2026-10-09/outputs/`.*

**Research objective (unchanged):** does weight quantization break self-improvement in small language models?

**How to read the labels.** Every substantive statement carries one of:

| Label | Meaning |
|---|---|
| **[Lit]** | Stated in a paper whose relevant sections I read directly from the full text |
| **[Lit-sum]** | From a tool-generated summary of the full text or from the abstract only; re-check before citing |
| **[Data]** | Computed from our stored generations; script named in section 8 |
| **[Int]** | My interpretation of [Lit] or [Data] |
| **[Hyp]** | Hypothesis that still needs an experiment |

---

## 1. Bottom line

**The general trend we can claim now.** At 4-bit, a small model is right less often and confident less often, but when its samples agree it is exactly as trustworthy as before. In numbers, over ten model × dataset cells (six models): per-sample accuracy −4.5 points and yield of confident labels −6.2 points, both in 10 of 10 cells; precision of confident labels +0.3 [−0.8, +1.3]; ranking quality of agreement unchanged. Three companion claims hold as generally: majority voting hides about 1.4 points of the loss; the drop in agreement estimates the drop in per-sample accuracy without labels (0.85 points error on held-out models against 3.36); and 8-bit leaves voted answers untouched. What we cannot claim is a size: the loss ranges from 1 to 19.5 points by model. [Data, section 2.10]

**Next step.** These claims rest on one quantizer, a 512-token limit and several 100–200-question cells. The next step is the clean confirmation run of section 6 (phase 1): full GSM8K test set, token limit set from a pilot, bitsandbytes plus one of AWQ/GPTQ at 4-bit and a 3-bit point, on the five anchor models. Nothing else is needed before starting it.

1. **The generator accuracy result is not new.** [Quantization Hurts Reasoning?](https://arxiv.org/abs/2504.04823) (April 2025) already tests DeepSeek-R1-Distill-Qwen-1.5B on GSM8K and MATH-500 with seven quantization methods, 3 seeds and a 32,768-token limit. Neither the proposal nor the audit cites it. Our DeepSeek numbers agree with theirs, which validates our pipeline but removes "4-bit hurts small reasoning models" as a contribution. [Lit][Data]
2. **"Nobody has combined quantization and self-improvement" is false.** [MACA](https://arxiv.org/abs/2509.15172) self-trains Qwen-2B, Llama-3B, Phi-4B and Llama-8B on majority-vote signals *with a 4-bit bitsandbytes base*, on GSM8K among others, and gains about 21 points there for Qwen-2B (23.0 to 44.2 after one iteration). What it never does is treat precision as an experimental factor. [Lit]
3. **Our data contain four findings that the literature does not have**, all about what quantization does to the *agreement signal* rather than to accuracy:
   - majority-vote accuracy hides per-sample damage;
   - answer "flips" at 8-bit equal the replicate noise floor exactly, and 4-bit excess flips are one-directional;
   - vote-filtered pseudo-labels keep their precision at 4-bit but lose yield and coverage;
   - the shift in agreement estimates the shift in *per-sample* accuracy (not majority accuracy), and this survives the truncation control.
4. **The audit's headline (agreement shift predicts majority-accuracy loss, r = 0.93) does not hold** as stated. It fails on untruncated questions and on CommonsenseQA. The per-sample version holds on both. [Data]
5. **Several stored cells are not valid precision comparisons**: prompts differed across precisions for Llama (run date in the template) and probably for one CommonsenseQA cell; MATH-500 Qwen files were overwritten after the audit; Gemma's 4-bit collapse is mostly a change of answer style. [Data]
6. **The CSVs saved on 4 Oct must not be quoted.** The generator report script (then `scripts/analyze_results.py`) averaged the metrics stored inside each run file, which were computed by whatever extractor existed on the day of the run. Against a rescoring of the same files they are off by up to 21 points in level, and they create precision effects that are not there (DeepSeek MATH-500 at 8-bit: CSV +14.0, rescored +5.0 with an interval that includes 0; Qwen-0.8B MATH-500 at 8-bit: CSV −16.0, rescored +1.0). Section 2.1 has the corrected table. [Data]
7. **What generalises and what does not** (section 2.10). Across 10 model × dataset cells at 4-bit, the *direction* of the damage is universal (per-sample accuracy and label yield fall in 10 of 10 cells) and two things do not move at all (label precision +0.3 [−0.8, +1.3] points; ranking quality of agreement). The *size* of the damage does not generalise: it ranges from −1 to −19.5 points, and which questions break is unrelated between models (correlation ≈ 0). [Data]
8. **One outlier cell, and it has an explanation** (section 2.11). Gemma-1B at 4-bit is the only cell that departs from the rest, on every metric, and it is the answer-style switch of section 2.4. No conclusion depends on it; pooled magnitudes shrink by about a third without it. [Data]
9. **Recommended paper:** an inference-only measurement study, *what a quantized small model's agreement signal can and cannot be trusted for*, with the self-training loop as a gated second phase that tests one sharp prediction. Details in section 6.

---

## 2. What our stored results actually show

All numbers: current extractor and matcher (with `sympy`); truncated samples never vote; "per-sample accuracy" counts a truncated sample as wrong; intervals are 95% paired bootstrap over questions; precisions are compared only on questions present at both and only when both runs used the same token limit. Installing `sympy` changed no GSM8K, SVAMP or CommonsenseQA number and moved MATH-500 per-sample accuracy by at most 0.1 points.

### 2.1 The saved CSVs (4 Oct) against a rescoring of the raw generations

**Status: fixed on 9 Oct.** The three report scripts (now `scripts/report_generator_metrics.py`, `scripts/report_verifier_metrics.py` and `scripts/report_verifier_shared_questions.py`) now rebuild every number from the raw outputs through one shared module, `src/experiment_d/rescoring.py`, and all CSVs and plots under `results/` were regenerated. The regenerated CSVs equal the "Rescored" column below. The rest of this subsection documents what the 4 Oct files said and why, because the web pages and the audit still quote them.

| Change | Where |
|---|---|
| Generator metrics are recomputed from `raw_samples` with the current extractor; truncated samples do not vote | `rescoring.rescore_record`, used by `report_generator_metrics.py` |
| Deltas against 16-bit are paired on shared questions; `Delta_N_Questions` says how many; blank when token limits differ | `report_generator_metrics.compute_quantization_deltas` |
| New CSV columns: `Sample_Acc_%`, `Sample_Acc_Delta_vs_16bit_%`, `Confidence_All_%` | `report_generator_metrics.py` |
| Verifier trace labels and voting answers are recomputed from `student_solution`; truncated traces are dropped and counted; precisions are compared on shared questions; new columns `Unparsed_Verdicts`, `Truncated_Traces_Dropped`, `Truncation_Unknown` | `rescoring.load_verifier_traces`, used by both verifier report scripts |
| The verifier runner no longer overwrites the trace's truncation flag with its own (`trace_cutoff`, `verifier_cutoff`, `verifier_length`) | `scripts/run_verifier.py`; exercised on CPU at 16-bit only |
| `_backup` files are skipped everywhere; the generator CSV is written into the dataset's own folder (the lower-cased path is what created `results/chilled_svamp (1)`) | all three scripts |
| Tests | `tests/test_rescoring.py`; the four test files pass in the `SLM` environment |

**How the 4 Oct CSVs were made.** `results/*/generator/metrics/csvs/results_*.csv` was written by `scripts/analyze_results.py`, which took the mean of `metrics.correctness` stored in each record. That field was computed at generation time, so it carries the extractor, vote rule and truncation rule of that day. The audit's fixes (last `####` marker, `\boxed{}` reading, normalised votes, truncated samples not voting) are therefore absent from every run made before 3 Oct, and present in runs made after. [Data: `09_reconcile_saved_csvs.py` reproduces every CSV value exactly from the stored field]

**Generator CSVs: accuracy (%) at 16 / 8 / 4-bit**

| Cell | Saved CSV | Rescored | CSV change at 8-bit, 4-bit | Rescored paired change at 8-bit, 4-bit |
|---|---|---|---|---|
| Qwen-0.8B GSM8K n8 | 65.0 / 63.7 / 55.1 | 66.6 / 65.0 / 59.4 | −1.4, −9.9 | −1.7, −7.2 |
| Qwen-0.8B GSM8K n16 | 66.5 / 66.3 / 57.8 | 68.1 / 68.6 / 60.7 | −0.2, −8.7 | +0.5, −7.4 |
| DeepSeek-1.5B GSM8K | 83.5 / 80.5 / 83.0 | 89.0 / 89.0 / 86.5 | −3.0, −0.5 | 0.0, −2.5 |
| Gemma-1B GSM8K | 51.0 / 48.5 / 35.5 | 60.0 / 58.0 / 37.0 | −2.5, −15.5 | −2.0, −23.0 |
| Llama-1B GSM8K | 46.7 / 45.6 / 45.3 | 48.0 / 46.9 / 46.2 | blank | −0.7, −0.8 (1,000 shared questions) |
| Llama-3B GSM8K | 80.5 / 80.7 / 80.6 | 80.7 / 81.1 / 80.7 | +0.2, +0.1 | +0.4, −0.1 |
| DeepSeek-1.5B MATH-500 | 47.0 / 61.0 / 55.0 | 68.0 / 73.0 / 68.0 | **+14.0, +8.0** | **+5.0 [−1, +11], 0.0 [−8, +8]** |
| Qwen-0.8B MATH-500 | 58.0 / 42.0 / 35.0 | 58.0 / 59.0 / 35.0 | −16.0, blank | not comparable at 8-bit (see below); −25.0 [−45, −5] at 4-bit on 20 questions |
| Qwen-0.8B SVAMP | 84.0 / 80.0 / 80.0 | 83.0 / 81.0 / 78.0 | −4.0, −4.0 | −2.0, −5.0 |
| CommonsenseQA, all cells | identical | identical | identical | identical |

- The error is not a constant offset. Gemma's 16-bit and 8-bit runs gain 9.0 and 9.5 points on rescoring, its 4-bit run gains 1.5, so the CSV *understates* the 4-bit loss by 7.5 points. DeepSeek GSM8K gains 5.5 / 8.5 / 3.5, which moves the 8-bit effect from −3.0 to 0.0. [Data]
- The MATH-500 "improvement under quantization" for DeepSeek is an artefact: 15.4% of the 16-bit run's samples get a different extracted answer under the current code (the old extractor cut answers at the first brace), against about 1% for the 8-bit and 4-bit runs. [Data]
- Qwen-0.8B MATH-500 mixes three protocols: 16-bit at a 4096-token limit (100 questions, scored with current code), 8-bit at 2048 (100 questions, stored with old code), 4-bit at 4096 (20 questions, run incomplete). The CSV's −16.0 at 8-bit compares old-scored with new-scored runs under different limits. [Data]
- `Confidence_%` in the CSV is the majority share among *valid* samples from stored metrics; the dossier uses the share among all samples. `Acc_Delta_vs_16bit_%` is unpaired and left blank when question counts differ.

**Verifier CSVs** (`verifier_output/metrics/csvs/*.csv`, `standardized_verifier_metrics/*.csv`)

| Issue | Size | Effect |
|---|---|---|
| Trace labels are the stored `is_correct`, from the extractor of the day | Labels change for 14.4% of DeepSeek MATH-500 16-bit traces, 4–9% of other MATH-500 traces, 1–4% of Gemma and DeepSeek GSM8K traces, under 0.2% elsewhere | MATH-500, Gemma and DeepSeek rows are unreliable |
| Truncated traces are counted | 7.5–18% of traces for Qwen on GSM8K/SVAMP, 29–32% for Gemma, 35–41% on MATH-500 | A truncated trace is wrong by construction and usually rejected, so false-positive rates are diluted, more so where truncation is higher |
| Per-model files use different question sets per precision | Qwen-2B: 200 / 100 / 100; DeepSeek rows drop unparsed traces and keep 77–97 questions; the Qwen-0.8B `n16` file covers 13 questions | Rows within one file are not comparable; only the `shared100` file is |
| Qwen-0.8B MATH-500 16-bit and 4-bit verdicts refer to generator runs that have since been overwritten | 100% of those traces are absent from the current generator files | That verifier CSV describes data no longer in the repository |

- The dilution changes a conclusion. On MATH-500 the `shared100` CSV shows Qwen-0.8B's false-positive rate *falling* with quantization (0.330 / 0.284 / 0.267). On complete traces it is 0.63 / 0.63 / 0.70. [Data]
- `results/rescored/*.csv` (3 Oct) predates the current files and should be regenerated or deleted.
- In the regenerated verifier CSV for Qwen-0.8B MATH-500, the 16-bit and 4-bit rows show `Truncation_Unknown = 800`: their traces cannot be matched to a generator file, so truncated traces could not be removed and those two rows are not comparable with the 8-bit row.
- Resolved on 12 Oct: `web/generation.html` and `web/verifier.html` now draw every number from `results/` (`scripts/export_web_data.py`, same re-scoring as the reports), so they no longer quote the 4 Oct numbers. `web/audit.html` is a local-only snapshot, kept out of git.
- Changed on 12 Oct, at Prachit's request: the Phase 2 web page scores every row on the same 100 questions of its model and on all 800 of their traces (N = 8), so truncated traces are kept as wrong traces instead of dropped, and runs with fewer than 100 complete questions (Qwen-0.8B GSM8K `n16`) are left out. Voting is unchanged (truncated traces never vote). The page's false-positive rates therefore carry the dilution described in the table above and are lower than the report CSVs wherever truncation is high (Qwen-2B GSM8K 16-bit: 0.20 on the page, 0.40 on complete traces). The 4-bit rise survives on GSM8K (Qwen-0.8B +14.5 points [+7.4, +20.8]; Qwen-2B +10.7 [+3.9, +18.4]). The report scripts still drop truncated traces. To switch the page back, pass `keep_truncated=False` in `web_export.collect_verifier_runs`.

### 2.2 Validity problems found in the stored runs

| # | Problem | Evidence | Consequence |
|---|---|---|---|
| V1 | `results/` changed after the audit CSV was written (Drive sync) | MATH-500 Qwen-0.8B 16-bit and 4-bit files are now 4096-token reruns; the 4-bit rerun stopped at 20 questions; 8-bit is still the 2048-token run; the old files and all `_backup` copies are gone | Audit MATH-500 Qwen numbers cannot be reproduced; the stored verifier verdicts for those two precisions judge traces that no longer exist |
| V2 | Llama prompts differ across precisions | Template prints the run date. Llama-1B GSM8K: 8/4-bit ran 22 Sep, 16-bit 24 Sep. Llama-3B: 16/8-bit 22 Sep, 4-bit 25 Sep | The audit's "same bug in all three precisions, so comparisons stay valid" is wrong for Llama. Effect size unknown, probably small |
| V3 | CommonsenseQA system prompt changed mid-campaign | Commit `7ed4374` (19 Sep 22:30) rewrote it; Qwen-0.8B 4-bit n8 started 17:06 that day. Its replicate differs from the n16 run by −1.1 [−2.2, −0.1] per-sample; no other replicate differs | That cell is suspect |
| V4 | 8-bit ran in fp16, 16-bit and 4-bit in bf16 | Model loading code (now `model_loading.py`), changed twice on 26–27 Sep | Any 8-bit effect is confounded with dtype |
| V5 | No commit hash in run headers; runs often predate the commit that contains their code | SVAMP ran 28 Sep, SVAMP support committed 29 Sep | Code version per run is unknowable |
| V6 | Truncation is tied to difficulty | Qwen-0.8B GSM8K, 16-bit: questions with any truncated sample score 0.433 on their *complete* samples, others 0.681; median length 306 tokens against a 512 cap | "Drop truncated" and the no-truncation view are both biased toward easy questions |
| V7 | Unequal question sets | Qwen-2B GSM8K: 200 / 100 / 112 questions at 16/8/4-bit | Paired comparison uses 100 |
| V8 | Statistical power | Flip rate ≈ 0.17 gives a minimum detectable majority-accuracy change of about 8–11 points at 100–200 questions and about 3 points at 1,319 | Every 100–200-question "no effect" cell is uninformative |

Clean same-day triples: Gemma GSM8K, DeepSeek GSM8K, SVAMP, Qwen-0.8B GSM8K n16. Everything else carries at least one of V2–V5. DeepSeek MATH-500 shares one token limit (2048) across precisions, but 35–41% of its samples are truncated there, against a 32,768-token limit in the published protocol [Lit: 2504.04823], and boxed-answer compliance differs between its 16-bit run (97%) and its 8/4-bit runs (78%, 84%), which suggests a prompt or stopping-rule change between them.

### 2.3 Generator: the audit's claims re-examined

| Audit claim | Verdict | Evidence [Data] |
|---|---|---|
| "8-bit is indistinguishable from 16-bit" | **True for voted answers, not established for single samples** | Majority accuracy: no cell differs. Per-sample accuracy: Qwen-0.8B GSM8K −1.5 [−2.5, −0.5] and −1.1 [−1.9, −0.4] in two independent run pairs; Llama-1B −1.6 [−2.5, −0.8]; Gemma −3.4 [−6.4, −0.6]. Replicate noise is up to 0.8, and V4 applies |
| "4-bit costs about 5 points on average, in 8 of 10 cells" | **The average mixes incomparable cells** | Solid: Qwen-0.8B GSM8K −7.2 [−9.4, −4.9], replicated at n16 (−7.4). Gemma −23.0 [−29.5, −16.5] but see 2.4. No majority effect: Llama-1B, Llama-3B. Underpowered (V8): Qwen-2B, DeepSeek, SVAMP. MATH-500: DeepSeek +5.0 [−1.0, +11.0] at 8-bit and 0.0 [−8.0, +8.0] at 4-bit on 100 questions, so no detectable effect; Qwen-0.8B −25.0 [−45.0, −5.0] at 4-bit but on 20 questions; Qwen-0.8B 8-bit is not a valid contrast (different token limit) |
| "Agreement predicts correctness equally well at every precision" | **Holds on the large GSM8K cells only** | AUROC Qwen-0.8B 0.880 / 0.864 / 0.869; Llama-3B 0.875 / 0.877 / 0.877. CommonsenseQA declines 0.728 to 0.700. Part of the signal is "did the sample finish": AUROC with truncation counted is 0.88, without it 0.85 |
| "Outcome 1 (confidently wrong) did not happen" | **Supported, one possible exception** | Error among questions with agreement ≥ 0.75: Qwen-0.8B 3.2% / 3.4% / 4.9% (n8) and 2.6% / 3.4% / 2.7% (n16); Llama-3B 5.2% / 4.6% / 5.2%. Gemma 10.1% (9/89) to 19.6% (10/51) |
| "Most answer flips are sampling noise" | **Confirmed and sharpened** | See 2.5 |

**New: majority voting masks quantization damage.** Per-sample versus majority change at 4-bit: Llama-1B −2.7* versus −0.8; Llama-3B −1.1* versus −0.1; CommonsenseQA Llama-1B −1.4* versus +0.2 (* = interval excludes 0). Where damage is large it passes through the vote: Qwen-0.8B −8.4 versus −7.2. [Data]

**New: two damage regimes, separable by coverage.** Coverage (at least one correct sample in 8) is unchanged at 4-bit for Llama-1B (−0.4) and Llama-3B (−0.5) but falls for Qwen-0.8B (−5.6 [−7.7, −3.6]) and Gemma (−17.0). [Data] Coverage is the quantity that sharpening theory ties to the cost of self-improvement [Lit: 2412.01951] and that self-evolution leaves unchanged while raising pass@1 [Lit: 2606.01075, Table 7]. So coverage loss is damage a self-training loop cannot repair. [Int]

**External check.** DeepSeek-R1-1.5B per-sample accuracy on complete GSM8K samples is 0.836 / 0.830 / 0.826; the published values are 84.5 (BF16), 83.3 (GPTQ 4-bit), 82.7 (AWQ 4-bit). [Data][Lit: 2504.04823, Table 1]

### 2.4 Gemma's 4-bit collapse is mostly a change of answer style

- At 16-bit Gemma answers 70% of the time in a LaTeX style ending `\boxed{N}` (accuracy 0.77) and 13% in prose (0.40). At 4-bit: 26% boxed (0.72), 74% prose (0.29), and the requested `####` format disappears. [Data]
- Within the *same question*, boxed samples are about 15 points more accurate than other samples at every precision (+0.155, +0.144, +0.140). [Data]
- A second extractor gives the same accuracy on prose answers (0.288 versus 0.279), so this is not an extraction artefact. [Data]
- Holding the 16-bit style mix fixed, the loss would be roughly 7 of the 19.5 per-sample points; the rest is the style shift. [Int]
- Llama-1B moves the other way: `####` compliance rises from 0.31 to 0.55 at 4-bit, which hides part of a 4–5 point within-style loss. [Data]
- Prior reports that quantization disrupts instruction following exist [Lit-sum: 2409.11055], but I found no study that decomposes an accuracy loss into style shift and within-style loss. [Int]
- **[Hyp]** Forcing the answer format at 4-bit (answer prefix or a second extraction turn) recovers most of Gemma's loss. This separates "lost capability" from "lost mode selection" and is cheap to test.

### 2.5 Flips against a true replicate floor (Qwen-0.8B GSM8K, 1,319 questions, four runs per precision pair)

| Comparison (8 samples each) | Correctness flips | Correct to wrong | Wrong to correct |
|---|---|---|---|
| Same precision, two independent runs (16-bit) | 14.6% | 8.1% | 6.4% |
| Same precision, two independent runs (4-bit) | 16.6% | 8.6% | 8.0% |
| 16-bit versus 8-bit | 14.9% | 7.8% | 7.1% |
| 16-bit versus 4-bit | 19.2% | 12.9% | 6.3% |

- 8-bit adds nothing to the floor. 4-bit adds about 4 points, all in one direction. [Data]
- The flips literature evaluates deterministically and has no such floor [Lit: 2407.09141; Lit-sum: 2607.08734]. Its own explanation is that quantization behaves like symmetric noise on log-probabilities [Lit: 2407.09141]. Our 4-bit excess is asymmetric, which symmetric noise does not predict. [Int]
- 4-bit damage is diffuse: with 24 samples per precision, no question that 16-bit solves at least half the time falls to zero; 3.4% of questions drop by 0.5 or more against 0.2% under the null. [Data]
- 74% of the lost per-sample accuracy comes from questions the 16-bit model already answers correctly at least half the time. [Data] [When Quantization Affects Confidence](https://arxiv.org/abs/2405.00632) reports that quantization mainly moves *low*-confidence items [Lit]; in relative terms we agree (22–27% relative loss in low bins, 8–13% in high bins), in absolute terms we do not. On CommonsenseQA low-confidence questions *improve* (+4.5, +5.5) while high-confidence ones drop, a flattening they also report as higher entropy. [Data][Lit]

### 2.6 Pseudo-label audit on stored data (vote filter, majority share ≥ 0.75)

| Cell | Yield 16 / 8 / 4-bit | Precision 16 / 8 / 4-bit |
|---|---|---|
| Qwen-0.8B GSM8K n8 | 0.383 / 0.356 / 0.279 | 0.968 / 0.966 / 0.951 |
| Qwen-0.8B GSM8K n16 | 0.327 / 0.331 / 0.252 | 0.974 / 0.966 / 0.973 |
| Llama-1B GSM8K | 0.288 / 0.269 / 0.240 | 0.868 / 0.881 / 0.900 |
| Llama-3B GSM8K | 0.679 / 0.675 / 0.665 | 0.948 / 0.954 / 0.948 |
| DeepSeek-1.5B GSM8K | 0.690 / 0.670 / 0.655 | 0.986 / 0.993 / 0.992 |
| Qwen-0.8B CommonsenseQA | 0.583 / 0.536 / 0.467 | 0.725 / 0.720 / 0.730 |
| DeepSeek-1.5B MATH-500 (100 questions) | 0.490 / 0.520 / 0.450 | 0.939 / 0.942 / 0.956 |
| Gemma-1B GSM8K | 0.445 / 0.385 / 0.255 | 0.899 / 0.909 / 0.804 [0.692, 0.907] |

- Precision of vote-filtered labels is invariant to bit-width in seven of eight cells; yield falls by up to 27% relative (Qwen-0.8B) and 43% (Gemma). [Data]
- Caveat: this uses the *test* split as a stand-in for an unlabeled pool. The paper version must use the train split.
- The precision/yield curve itself is prior art [Lit: 2505.21444, Appendix A.1; Lit: 2210.11610, Fig. 2]. Only the precision axis is new.

### 2.7 Agreement shift as a label-free estimator

| Target | Cells | r | Error (points) | "Assume no change" error |
|---|---|---|---|---|
| Majority accuracy, all cells | 23 | 0.92 | 1.8 | 2.9 |
| Majority accuracy, CommonsenseQA only | 5 | 0.91 | 2.7 | 1.2 |
| Majority accuracy, untruncated questions only | 13 | 0.91 | 2.0 | 1.4 |
| **Per-sample accuracy, all cells** | 23 | 0.96 | 0.9 | 3.4 |
| **Per-sample accuracy, untruncated questions only** | 13 | 0.95 | 0.8 | 3.0 |
| Per-sample accuracy, cells with change under 5 points | 20 | 0.91 | 0.6 | 2.1 |

- The majority version fails two controls; the per-sample version passes both. [Data]
- Why: on questions whose majority answer is correct, agreement *is* per-sample accuracy. The empirical content is that the surplus votes on wrong-majority questions stay constant under quantization. [Int]
- It is the Difference-of-Confidences estimator [Lit: 2107.03315] with self-consistency as the confidence and quantization as the shift. MACA's Table 16 contains three unremarked cells showing the same co-movement at 4-bit (for example Qwen-2B MATH: agreement −7.6, accuracy −9.3). [Lit]
- The 23 cells now include DeepSeek MATH-500 (8-bit: per-sample +2.4, agreement +2.3; 4-bit: −2.5, −1.7). [Data]
- Cells are not independent (six models, four of them contributing most cells). A paper needs more models and a held-out-model evaluation.

### 2.8 Verifier

The verifier emits one greedy line (median 26 characters), so it is a one-token classifier thresholded at zero. I applied signal detection theory (sensitivity d′ and criterion c; negative c = lenient) to separate skill from bias.

| Cell | d′ 16 / 8 / 4-bit | Criterion c 16 / 8 / 4-bit |
|---|---|---|
| Qwen-0.8B GSM8K | 1.13 / 1.11 / 1.17 | −0.36 [−0.49, −0.24] / −0.19 / −0.77 [−0.91, −0.64] |
| Qwen-0.8B SVAMP | 0.80 / 0.84 / 0.75 | −0.61 / −0.53 / −0.64 |
| Qwen-2B GSM8K | 1.28 / 1.20 / 1.16 | −0.39 / −0.25 / −0.40 |
| Llama-3B GSM8K | 0.88 / 0.71 / 0.50 (wide) | −1.60 / −1.27 / −1.69 |
| Qwen-0.8B MATH-500 (old 2048-token traces) | 0.88 / 0.53 / 1.24 (wide) | −0.77 [−0.99, −0.58] / −0.59 / −1.14 [−1.46, −0.91] |
| Gemma-1B GSM8K | 0.02 / 0.00 / undefined | −0.96 / −0.83 / −2.53 |

- **Qwen-0.8B GSM8K: sensitivity unchanged, criterion more lenient at 4-bit.** This is the cleanest verifier result we have. [Data]
- **Qwen-0.8B MATH-500 repeats the direction** (criterion −0.77 to −1.14 at 4-bit, stricter at 8-bit), but sensitivity is unstable there and the intervals nearly touch. These verdicts were given on the old 2048-token traces, whose generator files are gone, so truncation was inferred from the absence of a closed `\boxed{}`. [Data]
- It does not generalise across models or datasets: no shift on SVAMP or Qwen-2B. The cells with a real shift (Qwen-0.8B on GSM8K and MATH-500, Gemma) are the ones whose *generator* was damaged, and each precision judged its own traces, so the trace pool is confounded. [Data][Int]
- 8-bit is *stricter* than 16-bit in 6 of 6 cells. That is the opposite direction and coincides with the fp16/bf16 difference (V4). The criterion may simply be fragile to any numerical perturbation. [Data][Hyp]
- Weak independent support for verifier-side leniency: approval of truncated, answerless traces rises at 4-bit in all four Qwen cells (GSM8K 0.05 to 0.10; Qwen-2B 0.06 to 0.14; SVAMP 0.03 to 0.11; MATH-500 0.00 to 0.09). [Data]
- The audit's "never beats the plurality rule" needs qualifying. For Qwen models the verdict adds information *within* the plurality: P(correct | in plurality, approved / rejected) = 0.87 / 0.60 at 16-bit and 0.83 / 0.38 at 4-bit. About two thirds of its discrimination is within-question (+0.28 of +0.41). [Data]
- Gemma-1B has zero discrimination and approves 91–95% of truncated traces. [Data] Independent corroboration: Gemma-3-1B verifier accuracy 51% [Lit: 2606.01075, Table 6].
- Llama-1B rejects everything; DeepSeek re-solves instead of judging (26–33% parsed). These are prompt failures, not findings.

### 2.9 Corrections to my own 3 Oct plan

- "Repetition penalty 1.1 on math" affects `run_pipeline.sh` (then `fullrun.sh`) for future runs; no stored run records it.
- `rescore_paired.py` (now archived) does not pick up `_backup` files (its filename pattern excludes them).
- "Nobody measures sampled self-consistency on reasoning under quantization" was too strong: MACA Table 16 reports agreement at 4-bit.
- "Nobody varies precision in a label-free self-training loop" must become "nobody *trains* at two precisions and compares"; MACA evaluates a 4-bit-trained adapter at both precisions.

### 2.10 Generality: which findings hold beyond the cell they were found in

**Method.** Ten model × dataset cells have a valid 4-bit contrast and nine have a valid 8-bit contrast (six models; GSM8K, CommonsenseQA, SVAMP, MATH-500; 8-sample runs; at least 100 shared questions; matched token limits). Each effect was pooled with a random-effects model. τ is the standard deviation of the true effect between cells; the prediction interval is where a *new* cell is expected to fall. [Data: `10_generality.py`] Cells that share a model are not independent (Qwen-0.8B supplies three of the ten, DeepSeek two), so the pooled intervals are somewhat too narrow.

**4-bit versus 16-bit, in points**

| Effect | Pooled mean [95% CI] | τ | Prediction interval for a new cell | Cells with the pooled sign | Verdict |
|---|---|---|---|---|---|
| Per-sample accuracy | −4.5 [−6.7, −2.3] | 3.3 | −11.3 to +2.4 | 10 of 10 | Direction general, size not |
| Yield of labels at agreement ≥ 0.75 | −6.2 [−9.6, −2.8] | 4.8 | −16.2 to +3.8 | 10 of 10 | Direction general, size not |
| Majority-vote accuracy | −3.5 [−6.2, −0.7] | 3.9 | −11.6 to +4.7 | 7 of 10 | Not general |
| Coverage (any correct sample in 8) | −1.9 [−4.0, +0.1] | 2.7 | −7.6 to +3.8 | 8 of 10 | Model-specific |
| Damage hidden by the vote (majority minus per-sample) | +1.4 [+0.6, +2.2] | 0.4 | +0.3 to +2.5 | 7 of 10 (the three exceptions are 100–200-question cells) | General |
| Precision of labels at agreement ≥ 0.75 | +0.3 [−0.8, +1.3] | 0.8 | −1.7 to +2.2 | — | General null |
| AUROC of agreement (×100) | −0.9 [−2.4, +0.5] | 1.3 | −3.8 to +2.0 | — | General null |
| Estimator error (agreement shift minus per-sample shift) | +0.4 [−0.4, +1.1] | 0.9 | −1.6 to +2.3 | — | General |

**8-bit versus 16-bit:** per-sample accuracy −1.3 [−2.1, −0.5] (τ 0.9); majority accuracy −0.4 [−1.3, +0.5] (τ 0); yield −2.2 [−3.4, −0.9]; label precision +0.5 [−0.3, +1.2] (τ 0). Voted answers are unaffected in general; the small per-sample and yield losses are general but confounded with the fp16/bf16 difference (V4).

**Four further checks**

- **Small question sets do not generalise to the full test set.** For Qwen-0.8B GSM8K the full set gives −7.2 at 4-bit; the first 100 questions give −16.0 (this is the audit's "68 to 52"), the first 200 give −14.0, and 95% of random 100-question subsets fall between −15 and +1. For Llama-1B (true effect −0.8) the same range is −9 to +7. On CommonsenseQA the first 100 questions give +2.0 where the full set gives −2.1. Per-sample accuracy is steadier (−9.4 on the first 100 against −8.4). [Data] The cells that exist only at 100–200 questions (Qwen-2B, DeepSeek, Gemma, SVAMP, MATH-500) are therefore indicative, not established.
- **The estimator generalises to held-out models.** Leaving each model out in turn, the zero-parameter rule (per-sample change = agreement change) has a mean error of 0.85 points against 3.36 for "assume no change", and beats it for every one of the six models (worst case Gemma: 2.9 against 11.5). A line fitted on the other models does worse (1.42), so nothing should be fitted. [Data]
- **Sample count.** The 8-sample and 16-sample runs of Qwen-0.8B agree: 4-bit per-sample −8.4 and −7.6, majority −7.2 and −7.4. [Data]
- **Which questions break does not generalise between models.** The per-question change at 4-bit correlates +0.40 between two independent run pairs of the same model (Qwen-0.8B), but between −0.11 and +0.15 across models. Baseline difficulty, in contrast, is shared (r = 0.44 to 0.70 across models). [Data] So quantization damage is reproducible within a model and unrelated across models; it is not a property of the question. This fits the absence of any link between drop size and question length or solution steps (section 2.11) and argues against explanations based on arithmetic difficulty alone. [Int]

**Limits of generality that no analysis of the current data can lift:** one quantizer (bitsandbytes), nothing below 4-bit, models of 0.8–3B, one prompt per task, one temperature, and about 60% of the cells from GSM8K. Six models from four families is a small sample of models; τ is itself poorly estimated from ten cells.

### 2.11 Outliers

**Cell level.** Each 4-bit cell was compared with the others by a robust z-score (median and MAD). [Data: `11_outliers.py`]

| Cell | Per-sample | Majority | Coverage | Yield | Label precision | Flagged |
|---|---|---|---|---|---|---|
| Gemma-1B GSM8K | −19.5 | −23.0 | −17.0 | −19.0 | −9.5 | **Yes, on every metric (z from −3.9 to −9.4)** |
| Qwen-0.8B GSM8K | −8.4 | −7.2 | −5.6 | −10.4 | −1.7 | No |
| Qwen-0.8B CommonsenseQA | −4.8 | −2.1 | +1.8 | −11.6 | +0.5 | No |
| Qwen-0.8B SVAMP | −4.2 | −5.0 | −3.0 | −2.0 | +5.7 | No |
| Llama-1B GSM8K | −2.7 | −0.8 | −0.4 | −4.8 | +3.2 | No |
| Qwen-2B GSM8K | −2.5 | +0.9 | +1.8 | −4.5 | −1.8 | No |
| DeepSeek-1.5B MATH-500 | −2.5 | 0.0 | −1.0 | −4.0 | +1.7 | No |
| Llama-1B CommonsenseQA | −1.4 | +0.2 | −1.1 | −0.5 | +0.4 | No |
| Llama-3B GSM8K | −1.1 | −0.1 | −0.5 | −1.4 | 0.0 | No |
| DeepSeek-1.5B GSM8K | −0.9 | −2.5 | −1.0 | −3.5 | +0.7 | No |

- **Gemma-1B at 4-bit is the only outlier cell.** It is explained by the answer-style switch (section 2.4), it is the one cell where label precision falls, and it is the one cell where the vote *amplifies* the loss instead of hiding it. It should be kept and reported as a second failure mode, with the forced-format test as its follow-up.
- **No conclusion rests on it.** Without Gemma the pooled 4-bit per-sample loss is −3.2 instead of −4.5 and the majority loss −1.9 instead of −3.5; every verdict in section 2.10 stands. The estimator's correlation stays between 0.94 and 0.96 whichever single cell is removed, and is 0.91 (error 0.63 against 2.09) without the three largest-change cells. [Data]
- Qwen-0.8B at 4-bit is the largest effect that is *not* an outlier, and it is replicated in two independent run pairs.

**Question level**

- **Unanimous but wrong answers are rare, stable across precisions and specific to each model.** Among questions where all 8 samples agree, 0–5% are wrong (Llama-3B: 2.3% / 2.0% / 2.1% at 16 / 8 / 4-bit); the one exception is Gemma at 4-bit, 3 of 20. On the 200 questions answered by all six models, 24 have a confident wrong majority (agreement ≥ 0.75) in some model at 16-bit: 19 in exactly one model, 4 in two, 1 in four. Plain wrong majorities, in contrast, are widely shared (131 of 200 questions are missed by at least one model, 63 by three or more). [Data] Quantization does not manufacture confident errors, and since the errors are model-specific, agreement between two different models would catch most of them [Hyp; method in Lit-sum: 2603.25450].
- **No sign of faulty gold labels.** Only one question draws the same confident wrong answer from three or more models (`gsm8k_3`: gold 540, four of six models answer 180), and its gold label is right; the models share a misreading. Scope: 200 questions were seen by all six models and 1,000 by at least three. [Data]
- **Catastrophic 4-bit drops are few and partly an artefact.** With 24 samples per precision, 23 of 1,319 questions fall by 0.5 or more for Qwen-0.8B (3.4% of the questions it usually solves). They are not longer, have no more solution steps and no larger numbers than other questions (|r| ≤ 0.13). The one correlate is truncation: the drop tracks the *increase* in truncated samples at 4-bit (r = −0.40), and the 25 questions that improve are ones where 4-bit truncated less. [Data] Part of the question-level scatter is therefore the 512-token limit, not reasoning. The mean effect is not: among complete samples accuracy still falls from 0.581 to 0.487.

**Sample level**

- Truncation is the only frequent anomaly (14–30% of samples in four models, under 0.5% for Llama).
- Answers above one million occur in at most 0.3% of samples with no trend across precisions; almost no complete sample lacks an answer; exact duplicate samples are under 0.4% everywhere except Gemma (1.4–1.8%). [Data]

**Instrument failures, to be reported as such and not as findings**

| Case | Symptom | Why it is not a result about quantization |
|---|---|---|
| Llama-1B verifier | Rejects every trace | Prompt failure at full precision already |
| DeepSeek verifier | 22–36% of outputs contain a verdict | The model re-solves the problem instead of judging |
| Gemma-1B verifier | Approves 91–95% of truncated traces | No discrimination at any precision |
| Qwen-0.8B on CommonsenseQA | Median output is 5 tokens | No reasoning is produced, so it is not a self-consistency test |
| Qwen-0.8B MATH-500 | Three protocols across three precisions | Not a precision contrast |
| "First 100 questions" tables | −16.0 where the full set gives −7.2 | Sampling variation of a small subset |

---

## 3. Literature: critical review

The paper-by-paper review now lives in its own file: **[RESEARCH_DOSSIER_PAPERS.md](RESEARCH_DOSSIER_PAPERS.md)**. Section numbers are unchanged, so references to "section 3" or "3.x" in this document point there.

| Section | Covers |
|---|---|
| 3.1 | Quantization and reasoning accuracy |
| 3.2 | Quantization and reliability |
| 3.3 | Self-consistency and self-verification (full precision) |
| 3.4 | Self-training |
| 3.5 | Method references |

It reviews 34 arXiv papers and two textbooks. The main ones get their design, findings, weak points and what they mean for us; the rest get a line each. Every entry is marked with how it was read (**F** full text, **S** tool summary, **A** abstract only). What the literature adds up to is in section 4 below.

---

## 4. Synthesis

### 4.1 What is established and therefore not ours to claim

1. 4-bit weight quantization costs small reasoning models little on GSM8K and more on harder tasks; 3-bit is risky; small models lose more. [2504.04823]
2. A 4-bit base model can self-improve from majority-vote signals. [2509.15172]
3. Majority-vote self-training can collapse; gains are largely sharpening; sub-2B verifiers are near chance. [2505.21444, 2606.01075]
4. Agreement is roughly calibrated on arithmetic word problems and poorly on hard multiple choice. [2210.11610, 2608.11403, 2607.08065]

### 4.2 Where our data agree with the literature

- Small-model, math-reasoning sensitivity at 4-bit; near-zero 8-bit effect on voted answers.
- DeepSeek-R1-1.5B GSM8K accuracy at all precisions.
- Gemma-3-1B cannot verify.
- Quantization flattens answer distributions.

### 4.3 Where they diverge or where we add something

| Topic | Literature | Our data | Status |
|---|---|---|---|
| Flips | Counted without a sampling floor | 8-bit flips = floor; 4-bit excess one-directional | New control; one model so far |
| Damage location | Low-confidence items move most (2405.00632) | Absolute loss sits in mid/high-confidence questions | Partial disagreement; different task type |
| Pseudo-label quality by precision | Never varied | Precision flat, yield and coverage down | New; needs train split |
| Verifier under quantization | Precision treated as infrastructure (2605.02915) | Criterion shift without sensitivity loss in one cell | Suggestive; confounded |
| Style shift | Instruction following degrades (2409.11055) | Decomposed: most of Gemma's loss is style | New; one model |
| Which items break | Low-margin items flip (2407.09141); harder tasks lose more (2504.04823) | Fragile questions are reproducible within a model and uncorrelated across models; unrelated to length or steps | New; three full-set models |

### 4.4 Remaining gaps, with my confidence that they are real

| Gap | Confidence | Basis |
|---|---|---|
| G1. No paper analyses self-consistency *as a signal* (calibration, AUROC, pseudo-label precision and yield, coverage) across bit-widths | High | Full texts of 22 papers retrieved (relevant sections of 13 read directly, 9 through summaries), 9 targeted searches |
| G2. No flip study on sampled generation subtracts a replicate floor | High | Both flip papers evaluate deterministically |
| G3. No self-training study trains at two precisions and compares | Medium | MACA comes close; a 2026 paper could exist that my searches missed |
| G4. No study separates verifier sensitivity from bias across precisions | High | Existing self-verification studies treat precision as infrastructure rather than a variable |
| G5. No decomposition of quantization loss into style shift and within-style loss | Medium | Searched once; adjacent instruction-following reports exist |

### 4.5 What the data mean for the project

1. **For the research question.** The step that turns a model's samples into training labels survives 4-bit quantization: the precision of agreement-filtered labels is unchanged in general (section 2.10). What quantization removes is quantity and headroom: fewer confident labels in every cell, lower per-sample accuracy in every cell, and lower coverage in the fragile models. The expected consequence for a self-training loop is slower improvement and a lower ceiling, not poisoned training data. [Hyp, to be tested in phase 2] This agrees with the one published 4-bit self-training result [Lit: 2509.15172] and with coverage acting as the ceiling of self-evolution [Lit: 2606.01075].
2. **The size of the damage is a property of the model.** The same quantizer at the same bit-width costs 1 point in Llama-3B and DeepSeek and 19.5 in Gemma, and the questions that break in one model are unrelated to those that break in another. Damage cannot be predicted from another model or from the questions. That is the practical case for a label-free, per-model check, and the agreement shift is one that works on held-out models (0.85 points error against 3.36). [Data][Int]
3. **Voted accuracy is the wrong yardstick for small effects.** It hides about 1.4 points of per-sample loss, consistently (τ 0.4). Any study that reports only majority-vote or only greedy accuracy will call a degraded model unchanged. [Data][Int]
4. **Our measurement process varied more than our treatment.** Scoring version moved results by up to 21 points; the choice of 100 questions moves a 7-point effect anywhere between −15 and +1; token limits and answer style each account for effects as large as quantization itself. The quantization effects we care about are 1–8 points. The first deliverable is therefore a frozen protocol, and every number from a 100–200-question cell is provisional. [Data]
5. **What we can claim today, and where.** Supported on full test sets with replication: the Qwen-0.8B GSM8K results (per-sample and majority loss, flips against the floor, pseudo-label precision and yield). Supported across six models as direction or invariance: section 2.10. Supported by one model family only: everything about the verifier. Not supported: any statement about bit-widths below 4, quantizers other than bitsandbytes, models above 3B, or an actual training loop.
6. **What it changes in the plan.** Nothing in the direction of section 6; three points gain weight: full test sets instead of 100–200 questions, a token limit set from a pilot, and the forced-format test for Gemma as the cheapest experiment with the largest explanatory value.

---

## 5. Candidate contributions, ranked

| Rank | Contribution | Evidence in hand | What is missing | Kill criterion |
|---|---|---|---|---|
| 1 | **Replicate-floor-corrected flips** and the masking of per-sample damage by voting | Strong for Qwen-0.8B (four runs × 1,319 questions); masking pooled over 10 cells: +1.4 [+0.6, +2.2], τ 0.4 | Replicates for 3–4 more models; a second quantizer; one protocol | Excess flips or masking absent in at least half of new models |
| 2 | **Pseudo-label audit across bit-widths**: precision invariant, yield and coverage fall | Pooled over 10 cells: precision +0.3 [−0.8, +1.3]; yield −6.2 [−9.6, −2.8], negative in 10 of 10 | Train-split pool; sub-4-bit point where precision may finally break | Precision falls with bit-width as fast as yield |
| 3 | **Agreement shift estimates per-sample accuracy loss** | 23 cells, both controls passed; beats "no change" for each of six held-out models (0.85 against 3.36) | More models; held-out-model test; comparison with flip-rate and cross-model baselines | Error not below "no change" on held-out models |
| 4 | **Verifier criterion shift** under quantization | One clean cell, one supporting cell on MATH-500; confounded | Crossed design with logit scores on a frozen trace pool | No criterion shift when all precisions judge the same traces |
| 5 | **Style-shift decomposition** | Gemma, Llama-1B | Forced-format control; more models | Forced format does not recover accuracy |
| 6 | **Self-training at two precisions** | None | Whole experiment | — |

Ranks 1–3 are one coherent story and need only inference. Rank 4 is the highest-variance item: if it holds under the crossed design it is the most quotable result; if it does not, it becomes a short negative result that corrects the audit.

---

## 6. Recommended study

**Working title:** *What survives quantization? Agreement, coverage and pseudo-label quality in small language models.*

**Claim structure.** Weight quantization down to 4 bits leaves the *ranking* information in a small model's self-agreement intact and leaves vote-filtered pseudo-labels clean, but it lowers per-sample accuracy, yield and (in fragile models) coverage, and voted accuracy under-reports this. A same-model verifier keeps its sensitivity but shifts its criterion.

### Phase 1 — inference only (the paper)

| Decision | Choice | Justification |
|---|---|---|
| Models | DeepSeek-R1-Distill-Qwen-1.5B, Llama-3.2-1B, Llama-3.2-3B, Gemma-3-1B, Qwen3.5-0.8B | First three have published quantized checkpoints and reference numbers [2504.04823, 2501.03035, 2607.10855]; Gemma and Qwen-0.8B are the fragile cases in our data |
| Precisions | bf16; bitsandbytes 8-bit and 4-bit; AWQ or GPTQ 4-bit; GPTQ 3-bit | A sub-4-bit point is where published damage becomes large [2504.04823]; reviewers expect AWQ/GPTQ; GPTQ must use reasoning-domain calibration or a published checkpoint [2504.04823, §4.2] |
| Evaluation set | Full GSM8K test (1,319) | Our power analysis (V8, section 2.2): 100–200 questions cannot detect effects under 8 points; random 100-question subsets of our own full-set runs span −15 to +1 for a true −7.2 (section 2.10) |
| Pseudo-label pool | 1,500 GSM8K *train* questions | Removes the test-as-pool objection; same size as MACA |
| Samples | 8 per question, temperature fixed; two independent replicates of bf16 and of each 4-bit configuration | The replicate floor is contribution 1; our replicate noise at 1,319 questions is at most 0.8 points per-sample |
| Token limit | Set from a pilot so that truncation is under 2%; report answered / truncated / runaway separately | V6; published protocol uses 32,768 for reasoning models [2504.04823] |
| Answer extraction | Second-stage extraction prompt or forced answer prefix, plus the current regex as a robustness check; report format compliance per cell | Section 2.4; regex extraction is unreliable in general [2405.11874] |
| Verifier | Logit margin for Correct versus Incorrect on a frozen bf16 trace pool, judged by every precision; optional sampled-verdict threshold as in 2606.01075 | Removes the trace-pool confound; gives a full ROC [2605.02915] |
| Primary endpoints (fixed before running) | Δ per-sample accuracy; Δ majority accuracy; excess flips over floor; pseudo-label precision and yield at 0.75; Δ coverage; verifier AUROC and criterion | Prevents choosing metrics after seeing results |
| Statistics | Paired bootstrap over questions; Holm correction across the endpoint family; minimum detectable effect reported for every null | Pre-registration practice in 2608.11403 |
| Provenance | One frozen commit, hash and full config in every run header, pinned requirements, one dtype | V2–V5 |

**Not needed, with reasons.**
- CommonsenseQA for Qwen-0.8B: the model emits 5 tokens and no reasoning, so it measures something else. Keep one multiple-choice set only as a labelled "direct answer" control.
- SVAMP: ceiling, and 99% of answers came through the fallback extractor.
- MATH-500: at our 2048–4096-token limits 31–44% of Qwen-0.8B and DeepSeek samples are truncated, the token budget that published work uses (32,768) is out of reach on a free GPU, and the accuracy result exists [2504.04823]. The stored MATH-500 cells show no detectable precision effect for DeepSeek and cannot support one for Qwen-0.8B.
- "Three seeds × 100–200 questions" (audit plan): more questions buy more power than more seeds here.
- A new GPTQ calibration pipeline: use published checkpoints first.

### Phase 2 — one self-training round, only if phase 1 passes a gate

- **Gate:** phase 1 shows a yield or coverage difference between bf16 and 4-bit larger than the replicate floor for at least two models.
- **Design:** vote-filtered traces from the train pool; one LMSI-style round with QLoRA; base model at bf16 versus 4-bit; filter = vote versus vote-and-verifier; two models.
- **Prediction A [Hyp]:** equal label precision but lower yield at 4-bit gives a smaller gain per round, not a harmful one.
- **Prediction B [Hyp]:** adding a lenient 4-bit verifier to the filter lowers label precision relative to bf16.
- **Mandatory reporting:** completion / format / reasoning decomposition [2509.15172, Tables 21–22]; pass@1 versus pass@8 [2606.01075, Table 7]; answer diversity [2407.05013]; label accuracy per round [2505.21444].
- Without the decomposition, a loop under a token cap mostly teaches brevity and format, and the result would be uninterpretable.

---

## 7. Reviewer objections to pre-empt

| Objection | Current exposure | Answer to build |
|---|---|---|
| "Accuracy under quantization is known" | High | Cite 2504.04823 up front; position accuracy as replication, signal analysis as the contribution |
| "Self-improvement in the title, no training" | High | Title and claims about pseudo-labels and agreement; phase 2 optional |
| "Only bitsandbytes, nothing below 4-bit" | High | AWQ/GPTQ 4-bit and GPTQ 3-bit |
| "Truncation and token caps drive the results" | High | Pilot-set cap; outcome classes; results with and without truncated questions |
| "Regex extraction and format drift" | High | Second-stage extraction; compliance table; style decomposition as a finding |
| "Low power, one seed, many comparisons" | Medium | Full test set, replicates, Holm, minimum detectable effects |
| "GSM8K is saturated or contaminated" | Medium | Models score 38–84%; contamination affects levels, not paired precision contrasts. One harder set would still help |
| "Verifier is one prompt, one greedy token" | High | Logit ROC; second prompt; crossed design |
| "Estimator is arithmetic" | Medium | Say so; state the invariance it rests on; test on held-out models; compare with Difference of Confidences |
| "Causal language" | Medium | "Co-occurs with" unless the crossed design supports more |
| "Models loaded with remote code, hard to reproduce" | Low–medium | Anchor on Llama and DeepSeek; pin revisions |

---

## 8. Reproducing the numbers in section 2

```bash
cd analysis/reassessment_2026-10-09
PY=~/miniconda3/envs/SLM/bin/python            # or: conda activate SLM; PY=python
$PY 00_build_cache.py                          # per-sample cache from results/*/generator (skips *_backup*)
$PY 01_cell_table.py                           # 2.3 table inputs
$PY 02_extraction_paths.py                     # 2.4
$PY 03_paired_deltas_and_replicates.py         # 2.3 deltas, replicate noise
$PY 04_flips_and_damage_location.py            # 2.5
$PY 05_pseudolabels_estimators_truncation.py   # 2.6, 2.7, V6
$PY 06_estimator_target_and_fallback_check.py  # 2.7
$PY 07_truncation_control_and_gemma_style.py   # 2.7 truncation control, 2.4
$PY 08_verifier_signal_detection.py            # 2.8
$PY 09_reconcile_saved_csvs.py                 # 2.1 (stored versus rescored; the CSV-delta column now reads the regenerated files)
$PY 10_generality.py                           # 2.10
$PY 11_outliers.py                             # 2.11
```

- The output of each script from the 9 Oct runs is saved in `outputs/` (`09_reconcile_saved_csvs.txt` was produced before the CSVs were regenerated and records the 4 Oct values).
- To regenerate the project CSVs and plots: `$PY scripts/report_generator_metrics.py`, `$PY scripts/report_verifier_metrics.py`, `$PY scripts/report_verifier_shared_questions.py` from the repository root; tests: `for t in tests/test_*.py; do $PY $t; done`.
- **Reorganisation check (9 Oct).** After renaming and restructuring the code: the three pipeline stages run on CPU with a small model produce records identical to the pre-change run; all 23 report CSVs are byte-identical; the outputs of analysis scripts 01–08, 10 and 11 are byte-identical; the five test files pass.
- The scripts read `results/` as it stood on 9 Oct. Because that folder is still syncing (V1), copy it to a dated snapshot before the next rerun.
- Precision pairs whose runs record different token limits are skipped or flagged, not compared.

---

## 9. Publication path

- **Realistic first target:** a workshop paper (4–8 pages) on efficient or reliable ML at a major conference, built from phase 1.
- **Full version:** TMLR suits a careful measurement study whose value is correctness rather than a new method. ACL Rolling Review (Findings) is the stretch target and would need phase 2 or a harder benchmark.
- I have not checked 2026–27 calls or deadlines.

---

## 10. Open items

**Could not verify**
- `Docs/Self-Improvement Quantization - Google Docs.pdf` and `Docs/tab3.pdf` are image-only; I could not read them. If they hold mentor feedback, it is not reflected here.
- MATH-500 for Qwen-0.8B: the three precision files follow three different protocols (section 2.1), so no rerun of the analysis can fix it; it needs new runs.
- Papers marked S or A in section 3. Before citing, re-read MACA's main text, 2606.01850 and 2607.10855 tables, and 2505.11574.
- Whether AWQ/GPTQ tooling and vLLM support Qwen3.5 and Gemma-3.
- Gap G3: repeat the search immediately before writing the introduction.

**Decisions needed from you**
1. Compute and time budget, and target venue. This decides whether phase 2 is in scope.
2. Whether the Docs PDFs contain constraints from your mentor (model list, datasets, deadline).
3. ~~Whether to update the web pages~~ Done on 12 Oct: the Phase 1 and Phase 2 pages are generated from `results/`; run `python scripts/export_web_data.py` after any new or changed run file.

**Decision log**
| Date | Decision | Reason |
|---|---|---|
| 3 Oct | Audit proposes agreement-shift estimator (majority accuracy) as headline | r = 0.93 over 20 runs |
| 3 Oct | My plan demotes it, proposes pseudo-label audit + crossed verifier + gated loop | Truncation control |
| 9 Oct | Accuracy result reclassified as replication | 2504.04823 |
| 9 Oct | Estimator retargeted to per-sample accuracy | Passes truncation and CommonsenseQA controls |
| 9 Oct | Verifier claim downgraded to hypothesis pending crossed design | Shift appears only where generator was damaged; 8-bit moves the other way |
| 9 Oct | CommonsenseQA (Qwen), SVAMP, MATH-500 dropped from the plan | Sections 2.2, 6 |
| 9 Oct | Saved CSVs of 4 Oct declared not citable; dossier numbers are the record | Stored metrics predate the scoring fixes (section 2.1) |
| 9 Oct | Analysis moved to the `SLM` conda environment; MATH-500 included where token limits match | `sympy` available; no change to non-MATH numbers |
| 9 Oct | Report scripts changed to rescore from raw outputs; CSVs and plots regenerated | User decision; section 2.1 |
| 9 Oct | Gemma-1B at 4-bit kept as a reported outlier and second failure mode, not excluded | Only outlier cell; explained by style switch; no conclusion depends on it (section 2.11) |
| 9 Oct | Claims to be worded as general direction and invariances with model-specific size | Random-effects pooling over 10 cells (section 2.10) |
| 9 Oct | Codebase reorganised: descriptive file names, shared library modules, legacy files moved to `archive/`; result file names unchanged | User request; outputs verified identical (section 8) |
