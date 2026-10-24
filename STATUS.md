# Project status, issue register and next steps

| | |
|---|---|
| **Purpose** | Current state of the project: what works, what fails, open issues, next steps, and which document is authoritative for what. |
| **Audience** | Anyone new to the project, or returning to it. |
| **Scope and caveats** | Summary only; evidence lives in the documents listed in section 7. |
| **Related** | README.md, RESEARCH_DOSSIER.md, Docs/research_notes/ |

*Single entry point for "where are we". Written 17 Oct 2026 from the repository, `results/` and the documents listed in section 7; it adds no new measurements. Tags: **[Data]** a number in `results/` or a saved analysis output; **[Lit]** from a paper as summarised in our review documents (not re-checked here); **[Impl]** a fact about our code; **[Int]** interpretation; **[Hyp]** untested. Small cells (100-200 questions, one seed) cannot carry broad claims. Course-paper numbers come from the 100-question analysis (`analysis/paper_100q_2026-10-13/`), not from the full-set dossier tables.*

## The story in one page

**What we did.** We asked whether shrinking a small language model to 8 or 4-bit weights breaks its ability to check and improve its own answers. We ran six models on GSM8K (some also on MATH-500 and SVAMP), 8 samples per question, at 16, 8 and 4-bit. We measured three things: how often the samples agree (self-consistency), whether the model can judge its own solutions (self-verification), and whether training on its own majority votes helps (self-training, Qwen3.5-0.8B only).

**What we found** (details and tags in sections 2 and 5).
1. *Quantization hurts the single answer, not the vote's trustworthiness.* At 4-bit, accuracy per sample and the number of confident labels fall in all 10 model-dataset cells; when samples do agree, they are as often right as before. Majority voting hides about 1.4 points of the damage. 8-bit is indistinguishable from noise. [Data]
2. *The drop in agreement estimates the drop in accuracy without labels* (0.85 points error on held-out models). [Data]
3. *The model's own verifier adds almost nothing.* The one-line verifier is near chance on four of six models and adds nothing on top of the vote. [Data]
4. *Self-training depends on precision.* Training on votes helped the 4-bit model (+6.4 points per sample, one seed, 200 questions) but not the 16-bit one, where the gain was only "finishing more often". [Data; mechanism is a hypothesis]

**Flaws, in order of importance.**
- Our first results were wrong: the saved CSVs of 4 Oct came from stale scoring code and some cells compared different prompts. They were re-scored and must not be quoted. [Confirmed]
- The self-training evidence is one seed on 200 questions, and the loop corrupts its own labels over rounds. [Confirmed]
- The DPO run produced runaway generation and we do not know why. [Unresolved]
- Every adapter was trained with the wrong end token (`<|endoftext|>` instead of `<|im_end|>`); the effect is unmeasured, so the self-training results may be partly an artifact. [Fact confirmed, effect unknown]
- All claims rest on one quantizer, a 512-token cap and one prompt style; accuracy-under-quantization itself is already published. [Known limits]

**What we are doing now.** The mentor paused self-training on 16 Oct. We are re-checking the generator and verifier with prompts from the literature to see whether the weak verifier result was our prompt's fault: a critic-style verifier (runs partial), token-level confidence signals (data collected, not analysed), and a paper-by-paper comparison with a view to reproducing the Sharpening paper. Next steps with success criteria are in section 6.

## 1. The problem

Experiment D asks whether weight quantization (16 / 8 / 4-bit) breaks a small model's self-consistency (agreement of its own samples) and self-verification (its own verdicts on its own solutions), and whether a quantized model can still improve itself from its own votes. Models: Qwen3.5-0.8B and 2B, Gemma-3-1B, Llama-3.2-1B and 3B, DeepSeek-R1-Distill-Qwen-1.5B; mainly GSM8K, with MATH-500 and SVAMP for some cells.

## 2. Where the work stands

| Phase | What exists | State |
|---|---|---|
| 1. Generator (self-consistency) | Three-stage pipeline, re-scored reports, web page | Done. Direction of the 4-bit damage holds in 10 of 10 model x dataset cells; its size (1 to 19.5 points) does not generalise [Data: dossier 2.10]. Rests on one quantizer (bitsandbytes), a 512-token cap and 100-200-question cells. |
| 2. Verifier (self-verification) | One-line verifier on all models and precisions; `analysis/objective_review_2026-10-16/11_verifier_by_precision.py` | Done. Adds little or nothing beyond the vote; at or below chance for DeepSeek and Gemma-1B, near chance for Llama-3B (in-question AUROC 0.36-0.56); 0.62-0.73 for Qwen [Data]. |
| 3. Self-training (SFT, DPO) | `q08_sft_{16,4}bit_p500`, pilots, three DPO runs | **Paused by the mentor on 16 Oct** (see section 5). Results are one seed, 200 questions. |
| 4. Re-check with literature prompts | Token-signal pass (`scripts/score_token_signals.py`), critic verifier (`scripts/run_verifier_variants.py`), paper-vs-setup comparison | **In progress.** Token-signal run file for the 1,600 round-0 traces of `q08_sft_16bit_p500` is complete (1,601 lines); no analysis of it is saved yet. Critic runs are partial: 0.8B 241 lines, 0.8B forced-reasoning variant 214, 2B 9 (expected about 800+ per model) [Data: line counts of the run files]. |

Code state: branch `verifier-variants` (stacked on `token-signals`), not merged to `main`. Uncommitted at the time of writing: `src/experiment_d/web_export.py`, `tests/test_web_export.py`, `web/assets/js/results.js` and the new `analysis/objective_review_2026-10-16/`.

## 3. Prior work and replication status

Full tables: [Docs/research_notes/SETUP_COMPARISON.md](Docs/research_notes/SETUP_COMPARISON.md) (23 papers grouped by signal: vote, verifier, confidence, gold labels; its section 5 is the replication table) and [RESEARCH_DOSSIER_PAPERS.md](RESEARCH_DOSSIER_PAPERS.md) (paper-by-paper review). Both were built from tool summaries of the papers, not from careful reading; **check any number against the paper before quoting it.**

| Paper / method | Contribution relevant to us [Lit] | Our status |
|---|---|---|
| Quantization Hurts Reasoning? (2504.04823) | 4-bit weight-only near lossless on large models, worse on small ones; pass@1 only | **Agrees** with our DeepSeek numbers; the generator accuracy result is a replication, not a contribution [Data] |
| MACA (2509.15172) | Vote-based self-training on a 4-bit base; never varies precision | Closest paper; **not reproduced** (needs 3 agents, 2 debate rounds, 256-token cap) |
| Self-consistency | Vote beats a single sample | **Reproduced**: about +18 points at 0.8B [Data] |
| Sharpening (2412.01951) | Self-training sharpens toward the plurality; gain is small | Our SFT loop is a sharpening operator [Int]; **not reproduced** as specified (best-of-N by log-likelihood, greedy baseline, 3 seeds) |
| LMSI, ScPO, STaR, ReST-EM | Vote or gold-filtered self-training | Not reproduced; LMSI/STaR/ReST-EM judged feasible at our scale, ScPO partly |
| TTRL, EMPO, INTUITOR, RENT, SRT, QuRL | Online RL on votes or confidence | Not feasible at our compute |
| Mind the Gap, generalisation-gap critic (2606.01075), self-verification | Verifier signal needs CoT and many samples; small models show no gap | **Partly tested**: one-line verifier shows none; the critic run is incomplete |

## 4. Research gaps

- Neither MACA nor 2504.04823 treats agreement calibration or pseudo-label precision as a function of bit-width [Lit, `Docs/research_notes/paper_planning/01_verified_literature_gaps.md`; the dossier asks for this search to be repeated before the paper is written].
- Four findings the dossier says the literature lacks, all about the agreement signal [Data]: voting hides per-sample damage; 8-bit flips equal the replicate noise floor; vote-filtered labels keep precision at 4-bit but lose yield; the drop in agreement estimates the drop in per-sample accuracy without labels.
- The precision-dependent effect of consensus SFT (helps 4-bit, not 16-bit) is the one training result worth following up, and it is one seed [Data, OBJECTIVE_REVIEW 2.3].

## 5. Issue register

| # | Issue | Evidence | Status | Impact | Next diagnostic |
|---|---|---|---|---|---|
| 1 | Saved CSVs of 4 Oct are wrong | Stored metrics predate extractor fixes; off by up to 21 points vs re-scoring (dossier 2.1) | **Confirmed** | Never quote them; reports re-score | none |
| 2 | Prompts differed across precisions for Llama and one CQA cell; MATH-500 Qwen-0.8B files follow three protocols | Dossier 2.2 | **Confirmed** | These cells are not valid precision comparisons | New runs (cannot be fixed by analysis) |
| 3 | Gemma-1B 4-bit collapse | Answer-style switch (dossier 2.4, 2.11) | **Confirmed**, no conclusion depends on it | Pooled magnitudes shrink about a third without it | none |
| 4 | Label drift: labels added by a trained model are 57-81% correct; unanimous labels decay (61/62 to 148/167 at 16-bit) | `analysis/objective_review_2026-10-16/01_label_dynamics.py` | **Confirmed** in these runs; cause (agreement is a property of the model being trained) is [Int] | Rising agreement after training is not evidence of accuracy | Frozen-label run (`--labels frozen`, not implemented) |
| 5 | Brevity trap at 16-bit: gain comes from finishing more, accuracy when finished flat or down | OBJECTIVE_REVIEW 2.3, `04`, `05`, `06` | Measured; mechanism [Int] | Self-training gain at 16-bit is not a reasoning gain | Longer token cap or frozen-label run |
| 6 | 4-bit gain (+6.4 [+3.4, +9.6] single sample) concentrated on quantization-damaged questions | Same | Measured once; "variance vs bias" is [Hyp] | The headline training result | Second seed; matched controls |
| 7 | DPO runaway generation (cut-off 0.16 to 0.56) | `qwen08_4bit_dpo_similar1`; length does not explain it (chosen longer in 26/55 pairs) | **Cause unresolved**: weak anchoring and likelihood displacement are [Hyp]; EOS suppression not ruled out | One DPO run only | Log policy chosen/rejected/EOS log-probs; add anchor |
| 8 | **End-token mismatch**: all SFT and DPO adapters trained with `<|endoftext|>` as the end token, base model ends with `<|im_end|>` | `lora_training.end_token_id`; token-signal probe (TOKEN_SIGNALS_PLAN section 12) | Fact [Impl]; **effect not measured** | Could contribute to the "finishing" gain and to the DPO runaway | Retrain one round with `<|im_end|>` (needs approval: GPU) |
| 9 | `end_token_id` fix, `--pairing median_majority`, `--nll_weight`, `--labels frozen` | CLAUDE.md "planned fix" | **Not implemented, not run** | none yet | Only if self-training is resumed |
| 10 | Verifier prompt is the one-line verdict; literature designs differ (CoT, many verdicts, graded rate) | SETUP_COMPARISON sections 4, 8 | Critic implemented [Impl]; runs incomplete | The verifier conclusion ("no signal at 0.8B") is only for this prompt | Finish critic run for 0.8B and 2B; gate in VERIFIER_VARIANTS_PLAN section 4 |
| 11 | Single quantizer, 512-token cap, no greedy baseline, 1-shot `####` prompt | Dossier section 1; SETUP_COMPARISON 6 | Known limits | Claims limited to this setup | Greedy and zero-shot comparisons (cheap) |
| 12 | Course paper numbers vs dossier numbers | Memory note; `analysis/paper_100q_2026-10-13/` | Rule: paper uses the 100-question analysis | Avoid mixing | none |
| 13 | `Docs/*.pdf` are image-only and unread | Dossier section 10 | Open | Mentor constraints may be missing | Read manually |

## 6. Next steps, in priority order

The mentor's direction (16 Oct): stop self-training; re-check generator and verifier with prompts from the literature; reproduce published self-improvement setups, starting with Sharpening (2412.01951).

1. **Finish the critic verifier run** for Qwen3.5-0.8B (then 2B), 16-bit. Success: the gate in `Docs/research_notes/VERIFIER_VARIANTS_PLAN.md` section 4 (in-question AUROC at least 0.75, or a vote combined with the critic beating the plain vote with an interval above zero; bars to beat: vote 0.814, one-line verifier 0.685). Needs GPU; ask first.
2. **Analyse the existing token-signal file** (no GPU): entropy, self-certainty and sequence log-likelihood against correctness within questions, per `Docs/research_notes/TOKEN_SIGNALS_PLAN.md` section 7. Success: beats the vote's 0.814 in-question AUROC or adds to it.
3. **Cheap generator re-checks**: greedy decoding, zero-shot prompt, 1,024-token cap, at 16 and 4-bit (SETUP_COMPARISON section 6).
4. **Sharpening replication** with a greedy baseline and 3 seeds, 16 and 4-bit.
5. Only if self-training resumes: fix the end token (issue 8), then frozen labels (issue 4), then a second seed of the 4-bit result (issue 6).
6. Before writing the paper: repeat the gap search and re-read the papers marked S or A (dossier section 10).

## 7. Document map (one authoritative home for each topic)

| Topic | Authority |
|---|---|
| Layout, commands, how to run | [README.md](README.md), [CLAUDE.md](CLAUDE.md) |
| Current status, issues, next steps | this file |
| Findings from phases 1-2, decisions log, reproduction | [RESEARCH_DOSSIER.md](RESEARCH_DOSSIER.md) (last full revision 9 Oct; does not cover phase 3 or phase 4) |
| Literature review | [RESEARCH_DOSSIER_PAPERS.md](RESEARCH_DOSSIER_PAPERS.md) |
| Papers against our setup, replication feasibility, prompts | [Docs/research_notes/SETUP_COMPARISON.md](Docs/research_notes/SETUP_COMPARISON.md) |
| Self-training diagnosis (phase 3) | [Docs/research_notes/OBJECTIVE_REVIEW.md](Docs/research_notes/OBJECTIVE_REVIEW.md) |
| Token-signal plan and probe | [Docs/research_notes/TOKEN_SIGNALS_PLAN.md](Docs/research_notes/TOKEN_SIGNALS_PLAN.md) |
| Critic verifier plan | [Docs/research_notes/VERIFIER_VARIANTS_PLAN.md](Docs/research_notes/VERIFIER_VARIANTS_PLAN.md) |
| Paper outline, gap search, simulated reviewer report | `Docs/research_notes/paper_planning/` |
| Original proposal and mentor documents | `Docs/` (top level) |
| Archived material and why | [archive/README.md](archive/README.md) |
