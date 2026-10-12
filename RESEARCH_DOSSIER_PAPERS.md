# Research dossier: literature review (section 3)

*Companion to [RESEARCH_DOSSIER.md](RESEARCH_DOSSIER.md). This is section 3 of Part B, moved here unchanged so the main dossier stays shorter. Section numbers (3.1 to 3.5) and the [Lit] / [Lit-sum] labels are the same as in the dossier, and "section 2", "section 4" and "section 6" below refer to the dossier. The synthesis that draws on these papers is dossier section 4. A browsable version with more papers is `web/previous_works.html`.*

## 3. Literature: critical review

"How read": **F** = I read the relevant sections of the full text; **S** = tool summary of the full text; **A** = abstract only.

### 3.1 Quantization and reasoning accuracy

**Quantization Hurts Reasoning?** ([2504.04823](https://arxiv.org/abs/2504.04823)) — F
- *Design:* DeepSeek-R1-Distill 1.5B–70B, QwQ-32B, Qwen3-8B; AWQ, GPTQ, KV-cache and weight-activation methods at 3/4/8-bit; AIME-120, MATH-500, GSM8K, GPQA, LiveCodeBench; temperature 0.6, 32,768-token limit, 3 seeds, mean ± sd.
- *Finds:* 4-bit weight-only is near lossless (1.5B: GSM8K 84.5 to 82.7/83.3); 3-bit is risky (64.5 AWQ, 77.8 GPTQ); smaller models and harder tasks lose more; aggressive quantization lengthens outputs in small models; GPTQ needs reasoning-domain calibration data (+9.8 points at 3-bit).
- *Weak points:* pass@1 only; no voting, confidence, verification or training; three seeds but no paired test.
- *For us:* settles the accuracy question. Gives a reference number, public quantized checkpoints for our anchor model, and a warning that our 512–2048-token caps are far too small for reasoning models.

**Quantization Meets Reasoning** ([2501.03035](https://arxiv.org/abs/2501.03035); follow-up [2505.11574](https://arxiv.org/abs/2505.11574) A) — S
- *Finds:* Llama-3.2-1B/3B and 3.1-8B lose 5–32% on MATH under AWQ/GPTQ 4-bit; repair with DPO + QLoRA on 545 *labelled* failures.
- *Weak points:* error labels come from five judge models; recovery needs ground truth.
- *For us:* shows fine-tuning a quantized model on curated data works. Our question is whether *self-generated* labels suffice.

**Dettmers & Zettlemoyer** ([2212.09720](https://arxiv.org/abs/2212.09720)) A; **Lee et al.** ([2409.11055](https://arxiv.org/abs/2409.11055)) A
- 4-bit is near-optimal for zero-shot accuracy per bit; quantized models struggle with instruction following. Background only.

### 3.2 Quantization and reliability

**Does Compression Preserve Uncertainty?** ([2606.01850](https://arxiv.org/abs/2606.01850)) — F (design, limitations) + S
- *Design:* 12 models 1–70B; RTN/AWQ/GPTQ and weight-activation methods, pruning; five six-option multiple-choice tasks; conformal set size from one forward pass.
- *Finds:* accuracy and set size decouple; small models inflate more; inflation is threshold-like.
- *Weak points:* no seeds or intervals; needs labelled calibration data; authors state open-ended generation is future work.
- *For us:* closest "uncertainty under compression" benchmark; does not touch sampled reasoning.

**Reliability Scaling Laws** ([2607.10855](https://arxiv.org/abs/2607.10855)) — F (setup, future work) + S
- *Design:* six quantizers at 2–8 bit; TriviaQA and CoQA with 20-token answers; token-entropy AUROC.
- *Finds:* reliability is non-monotonic in total bits; 4-bit is a good trade-off.
- *Weak points:* short factual answers; authors list the effect of fine-tuning on reliability as future work.
- *For us:* includes bitsandbytes and Llama-3.2-1B/3B, so our quantizer is defensible as one of several.

**When Quantization Affects Confidence** ([2405.00632](https://arxiv.org/abs/2405.00632)) — F
- *Finds:* GPTQ 4-bit lowers true-class confidence, raises entropy, and moves low-confidence items most.
- *Weak points:* 4-bit GPTQ only; multiple choice; no intervals on the bin analysis.
- *For us:* partial disagreement with our absolute-loss pattern (section 2.5); agreement on flattening.

**Accuracy Is Not All You Need** ([2407.09141](https://arxiv.org/abs/2407.09141)) — F (definitions, noise argument) + S
- *Finds:* answers flip in both directions while accuracy is preserved; flips correlate with KL divergence (ρ = 0.98 on MMLU); models quantization as symmetric noise.
- *Weak points:* deterministic evaluation, so no sampling floor; calls predicting degradation "a challenging and open problem".
- *For us:* our replicate floor is the missing control for sampled generation.

**Illusion of Equivalency** ([2607.08734](https://arxiv.org/abs/2607.08734)) S; **Self-explanations** ([2601.00282](https://arxiv.org/abs/2601.00282)) S; **BitCal-TTS** ([2605.05561](https://arxiv.org/abs/2605.05561)) A
- GGUF correctness agreement without a noise floor; greedy self-explanations on 7–72B models; 4-bit miscalibrates early-halting confidence (54 and 35 GSM8K questions). Adjacent; none studies agreement or verification.

### 3.3 Self-consistency and self-verification (full precision)

**When Should a Language Model Trust Itself?** ([2605.02915](https://arxiv.org/abs/2605.02915)) — F (methods, limitations) + S
- *Design:* same-model True/False logit verifier; ARC-Challenge and TruthfulQA-MC; AUROC and risk–coverage with bootstrap.
- *Finds:* large gains over likelihood on ARC; harmful for TinyLlama and DeepSeek-R1-8B on TruthfulQA.
- *Weak points:* "larger" models are loaded in 4-bit and the paper does not say which, so precision is an uncontrolled factor in its own comparisons; multiple choice only; authors ask for open-ended tasks and self-consistency baselines.
- *For us:* a logit verifier on free-form GSM8K traces, compared against the vote, at controlled precision, answers three of their stated limitations.

**Language Models (Mostly) Know What They Know** ([2207.05221](https://arxiv.org/abs/2207.05221)) A — origin of the P(True) probe.

**When Self-Consistency Backfires** ([2608.11403](https://arxiv.org/abs/2608.11403)) S; **When LLMs Agree, Are They Right?** ([2607.08065](https://arxiv.org/abs/2607.08065)) S; **Decomposing Wrong-Consensus** ([2608.18795](https://arxiv.org/abs/2608.18795)) S
- On hard multiple-choice science, agreement is a weak or non-monotone confidence signal; wrong-run agreement is mostly per-question answer preference.
- *For us:* agreement is strong on GSM8K (AUROC ≈ 0.87) and weak on CommonsenseQA (0.72), matching their task dependence. A reviewer will ask whether our conclusions survive a hard benchmark.

**Cross-Model Disagreement** ([2603.25450](https://arxiv.org/abs/2603.25450)) S — verifier perplexity on another model's answer; never pairs a model with its own quantized copy. A natural baseline if a full-precision model is available.

### 3.4 Self-training

**Sharpening Mechanism** ([2412.01951](https://arxiv.org/abs/2412.01951)) — F (definitions) + S
- *Finds:* the sample cost of self-improvement scales with a coverage coefficient; theory assumes a clean self-reward.
- *For us:* justifies reporting coverage; says nothing about a biased verifier.

**LMSI** ([2210.11610](https://arxiv.org/abs/2210.11610)) — F (method) 
- Keeps every path that reaches the majority answer, with no threshold, relying on agreement being calibrated (their Fig. 2); a 540B model.
- *For us:* the canonical vote filter. Calibration at 540B does not transfer to 1B models; ours is the small-model check.

**Can Large Reasoning Models Self-Train?** ([2505.21444](https://arxiv.org/abs/2505.21444)) — F (§4.2, limitations, A.1, G.4) + S
- *Finds:* majority-vote reward matches ground-truth RL, then collapses into a template answer; fewer rollouts per prompt delay collapse (untested hypothesis: noisier votes are harder to hack); asks for better verifiers.
- *Weak points:* 4–14B models at full precision on large GPUs.
- *For us:* two-sided prediction for a quantized loop (section 6, phase 2).

**MACA** ([2509.15172](https://arxiv.org/abs/2509.15172)) — F (appendices D, E, G, I, L, N) + S
- *Design:* 4-bit bitsandbytes QLoRA, 256-token cap, 1,500 train / 500 test per dataset.
- *Finds:* majority-vote labels train as well as ground truth; iterations 2–3 add 1–3 points; no collapse in three iterations; Table 16 shows gains of similar size evaluated at 4-bit and at full precision.
- *Weak points:* base truncation rates of 75–83% under the 256-token cap; up to 38 of 60 points of one gain is formatting (their own Table 22); precision never varied in training.
- *For us:* establishes that a 4-bit model can self-train. Supplies the decomposition (completion / format / reasoning) any loop of ours must report.

**On the Generalization Gap in Self-Evolving LMs** ([2606.01075](https://arxiv.org/abs/2606.01075)) — F
- *Design:* same generator–verifier game as ours, but 8 candidates × 16 sampled verdicts with a threshold; DPO; full fine-tuning.
- *Finds:* Gemma-3-1B gains 0.6 points, 4B gains 31% relative; verifier recall is the bottleneck; pass@1 rises while pass@32 stays flat.
- *Weak points:* verifier accuracy measured on 60–100 examples.
- *For us:* a threshold over sampled verdicts is the literature's way to get a graded verifier score. Coverage as ceiling supports our reading of 2.3.

**Self-Improvement Reversal** ([2407.05013](https://arxiv.org/abs/2407.05013)) S; **Rise-and-Collapse** ([2606.21090](https://arxiv.org/abs/2606.21090)) S; **QuRL** ([2602.13953](https://arxiv.org/abs/2602.13953)) S; **STaR** ([2203.14465](https://arxiv.org/abs/2203.14465)) A
- pass@1 gains can hide falling diversity and OOD accuracy; collapse within a training campaign; with a quantized actor, weight updates are far smaller than quantization error (ground-truth rewards). Supply metrics and cautions for phase 2.

### 3.5 Method references

- Difference of Confidences ([2107.03315](https://arxiv.org/abs/2107.03315)) F (definition); Agreement-on-the-Line ([2206.13089](https://arxiv.org/abs/2206.13089)) A — label-free accuracy estimation.
- xFinder ([2405.11874](https://arxiv.org/abs/2405.11874)) A — regex answer extraction reached 74% accuracy in the best framework they tested.
- Adding Error Bars to Evals ([2411.00640](https://arxiv.org/abs/2411.00640)) A — formulas for comparing two models and planning sample size.
