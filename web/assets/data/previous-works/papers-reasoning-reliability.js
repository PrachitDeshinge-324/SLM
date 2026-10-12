/* Quantization and reasoning models; quantization, calibration and reliability. */
window.PW.papers.push(

/* ================= QUANTIZATION AND REASONING MODELS ================= */
{
    id: 'quant-hurts-reasoning', title: 'Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models',
    year: 2025, posted: '2025-04', authors: 'Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, Chun Yuan, Lu Hou',
    venue: 'COLM 2025', status: 'published', cat: 'quant-reasoning', origin: 'dossier', dossier: '§3.1', prio: 1, arxiv: '2504.04823',
    tags: ['deepseek-r1-distill', 'awq', 'gptq', 'kv cache', 'w4a4', 'aime', 'math-500', 'gsm8k', 'qwq', 'qwen3'],
    summary: 'Reasoning models produce long chains of thought, but quantization methods were built for non-reasoning LLMs. This is the first systematic study of weight-only, weight-activation and KV-cache quantization on reasoning models, with bit-width and algorithm recommendations.',
    method: [
        'Models: DeepSeek-R1-Distill-Qwen 1.5B, 7B, 14B, 32B; DeepSeek-R1-Distill-Llama 8B and 70B; QwQ-32B; Qwen3-8B.',
        'Weight-only GPTQ and AWQ at W4 and W3 (group size 128); KV cache KVQuant* and QuaRot at 4 and 3 bits; weight-activation SmoothQuant, QuaRot, FlatQuant at W8A8KV8 and W4A4KV4, plus MXFP4.',
        'Calibration: GPTQ and SmoothQuant on 128 self-generated reasoning sequences of 2048 tokens from Numina-Math-1.5; AWQ and KVQuant* on 128 Pile sequences of 512 tokens; FlatQuant on WikiText2.',
        'Benchmarks AIME-120, MATH-500, GSM8K, GPQA-Diamond, LiveCodeBench; temperature 0.6, top-p 0.95, up to 32,768 generated tokens, three seeds, mean and standard deviation.'
    ],
    results: [
        'R1-Distill-Qwen-1.5B average over benchmarks (BF16 49.0): AWQ W4 46.9, GPTQ W3 41.4, AWQ W3 30.4, QuaRot W8A8KV8 49.0, FlatQuant W4A4KV4 38.2 (Table 1).',
        'R1-Distill-Qwen-32B (BF16 74.8): AWQ W4 74.4, GPTQ W3 69.6, FlatQuant W4A4KV4 71.9, MXFP4 62.9.',
        'Switching 1.5B GPTQ W3 calibration from WikiText2 to Numina-Math-1.5 raises the average from 28.3 to 38.1 (Table 2).',
        'Conclusions: W8A8KV8 is lossless; 4-bit weight-only is nearly lossless and 3-bit is risky; smaller models and harder tasks (AIME) lose more; aggressive quantization can lengthen outputs; quantized models gain less from test-time scaling (Sec. 4.4).'
    ],
    limits: [
        'The authors say the underlying mechanisms "remain incompletely understood"; improving 4-bit weight-activation methods for reasoning models is an open challenge.',
        'pass@1 only, three seeds with no paired test; no voting, confidence, verification or training; Qwen3 is only evaluated at 8B.'
    ],
    relevance: 'Settles the accuracy question, so "4-bit hurts small reasoning models" is replication for us, not a contribution. It also provides reference numbers for DeepSeek-R1-Distill-Qwen-1.5B, public quantized checkpoints, and the warning that our 512 to 2048 token caps are far below the 32,768 used for reasoning models.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2504.04823' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2504.04823' },
        { l: 'Code', u: 'https://github.com/ruikangliu/Quantized-Reasoning-Models' }
    ],
    read: 'main', note: 'Venue from the arXiv comments field ("COLM 2025"), the PDF header ("Published as a conference paper at COLM 2025") and the authors\' repository; a COLM or OpenReview page could not be located. The extracted Table 1 gave benchmark averages, so the per-benchmark GSM8K cells the dossier quotes (84.5 to 82.7/83.3 for the 1.5B model) were not re-checked in this pass.'
},
{
    id: 'quant-meets-reasoning-v1', title: 'Quantization Meets Reasoning: Exploring LLM Low-Bit Quantization Degradation for Mathematical Reasoning',
    year: 2025, posted: '2025-01', authors: 'Zhen Li, Yupeng Su, Runming Yang, Congkai Xie, Zheng Wang, Zhongwei Xie, Ngai Wong, Hongxia Yang',
    venue: 'arXiv preprint', status: 'preprint', cat: 'quant-reasoning', origin: 'dossier', dossier: '§3.1', alsoSite: true, prio: 1, arxiv: '2501.03035',
    tags: ['llama-3', 'awq', 'gptq', 'math', 'error taxonomy', 'dpo', 'qlora', 'restoration'],
    summary: 'Low-bit AWQ and GPTQ degrade Llama-3 on MATH. The paper builds a multidimensional error-analysis pipeline over the failures and repairs quantized models by DPO fine-tuning on a small set of error-specific examples.',
    method: [
        'Llama-3.1-8B, Llama-3.2-3B and Llama-3.2-1B Instruct; GPTQ and AWQ at W4A16; format alignment with LoRA (full precision) or QLoRA (quantized), rank 32, on PRM800K.',
        'Failures are judged by five LLMs with majority vote into seven error types in four categories (conceptual, method, execution, reasoning); about 2% human verification. Benchmarks MATH and MATH-500 via OpenCompass.',
        'Restoration: DPO with QLoRA on 545 examples sampled from 3,329 quantized failure cases, learning rate 1e-6, global batch 8.'
    ],
    results: [
        'MATH accuracy, BF16 against AWQ / GPTQ: 8B 47.2 / 41.8 / 43.06; 3B 40.2 / 38.8 / 31.6; 1B 18.6 / 14.26 / 14.94 (Table 1). Abstract: up to 32.39% degradation, 11.31% on average.',
        'Conceptual errors are more than half of mistakes; method and execution errors rise under quantization.',
        'After DPO the 1B models reach 18.2 and 18.8 (full precision 21) and 3B reaches 41.6 and 38.2 (full precision 37.4) on MATH-500 (Table 4).'
    ],
    limits: [
        'The authors cite limited time and compute; only W4A16 with two methods and only the Llama-3 family.',
        'Failure labels come from LLM judges with about 2% human checking; recovery uses a ground-truth-based preference set. The paper has internal inconsistencies (several Table 1 deltas do not match the score differences; 3,329 against 3,366 failure cases).'
    ],
    relevance: 'Shows that fine-tuning a quantized model on curated data repairs math damage. Our question is the self-supervised counterpart: do the model\'s own voted labels suffice, and does quantization change how many of them there are.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2501.03035' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2501.03035' }
    ],
    read: 'full', note: 'This and 2505.11574 are two separate arXiv entries with different author lists, not versions of one paper.'
},
{
    id: 'quant-meets-reasoning-v2', title: 'Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning',
    year: 2025, posted: '2025-05', authors: 'Zhen Li, Yupeng Su, Songmiao Wang, Runming Yang, Congkai Xie, Aofan Liu, Ming Li, Jiannong Cao, Yuan Xie, Ngai Wong, Hongxia Yang',
    venue: 'arXiv preprint (ICLR 2026 submission)', status: 'preprint', cat: 'quant-reasoning', origin: 'dossier', dossier: '§3.1', prio: 2, arxiv: '2505.11574',
    tags: ['silver bullet', 'dpo', 'qwen2.5', 'llama-3', 'smoothquant', 'step-level error'],
    summary: 'Extends the same line to Qwen2.5 and Llama-3 with a "measure, locate, restore" loop: find the first reasoning step where a quantized model goes wrong, then fine-tune with DPO on "Silver Bullet" preference data built from that step.',
    method: [
        'Qwen2.5-Instruct 0.5B to 7B and Llama-3 1B, 3B, 8B; AWQ and GPTQ at W4A16 and SmoothQuant at W8A8; evaluated on GSM8K, MATH and AIME (90 problems), restored on MATH-500, GSM8K and MMLU.',
        'Step-level error attribution with DeepSeek-R1 as pivot judge, four more LLMs voting and two human annotators for disagreements; accepted cases audited at 2%.',
        'Silver Bullet data: erroneous traces truncated at the first error step and completed by Llama-3.2-70B and Qwen2.5-Max; DPO with beta 1, LoRA rank 32, learning rate 1e-6.'
    ],
    results: [
        'Qwen2.5-0.5B MATH: 23.98 vanilla against 7.24 under GPTQ, a 69.81% relative drop; Qwen2.5-7B MATH 72.48 against 69.84 under AWQ (Table 1).',
        'Average degradation 7.16% on GSM8K and 15.18% on MATH; full-precision errors are 59.6% conceptual, 12.5% method, 13.0% execution, 13.1% reasoning.',
        'Targeted data (ALL-S) averages 57.42 against 55.73 for random non-targeted data and 51.83 for all non-silver data (Table 2); 332 examples and 3 to 5 minutes on one GPU are claimed to recover W4A16 math reasoning.'
    ],
    limits: [
        'Only PTQ at W4A16 and W8A8 with three methods; mostly models of 8B or less; no variance reporting in the ablation table.',
        'The judge pipeline favours precision over recall; completions come from very large models, so the repair depends on access to them.'
    ],
    relevance: 'Gives a step-level view of where quantized small models first go wrong, useful for explaining why agreement among samples drops. Like the first entry, the repair needs an external answer key or a bigger model, unlike a self-labelled loop.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2505.11574' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2505.11574' }
    ],
    read: 'main', note: 'Version history: v1 (May 2025) had a different title; v3\'s PDF header says "Published as a conference paper at ICLR 2026" while v4 (Jan 2026) says "Under review as a conference paper at ICLR 2026". Acceptance could not be confirmed, so it is shown as a preprint.'
},
{
    id: 'quantized-think-longer', title: 'Quantized Reasoning Models Think They Need to Think Longer, but They Do Not',
    year: 2026, posted: '2026-05', authors: 'Sanae Lotfi, Polina Kirichenko, Steven Li, Zechun Liu',
    venue: 'arXiv preprint', status: 'preprint', cat: 'quant-reasoning', origin: 'new', prio: 1, arxiv: '2606.00206',
    tags: ['overthinking', 'chain of thought length', 'logit penalty', 'awq', 'gptq', 'flatquant', 'wait'],
    summary: 'Quantization lowers accuracy and lengthens chains of thought. The authors argue quantized models often reach a correct intermediate answer and then fail to commit to it, and show a training-free logit penalty on 50 "overthinking" tokens shortens traces while preserving accuracy.',
    method: [
        'DeepSeek-R1-Distill-Qwen 1.5B, 7B, 14B, R1-Distill-Llama-8B and QwQ-32B; GPTQ and AWQ at 3 and 4 bits, FlatQuant at W4A4KV4 and W8A8KV8; GSM8K, MATH-500, AIME-120, LiveCodeBench, GPQA-Diamond; temperature 0.6, top-p 0.95.',
        'Error categories (overthinking, logical, arithmetic, formatting) assigned by GPT-5 as judge with over 95% agreement with humans after prompt tuning; token-level KL between BF16 and 3-bit AWQ.',
        'Fix: subtract a fixed penalty lambda (0.5 to 4.0) from the logits of 50 markers such as "Wait", "But", "Alternatively" at every decoding step; ablations penalize high-KL, low-KL and random tokens.'
    ],
    results: [
        'Qwen-1.5B on MATH-500: BF16 85.6% with 5.2K tokens; 3-bit AWQ 47.0% with 23.4K tokens (4.5x longer); GPTQ W3 71.6% with 6.8K; FlatQuant W4A4KV4 66.4% with 8.0K.',
        'Accuracy drop and CoT growth correlate with Spearman rho = -0.73 over 28 model-quantization pairs; overthinking accounts for 52% of 3-bit AWQ errors against 26% in BF16.',
        'The penalty cuts CoT length by 4.1% to 28.0% in Table 1, with accuracy preserved or improved in most configurations (+6.2 points for Qwen-1.5B AWQ W3).'
    ],
    limits: [
        'Table 1 reports the best lambda per configuration, selected on the same benchmarks, so gains are optimistic; a fixed lambda and a hand-curated English marker list.',
        'Error categories rely on an LLM judge; the KL analysis uses one quantized model on one benchmark.'
    ],
    relevance: 'Directly relevant to our truncation problem: if a quantized reasoning model writes longer, a fixed token cap cuts off more of its answers and looks like lost accuracy. It supports the dossier\'s plan to set the token limit from a pilot and report truncated, answered and runaway samples separately.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2606.00206' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2606.00206' }
    ],
    read: 'main'
},
{
    id: 'quant-inflates-reasoning', title: 'Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models',
    year: 2026, posted: '2026-06', authors: 'Xinyu Lian, Walid Krichene, Beichen Huang, Masahiro Tanaka, Olatunji Ruwase, Li Zhang, Minjia Zhang',
    venue: 'arXiv preprint', status: 'preprint', cat: 'quant-reasoning', origin: 'new', prio: 2, arxiv: '2606.25519',
    tags: ['token inflation', 'ctir', 'chain of thought length', 'int3', 'int4', 'serving latency'],
    summary: 'Low-bit quantization of reasoning models can preserve final-answer accuracy while lengthening chains of thought, which cancels the expected per-token speedup. The paper defines a CoT Token Inflation Ratio (CTIR) to measure it.',
    method: [
        'Qwen3-4B-Thinking-2507, Phi-4-Reasoning-Plus (14B) and Qwen3-30B-A3B-Thinking-2507; weight-only RTN, GPTQ, AWQ, HQQ and ParoQuant at INT4 and INT3, group size 128; vLLM, temperature 0.',
        'AIME25, GSM8K, LiveCodeBench, MBPP, GPQA-Diamond, BBH (100-question subsets except AIME25) and general tasks; reasoning-span token counts; step counts and repetitiveness via embeddings; serving TTFT and throughput on H100.',
        'Mitigations tested: repetition penalty 1.1 to 1.4, conciseness prompts, different calibration corpora, DiscQuant QAT.'
    ],
    results: [
        'CTIR: Qwen3-4B INT4 RTN +42.5% on average; INT3 +276.1% (AWQ) and +292.1% (HQQ); Qwen3-30B INT3 +32.8% (AWQ) and +60.1% (HQQ) (Table 2).',
        'A 1.1x CTIR increase raises p90 time-to-first-token from 1050 s to 1400 s and lowers throughput from 0.055 to 0.047 requests/s.',
        'Switching INT3 GPTQ calibration from WikiText to math chain-of-thought data raises accuracy from 28% to 65% and cuts CoT length from 30k to 12k tokens; QAT is the only Pareto-dominant mitigation.'
    ],
    limits: [
        'Weight-only PTQ only, three models, 100-question benchmark subsets, single runs at temperature 0.',
        'The QAT claim rests on one method and one comparison; Phi-4 scores 0% on LiveCodeBench because of format mismatches.'
    ],
    relevance: 'Provides a way to measure the length effect separately from correctness, and shows calibration data changes it a lot. Both feed our token-limit pilot and the choice of calibration data for AWQ/GPTQ.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2606.25519' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2606.25519' }
    ],
    read: 'full'
},
{
    id: 'reasoning-qat', title: 'What Makes Low-Bit Quantization-Aware Training Work for Reasoning LLMs? A Systematic Study',
    year: 2026, posted: '2026-01', authors: 'Keyu Lv, Manyi Zhang, Xiaobo Xia, Jingchen Ni, Shannan Yan, Xianzhi Yu, Lu Hou, Chun Yuan, Haoli Bai',
    venue: 'arXiv preprint', status: 'preprint', cat: 'intersection', origin: 'new', prio: 2, arxiv: '2601.14888',
    tags: ['qat', 'knowledge distillation', 'grpo', 'w3', 'w2', 'cold start', 'rtn', 'gptq'],
    summary: 'Post-training quantization at 3 and 2 bits collapses reasoning models. This study compares quantization-aware training recipes and settles on a three-stage workflow, PTQ initialization, distillation recovery from the full-precision teacher, then cold-start GRPO.',
    method: [
        'DeepSeek-R1-Distill-Qwen-1.5B (SFT-trained), Qwen3-0.6B and Qwen3-4B (RL-trained); W3G128 and W2G128 weight-only, plus W4A4KV4 in an appendix.',
        'Objectives SFT, KD (KL to the full-precision teacher) and GRPO; RTN and GPTQ initialization (FlatQuant for W4A4KV4); OpenR1-Math 94k for SFT and KD; 6,000 KD steps, GRPO 250 steps with group size 8.',
        'LightEval with vLLM, temperature 0.6, up to 32,768 tokens, three seeds; baselines RTN, GPTQ, AWQ, EfficientQAT, BitDistiller, QuaRot, FlatQuant.'
    ],
    results: [
        'KD loses 8.06 average points against 10.51 for SFT on R1-Qwen-1.5B (W3G128) and 9.26 against 21.40 on Qwen3-4B (Table 1).',
        'GPTQ initialization reaches 52.42 average against 10.41 for RTN on R1-Qwen-1.5B; zero-RL on RTN does not recover (10.41 to 10.84).',
        'Qwen3-0.6B W3G128: GPTQ 12.61 against 31.67 for the full workflow (BF16 41.10); Qwen3-4B W2G128: GPTQ 6.16 against 41.89.'
    ],
    limits: [
        'Training is math-centred, so coding and science generalization is limited, especially for small models at extreme low bit.',
        'QuaRot-initialized W4A4 models collapse under RL, so the workflow needs a strong PTQ start; baselines compared on one model and setting; the HTML reference list did not render.'
    ],
    relevance: 'Supplies evidence that RL on a quantized small reasoning model can recover accuracy given a distillation cold start, and that some starts collapse. Both are relevant if a phase-2 loop uses a quantized actor.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2601.14888' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2601.14888' }
    ],
    read: 'full', note: 'Table 6 (hyperparameter column mapping) was garbled in retrieval; values come from the main text.'
},
{
    id: 'compression-uncertainty', title: 'Does Compression Preserve Uncertainty? A Unified Benchmark for Quantized and Sparse LLMs via Conformal Prediction',
    year: 2026, posted: '2026-06', authors: 'Yujia Tong, Yuxi Wang, Yunyang Wan, Tian Zhang, Junhao Dong, Jingling Yuan',
    venue: 'arXiv preprint', status: 'preprint', cat: 'quant-reliability', origin: 'dossier', dossier: '§3.2', prio: 1, arxiv: '2606.01850',
    tags: ['conformal prediction', 'set size', 'uncertainty', 'rtn', 'awq', 'gptq', 'quarot', 'spinquant', 'pruning'],
    summary: 'Compression is judged by accuracy. This benchmark asks whether compressed LLMs still quantify uncertainty, using conformal prediction set size from one forward pass over 12 models and many compression configurations.',
    method: [
        '12 LLMs from four families, 1B to 70B; W4A16 with RTN, AWQ, GPTQ; W4A4 with QuaRot, SpinQuant, FlatQuant; 50% unstructured pruning (magnitude, SparseGPT, Wanda) and 20% structured pruning; a combined setting.',
        'MMLU, CosmosQA, HellaSwag, HaluDial, HaluSum; 2,000 instances per dataset split 50/50 into calibration and test; six options including "I don\'t know"; alpha = 0.1; predictions from option-letter logits, averaged over two score functions and three prompts.',
        'Metrics accuracy, coverage rate and set size on eight H100 GPUs.'
    ],
    results: [
        'W4A16 is most stable: Llama2-70B AWQ 73.20 accuracy / 2.18 set size against FP16 72.48 / 2.17; W4A4 is disruptive: Llama2-7B QuaRot accuracy 47.09 to 22.04 and set size 3.09 to 5.65 (Table 1).',
        'Accuracy and set size decouple: Llama3.1-8B under 50% Wanda drops 62.93% to 45.97% on QA accuracy while set size rises only 2.98 to 3.09.',
        'Larger models absorb inflation better (QuaRot set-size increase +2.56 for 7B, +0.72 for 13B, +0.37 for 70B); inflation is threshold-like in sparsity.'
    ],
    limits: [
        'Multiple-choice only; open-ended generation is future work; fixed error rate and split; no seeds or intervals.',
        'Only 2,000 instances per dataset; some cells are degenerate (SliceGPT on Qwen3-8B, coverage about 12%) and some rows are missing.'
    ],
    relevance: 'The closest benchmark of uncertainty under compression, and a template for our label-free angle. It uses one forward pass on multiple-choice, so it says nothing about agreement across sampled reasoning chains or verifier behaviour.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2606.01850' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2606.01850' }
    ],
    read: 'main'
},
{
    id: 'reliability-scaling', title: 'Reliability Scaling Laws for Quantized Large Language Models',
    year: 2026, posted: '2026-07', authors: 'Sirine Ayadi, Sándor Daróczi, Stephan Günnemann, Bertrand Charpentier',
    venue: 'ICML 2026 SCALE workshop (poster); arXiv comments: TMLR 2026', status: 'workshop', cat: 'quant-reliability', origin: 'dossier', dossier: '§3.2', alsoSite: true, prio: 1, arxiv: '2607.10855',
    tags: ['bitsandbytes', 'awq', 'gptq', 'hqq', 'aqlm', 'calibration', 'token entropy', 'perturbations', 'llama-3.2'],
    summary: 'Quantized LLMs are judged by accuracy on clean inputs. This paper adds uncertainty, calibration and robustness to 15 input perturbations and fits bit-level scaling laws, finding reliability peaks near 4 bits while accuracy rises monotonically with total bits.',
    method: [
        'LLaMA-3.2-1B, LLaMA-3.2-3B, LLaMA-3-8B, LLaMA-3-70B (plus OPT and Qwen3 checks); bitsandbytes, AWQ, GPTQ, HQQ, Quanto and AQLM at 2, 3, 4 and 8 bits, 63 configurations; Wanda and SparseGPT pruning; EfficientQAT.',
        'TriviaQA, CoQA, CommonsenseQA, 1,000 prompts over 5 seeds; multinomial sampling at temperature 0.7 with at most 20 new tokens; metrics accuracy, token-entropy AUROC and AUPR, Brier score, KL to the base model; empirical form L(B) = a(log B)^2 + b log B + c.'
    ],
    results: [
        'TriviaQA accuracy unperturbed 0.338 (1B), 0.549 (3B), 0.691 (8B), 0.728 (70B); entropy AUROC 0.683, 0.722, 0.715, 0.503 (Table 5).',
        'At a fixed bit budget, going from 16 to 4 bits improves perplexity and accuracy; this reverses at 3 and 2 bits; 4-bit models keep small KL to the base model.',
        'Quantization (GPTQ, AWQ, HQQ) improves AUROC under natural perturbations while accuracy stays close to the base; a 4-bit sweet spot also appears for OPT and Qwen3.'
    ],
    limits: [
        'Post-training quantization focus; short 20-token factual answers; 1,000-prompt subsamples; no formal predictive law; EfficientQAT only for QAT.',
        'Analysis is centred on LLaMA-3; latency measured on one GPU at batch size 1.'
    ],
    relevance: 'Includes bitsandbytes and Llama-3.2-1B/3B, so our quantizer and models are defensible as one of several in the literature. It measures token-level uncertainty on short answers, not agreement across long reasoning samples.',
    links: [
        { l: 'ICML 2026 workshop poster page', u: 'https://icml.cc/virtual/2026/67411', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2607.10855' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2607.10855' }
    ],
    read: 'main', note: 'Venue is ambiguous: the arXiv comments field says "Transactions on Machine Learning Research (TMLR), 2026" and the paper page says "Reviewed on OpenReview", but the ICML 2026 site lists it as a poster in the SCALE workshop. A TMLR acceptance page could not be opened.'
},
{
    id: 'quant-affects-confidence', title: 'When Quantization Affects Confidence of Large Language Models?',
    year: 2024, posted: '2024-05', authors: 'Irina Proskurina, Luc Brun, Guillaume Metzler, Julien Velcin',
    venue: 'Findings of NAACL 2024', status: 'published', cat: 'quant-reliability', origin: 'dossier', dossier: '§3.2', prio: 1, arxiv: '2405.00632',
    tags: ['gptq', '4-bit', 'confidence', 'calibration error', 'entropy', 'bloom', 'opt', 'mistral', 'llama'],
    summary: 'Studies how 4-bit post-training quantization changes model confidence and calibration, with attention to model family and size. Quantization lowers confidence on the true label, and the change is concentrated in samples where the full-precision model was already least confident.',
    method: [
        'Post-training GPTQ to 4-bit on causal LMs including BLOOM, OPT, Mistral and Llama families at several scales (BLOOM-7B1 is the repository\'s worked example).',
        'Zero-shot multiple-choice tasks through the EleutherAI harness: ARC-Easy, BoolQ, HellaSwag, OpenBookQA, PIQA, XStoryCloze, TruthfulQA (MC).',
        'Metrics: confidence assigned to the true label, calibration error, entropy and divergence of the output distribution; analysis of absolute confidence shifts by the original confidence bin.'
    ],
    results: [
        'Abstract: 4-bit GPTQ lowers the confidence on true labels, with effects varying across models and scales, and "disproportionately affects samples where the full model exhibited low confidence".',
        'Dossier cross-check: relative loss is larger in low-confidence bins, but in absolute terms our loss sits in mid and high-confidence questions.'
    ],
    limits: [
        'GPTQ at 4-bit only; multiple-choice tasks, so no free-form generation or sampling; per-bin analysis lacks intervals (dossier assessment).'
    ],
    relevance: 'Supports our finding that quantization flattens answer distributions, and partly disagrees with our absolute-loss pattern (dossier 2.5). It is the reason the dossier checks where on the confidence scale the damage falls.',
    links: [
        { l: 'ACL Anthology (Findings NAACL 2024)', u: 'https://aclanthology.org/2024.findings-naacl.124/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2405.00632' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2405.00632' },
        { l: 'Code and quantized models', u: 'https://github.com/upunaprosk/quantized-lm-confidence' }
    ],
    read: 'abstract', note: 'The dossier read this paper in full (label F). For this card only the abstract, the proceedings record, the authors\' README and fragments of the HTML tables could be retrieved (the HTML is mostly table markup), so numbers are not repeated here and the setup is taken from the README.'
},
{
    id: 'accuracy-not-all', title: 'Accuracy is Not All You Need',
    year: 2024, posted: '2024-07', authors: 'Abhinav Dutta, Sanjeev Krishnan, Nipun Kwatra, Ramachandran Ramjee',
    venue: 'NeurIPS 2024', status: 'published', cat: 'quant-reliability', origin: 'dossier', dossier: '§3.2', prio: 1, arxiv: '2407.09141',
    tags: ['flips', 'kl divergence', 'compression', 'mt-bench', 'quantization', 'pruning', 'slicegpt'],
    summary: 'Compressed LLMs are validated by checking that accuracy stays close. The paper shows accuracy can hide many answers flipping between right and wrong, proposes flips and KL divergence as distance metrics, and shows compressed models can be visibly worse on free-form generation despite equal accuracy.',
    method: [
        'Llama2-chat 7B, 13B, 70B and Yi-chat 6B and 34B; LLM.int8() via bitsandbytes (W8A8 and W4A4), GPTQ, AWQ, SmoothQuant, layer dropping, Wanda and SliceGPT.',
        'MMLU, PIQA, HellaSwag, ARC, LAMBADA, Winogrande, GSM8K, TriviaQA and WikiText-2 perplexity; MT-Bench scored by GPT-4; flips counted as answers changing from correct to incorrect and the reverse; KL divergence averaged over option tokens.'
    ],
    results: [
        'Llama2-70B-chat MMLU: BnB W8A8 changes accuracy by -0.55 with 3.32% flips and BnB W4A4 by -0.78 with 5.65% flips; the largest flip rate is 13.62% (Llama2-7B, SmoothQuant W8A8) (Table 5).',
        'Flips and KL divergence correlate at Spearman 0.981 on MMLU; harder tasks flip more (GSM8K 12 to 30% for BnB W8A8 and W4A4).',
        'MT-Bench for Llama2-70B-chat drops from 7.431 to 7.059 (W8A8) and 7.018 (W4A4), about 5%.',
        'Adding Gaussian noise to log-probabilities keeps perplexity at 5.70 while the share of greedy-matching tokens falls from 61.3% to 21.5%.'
    ],
    limits: [
        'Deterministic evaluation, so no sampling noise floor; qualitative MT-Bench review covers about 30 questions with about 10 shown.',
        'Flip counts in the text and tables are not fully consistent; some entries are missing (GPTQ on Yi).'
    ],
    relevance: 'The dossier\'s flip analysis is built on this paper, and the replicate noise floor we measure is the missing control for sampled generation. Its symmetric-noise explanation is what our one-directional 4-bit excess flips contradict.',
    links: [
        { l: 'NeurIPS 2024 proceedings', u: 'https://proceedings.neurips.cc/paper_files/paper/2024/hash/e0e956681b04ac126679e8c7dd706b2e-Abstract-Conference.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2407.09141' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2407.09141' }
    ],
    read: 'main'
},
{
    id: 'illusion-equivalency', title: 'The Illusion of Equivalency: Statistical Characterization of Quantization Effects in LLMs',
    year: 2026, posted: '2026-07', authors: 'Baha Rababah, Shahzeb Qamar, Lorenz Sparrenberg, Rafet Sifa, Murat Kantarcioglu, Cuneyt Gurcan Akcora, Carson K. Leung',
    venue: 'arXiv preprint', status: 'preprint', cat: 'quant-reliability', origin: 'dossier', dossier: '§3.2', alsoSite: true, prio: 2, arxiv: '2607.08734',
    tags: ['gguf', 'llama.cpp', 'correctness agreement', 'cohen kappa', 'attention weights', 'q2_k'],
    summary: 'Perplexity and accuracy parity do not imply decision-level equivalence. The paper defines Correctness Agreement, the probability that base and quantized models are both correct on the same prompts, and links attention-weight drift under 8-bit to 2-bit quantization to behaviour change.',
    method: [
        'Llama-3.2-3B, Vicuna-7B-v1.5, Mistral-7B-v0.1, Llama-3.1-8B; llama.cpp legacy Q8_0, Q5_0, Q4_0 and K-quants Q6_K down to Q2_K; weights read after dequantization.',
        'Deterministic log-likelihood scoring on zero-shot ARC, HellaSwag and Winogrande (no sampling, so sigma = 0 across repeats); block statistics (skewness, kurtosis), KS, cosine and KL distances, Correctness Agreement and Cohen\'s kappa.'
    ],
    results: [
        'Llama-3B WikiText-2 perplexity: base 8.16, Q8_0 10.23, Q2_K 16.95; ARC accuracy 0.741 falls to 0.542 at Q2_K (Tables III and V).',
        'Vicuna-7B Winogrande: accuracy 0.52 at both base and Q6_K, yet Correctness Agreement is 0.294 and kappa 0.106.',
        'Q and K projections are the most sensitive at Q3_K and Q2_K (KL above 5.0 for Q/K at Q3_K).'
    ],
    limits: [
        'No explicit limitations section; four models and three zero-shot multiple-choice tasks, Winogrande near chance.',
        'Calibration data and hyperparameters are missing; the "31.8% decision flip rate" is inferred from Correctness Agreement, not measured, and the "prove" in the abstract is stronger than the Theorem 3 mapping.'
    ],
    relevance: 'Same concern as our flip analysis (same accuracy, different answers), but deterministic, so it cannot separate quantization flips from sampling flips. Our replicate noise floor is exactly the control it lacks.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2607.08734' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2607.08734' }
    ],
    read: 'full'
},
{
    id: 'quantized-calibrated', title: 'Quantized Can Still Be Calibrated: A Unified Framework to Calibration in Quantized Large Language Models',
    year: 2025, posted: '2025-07', authors: 'Mingyu Zhong, Guanchu Wang, Yu-Neng Chuang, Na Zou',
    venue: 'ACL 2025 (Long Papers)', status: 'published', cat: 'quant-reliability', origin: 'new', prio: 1, arxiv: '',
    tags: ['calibration', 'upper bound calibration error', 'soft prompt', 'weight quantization', 'ece'],
    summary: 'Asks how weight quantization affects uncertainty calibration. It derives an analytic upper bound on calibration error that treats correct and incorrect predictions separately, finds quantized models are consistently less well calibrated, and recovers calibration by tuning soft prompt tokens after the embedding layer.',
    method: [
        'Analytic upper bound of calibration error (UBCE) for LLMs, with separate terms for correct and incorrect predictions.',
        'Post-hoc calibration by inserting trainable soft tokens after the embedding layer of the quantized model and optimizing them to recover the calibration lost to weight quantization.',
        'Evaluated on multiple LLMs and datasets (details are in the paper; only the abstract could be read here).'
    ],
    results: [
        'Quantized models are consistently less well calibrated than full-precision models across the LLMs and datasets tested (abstract).',
        'Soft-prompt calibration improves uncertainty calibration of quantized models on several datasets (abstract). Numeric results were not retrievable.'
    ],
    limits: [
        'Not assessed here: the PDF could not be machine-read, so limitations are those of the abstract-level claim only.',
        'Calibration here is about label probabilities, not agreement across sampled answers.'
    ],
    relevance: 'The ACL 2025 paper that most directly supports "quantization worsens calibration" and gives a repair. It sits on token probabilities of single answers, whereas our agreement signal is a property of repeated sampling, which this work does not examine.',
    links: [
        { l: 'ACL Anthology (ACL 2025)', u: 'https://aclanthology.org/2025.acl-long.1473/', o: true },
        { l: 'PDF', u: 'https://aclanthology.org/2025.acl-long.1473.pdf' }
    ],
    read: 'abstract', note: 'Title wording: the Anthology lists "A Unified Framework to Calibration in Quantized Large Language Models"; the PDF header uses "Understanding and Improving Uncertainty Calibration for Quantized Large Language Models". No arXiv entry was found.'
},
{
    id: 'interpreting-quant', title: 'Interpreting the Effects of Quantization on LLMs',
    year: 2025, posted: '2025-08', authors: 'Manpreet Singh, Hassan Sajjad',
    venue: 'IJCNLP-AACL 2025 (Long Papers)', status: 'published', cat: 'quant-reliability', origin: 'new', prio: 2, arxiv: '2508.16785',
    tags: ['interpretability', 'calibration', 'ace', 'integrated gradients', 'dead neurons', 'bitsandbytes', 'nf4'],
    summary: 'Uses interpretability tools to ask how 4-bit and 8-bit quantization change confidence, calibration, salient neurons, dead neurons and neuron redundancy, and finds no drastic change that would argue against quantization.',
    method: [
        'Phi-2, Llama-2-7B, Qwen-2.5-3B and 7B, Mistral-7B (OPT-6.7B for ReLU); bitsandbytes 8-bit and NF4 with double quantization, fp16 compute.',
        'BoolQ, Jigsaw Toxicity, PIQA, HellaSwag, IMDB with single-token outputs; confidence as mean maximum softmax probability; calibration by Adaptive Calibration Error; Integrated Gradients on the first MLP; dead neurons by activation near zero; redundancy by correlation above 0.8.'
    ],
    results: [
        'Accuracy usually within 1 to 4%, with exceptions such as Llama-2-7B 4-bit on Toxicity (-14%) and Phi-2 8-bit on IMDB (-17%).',
        'Confidence mostly changes little, 4-bit often slightly lower (Llama-2-7B 4-bit -13% on Toxicity); calibration effects are mixed and model and dataset dependent.',
        'Dead-neuron counts are essentially unchanged by quantization.'
    ],
    limits: [
        'Authors list limited compute, no 2- or 3-bit and only single-token outputs, not generation.',
        'Neuron attribution only on BoolQ and redundancy only on Phi-2 and Llama-2-7B; no variance or significance tests; ACE rather than the more common ECE.'
    ],
    relevance: 'Uses the same bitsandbytes 8-bit and NF4 settings we use and reports mostly small calibration changes at 4 and 8 bits, which agrees with our finding that label precision survives. It is limited to one-token answers, so it does not cover sampled reasoning.',
    links: [
        { l: 'ACL Anthology (IJCNLP-AACL 2025)', u: 'https://aclanthology.org/2025.ijcnlp-long.123/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2508.16785' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2508.16785' }
    ],
    read: 'full'
}
);
