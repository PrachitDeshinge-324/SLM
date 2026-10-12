/* Quantization meets voting, verification, self-improvement and RL: the closest prior work to Experiment D. */
(window.PW_PAPERS = window.PW_PAPERS || []).push(

{
    id: 'not-all-bits', title: 'Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models',
    year: 2026, posted: '2025-10', authors: 'Junhyuck Kim, Ethan Ewer, Taehong Moon, Jongho Park, Dimitris Papailiopoulos',
    venue: 'ICLR 2026 (poster)', status: 'published', cat: 'intersection', origin: 'dossier', dossier: '§3.1', alsoSite: true, prio: 1, arxiv: '2510.10964',
    tags: ['majority voting', 'gptq', 'kv cache', 'budget forcing', 'test-time scaling', 'memory', 'qwen3', 'aime25', 'gpqa'],
    summary: 'For reasoning models the KV cache can exceed the weights in memory. The paper maps, over 1,700 inference scenarios, how to spend a fixed memory budget across model size, weight precision, generation length and number of parallel votes, and finds the best allocation changes at an effective size of about "8-bit 4B".',
    method: [
        'Qwen3 0.6B, 1.7B, 4B, 8B, 14B, 32B; GPTQ at 4 and 8 bits (group size 128, symmetric) against 16-bit; KV eviction with R-KV and StreamingLLM; KV quantization with HQQ at 2, 4, 8 bits.',
        'Serial scaling by budget forcing (replace the end token with "Wait", budgets 2k to 30k); parallel scaling by majority voting with group sizes up to 16; temperature 0.6; accuracy averaged over 32 generations (8 in the KV section).',
        'AIME25 and GPQA-Diamond; total memory = weights + KV cache; latency measured with vLLM on one A100 80 GB.'
    ],
    results: [
        'Below an effective 8-bit 4B, spending memory on more weights beats longer generation: a 1.7B 8-bit model at 6k tokens beats a 0.6B 8-bit model at 18k tokens.',
        'On AIME25, 8-bit is memory-optimal for 0.6B and 1.7B models and an 8B 8-bit model beats a 14B 4-bit one; on GPQA-Diamond 4-bit is broadly memory-optimal.',
        'Parallel voting is not memory-efficient below the threshold; above it, group sizes 4 to 8 are optimal at 16 to 29 GB and 8 or more above 29 GB.',
        '4-bit is never on the latency/throughput Pareto frontier; 2-bit KV quantization loses significant accuracy.'
    ],
    limits: [
        'Authors list: majority voting only (no external verifiers), no comparison of other PTQ or eviction algorithms, no quantization-aware training, one model family and two benchmarks.',
        'The threshold is stated inconsistently in the text ("8-bit 4B" and "8-bit 8B"); one GPU setup.'
    ],
    relevance: 'The closest published work to our framing: quantization crossed with sample-and-vote at inference. It only asks whether voting pays for its memory, never whether agreement still tracks correctness, and it is inference-only. Its finding that 8-bit is optimal for 0.6B and 1.7B matches our Qwen-0.8B result.',
    links: [
        { l: 'ICLR 2026 poster page', u: 'https://iclr.cc/virtual/2026/poster/10008643', o: true },
        { l: 'OpenReview PDF (camera-ready)', u: 'https://openreview.net/pdf/e4c6cd05fb9c7ca57c8fd4185c4974ae299fafb4.pdf' },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2510.10964' },
        { l: 'PDF (v1)', u: 'https://arxiv.org/pdf/2510.10964' }
    ],
    read: 'full', note: 'The method and numbers above are from arXiv v1, which the dossier read. The ICLR camera-ready describes mathematical, code-generation and knowledge-intensive reasoning tasks, so the final experimental scope differs from v1.'
},
{
    id: 'maca', title: 'Self-Improvement of Language Models by Post-Training on Multi-Agent Debate',
    year: 2026, posted: '2025-09', authors: 'Ankur Samanta, Akshayaa Magesh, Runzhe Wu, Ayush Jain, Youliang Yu, Daniel Jiang, Boris Vidolov, Paul Sajda, Yonathan Efroni, Kaveh Hassani',
    venue: 'ICLR 2026 workshop (Multi-Agent Learning), poster', status: 'workshop', cat: 'intersection', origin: 'dossier', dossier: '§3.4', prio: 1, arxiv: '2509.15172',
    tags: ['maca', 'multi-agent debate', 'consensus', 'qlora', '4-bit', 'dpo', 'kto', 'grpo', 'majority vote', 'self-consistency'],
    summary: 'Multi-Agent Consensus Alignment (MACA) post-trains a model on the consensus of its own debating clones. Preference learning over full reasoning traces turns ground-truth-free majority labels into gains in accuracy and in sampling consistency.',
    method: [
        'Instruction-tuned Qwen-2B, Llama-3B, Phi-4B and Llama-8B; main experiments use 4-bit quantization with QLoRA, with full-precision results in an appendix; 256-token response cap.',
        'Debate with M = 3 clones for R = 2 rounds at temperature 1.0; the majority answer defines agreeing (G+) and dissenting (G-) trajectories, from which MV-SFT, MV-GRPO, MV-DPO and MV-KTO objectives are built.',
        'MATH, GSM8K, MathQA, SVAMP, GPQA and CommonsenseQA, 1,500 train and 500 test questions each; three seeds; baselines include TTRL-like GRPO on first-round majority and ScPO-like DPO.'
    ],
    results: [
        'Debate accuracy: Llama-8B on MATH 32.80 to 59.67 with MV-DPO (+26.87); Llama-3B on MathQA 21.60 to 64.00 with MV-KTO (+42.73) (Table 1).',
        'Single-agent zero-shot: Llama-8B MathQA 29.67 to 51.18 with MV-DPO; sampling consistency rises up to 27.6 points on GSM8K.',
        'Qwen-2B GSM8K debate: unanimous agreement 13.4% to 43.4% and non-parseable responses 13.8% to 0.6%.',
        'Abstract reports +16.3% on GPQA and +11.6% on CommonsenseQA from generalization.'
    ],
    limits: [
        'Authors: needs sufficient base-model competence to produce meaningful consensus; may amplify existing biases; no supervision of intermediate steps.',
        'Small models, 256-token cap, 4-bit base in main results. Dossier reading of the appendices (not re-verified here): base truncation of 75 to 83% under the 256 cap, and up to 38 of one 60-point gain is formatting (their Table 22).'
    ],
    relevance: 'The one paper that already self-trains a 4-bit model on majority-vote signals, so "nobody combined quantization and self-improvement" is false. It never varies precision, so it cannot say whether the 4-bit base shaped the labels; evaluating a 4-bit-trained adapter at full precision is the nearest it comes (dossier Table 16).',
    links: [
        { l: 'ICLR 2026 workshop poster page', u: 'https://iclr.cc/virtual/2026/10018396', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2509.15172' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2509.15172' }
    ],
    read: 'main', note: 'The arXiv entry shows no venue. The ICLR 2026 page lists it as "Poster in Workshop" (Multi-Agent Learning and Its Opportunities in the Era of Generative AI), April 27, 2026, so it is not a main-conference paper. Appendix tables were cut off in retrieval.'
},
{
    id: 'qurl', title: 'QuRL: Efficient Reinforcement Learning with Quantized Rollout',
    year: 2026, posted: '2026-02', authors: 'Yuhang Li, Reena Elangovan, Xin Dong, Priyadarshini Panda, Brucek Khailany',
    venue: 'ICLR 2026 (poster)', status: 'published', cat: 'intersection', origin: 'dossier', dossier: '§3.4', prio: 1, arxiv: '2602.13953',
    tags: ['quantized rollout', 'int8', 'fp8', 'ppo', 'grpo', 'dapo', 'adaptive clipping range', 'update-aware quantization'],
    summary: 'Rollout generation is up to 70% of RL training time for reasoning models. QuRL generates rollouts with a quantized actor and updates a full-precision copy, with two fixes for the resulting mismatch: an adaptive clipping range and update-aware quantization that preserves small weight updates.',
    method: [
        'Qwen2.5-0.5B-Instruct (PPO, GSM8K), Qwen2.5-7B-Math (DAPO, DAPO-17k) and DeepSeek-R1-Distill-Qwen-1.5B (GRPO, DeepScaleR); INT8 and FP8 weights with channel-wise scales and token-wise activation scales; VeRL with vLLM.',
        'Adaptive Clipping Range modifies the clipping bound using the full-precision to quantized policy ratio (Eq. 9); Update-Aware Quantization applies invariant scaling with s = 1.5 so quantization resolves weight changes smaller than the quantization step.',
        'Baselines: BF16 RL, naive INT8/FP8 RL, FlashRL (truncated importance sampling).'
    ],
    results: [
        'GSM8K (PPO): BF16 55.35, naive INT8 RL 48.78, FlashRL INT8 51.40, QuRL INT8 53.55, QuRL FP8 54.28 (Table 1).',
        'AIME 2024 (DAPO, Avg@32): BF16 31.67, QuRL INT8 with UAQ 31.25, QuRL FP8 with UAQ 33.27; naive quantized RL collapses to about 0.001 to 0.003 (Table 2).',
        'Rollout throughput gains of 20 to 30% on 7B and about 70% (A100) to 90% (H100) on 32B for INT8.'
    ],
    limits: [
        'FP8 KV cache excluded; larger UAQ scales or learning rates destabilize training; the UAQ scale is tuned empirically.',
        'Table 1 contains an unexplained 0.0 entry, Table 3 has no FP8 DeepScaleR results, and no variance across runs is shown. Rewards are ground-truth, not self-generated.'
    ],
    relevance: 'The dossier cites it for the point that, with a quantized actor, weight updates are far smaller than quantization error. That is the mechanism that could stop a quantized model from improving in a self-training round, independent of label quality.',
    links: [
        { l: 'ICLR 2026 poster page', u: 'https://iclr.cc/virtual/2026/poster/10008335', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2602.13953' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2602.13953' }
    ],
    read: 'full', note: 'The ICLR 2026 listing uses the title "QuRL: Low-Precision Reinforcement Learning for Efficient Reasoning"; arXiv v1 uses "Efficient Reinforcement Learning with Quantized Rollout".'
},
{
    id: 'qerl', title: 'QeRL: Beyond Efficiency: Quantization-enhanced Reinforcement Learning for LLMs',
    year: 2025, posted: '2025-10', authors: 'Wei Huang, Yi Ge, Shuai Yang, Yicheng Xiao, Huizi Mao, Yujun Lin, Hanrong Ye, Sifei Liu, Ka Chun Cheung, Hongxu Yin, Yao Lu, Xiaojuan Qi, Song Han, Yukang Chen',
    venue: 'arXiv preprint', status: 'preprint', cat: 'intersection', origin: 'new', prio: 1, arxiv: '2510.11696',
    tags: ['nvfp4', 'lora', 'grpo', 'dapo', 'adaptive quantization noise', 'exploration', 'entropy', 'qwen2.5'],
    summary: 'QeRL runs reinforcement learning for reasoning on an NVFP4-quantized base with LoRA adapters. It argues that quantization noise raises policy entropy and aids exploration, and adds an Adaptive Quantization Noise schedule to control it.',
    method: [
        'Qwen2.5-Instruct 3B, 7B, 14B, 32B; NVFP4 main format (also MXFP4 and NF4) with AWQ calibration on 256 sequences from OpenThoughts-114k; GRPO on GSM8K and DAPO on BigMath.',
        'Adaptive Quantization Noise: per-forward-pass Gaussian noise merged into RMSNorm weights, sigma decayed exponentially over about 10 stages starting at 1e-2.',
        'AdamW-8bit, LoRA rank 32, rollout temperature 1.0; pass@1 at temperature 0.6 on GSM8K, MATH500, AIME 2024/2025, AMC23; baselines BF16 full-parameter, BF16 LoRA, NF4 QLoRA, MXFP4 LoRA.'
    ],
    results: [
        'GSM8K, 7B: BF16 LoRA 88.1, NF4 LoRA 85.0, NVFP4 LoRA 88.5, NVFP4 with AQN 90.8, full-parameter 91.2 (Table 1).',
        'BigMath average: 32B BF16 LoRA 42.2, NVFP4 with AQN 45.6, full 46.2; 14B AMC23 57.5 against 55.0 for full-parameter (Table 2).',
        'End-to-end GRPO speedup x1.5 over BF16 LoRA for 7B at batch size 2 while QLoRA is x0.8 (Table 3); the 32B model trains on one H100 80GB.'
    ],
    limits: [
        'Authors: models above 70B untested, math and reasoning benchmarks only. The text states +1.7 points on 7B GSM8K where Table 1 gives +2.7, and the noise range differs between Sec. 4.1 and Table 4.',
        'One model family, single runs without variance, speedups measured over the first 30 steps when outputs are short.'
    ],
    relevance: 'Argues the opposite of our worry: quantization noise might help RL exploration rather than hurt. It uses ground-truth rewards on a 4-bit-format actor, so it speaks to trainability, not to the quality of self-generated labels. A useful counter-hypothesis for phase 2.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2510.11696' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2510.11696' },
        { l: 'Code (NVlabs)', u: 'https://github.com/NVlabs/QeRL' }
    ],
    read: 'main'
},
{
    id: 'bitcal-tts', title: 'BitCal-TTS: Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models',
    year: 2026, posted: '2026-05', authors: 'Sai Babu Patarlapalli, Surya Teja Avvaru',
    venue: 'arXiv preprint', status: 'preprint', cat: 'intersection', origin: 'dossier', dossier: '§3.2', prio: 2, arxiv: '2605.05561',
    tags: ['nf4', 'bitsandbytes', 'adaptive halting', 'token entropy', 'gsm8k', 'qwen2.5', 'early stopping'],
    summary: 'In 4-bit models, online signals used to halt adaptive reasoning (token entropy, trace stability) are distorted and cause premature stops under a fixed token cap. BitCal-TTS adds a bit-conditioned confidence scale and a bit-dependent confirmation tail.',
    method: [
        'Qwen2.5-3B, 7B, 14B Instruct in 4-bit NF4 via bitsandbytes with BF16 compute; greedy decoding, seed 4242; GSM8K test, first N items only (N = 50, 54, 35).',
        'Controller: chunk k = 16, floor 128 tokens, thresholds on entropy, confidence and trace energy, signal weights (0.40, 0.35, 0.25), scale s(b) = 0.85 for b <= 4 and a 32-token post-marker tail; compared with fixed budget and a precision-agnostic adaptive controller; Wilson 95% intervals on a single T4 GPU.'
    ],
    results: [
        '7B at budget 512: fixed 90.7%, adaptive 79.6%, BitCal-TTS 83.3%; average tokens 466, 286, 316; premature stops 14.8% to 11.1%.',
        '14B: fixed 88.6%, adaptive 82.9%, BitCal-TTS 85.7%, token savings 40.8% against fixed.',
        '3B: both adaptive variants fail (premature stop 63%), BitCal-TTS is slightly below adaptive (20.0% against 22.0%).'
    ],
    limits: [
        'Authors call the gains suggestive: small shards, overlapping intervals, the 14B gain is one example, no paired test and no component ablation.',
        'GSM8K only, hand-tuned constants, no wall-clock or energy measurement; bitsandbytes version sensitivity up to about 0.5 points.'
    ],
    relevance: 'A 4-bit bitsandbytes setting like ours, on GSM8K, where a token cap interacts with how a quantized model stops. Its evidence rests on 35 to 54 GSM8K questions (as the dossier notes), so its premise that 4-bit distorts halting signals is suggestive rather than established.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2605.05561' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2605.05561' },
        { l: 'Code', u: 'https://github.com/Saibabu7770/bitcal-tts' }
    ],
    read: 'full'
},
{
    id: 'self-explanations-quant', title: 'Can Large Language Models Still Explain Themselves? Investigating the Impact of Quantization on Self-Explanations',
    year: 2026, posted: '2026-01', authors: 'Qianli Wang, Nils Feldhus, Pepa Atanasova, Fedor Splitt, Simon Ostermann, Sebastian Möller, Vera Schmitt',
    venue: 'Findings of EMNLP 2026 (camera-ready)', status: 'accepted', cat: 'intersection', origin: 'dossier', dossier: '§3.2', alsoSite: true, prio: 1, arxiv: '2601.00282',
    tags: ['self-explanations', 'faithfulness', 'natural language explanations', 'counterfactuals', 'gptq', 'awq', 'bitsandbytes', 'user study'],
    summary: 'Asks whether post-training weight quantization degrades the quality and faithfulness of a model\'s own explanations (natural-language explanations and counterfactuals), using automatic metrics, faithfulness tests and a 48-participant user study.',
    method: [
        'Llama3-8B and 70B and Qwen2.5-7B, 14B, 32B, 72B at full precision and with GPTQ (4 and 8 bit), AWQ and bitsandbytes (4 and 8 bit); temperature 0, three seeds.',
        'Natural-language explanations by zero-shot "Let\'s think step by step"; counterfactuals by the two-stage FIZLE method; eSNLI, HealthFC, ECQA, AG News, IMDb.',
        'Quality: BARTScore, TIGERScore, label flip rate, perplexity, textual similarity; faithfulness: biasing features, CC-SHAP, self-consistency checks; human ratings of trustworthiness and coherence (Krippendorff alpha 0.71 and 0.64).'
    ],
    results: [
        'Quantization typically leads to moderate declines in explanation quality (up to 4.4%) and faithfulness (up to 3.9%); the user study shows coherence and trustworthiness drops of up to 8.5%.',
        'Fluency is the most affected counterfactual metric (6.3% on average); larger models are less resilient in quality but preserve faithfulness better.',
        'LLM-as-a-judge agreement with humans is weak or negative and not significant; no quantization method wins on every metric.'
    ],
    limits: [
        'Authors: only 4 and 8 bits, PTQ only, black-box confounds. The user study covers one dataset, three Qwen models and 48 participants.',
        'Faithfulness relies on proxy metrics; the abstract (up to 4.4%) and Sec. 5.3 (up to 8.5%) quote different maxima for quality loss.'
    ],
    relevance: 'The closest paper in spirit: whether a quantized model can still judge and explain its own outputs. It studies explanation quality at greedy decoding, not sampled agreement or whether the judgment improves training.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2601.00282' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2601.00282' }
    ],
    read: 'main', note: 'arXiv comments: "EMNLP 2026 Findings; camera-ready version". The proceedings page was not available to check, so status is Accepted. The retrieved text was cut in the references and later appendices.'
},
{
    id: 'multipass-verification', title: 'Improving Quantized Model Performance in Qualitative Analysis with Multi-Pass Prompt Verification',
    year: 2026, posted: '2026-04', authors: 'Aisvarya Adeseye, Jouni Isoaho, Adeyemi Adeseye',
    venue: '12th Intelligent Systems Conference 2026 (Amsterdam)', status: 'accepted', cat: 'intersection', origin: 'dossier', dossier: '§4.4', prio: 3, arxiv: '2605.20193',
    tags: ['multi-pass verification', 'quantized llama', 'qualitative coding', 'hallucination', 'semantic drift', 'gguf', 'smoothquant', 'hqq'],
    summary: 'Quantized LLaMA-3.1 8B models hallucinate and drift when coding interview transcripts. A quantization-aware multi-pass prompt workflow that extracts, verifies and re-counts themes stabilizes low-bit models.',
    method: [
        'LLaMA-3.1 8B in BF16, 8-bit (SmoothQuant, INT8, GGUF Q8_0), 4-bit (Q4_K_M, NF4, GPTQ), 3-bit (HQQ, AWQ) and 2-bit (SPQR, Q2_K); llama.cpp on an RTX 4090, temperature 0.2, five runs per configuration.',
        'Passes: analysis prompt (themes, sub-themes, quotes as JSON), verification prompt removing unsupported items, then frequency counting and verification; at most three passes, 93% converge within two.',
        '82 semi-structured interviews (33 experts, 49 non-experts); eight metrics including F1, semantic drift, hallucination rate; gold standard from human NVivo coding.'
    ],
    results: [
        'Non-expert F1 before and after verification: BF16 0.65 to 0.82; Q4_K_M 0.58 to 0.77; GPTQ-4bit 0.55 to 0.74; HQQ (3-bit) 0.37 to 0.58; Q2_K 0.22 to 0.40; human reference 0.80 (Table 1).',
        'HQQ hallucination rate (non-expert) falls from 0.40 to 0.22; runtime increases 34.7%.'
    ],
    limits: [
        'Only LLaMA-3.1 8B and one qualitative domain; statistical tests are described but no p-values or effect sizes appear.',
        'Conflicting statements about whether corrected BF16 outputs entered the gold standard; some listed methods are not native llama.cpp formats; claimed 40 to 80% gains exceed Table 1 (19 to 33% F1).'
    ],
    relevance: 'The dossier cites it (abstract only) as an adjacent verification-under-quantization study. It verifies extracted themes, not reasoning correctness, and measures accuracy rather than verifier bias, so it does not address our verifier-criterion question.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2605.20193' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2605.20193' }
    ],
    read: 'main', note: 'arXiv comments: "Accepted to publish in 12th Intelligent Systems Conference 2026; 3-4 September 2026 in Amsterdam". The conference proceedings page was not checked.'
}
);
