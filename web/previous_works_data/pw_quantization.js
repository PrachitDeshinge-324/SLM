/* Quantization methods and tooling; quantization scaling laws and capability evaluation. */
(window.PW_PAPERS = window.PW_PAPERS || []).push(

/* ================= QUANTIZATION METHODS AND TOOLING ================= */
{
    id: 'llm-int8', title: 'LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale',
    year: 2022, posted: '2022-08', authors: 'Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer',
    venue: 'NeurIPS 2022 (Main Conference)', status: 'published', cat: 'quant-methods', origin: 'new', prio: 1, arxiv: '2208.07339',
    tags: ['bitsandbytes', '8-bit', 'outliers', 'int8', 'weights and activations'],
    summary: 'Plain 8-bit quantization of transformer weights and activations breaks above roughly 6.7B parameters because a few hidden dimensions carry very large outlier values. The paper makes 8-bit inference lossless up to 175B parameters by quantizing almost everything to int8 and keeping the outlier dimensions in 16-bit.',
    method: [
        'Vector-wise quantization (Sec. 3.1): each row of the activations and each column of the weights gets its own absmax scale; the output is rescaled by the outer product of the scales.',
        'Mixed-precision decomposition (Sec. 3.2, Eq. 8): feature dimensions with an outlier above magnitude 6.0 are multiplied in 16-bit; the remaining 99.9% or more of values are multiplied in int8. At most 7 outlier dimensions up to 13B parameters, about 0.1% extra memory.',
        'Perplexity on C4 validation for fairseq dense transformers from 125M to 13B; zero-shot accuracy with the EleutherAI harness on OPT models up to 175B (WinoGrande, HellaSwag, PIQA, LAMBADA).',
        'Baselines: int8 absmax, zero-point, row-wise and vector-wise variants, each with and without decomposition (Table 1), against 16/32-bit floats.'
    ],
    results: [
        'C4 perplexity (125M / 1.3B / 2.7B / 6.7B / 13B): 32-bit 25.65 / 15.91 / 14.43 / 13.30 / 12.45; int8 absmax 87.76 / 16.55 / 15.11 / 14.59 / 19.08; LLM.int8() 25.83 / 15.93 / 14.44 / 13.24 / 12.45 (Table 1).',
        'Outliers appear in all layers and about 75% of sequence positions at roughly 6.7B parameters; removing outlier dimensions raises perplexity by 600 to 1000%, against about 0.1% for random dimensions (Sec. 4).',
        'About 1.96x memory reduction for BLOOM-176B; int8 matmul up to 2.29x faster at model dimension 12288, but LLM.int8() is slower than fp16 below dimension 2560 and end-to-end BLOOM-176B is slightly slower (Tables 5 and 6).'
    ],
    limits: [
        'Studies only the int8 data type, not FP8, and nothing above 175B (Sec. 6).',
        'Attention matmuls stay in 16-bit; training and fine-tuning are only preliminary appendices.',
        'Overhead of the decomposition makes small models slower than fp16; the evidence is perplexity plus a handful of zero-shot tasks, not generation quality.'
    ],
    relevance: 'This is the 8-bit path of Experiment D: our `8bit` runs call bitsandbytes `load_in_8bit`, which is LLM.int8(). Keeping outlier dimensions in 16-bit is consistent with 8-bit changing voted answers so little in our data, and its fp16 compute path is the dtype difference the dossier lists as validity problem V4.',
    links: [
        { l: 'NeurIPS 2022 proceedings', u: 'https://proceedings.neurips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2208.07339' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2208.07339' }
    ],
    read: 'full', note: 'The NeurIPS proceedings page is titled "GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale" while the paper and arXiv use "LLM.int8()".'
},
{
    id: 'gptq', title: 'GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers',
    year: 2023, posted: '2022-10', authors: 'Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh',
    venue: 'ICLR 2023', status: 'published', cat: 'quant-methods', origin: 'new', prio: 1, arxiv: '2210.17323',
    tags: ['gptq', 'post-training quantization', '3-bit', '4-bit', 'second-order'],
    summary: 'One-shot weight quantization of 100B-scale language models was only accurate at 8 bits. GPTQ uses approximate second-order information to quantize OPT-175B and BLOOM-176B to 3 to 4 bits per weight in about four GPU hours with small perplexity loss.',
    method: [
        'Layer-wise objective: minimize ||WX - W_hat X||^2 with a fixed quantization grid (Sec. 3, Eq. 1), starting from Optimal Brain Quantization.',
        'Three changes make it scale (Sec. 4): quantize all rows in the same column order, lazy batch updates in blocks of 128 columns, and a Cholesky reformulation of the inverse Hessian with dampening of 1% of the mean diagonal.',
        'Calibration: 128 random 2048-token segments of C4; per-row asymmetric min-max grids; blocks quantized sequentially on a single 80GB A100.',
        'Models OPT 125M to 175B and BLOOM 560M to 176B; WikiText2, PTB and C4 perplexity; LAMBADA, ARC, PIQA, StoryCloze; group sizes from 1024 down to 8; baselines RTN, AdaRound, AdaQuant, BRECQ, OBQ.'
    ],
    results: [
        'OPT-175B WikiText2 perplexity: FP16 8.34, RTN 4-bit 10.54, GPTQ 4-bit 8.37, GPTQ 3-bit 8.68, GPTQ 3-bit with group size 128 8.45 (Table 5).',
        'RTN collapses at 3 bits (OPT-175B WikiText2 about 7.3e3) where GPTQ stays usable.',
        'Quantization time 4.2 h for OPT-175B and 3.8 h for BLOOM-176B (Table 2); generation speedups of 3.24x on A100 and 4.53x on A6000 from fewer GPUs and less memory traffic (Table 6).',
        'Around 2.2 bits (group size 128) perplexity rises by under 1.5; ternary with group size 8 reaches 9.20 WikiText2 on OPT-175B (Sec. 5).'
    ],
    limits: [
        'Speedups come from reduced memory movement, not reduced compute; no activation quantization (Sec. 6).',
        'Results are perplexity-heavy and need C4 calibration data, so the method is not fully zero-shot.',
        'OPT-66B degrades at 3 bits, which the authors link to dead units; bias effects are not studied.'
    ],
    relevance: 'The dossier proposes GPTQ for the 3-bit point and as one candidate (with AWQ) for the extra 4-bit method in the confirmation run. Quantization Hurts Reasoning? found GPTQ needs reasoning-domain calibration data at 3 bits, so the choice of calibration data becomes a design decision for us.',
    links: [
        { l: 'ETH Research Collection record (ICLR 2023)', u: 'https://www.research-collection.ethz.ch/handle/20.500.11850/656510', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2210.17323' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2210.17323' },
        { l: 'Code (IST-DASLab)', u: 'https://github.com/ist-daslab/gptq' }
    ],
    read: 'full', note: 'Venue taken from the arXiv comments field ("ICLR 2023") and the ETH publication record; the OpenReview page could not be opened (bot check).'
},
{
    id: 'awq', title: 'AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration',
    year: 2024, posted: '2023-06', authors: 'Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han',
    venue: 'MLSys 2024 (Best Paper Award)', status: 'published', cat: 'quant-methods', origin: 'new', prio: 1, arxiv: '2306.00978',
    tags: ['awq', 'weight-only', 'salient channels', 'tinychat', '4-bit'],
    summary: 'Not all weights matter equally: protecting about 1% of channels, chosen by activation magnitude, removes most of the error of low-bit weight-only quantization. AWQ does this by per-channel scaling rather than mixed precision, and ships TinyChat to turn 4-bit memory savings into real speedups.',
    method: [
        'Salient channels are found from activation statistics, not weight norm; keeping 0.1 to 1% in FP16 helps only under activation-based selection (Sec. 3.1, Table 1).',
        'Scale search: s = s_X^alpha with alpha grid-searched over [0, 1] in 20 steps, then weight clipping by MSE; no backpropagation or reconstruction (Sec. 3.2, Eqs. 4 and 5).',
        'Calibration set is a small slice of the Pile (16 sequences for OPT-6.7B against 192 for GPTQ in Fig. 8); weight-only, group size 128, INT4 and INT3.',
        'Models: LLaMA, Llama-2, OPT, Vicuna, Mistral-7B, Mixtral-8x7B, CodeLlama, OpenFlamingo, LLaVA, VILA. Baselines RTN, GPTQ, GPTQ with reordering. TinyChat: fused dequantization kernels and SIMD-aware weight packing.'
    ],
    results: [
        'INT3-g128 perplexity: Llama-2-7B FP16 5.47, RTN 6.66, GPTQ 6.43, AWQ 6.24; LLaMA-65B FP16 3.53, RTN 4.24, GPTQ 4.17, AWQ 3.95 (Table 4).',
        'Under a calibration-domain mismatch AWQ perplexity rises by 0.5 to 0.6 against 2.3 to 4.9 for GPTQ (Sec. 5.3).',
        'INT4 on programming and math: GSM8K 7B FP16 13.87, AWQ 13.57; MBPP 7B pass@1 FP16 38.53, AWQ 40.64 (Table 8).',
        'Up to 3.9x over HuggingFace FP16 on an RTX 4090 and 3.5x on a Jetson Orin.'
    ],
    limits: [
        'The calibration set still defines the scale search, so "no overfitting" is relative; RTN fails at INT2 and AWQ needs GPTQ on top there (Table 9).',
        'Main evaluation is perplexity centred; some multimodal scores drop slightly, so "lossless" is approximate.',
        'Speed numbers are batch size 1 with a short prompt on specific GPUs.'
    ],
    relevance: 'AWQ is the other candidate (with GPTQ) the dossier names for the extra 4-bit method in phase 1. Quantization Hurts Reasoning? prefers AWQ over GPTQ for weight-only quantization, so it is a natural second quantizer for checking whether our agreement results are bitsandbytes-specific.',
    links: [
        { l: 'MLSys 2024 proceedings', u: 'https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2306.00978' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2306.00978' },
        { l: 'Code (MIT HAN Lab)', u: 'https://github.com/mit-han-lab/llm-awq' }
    ],
    read: 'full', note: 'The MLSys title says "On-Device LLM Compression and Acceleration"; arXiv says "LLM Compression and Acceleration".'
},
{
    id: 'smoothquant', title: 'SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models',
    year: 2023, posted: '2022-11', authors: 'Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han',
    venue: 'ICML 2023', status: 'published', cat: 'quant-methods', origin: 'new', prio: 3, arxiv: '2211.10438',
    tags: ['smoothquant', 'w8a8', 'activation outliers', 'int8'],
    summary: 'Activation outliers make W8A8 quantization hard. SmoothQuant migrates the difficulty from activations to weights with an offline, mathematically equivalent per-channel scaling, enabling INT8 for all matrix multiplications without training.',
    method: [
        'Smoothing factor s_j = max|X_j|^alpha / max|W_j|^(1-alpha); activations are divided by s and weights multiplied by s so the layer output is unchanged (Sec. 4, Eqs. 3 and 4).',
        'Migration strength alpha: 0.5 default for OPT and BLOOM, 0.75 for GLM-130B, 0.6 to 0.9 for Llama-2, Falcon and Mistral; calibration from 512 Pile sentences.',
        'Three W8A8 schemes of increasing efficiency: per-token dynamic (O1), per-tensor dynamic (O2), per-tensor static (O3).',
        'Models from OPT to MT-NLG 530B, LLaMA, Llama-2, Falcon, Mistral, Mixtral; baselines naive W8A8, ZeroQuant, LLM.int8(), Outlier Suppression.'
    ],
    results: [
        'OPT-175B: SmoothQuant-O3 averages 66.8% against 66.9% FP16; naive W8A8 averages 35.5% (Table 3).',
        'BLOOM-176B and GLM-130B: O1 matches FP16; static O3 loses about 0.8 to 1 point.',
        'Up to 1.51x speedup and 1.96x memory saving in PyTorch; OPT-175B runs on half as many GPUs with FasterTransformer (Sec. 5.3).'
    ],
    limits: [
        'Static quantization (O3) drops accuracy on some models; alpha has to be searched per model.',
        'Only INT8; W4A4 is left to future work; speed results come from A100 servers only.'
    ],
    relevance: 'Background for the weight-activation family. Quantization Hurts Reasoning? and Evaluating Quantized LLMs show W8A8 is near lossless, so the weight-only bitsandbytes setting we use is the more informative stress test; SmoothQuant is the reference if a W8A8 point is ever added.',
    links: [
        { l: 'ICML 2023 proceedings (PMLR 202)', u: 'https://proceedings.mlr.press/v202/xiao23c.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2211.10438' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2211.10438' }
    ],
    read: 'full'
},
{
    id: 'qlora', title: 'QLoRA: Efficient Finetuning of Quantized LLMs',
    year: 2023, posted: '2023-05', authors: 'Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer',
    venue: 'NeurIPS 2023 (Main Conference)', status: 'published', cat: 'quant-methods', origin: 'new', prio: 1, arxiv: '2305.14314',
    tags: ['nf4', 'double quantization', 'lora', 'bitsandbytes', '4-bit', 'fine-tuning'],
    summary: 'QLoRA backpropagates through a frozen 4-bit quantized model into low-rank adapters, which lets a 65B model be fine-tuned on one 48 GB GPU with no reported loss against 16-bit fine-tuning. It introduces the NF4 data type, double quantization and paged optimizers.',
    method: [
        '4-bit NormalFloat (NF4), a quantile data type described as information-theoretically optimal for normally distributed weights; double quantization of the constants (block size 64 for weights, 256 for constants) cuts overhead from 0.5 to 0.127 bits per parameter.',
        'Paged optimizers use unified memory to absorb gradient-checkpointing memory spikes.',
        'LoRA adapters on all linear layers are needed to match full fine-tuning; rank is unrelated to quality when all layers are adapted (App. A.1).',
        'LLaMA 7B to 65B tuned on OASST1, HH-RLHF, FLAN v2 and others; evaluated with MMLU 5-shot, the Vicuna benchmark scored by GPT-4, and human pairwise Elo.'
    ],
    results: [
        'Mean perplexity by 4-bit data type: Int4 34.34, Float4 E2M1 31.07, Float4 E3M0 29.48, NF4 with double quantization 27.41 (Table 2).',
        'MMLU 5-shot mean: BFloat16 53.0 against NF4 with double quantization 53.1 (Table 4).',
        'Guanaco 65B reaches 99.3% of ChatGPT on the Vicuna benchmark with 41 GB; Guanaco 7B reaches 87.0% in 5 GB (Table 6).',
        'GPT-4 and human judges agree only moderately (Kendall tau 0.43 at system level, Fleiss kappa 0.25 at sample level).'
    ],
    limits: [
        'Equivalence with 16-bit fine-tuning was not established at 33B and 65B because of cost; no 3-bit bases, other adapters or RLHF were tested.',
        'Wide, overlapping confidence intervals on Vicuna scores; GPT-4 shows order effects and self-preference.'
    ],
    relevance: 'Our 4-bit setting is exactly this configuration: `bnb_4bit_quant_type="nf4"` with double quantization (see `model_loading.py`). It is also the training route for any phase-2 self-training round on a quantized base, which is what MACA does.',
    links: [
        { l: 'NeurIPS 2023 proceedings', u: 'https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2305.14314' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2305.14314' }
    ],
    read: 'main'
},
{
    id: 'llm-qat', title: 'LLM-QAT: Data-Free Quantization Aware Training for Large Language Models',
    year: 2024, posted: '2023-05', authors: 'Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, Vikas Chandra',
    venue: 'Findings of ACL 2024', status: 'published', cat: 'intersection', origin: 'new', prio: 2, arxiv: '2305.17888',
    tags: ['qat', 'self-generated data', 'distillation', 'kv cache', 'data-free'],
    summary: 'Post-training quantization breaks below 8 bits. LLM-QAT trains the quantized model by distilling from the full-precision model on data that model generated itself, so no original training data is needed, and also quantizes the KV cache.',
    method: [
        'Data generation (Sec. 2.1): start from a random token and sample autoregressively; a hybrid scheme takes the top-1 token for the first 3 to 5 tokens and samples the rest, up to 1024 tokens.',
        'Symmetric MinMax quantization with outliers kept, per-channel weights and per-token activations and KV cache; cross-entropy on the full teacher logits (attention or hidden-state distillation hurt, Table 5).',
        'LLaMA-7B, 13B, 30B; AdamW, learning rate 2e-5 cosine; evaluated on common-sense tasks, TriviaQA, MMLU and WikiText2/C4 perplexity; baselines RTN, GPTQ, SmoothQuant.'
    ],
    results: [
        '30B at 4-bit weights, 8-bit activations, 4-bit KV (4-8-4): 69.9 average against 71.4 for FP16; SmoothQuant at 8-8-4 averages 50.7 against 69.7 for LLM-QAT (Table 1).',
        '7B at 4-8-4: 60.7 against 41.2 for RTN (Table 1).',
        'Generated data beats real corpora for QAT: 63.1 average against 61.5 (C4) and 58.1 (WikiText2) at 4-6-16 on 7B (Table 3).'
    ],
    limits: [
        '4-bit activations did not work in the settings tried (Sec. 3.4); no hardware kernels or latency numbers.',
        'Only LLaMA models; instruction-tuned and RLHF models are left to future work.'
    ],
    relevance: 'A rare case where a model\'s own samples are used to repair quantization damage. It shows self-generated text is a usable training signal under quantization, which is the loop we want to study in the other direction (does quantization damage the self-generated signal).',
    links: [
        { l: 'ACL Anthology (Findings ACL 2024)', u: 'https://aclanthology.org/2024.findings-acl.26/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2305.17888' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2305.17888' },
        { l: 'Code (Meta)', u: 'https://github.com/facebookresearch/LLM-QAT' }
    ],
    read: 'full'
},

/* ================= QUANTIZATION: SCALING LAWS AND EVALUATION ================= */
{
    id: 'kbit-scaling', title: 'The case for 4-bit precision: k-bit Inference Scaling Laws',
    year: 2023, posted: '2022-12', authors: 'Tim Dettmers, Luke Zettlemoyer',
    venue: 'ICML 2023', status: 'published', cat: 'quant-scaling', origin: 'dossier', dossier: '§3.1', prio: 2, arxiv: '2212.09720',
    tags: ['scaling law', '4-bit', 'zero-shot', 'bit-level', 'block size', 'data type'],
    summary: 'For a fixed memory budget, should you use a bigger model at fewer bits or a smaller one at more bits? Across more than 35,000 zero-shot experiments the paper finds 4-bit is almost universally optimal for total model bits versus accuracy.',
    method: [
        'Families OPT, BLOOM, BLOOMZ, Pythia/NeoX and GPT-2, 19M to 176B parameters; 16-bit inputs with k-bit weights for k = 3 to 8; attention matrices not quantized.',
        'Data types integer, float, dynamic exponent and quantile; block-wise quantization with block sizes 1024 down to 64; "proxy quantization" keeps the top 2% of outlier dimensions in 16-bit.',
        'Metric: mean zero-shot accuracy on LAMBADA, Winogrande, HellaSwag and PiQA plus CommonCrawl perplexity; scaling curves are interpolated because bivariate power laws fit poorly.'
    ],
    results: [
        '4-bit is optimal for almost every family and scale; BLOOM-176B is the exception, where 3-bit is slightly but not significantly better.',
        'A block size of 64 to 128 is best at 4-bit; going from block size 1024 to 64 adds 0.24 bits and gains about as much as moving from 4 to 5 bits for Pythia.',
        'Quantile and float types beat integer and dynamic exponent at 4-bit; 3-bit OPT and Pythia are unstable, with the largest models near random.',
        'Perplexity and zero-shot accuracy correlate at -0.94 across the experiments.'
    ],
    limits: [
        'Learned or data-optimized methods such as GPTQ are outside the main grid; no optimized kernels were built, so lookup-table types may not get speedups.',
        'Zero-shot benchmarks are noisy and the laws apply only to small-batch (memory-bound) inference.'
    ],
    relevance: 'The classic argument for 4-bit as the default precision and the reason our lowest precision is 4-bit. It measures capability per bit, not self-consistency, so it leaves open what 4-bit does to sampled agreement.',
    links: [
        { l: 'ICML 2023 proceedings (PMLR 202)', u: 'https://proceedings.mlr.press/v202/dettmers23a.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2212.09720' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2212.09720' }
    ],
    read: 'full', note: 'Venue confirmed by the PMLR page title and search listing; the arXiv page itself shows no venue.'
},
{
    id: 'precision-scaling', title: 'Scaling Laws for Precision',
    year: 2025, posted: '2024-11', authors: 'Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Ré, Aditi Raghunathan',
    venue: 'ICLR 2025 (Oral)', status: 'published', cat: 'quant-scaling', origin: 'new', prio: 2, arxiv: '2411.04330',
    tags: ['scaling law', 'precision', 'post-training quantization', 'overtrained', 'effective parameters'],
    summary: 'Scaling laws ignore numeric precision. This paper models low-precision training as reducing a model\'s effective parameter count and models post-training-quantization damage as growing with the amount of pretraining data.',
    method: [
        'OLMo-style models of 30M to 220M non-embedding parameters trained on Dolma on 1.5B to 26B tokens, 465+ runs with weight, activation and KV precision swept over 3 to 12 bits.',
        'Post-training quantization degradation modelled as C_T D^gamma_D / N^gamma_N e^(-P_post/gamma_post) (Eq. 2); quantization-aware training with N_eff = N(1 - e^(-P_w/gamma_w)) (Eq. 3); a unified law combines both (Eq. 9).',
        'Quantization is simulated; integer types used for fitting and floating-point runs up to 1.6B parameters for validation.'
    ],
    results: [
        'Post-training quantization degradation grows with training tokens, so an overtrained model can get worse after quantization and extra data can become harmful (Finding 1; R^2 = 0.97).',
        'Weight, activation and KV precision effects are approximately independent and multiplicative.',
        'Compute-optimal pretraining precision is about 7 to 8 bits; training at 4 bits would need models more than 4x larger.',
        'The unified fit reaches R^2 = 0.90 on over 1,000 data points.'
    ],
    limits: [
        'Fixed architecture, simulated quantization, loss only (no downstream accuracy), and models up to about 1.7B parameters and 26B tokens.',
        'The authors call the trends "suggestive rather than prescriptive".'
    ],
    relevance: 'Offers a candidate mechanism for why some small models are more fragile than others in our data: quantization damage grows with training tokens per parameter. A hypothesis worth testing against our per-model damage sizes once training-token counts are collected.',
    links: [
        { l: 'ICLR 2025 proceedings', u: 'https://proceedings.iclr.cc/paper_files/paper/2025/hash/b2cac94f82928a85055987d9fd44753f-Abstract-Conference.html', o: true },
        { l: 'ICLR 2025 oral page', u: 'https://iclr.cc/virtual/2025/oral/31737' },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2411.04330' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2411.04330' }
    ],
    read: 'main'
},
{
    id: 'undertrained', title: 'Low-Bit Quantization Favors Undertrained LLMs',
    year: 2025, posted: '2024-11', authors: 'Xu Ouyang, Tao Ge, Thomas Hartvigsen, Zhisong Zhang, Haitao Mi, Dong Yu',
    venue: 'ACL 2025 (Long Papers)', status: 'published', cat: 'quant-scaling', origin: 'new', prio: 2, arxiv: '2411.17691',
    tags: ['scaling law', 'quantization-induced degradation', 'undertrained', 'pythia', '100T tokens'],
    summary: 'Larger models and models trained on fewer tokens lose less to low-bit quantization, while small, well-trained models lose the most. The paper fits scaling laws for quantization-induced degradation and uses them to project to 100T training tokens.',
    method: [
        'Pythia 160M to 12B with 20 checkpoints each (over 1,500 quantized checkpoints) quantized with GPTQ at 2, 3 and 4 bits; AWQ and bitsandbytes used for validation.',
        'QiD = loss of quantized model minus loss of the 16-bit model on 1,000 RefinedWeb texts; fitted dQLoss = k D^0.5251 / (N^0.2261 P^5.4967) (Eq. 8).',
        'Generalization checked on the Spectra suite, Llama and Qwen; a replicated BitNet b1.58 run at 120M and 1.2B parameters.'
    ],
    results: [
        'QiD scales with tokens as D^0.53, with model size as N^-0.23 and with bit width as P^-5.5.',
        'A 70B model at 4 bits needs over 17T tokens to reach a loss increase of 0.2 (Table 1); at 2 to 3 bits models trained on 100T tokens are projected to degrade severely.'
    ],
    limits: [
        'The 100T projection extrapolates about 500x beyond the roughly 206B-token fitting range; only single-stage pretraining is covered.',
        'The BitNet comparison is small; the authors call the checkpoint sample "not extensive enough".'
    ],
    relevance: 'Offers a testable explanation for the model-to-model spread of damage we observed (1 point for Llama-3B and DeepSeek, 19.5 for Gemma-1B at 4-bit): training tokens per parameter, not the questions.',
    links: [
        { l: 'ACL Anthology (ACL 2025)', u: 'https://aclanthology.org/2025.acl-long.1555/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2411.17691' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2411.17691' }
    ],
    read: 'full', note: 'The ACL version uses the shorter title; the arXiv title ends "Scaling Laws for Quantized LLMs with 100T Training Tokens". The arXiv comments field still says "Work in Progress".'
},
{
    id: 'evaluating-quantized', title: 'Evaluating Quantized Large Language Models',
    year: 2024, posted: '2024-02', authors: 'Shiyao Li, Xuefei Ning, Luning Wang, Tengxuan Liu, Xiangsheng Shi, Shengen Yan, Guohao Dai, Huazhong Yang, Yu Wang',
    venue: 'ICML 2024', status: 'published', cat: 'quant-scaling', origin: 'new', prio: 2, arxiv: '2402.18158',
    tags: ['benchmark', 'weight-only', 'kv cache', 'emergent abilities', 'multi-step reasoning', 'self-calibration'],
    summary: 'A broad evaluation of post-training quantization of weights, activations and KV cache across 11 model families and five task groups, ending in bit-width recommendations per model size and task.',
    method: [
        'Families OPT, LLaMA2, Falcon, Bloomz, Mistral, ChatGLM, Vicuna, LongChat, StableLM, Gemma and Mamba, 125M to 180B parameters.',
        'Weight-only W2 to W8, weight-activation W4A4 to W8A8 and KV cache KV2 to KV8; RTN, AWQ and SmoothQuant; asymmetric group-wise weights and symmetric per-token activations.',
        'Tasks: basic NLP, emergent abilities (MMLU, GSM8K, StrategyQA, ...), trustworthiness (ETHICS, TruthfulQA, AdvGLUE), dialogue (MT-Bench) and long context (LongEval).'
    ],
    results: [
        'Larger models tolerate weight and KV quantization better but tolerate activation quantization worse; activation kurtosis exceeds 1000 against about 10 for weights.',
        'Multi-step reasoning and self-calibration degrade more than instruction following and in-context learning (Sec. 4.2).',
        'Within a 2% loss: W8, W8A8, KV8 for models under 13B and W4, W4A8, KV4 for 13B and above.',
        'W2 collapses to near zero with RTN and AWQ; W4A4 hurts most models.'
    ],
    limits: [
        'Post-training quantization only; no group-size ablations; some 1B to 3B models not fully covered.',
        'Dialogue tested at two turns; multiple-choice accuracy can rise by chance after quantization.'
    ],
    relevance: 'One of the few large studies to report that multi-step reasoning and self-calibration are the abilities that break first, which is the premise of Experiment D. Its small-model bit-width advice (8-bit under 13B) matches our observation that 8-bit is nearly harmless.',
    links: [
        { l: 'ICML 2024 proceedings (PMLR 235)', u: 'https://proceedings.mlr.press/v235/li24bb.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2402.18158' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2402.18158' },
        { l: 'Code (THU-NICS)', u: 'https://github.com/thu-nics/qllm-eval' }
    ],
    read: 'main'
},
{
    id: 'emergent-quantized', title: 'Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study',
    year: 2024, posted: '2023-07', authors: 'Peiyu Liu, Zikang Liu, Ze-Feng Gao, Dawei Gao, Wayne Xin Zhao, Yaliang Li, Bolin Ding, Ji-Rong Wen',
    venue: 'LREC-COLING 2024', status: 'published', cat: 'quant-scaling', origin: 'new', prio: 2, arxiv: '2307.08072',
    tags: ['in-context learning', 'chain of thought', 'instruction following', 'gptq', '2-bit', 'lora'],
    summary: 'Tests whether in-context learning, chain-of-thought reasoning and instruction following survive weight quantization of LLaMA, and how to recover low-bit models by sensitivity analysis and fine-tuning.',
    method: [
        'LLaMA 7B, 13B, 30B and 65B; GPTQ weight-only at 2, 4, 8 and 16 bits on all attention and FFN projections.',
        'ICL on MMLU and BBH, CoT on GSM8K, instruction following on Vicuna\'s 80 questions scored by GPT-3.5; total model bits as the memory axis.',
        'Sensitivity tests by component, outlier dimension and substructure; recovery by full fine-tuning or LoRA before quantization and by LoRA on the quantized weights after.'
    ],
    results: [
        '4-bit keeps the abilities: LLaMA-7B MMLU 5-shot 35.2 (16-bit) against 34.2 (4-bit); 13B 47.0 against 45.9; 30B 58.4 against 57.3.',
        '2-bit collapses: MMLU 5-shot 3.8, 14.8, 26.1 for 7B, 13B, 30B and GSM8K 0.0 for 7B and 13B.',
        'Keeping the FFN in FP16 raises 2-bit LLaMA-7B MMLU from 0.038 to 0.225; post-quantization LoRA lifts 2-bit 65B 5-shot MMLU from 22.6 to 44.4.'
    ],
    limits: [
        'One method (GPTQ weights only) and one family (LLaMA).',
        'The instruction-following test is 80 questions with a GPT-3.5 judge; some 2-bit scores sit near chance.'
    ],
    relevance: 'Directly asks our question for chain-of-thought: 4-bit keeps GSM8K chain-of-thought for large LLaMA, 2-bit does not. It gives the chain-of-thought baseline and the 2-bit failure floor for the sub-4-bit point we plan to add.',
    links: [
        { l: 'ACL Anthology (LREC-COLING 2024)', u: 'https://aclanthology.org/2024.lrec-main.461/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2307.08072' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2307.08072' }
    ],
    read: 'full'
},
{
    id: 'compressing-truth', title: 'Compressing LLMs: The Truth is Rarely Pure and Never Simple',
    year: 2024, posted: '2023-10', authors: 'Ajay Jaiswal, Zhe Gan, Xianzhi Du, Bowen Zhang, Zhangyang Wang, Yinfei Yang',
    venue: 'ICLR 2024', status: 'published', cat: 'quant-scaling', origin: 'new', prio: 3, arxiv: '2310.01382',
    tags: ['llm-kick', 'perplexity is not enough', 'pruning', 'knowledge-intensive', 'vicuna'],
    summary: 'Perplexity hides capability loss in compressed models. The paper introduces LLM-KICK, a knowledge-intensive benchmark, and shows that pruning methods degrade at modest sparsity where perplexity barely moves, while quantization holds up better.',
    method: [
        'Vicuna 7B, 13B and 33B compressed by magnitude pruning, SparseGPT and Wanda (unstructured and N:M) and by GPTQ, with AWQ and LLM.int8() at 8 bits.',
        'Tasks: factoid QA (FreebaseQA), MMLU, in-context retrieval QA (TriviaQA), CNN/DailyMail summarization and 80 MT-Bench instructions judged by GPT-4.',
        'A compressed model "matches" if it stays within 5% of the dense one; averages over three runs.'
    ],
    results: [
        'Vicuna-7B Factoid-QA: dense 65.44, Wanda at 50% sparsity 44.66, SparseGPT 42.86, magnitude 13.99 (Table 3).',
        '8-bit quantization of Vicuna-7B stays close to dense: GPTQ / AWQ / LLM.int8() give Factoid-QA 60.14 / 60.31 / 61.02 and MMLU 47.10 / 47.18 / 47.82 (Table 2).',
        'No N:M sparse model matches the dense one on factoid QA, MMLU or long-story summarization.'
    ],
    limits: [
        'Only Vicuna (licensing); recovery methods such as LoRA are not tested; GPT-4 judging on 80 and 100 item samples.',
        'Quantization coverage is mostly 8-bit; lower bit-widths are not compared across methods.'
    ],
    relevance: 'Argues for the same stance as our reports: do not trust a single headline metric after compression. Its "within 5%" matching rule is a reasonable pre-registered equivalence margin for our agreement and accuracy endpoints.',
    links: [
        { l: 'ICLR 2024 poster page', u: 'https://iclr.cc/virtual/2024/poster/19229', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2310.01382' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2310.01382' },
        { l: 'Code (VITA-Group)', u: 'https://github.com/VITA-Group/llm-kick' }
    ],
    read: 'full'
},
{
    id: 'lee-tradeoffs', title: 'Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and Model Size in Large Language Models From Edge to Giant',
    year: 2025, posted: '2024-09', authors: 'Jemin Lee, Sihyeong Park, Jinse Kwon, Jihun Oh, Yongin Kwon',
    venue: 'IJCAI 2025', status: 'published', cat: 'quant-scaling', origin: 'dossier', dossier: '§3.1', prio: 2, arxiv: '2409.11055',
    tags: ['instruction-tuned', 'gptq', 'awq', 'smoothquant', 'fp8', '405b', 'mt-bench'],
    summary: 'Existing quantization studies lean on perplexity and easy benchmarks. This paper evaluates quantized instruction-tuned models from 1B to 405B on 13 datasets plus an MT-Bench judge analysis, comparing four methods.',
    method: [
        '12 instruction-tuned models from Vicuna, Gemma and Llama (Llama-3.2-1B/3B up to Llama-3.1-405B); GPTQ and AWQ at 4-bit weights, SmoothQuant and FP8 at W8A8; greedy decoding.',
        'Open LLM Leaderboard v1 (ARC-c, GSM8k, HellaSwag, MMLU, TruthfulQA, Winogrande) and v2 (BBH, GPQA, IFEval, MATH-Lvl-5, MMLU-PRO, MuSR) plus 80 MT-Bench questions with a GPT-4 judge; vLLM serving.'
    ],
    results: [
        'Llama-2-13B AWQ 4-bit (6.5 GB) averages 57.77 on leaderboard v1, above FP16 Llama-2-7B (53.11, 14 GB).',
        'Llama-3.2-1B GPTQ 4-bit (AutoRound) averages 37.67 against 47.81 FP16; GSM8k falls from 33.36 to 8.04.',
        'SmoothQuant on Llama-3.1-405B falls to 35.56 on v2 against 47.70 FP16; FP8 is the most stable method; AWQ generally beats GPTQ; small models suffer most at 4-bit.',
        'Task difficulty does not reliably predict accuracy loss.'
    ],
    limits: [
        'One deterministic greedy run per setting, so no variance is reported; the dataset count is inconsistent (13 against 12) and a 405B FP16 row appears despite a statement that it was not measured.',
        'GPT-4 judging is lenient and less sensitive for strong models; post-training quantization only.'
    ],
    relevance: 'Source of the dossier\'s statement that quantized models struggle with instruction following. Its observation that 1B models lose GSM8k sharply at 4-bit (33.4 to 8.0) is consistent with our Gemma-1B style-shift outlier.',
    links: [
        { l: 'IJCAI 2025 proceedings (PDF)', u: 'https://www.ijcai.org/proceedings/2025/0902.pdf', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2409.11055' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2409.11055' }
    ],
    read: 'main', note: 'The first arXiv version was titled "A Comprehensive Evaluation of Quantized Instruction-Tuned Large Language Models: An Experimental Analysis up to 405B". The dossier read only the abstract.'
},
{
    id: 'bf16-or-death', title: '"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization',
    year: 2025, posted: '2024-11', authors: 'Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, Dan Alistarh',
    venue: 'ACL 2025 (Long Papers)', status: 'published', cat: 'quant-scaling', origin: 'new', prio: 2, arxiv: '2411.02355',
    tags: ['fp8', 'int8', 'int4', 'llama 3.1', 'vllm', 'arena-hard', 'text similarity', 'gptq', 'awq'],
    summary: 'A large empirical study (500,000+ evaluations) of FP8, INT8 W8A8 and INT4 weight-only quantization on the Llama-3.1 family, covering academic and real-world benchmarks, similarity of generated text and deployment cost.',
    method: [
        'Llama-3.1-Instruct 8B, 70B and 405B, plus DeepSeek-R1-Distill models; W8A8-FP by round-to-nearest, W8A8-INT by GPTQ with SmoothQuant at 70B, W4A16-INT by GPTQ with MSE clipping, group size 128.',
        'Open LLM Leaderboard v1 and v2, Arena-Hard-Auto, HumanEval and HumanEval+, RULER up to 128k, and AIME, MATH-500, GPQA-Diamond with 20 samples per query at temperature 0.6.',
        'ROUGE, BERTScore and STS between quantized and uncompressed generations; vLLM inference benchmarks on A6000, A100 and H100.'
    ],
    results: [
        'FP8 is effectively lossless at all scales; INT8 loses about 1 to 3% when tuned; INT4 weight-only recovers 99.36% on leaderboard v1.',
        'Reasoning models: most variants recover over 99%; the exceptions are W4A16 on Llama-8B (97.2%) and Qwen-1.5B (93.5%).',
        'On the smallest models the generated text diverges more (ROUGE-1 0.62 at 8B against 0.70 at 70B and 405B).',
        'W4A16 is the most cost-efficient for synchronous serving; W8A8 wins for large asynchronous deployments.'
    ],
    limits: [
        'KV cache, input embeddings and LM head not studied; no multilingual tasks; RULER not run at 405B.',
        'GPQA and MuSR are high variance for small models; inference numbers depend on one vLLM version and cloud pricing.'
    ],
    relevance: 'Confirms from 500k evaluations that 4-bit weight-only recovers over 99% on average benchmark scores, so any damage we report is a sampled-reasoning effect that benchmark averages hide. Its text-similarity measurement is a template for comparing our precisions on generated traces.',
    links: [
        { l: 'ACL Anthology (ACL 2025)', u: 'https://aclanthology.org/2025.acl-long.1304/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2411.02355' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2411.02355' }
    ],
    read: 'full'
},
{
    id: 'deepseek-quant-drop', title: 'Quantitative Analysis of Performance Drop in DeepSeek Model Quantization',
    year: 2025, posted: '2025-05', authors: 'Enbo Zhao, Yi Shen, Shuming Shi, Jieyun Huang, Zhihao Chen, Ning Wang, Siqi Xiao, Jian Zhang, Kai Wang, Shiguo Lian',
    venue: 'arXiv preprint', status: 'preprint', cat: 'quant-scaling', origin: 'new', prio: 3, arxiv: '2505.02390',
    tags: ['deepseek-r1', 'deepseek-v3', 'llama.cpp', 'q4_k_m', 'dq3_k_m', '671b'],
    summary: 'Evaluates multi-bit quantization of the 671B DeepSeek-R1 and V3 models in llama.cpp so they fit a single 8-GPU machine, and proposes DQ3_K_M, a dynamic 3-bit mix of q3, q4 and q6 tensors.',
    method: [
        'DeepSeek-R1, V3, V3-0324 and R1-distill-Qwen-32B; Q8_0, Q4_K_M, Q3_K_M, Q2_K, UD-Q2_K_XL, Q2_K_L and the proposed DQ3_K_M, all weight-only.',
        'MATH-500, AIME 2024, GPQA, LiveCodeBench, MBPP, MBPP+, MMLU, CMMLU, C-Eval; temperature 0.6, top-p 0.95, up to 32,768 tokens, 4 to 8 samples on small sets.'
    ],
    results: [
        'DeepSeek-R1 weighted average: FP8 85.82, Q4_K_M 85.24 (0.68% drop), Q3_K_M 84.28 (1.80%), DQ3_K_M 85.53 (0.34%).',
        'DeepSeek-V3-0324: DQ3_K_M 0.30% drop, Q2_K_L 14.66% drop (Table 4).'
    ],
    limits: [
        'Weight-only llama.cpp quantization only; AIME has 30 questions and some standard deviations are large.',
        'The V3 FP8 baseline is a third-party API; the DQ3_K_M layer rule is a heuristic with no ablation; Table 4 and Table 5 drop figures do not recompute exactly.'
    ],
    relevance: 'Gives a second published check that DeepSeek-R1-family models lose little at 4-bit, here at 671B. Our DeepSeek-R1-Distill-Qwen-1.5B result (a 1 point loss) sits in the same direction, but at a scale where the effect is expected to be larger.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2505.02390', o: true },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2505.02390' }
    ],
    read: 'full'
}
);
