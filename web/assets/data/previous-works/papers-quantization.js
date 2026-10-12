/* Quantization methods and tooling; quantization scaling laws and capability evaluation. */
window.PW.papers.push(

/* ================= QUANTIZATION METHODS AND TOOLING ================= */
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
}
);
