/* Self-consistency and confidence; self-correction and self-verification; verifiers and test-time compute. All at full precision unless noted. */
(window.PW_PAPERS = window.PW_PAPERS || []).push(

/* ================= SELF-CONSISTENCY, CONFIDENCE, AGREEMENT ================= */
{
    id: 'self-consistency', title: 'Self-Consistency Improves Chain of Thought Reasoning in Language Models',
    year: 2023, posted: '2022-03', authors: 'Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou',
    venue: 'ICLR 2023', status: 'published', cat: 'consistency', origin: 'new', prio: 1, arxiv: '2203.11171',
    tags: ['majority vote', 'self-consistency', 'chain of thought', 'sampling', 'gsm8k', 'palm', 'gpt-3'],
    summary: 'Greedy chain-of-thought decoding limits reasoning accuracy. Self-consistency samples many diverse reasoning paths and returns the most frequent final answer, with no training or verifier.',
    method: [
        'Temperature sampling with top-k: UL2-20B and LaMDA-137B at T = 0.5, k = 40; PaLM-540B at T = 0.7, k = 40; GPT-3 at T = 0.7. 40 outputs per question, results averaged over 10 runs.',
        'Aggregation by unweighted majority vote over final answers; weighted variants (normalized and unnormalized sums of model probabilities) were tested and did not help.',
        'Arithmetic (GSM8K, MultiArith, AQuA, SVAMP, ASDiv, AddSub), commonsense (CSQA, StrategyQA, ARC), symbolic and NLP tasks; baselines greedy CoT, sample-and-rank, beam search, prompt and model ensembles.'
    ],
    results: [
        'PaLM-540B GSM8K 56.5 to 74.4 (+17.9); abstract gains such as GSM8K +17.9, SVAMP +11.0, AQuA +12.2, StrategyQA +6.4, ARC-c +3.9 are quoted from different models.',
        'Beam search is worse than sampling: UL2-20B AQuA top-beam 23.6 at beam 1 falling to 10.2 at beam 40 against 26.9 for self-consistency at 40 paths (Table 6).',
        'LaMDA-137B GSM8K: CoT 17.1, 40 prompt permutations 19.2, self-consistency with 40 paths 27.7 (Table 7).',
        'Gains saturate at about 5 to 10 paths; on tasks where CoT hurts (RTE, ANLI) it recovers (RTE 79.1 to 86.3).'
    ],
    limits: [
        'Higher compute cost; requires a fixed answer set; model probabilities are poorly calibrated so weighting does not help (Sec. 2).',
        'Models like LaMDA-137B and PaLM-540B are not public; models still produce nonsensical paths whose answers happen to agree.'
    ],
    relevance: 'The procedure our pipeline uses (8 samples, majority vote, agreement share). Its observation that sequence probabilities are poorly calibrated is why agreement, not likelihood, is our confidence signal. All of its results are at full precision.',
    links: [
        { l: 'ICLR 2023 (slides hosted by iclr.cc)', u: 'https://ICLR.cc/media/iclr-2023/Slides/11718.pdf', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2203.11171' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2203.11171' }
    ],
    read: 'main', note: 'Venue from the arXiv comments ("Published at ICLR 2023 ... V4: camera ready version at ICLR 2023") and the ICLR-hosted slides; the OpenReview page was not opened. Notably, this foundational paper was missing from the dossier\'s reference list.'
},
{
    id: 'cot-prompting', title: 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
    year: 2022, posted: '2022-01', authors: 'Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou',
    venue: 'NeurIPS 2022 (Main Conference)', status: 'published', cat: 'consistency', origin: 'new', prio: 2, arxiv: '2201.11903',
    tags: ['chain of thought', 'few-shot prompting', 'emergent ability', 'palm', 'gsm8k', 'scale'],
    summary: 'Few-shot prompts whose exemplars include intermediate reasoning steps unlock multi-step reasoning in sufficiently large language models without any fine-tuning.',
    method: [
        'Eight hand-written exemplars (four for AQuA) of the form input, chain of thought, output; no prompt engineering.',
        'GPT-3 (350M to 175B), LaMDA (422M to 137B), PaLM (8B, 62B, 540B), UL2 20B and Codex; greedy decoding.',
        'Arithmetic (GSM8K, SVAMP, ASDiv, AQuA, MAWPS), commonsense (CSQA, StrategyQA, Date, Sports, SayCan) and symbolic tasks; baselines standard prompting and prior supervised results.'
    ],
    results: [
        'PaLM-540B GSM8K: standard 17.9%, CoT 56.9%, CoT with calculator 58.6%, above the prior best 55% (Table 1); SVAMP 69.4 to 79.0, MAWPS 79.2 to 93.3.',
        'Gains appear around 100B parameters; CoT can hurt smaller models ("chain-of-thought prompting is an emergent ability of model scale").',
        'Last-letter concatenation 2-word 7.6 to 99.4 and out-of-domain 4-word 0.0 to 63.0 on PaLM-540B.'
    ],
    limits: [
        'Whether the model really reasons is open; reasoning paths can be wrong while the answer is right; helps only at large scale.',
        'Prompt and exemplar order matter (coin-flip accuracy 99.6% to 71.4% across annotators); the text and Table 4 disagree on StrategyQA.'
    ],
    relevance: 'The basis for sampled chain-of-thought as our unit of analysis. Because CoT is reported as an ability that emerges with scale, 0.8B to 3B models sit near the edge of where it reliably helps, which is where extra noise from quantization could matter most.',
    links: [
        { l: 'NeurIPS 2022 proceedings', u: 'https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2201.11903' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2201.11903' }
    ],
    read: 'main'
},
{
    id: 'kadavath-know', title: 'Language Models (Mostly) Know What They Know',
    year: 2022, posted: '2022-07', authors: 'Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, Jared Kaplan',
    venue: 'arXiv preprint (Anthropic)', status: 'preprint', cat: 'consistency', origin: 'dossier', dossier: '§3.3', prio: 2, arxiv: '2207.05221',
    tags: ['p(true)', 'p(ik)', 'calibration', 'self-evaluation', 'multiple choice', 'honesty'],
    summary: 'Asks whether language models can judge their own answers. Large models are well calibrated on multiple-choice and True/False formats, can self-evaluate sampled answers via P(True), and a trained P(IK) probe predicts "I know" and partly generalizes across tasks.',
    method: [
        'Pretrained LMs of 800M, 3B, 12B and 52B parameters; calibration on BIG-Bench, MMLU, TruthfulQA, QuALITY and LogiQA with lettered options, "none of the above" and True/False variants.',
        'P(True): show the question and a proposed answer and read the probability of "(A) True"; samples at T = 1 on TriviaQA, Lambada, arithmetic, GSM8K and HumanEval, with few-shot and with five other samples in context.',
        'P(IK): a value head trained against the fraction of 30 T = 1 samples that are correct, trained mainly on TriviaQA; metrics ECE, RMS calibration error, AUROC, Brier score.'
    ],
    results: [
        'Calibration improves with scale and few-shot prompting; "none of the above" lowers MMLU accuracy and calibration; RLHF policies look miscalibrated but temperature 2.5 largely fixes it.',
        'P(IK) at 52B trained on TriviaQA against all tasks except GSM8K, AUROC: GSM8K 0.624 against 0.752, Lambada 0.606 against 0.853, arithmetic 0.928 against 0.987 (Table 1).',
        'Providing source material raises P(IK) from 18% to 78% on one example question.'
    ],
    limits: [
        'Authors: pretrained models only (RLHF briefly), no deception or honesty beyond calibration, only five sampling datasets.',
        'P(IK) may track question difficulty rather than self-knowledge (footnote 2); cross-model effect is small (about 6%).'
    ],
    relevance: 'Origin of the P(True) probe that the dossier proposes as a logit-based verifier at controlled precision. It tests 800M to 52B pretrained models at full precision, never quantized ones.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2207.05221', o: true },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2207.05221' }
    ],
    read: 'main', note: 'No peer-reviewed venue was found for this paper; it is cited as an arXiv preprint.'
},
{
    id: 'just-ask-calibration', title: 'Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback',
    year: 2023, posted: '2023-05', authors: 'Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, Christopher D. Manning',
    venue: 'EMNLP 2023 (Main)', status: 'published', cat: 'consistency', origin: 'new', prio: 2, arxiv: '2305.14975',
    tags: ['verbalized confidence', 'calibration', 'rlhf', 'ece', 'triviaqa', 'truthfulqa', 'sciq'],
    summary: 'For RLHF-tuned models whose token probabilities are poorly calibrated, asking the model to state its confidence in words is typically better calibrated than its conditional probabilities, often cutting expected calibration error by a relative 50%.',
    method: [
        'gpt-3.5-turbo, gpt-4, claude-1, claude-2 and Llama-2-70b-chat; 1,000 TriviaQA and 1,000 SciQ questions plus all 817 TruthfulQA questions.',
        'Baselines: label probability from 10 samples and an "Is True" probability. Verbalized methods: one- and two-stage numerical probabilities with top-k hypotheses, chain-of-thought, and linguistic expressions with human or optimized probability mappings.',
        'ECE, ECE after temperature scaling, Brier score and AUC of selective accuracy; correctness judged by GPT-4 or GPT-3.5 equivalence checks.'
    ],
    results: [
        'gpt-3.5-turbo TruthfulQA ECE 0.451 for label probability against 0.125 for linguistic one-stage; SciQ 0.256 against 0.065; gpt-4 TruthfulQA 0.445 against 0.082 (Tables 1 and 2).',
        'Generating several hypotheses before giving a confidence improves calibration; chain-of-thought does not.',
        'Llama-2-70b-chat shows less consistent gains.'
    ],
    limits: [
        'Factual-recall QA only, short-form; reasoning-heavy settings untested; verbalized methods are prompt-sensitive; claude-1 is deprecated.',
        'Correctness depends on LLM-based equivalence judgments.'
    ],
    relevance: 'Gives the verbalized-confidence alternative to our one-token verifier verdict. Its models are large API models; in the vision-language study Bigger or Cheaper?, 4-bit also lowered the verbalized-confidence parse rate from 0.99 to 0.64, one reason to prefer logit margins for a quantized verifier.',
    links: [
        { l: 'ACL Anthology (EMNLP 2023)', u: 'https://aclanthology.org/2023.emnlp-main.330/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2305.14975' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2305.14975' }
    ],
    read: 'full'
},
{
    id: 'cisc', title: 'Confidence Improves Self-Consistency in LLMs',
    year: 2025, posted: '2025-02', authors: 'Amir Taubenfeld, Tom Sheffer, Eran Ofek, Amir Feder, Ariel Goldstein, Zorik Gekhman, Gal Yona',
    venue: 'Findings of ACL 2025', status: 'published', cat: 'consistency', origin: 'new', prio: 1, arxiv: '2502.06233',
    tags: ['cisc', 'confidence-weighted voting', 'p(true)', 'within-question discrimination', 'sample efficiency'],
    summary: 'Majority voting needs many long reasoning paths. Confidence-Informed Self-Consistency (CISC) weights each path by the model\'s own confidence in it, cutting the number of paths needed by over 40% on average.',
    method: [
        'Confidence from the model itself: response probability, verbal binary, verbal 0 to 100, and P(True) (probability of "1" in a binary self-rating) obtained with a cached second prompt, so reasoning chains are identical across methods.',
        'Softmax normalization with temperature T tuned per model and method (T to infinity recovers plain voting); selection by the highest summed normalized confidence.',
        'Nine open-weight instruction-tuned models (Gemma2 2B/9B/27B, Qwen2.5 3B/14B/72B, Mistral 8B/22B/123B); GSM8K, MATH, MMLU-Pro and BBH; budgets of 5 and 10 against self-consistency up to 30.'
    ],
    results: [
        'P(True) gives 41% cost reduction at budget 5 and 46% at budget 10 with 1.6 and 1.1% accuracy gains; response probability 22% and 31%; verbal 0 to 100 22% and 30% (Table 1).',
        'Within-Question Discrimination ranks confidence methods in the same order as CISC performance, while calibration error (ECE) does not (Table 3).',
        'Eight CISC samples exceed 30-sample self-consistency for Gemma2-9B on MATH.'
    ],
    limits: [
        'Authors: two-step prompting may not fit all frameworks; P(True) needs token probabilities; small human evaluation; no trained confidence models.',
        'The within-question discrimination claim rests on four confidence methods; open-weight models only.'
    ],
    relevance: 'A directly usable idea for our setup: a within-question ranking of confidence matters more than calibration. It supports computing a verifier margin per sample and testing it inside the vote, and gives a baseline weighting for quantized models.',
    links: [
        { l: 'ACL Anthology (Findings ACL 2025)', u: 'https://aclanthology.org/2025.findings-acl.1030/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2502.06233' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2502.06233' }
    ],
    read: 'full'
},
{
    id: 'unsupervised-confidence', title: 'Unsupervised Confidence Calibration for Reasoning LLMs from a Single Generation',
    year: 2026, posted: '2026-04', authors: 'Thomas Zollo, Jimmy Wang, Richard Zemel',
    venue: 'ICML 2026 workshop (Statistical Frameworks for Uncertainty in Agentic Systems), poster', status: 'workshop', cat: 'consistency', origin: 'new', prio: 2, arxiv: '2604.19444',
    tags: ['self-consistency', 'label-free', 'calibration', 'distillation', 'single generation', 'qwen3', 'deepseek-r1-distill'],
    summary: 'Uses offline self-consistency on unlabeled data as a proxy target and trains a lightweight predictor that scores the confidence of a single generated response, removing the need for labels or repeated sampling at deployment.',
    method: [
        'Offline: k = 100 samples per unlabeled input at temperature 0.7; the self-consistency score is the share of samples giving each answer (Eq. 2).',
        'A linear model on last-token embeddings of the reasoning model, trained on a 50/50 calibration split, followed by isotonic regression onto the self-consistency targets.',
        'Nine reasoning models (Qwen3 0.6B to 14B, OpenReasoning-Nemotron-7B, Nemotron-Cascade-8B, two R1-Distill models); GSM8K, PolyMath, SciQ, TriviaQA, WebQuestions; ECE, Brier, AUROC.'
    ],
    results: [
        'Qwen family ECE2 / Brier: 0.089 / 0.161 against 0.217 / 0.223 for token probabilities and 0.304 / 0.296 for verbalized confidence (Table 1).',
        'Under language shift ECE2 / Brier 0.092 / 0.123 against 0.153 / 0.165; with only 5 samples it cuts ECE by over 50% relative to the best unsupervised baseline.'
    ],
    limits: [
        'Offline sampling is costly and inherits self-consistency\'s failure on confidently wrong models; only tasks with automatic evaluation.',
        'Three of nine models were excluded from QA for formatting failures; downstream studies use one decision-maker and one dataset.'
    ],
    relevance: 'Another use of self-consistency as a label-free target, here to calibrate other signals. It is a template for the "agreement as a free label" idea, and an open question is whether a quantized model\'s consistency target is as good as the full-precision one.',
    links: [
        { l: 'ICML 2026 workshop poster page', u: 'https://icml.cc/virtual/2026/74537', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2604.19444' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2604.19444' }
    ],
    read: 'main'
},
{
    id: 'trust-itself', title: 'When Should a Language Model Trust Itself? Same-Model Self-Verification as a Conditional Confidence Signal',
    year: 2026, posted: '2026-04', authors: 'Aditya Ajay Phalod',
    venue: 'arXiv preprint', status: 'preprint', cat: 'consistency', origin: 'dossier', dossier: '§3.3', alsoSite: true, prio: 1, arxiv: '2605.02915',
    tags: ['p(true)', 'self-verification', 'selective prediction', 'auroc', 'aurc', 'nf4', 'arc-challenge', 'truthfulqa'],
    summary: 'Tests whether a model\'s own True/False judgment of its selected answer adds value for selective prediction beyond cheap likelihood baselines. The signal helps on ARC-Challenge for several models but is unreliable on TruthfulQA-MC.',
    method: [
        'phi-2, Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, TinyLlama-1.1B-Chat, DeepSeek-R1-Distill-Llama-8B; score-based, no sampling; float16 by default, 4-bit NF4 with double quantization for larger models when CUDA is present.',
        'Self-verify score from the True/False token logits through a sigmoid of the logit difference, against LL-AVG, LL-SUM, margin and entropy; ARC-Challenge test and TruthfulQA-MC validation; batch 8, max 256 tokens, seed 4242, 2,000 bootstrap replicates.',
        'Metrics AUROC, AURC, error at 80% and 50% coverage, Brier score, ECE.'
    ],
    results: [
        'ARC-Challenge, Qwen-7B: AUROC 0.555 (LL-AVG) to 0.886 (Self-Verify), LL-SUM 0.753; AURC 0.364 to 0.143; error at 80% coverage 0.369 to 0.271 (Tables 1 and 4).',
        'TruthfulQA-MC: Qwen-7B AUROC 0.593 to 0.667 but LL-SUM reaches 0.742; TinyLlama falls to 0.363 and DeepSeek-R1-Distill-8B to 0.494.',
        'Ranking gains do not imply better probabilities: Brier and ECE can worsen.'
    ],
    limits: [
        'Authors: few scale points, multiple choice only, no semantic entropy or self-consistency baselines, two prompts, descriptive not mechanistic, uncorrected intervals.',
        'The 256-token cap and mixed float16 / 4-bit precision across models confound cross-model comparisons (not analyzed by the paper).'
    ],
    relevance: 'The paper whose limitations our verifier design addresses: free-form GSM8K traces, comparison with the vote, and controlled precision. It loads larger models in 4-bit without analyzing it, so precision is an uncontrolled factor in its own comparisons.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2605.02915' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2605.02915' }
    ],
    read: 'full'
},
{
    id: 'sc-backfires', title: 'When Self-Consistency Backfires: Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs',
    year: 2026, posted: '2026-08', authors: 'Utkarsh Bahuguna',
    venue: 'COLM 2026 Workshop on Efficient Reasoning (v1)', status: 'workshop', cat: 'consistency', origin: 'dossier', dossier: '§3.3', alsoSite: true, prio: 1, arxiv: '2608.11403',
    tags: ['majority vote', 'gpqa diamond', 'pre-registered', 'agreement gate', 'entropy', 'answer-token margin', 'qwen2.5-7b', 'llama-3-8b'],
    summary: 'On GPQA Diamond, majority vote lowers per-problem accuracy for most problems for two small instruction-tuned models. Three cheap confidence signals (agreement, entropy, answer-token margin) fail to recover the available headroom.',
    method: [
        'Qwen2.5-7B-Instruct-Turbo and Meta-Llama-3-8B-Instruct-Lite on Together AI; 198 GPQA Diamond questions with a pre-registered split (47 exploratory, 151 confirmatory) and 69 held-out problems for the margin claims.',
        '64 samples per problem at temperature 0.7 with one locked prompt; expected majority-vote accuracy by Monte Carlo; "backfire" is MV(64) minus MV(1) below zero; problem-level bootstrap with 1,000 resamples.',
        'Gates: agreement (k = 8, tau = 0.75), mean entropy (k = 4) and answer-token log-probability margin; grid oracle over N from 1 to 64 as an upper bound.'
    ],
    results: [
        'Backfire rate 56.6% (Qwen) and 65.7% (Llama); aggregate accuracy barely moves (Qwen 0.342 to 0.369).',
        'Grid oracle reaches 0.482 (Qwen) and 0.439 (Llama), 14 and 17 points above N = 1; agreement gate 0.368 against 0.369 for flat N = 64.',
        'Answer-token margin carries cross-question signal (+0.0604 [0.0183, 0.1017]) but not within-question signal (-0.0168 [-0.0527, 0.0182]); median margin of dissenting samples 20.52 nats.'
    ],
    limits: [
        'Authors: Llama is near chance (0.273 against 0.25), one benchmark, two small non-reasoning models, margin claims on one model, no mechanism for why agreement fails, no compute-matched baseline, in-sample entropy threshold.',
        'The registered second-model replication was rejected by the paper\'s own comparability rule; reasoning-native models could not be evaluated on metered serverless inference.'
    ],
    relevance: 'The cautionary counterpart to our result: agreement is a good signal on GSM8K (AUROC about 0.87) and weak on hard multiple choice, so a reviewer will ask whether our conclusions survive a hard benchmark. Its pre-registration and locked-prompt practice is what the dossier adopts.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2608.11403' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2608.11403' }
    ],
    read: 'full', note: 'arXiv comments: v1 accepted at the COLM 2026 Workshop on Efficient Reasoning; v2 additions are not peer reviewed. The numbers above are from v2, which keeps the v1 results unchanged.'
},
{
    id: 'llms-agree', title: 'When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement as Confidence Signals',
    year: 2026, posted: '2026-07', authors: 'Kaihua Ding',
    venue: 'arXiv preprint', status: 'preprint', cat: 'consistency', origin: 'dossier', dossier: '§3.3', alsoSite: true, prio: 1, arxiv: '2607.08065',
    tags: ['self-consistency', 'gpt-4.1', 'gpqa', 'aime', 'ece', 'aurc', 'spearman', 'cross-family'],
    summary: 'Audits whether agreement among a model\'s samples, or among models, is a reliable confidence signal. Agreement is a positive but weak, regime-dependent proxy, and is least reliable for the most self-consistent frontier model.',
    method: [
        'gpt-4.1-nano, -mini and gpt-4.1 at temperature 1.0 with K = 50 samples; exploratory check on Claude haiku-4.5, sonnet-4.6 and opus-4.7 with K = 10; GPQA Diamond and AIME.',
        'Data from 53 course participants (about 265,000 samples, 394 unique cases) analysed as a de-identified meta-analysis; three paired axes (model size, zero-shot against CoT, mini against full).',
        'Spearman correlation between self-consistency and majority-correct, ECE, Brier, AURC; hierarchical runner- and case-clustered bootstrap, Holm correction over twelve tests.'
    ],
    results: [
        'Correlation between consistency and majority-correct is positive in all twelve cells (0.20 to 0.59) and survives Holm correction.',
        'gpt-4.1 has the highest GPQA consistency (0.89) but the lowest correlation (0.20), ECE 0.41; when consistency is at least 0.8 it is wrong 48% of the time.',
        'Chain-of-thought raises accuracy (+0.067 on GPQA) but not the consistency-correctness link; adaptive stop-when-consensus uses 19.8 samples against 50 at equal error.'
    ],
    limits: [
        'Authors: single provider for the main audit, two benchmarks, runners wrote their own prompts, snapshots not logged, endpoints declared post hoc, CIs wide for baselines.',
        'K = 50 leaves measurement noise; shared wrong answers across models mean cross-model agreement does not certify correctness.'
    ],
    relevance: 'Gives the metric set (Spearman, ECE, AURC with clustered bootstrap) used in the dossier and the reminder that high agreement can be confidently wrong. It audits only closed API models, with no compression.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2607.08065' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2607.08065' }
    ],
    read: 'full'
},
{
    id: 'wrong-consensus', title: 'Decomposing Wrong-Consensus Agreement in LLM Self-Consistency',
    year: 2026, posted: '2026-08', authors: 'Lizhuo Zhang, Mengmeng Tang, Chenfeng Long, Xiaoyong Tang, Xiang Luo',
    venue: 'arXiv preprint', status: 'preprint', cat: 'consistency', origin: 'dossier', dossier: '§3.3', prio: 2, arxiv: '2608.18795',
    tags: ['wrong consensus', 'agreement index', 'answer preference', 'gpqa', 'aime', 'open-weights', 'gpt-4.1'],
    summary: 'Samples that agree on a wrong answer are common. The paper splits wrong-consensus agreement into a part explained by a per-question answer preference and a residual, defining an agreement index and a mechanical-coverage measure.',
    method: [
        'Main analysis on released GPT-4.1-family data (K = 50); controlled replication on five open-weights models (qwen3.5-9b, qwen3.8-27b, gemma4-26b, gemma4-31b, qwen3.5-122b) with 4 runs of K = 32 at temperature 0.7 through Ollama.',
        'GPQA-Diamond (198 questions) and a 200-question AIME sample; index Gamma = E[agreement | wrong run] / d, mechanical coverage phi, residual 1 - phi, uniform-null and Dirichlet-multinomial nulls; bootstrap with B = 10^4.'
    ],
    results: [
        'GPT-4.1 on GPQA-Diamond: phi between 0.806 and 0.927, so most wrong-consensus agreement is explained by per-question preference; on AIME phi between 0.586 and 0.781.',
        'Open-weights replication: phi between 1.001 and 1.056 in all ten cells, attributed to a finite-donor plug-in bias.',
        'Conclusion: agreement is graded evidence of correctness, not certification.'
    ],
    limits: [
        'i.i.d. voting assumption; model family, openness and protocol are perfectly confounded; four cells per benchmark; leave-one-out preferences from few donors bias phi upward.',
        'Appendices B to J were cut off in retrieval; no mechanism is identified.'
    ],
    relevance: 'Explains why confidently wrong majorities are question-specific: they are mostly per-question answer preference. That matches our finding that wrong unanimous answers are model-specific and that quantization does not manufacture them.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2608.18795' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2608.18795' }
    ],
    read: 'main'
},
{
    id: 'cross-model-disagreement', title: 'Cross-Model Disagreement as a Label-Free Correctness Signal',
    year: 2026, posted: '2026-03', authors: 'Matt Gorbett, Suman Jana',
    venue: 'arXiv preprint', status: 'preprint', cat: 'consistency', origin: 'dossier', dossier: '§3.3', prio: 2, arxiv: '2603.25450',
    tags: ['label-free', 'cross-model', 'perplexity', 'entropy', 'verifier', 'routing', 'mmlu', 'gsm8k'],
    summary: 'Detects when a model is wrong without labels by scoring the generator\'s greedy answer with a second model: cross-model perplexity and entropy from one forward pass, which catch confident errors that within-model uncertainty misses.',
    method: [
        'Generators Qwen2.5 0.5B/7B, Llama-3 1B/8B, Gemma-3 270M, Mistral 7B, OLMo-3 7B; verifiers such as Qwen2.5-7B-Instruct and Llama-3-8B-Instruct; bf16, greedy, up to 2,000 examples of MMLU, TriviaQA and GSM8K.',
        'Baselines: generator entropy and perplexity, P(True), verifier answer agreement, semantic entropy (k = 10); metrics AUROC and a gap-weighted gain on a single A10G.'
    ],
    results: [
        'MMLU mean AUROC 0.727 for cross-model perplexity against 0.595 for generator entropy; wins 12 of 15 pairs (Table 3).',
        'GSM8K: cross-model perplexity 0.623, cross-model entropy 0.621, generator entropy 0.584; P(True) is below chance at 0.463.',
        'The signal collapses when verifier and generator share failure modes (Mistral to Qwen on TriviaQA, CMP AUROC 0.42).'
    ],
    limits: [
        'Authors: verifier must be a better correctness proxy than the generator; greedy decoding only; static benchmarks.',
        'Inconsistent model lists and some numbers differ between tables and text; very weak generators (about 5% GSM8K) give no signal.'
    ],
    relevance: 'A natural baseline if a full-precision copy is available: score the 4-bit model\'s answers with the 16-bit model. It never pairs a model with its own quantized copy, which is the experiment the dossier flags as untested.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2603.25450' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2603.25450' }
    ],
    read: 'full'
},

/* ================= SELF-CORRECTION AND SELF-VERIFICATION ================= */
{
    id: 'self-correct-yet', title: 'Large Language Models Cannot Self-Correct Reasoning Yet',
    year: 2024, posted: '2023-10', authors: 'Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou',
    venue: 'ICLR 2024', status: 'published', cat: 'correction', origin: 'site', prio: 1, arxiv: '2310.01798',
    tags: ['intrinsic self-correction', 'oracle labels', 'gpt-4', 'multi-agent debate', 'self-consistency'],
    summary: 'Tests whether a model can fix its reasoning answers using only its own judgment. Without oracle labels, accuracy usually gets worse, and earlier reports of gains relied on ground-truth stopping signals or weak baselines.',
    method: [
        'Three-step intrinsic self-correction (answer, self-review, re-answer) for up to two rounds; oracle-label stopping versus no oracle; three feedback prompts.',
        'GPT-3.5-Turbo, GPT-4, GPT-4-Turbo and Llama-2-70b-chat; GSM8K test, CommonSenseQA dev and HotpotQA (100 questions); 200 random questions for non-GPT-3.5 models.',
        'Multi-agent debate compared with self-consistency at matched response counts; a constrained-generation prompt-design study.'
    ],
    results: [
        'With oracle labels GPT-3.5 GSM8K rises 75.9 to 84.3; without oracle it falls to 75.1 then 74.7; CSQA drops 75.8 to 38.1 (Tables 2 and 3).',
        'GPT-4 GSM8K 95.5 to 91.5 to 89.0; Llama-2-70b-chat GSM8K 62.0 to 43.5 to 36.5.',
        'At nine responses self-consistency reaches 88.2 against 83.0 for multi-agent debate on GSM8K (Table 7).'
    ],
    limits: [
        'Authors: scope is reasoning only; self-correction may work for style or safety alignment.',
        'Single runs on 100 to 200 question subsets for several models and only three feedback prompts.'
    ],
    relevance: 'The reason a small model should not be trusted to correct itself and why Experiment D measures self-judgment rather than assuming it. Its finding that voting beats debate at matched budget supports using plain majority vote as the baseline filter.',
    links: [
        { l: 'ICLR 2024 poster page', u: 'https://iclr.cc/virtual/2024/poster/18956', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2310.01798' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2310.01798' }
    ],
    read: 'full'
},
{
    id: 'self-refine', title: 'Self-Refine: Iterative Refinement with Self-Feedback',
    year: 2023, posted: '2023-03', authors: 'Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark',
    venue: 'NeurIPS 2023 (Main Conference)', status: 'published', cat: 'correction', origin: 'new', prio: 2, arxiv: '2303.17651',
    tags: ['self-refine', 'self-feedback', 'iterative refinement', 'gpt-4', 'test-time'],
    summary: 'A single LLM generates an output, critiques it and refines it in a loop at test time, with no supervised data, extra training or reinforcement learning, improving outputs by about 20 points on average across seven tasks.',
    method: [
        'Initial generation, then FEEDBACK and REFINE prompts with the full history in context; stop by a fixed step or a stop indicator, capped at 4 iterations.',
        'GPT-3.5, ChatGPT and GPT-4 (Codex for code); seven tasks: dialogue, code optimization, code readability, math reasoning, sentiment reversal, acronyms, constrained generation; human and GPT-4 preference alongside task metrics.'
    ],
    results: [
        'GPT-4: dialogue response 25.4 to 74.6, sentiment reversal 3.8 to 36.2, constrained generation 15.0 to 45.0, code optimization 27.3 to 36.0, math reasoning 92.9 to 93.1 (Table 1).',
        'Math gains are small: ChatGPT judged 94% of instances "everything looks good"; with oracle feedback they are larger.',
        'Vicuna-13B could not follow the feedback and refine prompts; failure analysis: 33% mislocalized feedback, 61% inappropriate fixes.'
    ],
    limits: [
        'Authors: base models need strong few-shot and instruction following; closed models; English only; no misuse guard.',
        'Human preference done by the authors on 150 examples per dataset; the decoding setting is described inconsistently.'
    ],
    relevance: 'The canonical positive claim for self-feedback, with the telling exception that math reasoning barely improves and weak models cannot follow the loop. Both points set expectations for a 0.8B to 3B verifier.',
    links: [
        { l: 'NeurIPS 2023 proceedings', u: 'https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2303.17651' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2303.17651' }
    ],
    read: 'main'
},
{
    id: 'correct-own-mistakes', title: 'When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs',
    year: 2024, posted: '2024-06', authors: 'Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, Rui Zhang',
    venue: 'Transactions of the ACL (TACL) 2024', status: 'published', cat: 'correction', origin: 'new', prio: 2, arxiv: '2406.01297',
    tags: ['survey', 'self-correction', 'external feedback', 'oracle', 'fair comparison', 'taxonomy'],
    summary: 'A critical review of self-correction: it defines three research questions, a taxonomy of frameworks and feedback sources, and a checklist, and finds no reliable evidence of successful prompted self-correction without external information.',
    method: [
        'Literature analysis to May 2024, no new experiments; categories explicit feedback or direct refinement, post-hoc or generation-time, and feedback source (intrinsic, external information, fine-tuning).',
        'Each study is classed realistic or unrealistic and fair or unfair by whether it uses oracle information, weak initial prompts or asymmetric information.'
    ],
    results: [
        'Intrinsic self-correction works for decomposable responses and when reliable external feedback (code interpreter, search) exists, not for general prompted reasoning (Sec. 4 and 5.1).',
        'Many SFT-based methods use more than 100K training instances (Table 6); feedback generation is the bottleneck; self-consistency and generate-and-rank are recommended baselines.'
    ],
    limits: [
        'Coverage ends May 2024 and focuses on inference-time self-correction.',
        'The fair and realistic classification depends on authors\' judgment of "best possible" prompts; no new experiments.'
    ],
    relevance: 'Gives the vocabulary and the fairness checklist that a self-verification study on quantized models should follow, notably comparing against self-consistency as the strong baseline.',
    links: [
        { l: 'TACL 2024 (ACL Anthology)', u: 'https://aclanthology.org/2024.tacl-1.78/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2406.01297' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2406.01297' }
    ],
    read: 'main'
},
{
    id: 'small-lms-strong-verifiers', title: 'Small Language Models Need Strong Verifiers to Self-Correct Reasoning',
    year: 2024, posted: '2024-04', authors: 'Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, Lu Wang',
    venue: 'Findings of ACL 2024', status: 'published', cat: 'correction', origin: 'new', prio: 1, arxiv: '2404.17140',
    tags: ['score', 'self-correction', 'small language models', 'verifier', 'critique', 'llama-2-13b', 'gemma-7b'],
    summary: 'Can models of 13B parameters or fewer fix their own reasoning without a stronger teacher? SCORE builds self-correction data from the small model\'s own correct and incorrect solutions, but gains depend on having a strong verifier to decide when to correct.',
    method: [
        'Sample 10 CoT solutions per training question; pair each incorrect with each correct solution and have the same model critique the incorrect one using the correct one as a hint; filter by rules and by whether the model can then produce a correct revision.',
        'Fine-tune the same model with LoRA (rank 32, 3 epochs) to produce critique and correction in one pass; LLaMA-2-13B-chat and Gemma-7B-it; GSM8K and CommonsenseQA with transfer to MATH Level 1, QASC and RiddleSense.',
        'Verifier settings: fine-tuned self-verifier, GPT-4 few-shot verifier, oracle.'
    ],
    results: [
        'Gemma-7B-it with a GPT-4 verifier gains +8.3 on average across five datasets; LLaMA-2-13B-chat gains +4.2 on GSM8K (37.2 to 41.4).',
        'With the fine-tuned self-verifier gains are minimal: +0.3 on GSM8K for LLaMA and +0.4 for Gemma; the oracle verifier reaches 46.0 on GSM8K for LLaMA.',
        'Prompted self-correction hurts in several settings (LLaMA CSQA to QASC -7.5).'
    ],
    limits: [
        'Authors: needs GPU compute for data generation and adds latency (about 1.3 to 1.4x).',
        'One correction iteration; the strong verifier is GPT-4, which conflicts with avoiding API dependence; dev sets used as test sets.'
    ],
    relevance: 'The most direct support for the dossier\'s finding that sub-2B self-verifiers are near chance: even for 7B to 13B models, verification, not refinement, is the bottleneck. A quantized small model starts from that weak position.',
    links: [
        { l: 'ACL Anthology (Findings ACL 2024)', u: 'https://aclanthology.org/2024.findings-acl.924/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2404.17140' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2404.17140' }
    ],
    read: 'full'
},
{
    id: 'self-verification-weng', title: 'Large Language Models are Better Reasoners with Self-Verification',
    year: 2023, posted: '2022-12', authors: 'Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, Jun Zhao',
    venue: 'Findings of EMNLP 2023', status: 'published', cat: 'correction', origin: 'new', prio: 2, arxiv: '2212.09561',
    tags: ['self-verification', 'backward verification', 'condition mask', 'reranking', 'gpt-3', 'gsm8k'],
    summary: 'Chain-of-thought is sensitive to single mistakes. The paper has the model verify its own candidate answers by treating each answer as a condition and re-predicting a masked original condition, then picks the best-verified candidate, with no training.',
    method: [
        'Forward reasoning: K = 5 sampled CoT candidates with Wei et al. prompts; backward verification by rewriting question and answer as a declarative conclusion and either True/False item verification or condition-mask verification (mask a number, re-predict it).',
        'Each candidate is verified P = 10 times and scored by consistent outcomes; GPT-3 code-davinci-001 and Instruct-GPT code-davinci-002 (175B); six arithmetic sets, CommonsenseQA and Date Understanding; baselines CoT, self-consistency, PAL.'
    ],
    results: [
        'Instruct-GPT, CoT to CoT with self-verification: GSM8K 60.81 to 65.14, SingleEq 91.01 to 93.40, AddSub 82.78 to 86.33, MultiArith 96.13 to 99.15 (Table 1).',
        'Smaller gains with weaker models (code-davinci-001 GSM8K 13.84 to 13.92); verification accuracy on GSM8K 58.9% against 53.4% for CoT answers and 35.7% random.',
        'Combined with self-consistency on GSM8K 78.00 to 78.32.'
    ],
    limits: [
        'Authors: hand-built prompts may bias; helps only if a correct answer is among the candidates; weaker for small models; evaluates conclusions not reasoning; multiplies sampling cost.',
        'Relies on 2022-era OpenAI API models; internal inconsistencies in terminology and table descriptions.'
    ],
    relevance: 'An early, cheap form of self-verification layered on voting. Its gain shrinks with model size, which is the expected pattern for our 1B to 3B models and a reminder that the verifier adds information only within the plurality.',
    links: [
        { l: 'ACL Anthology (Findings EMNLP 2023)', u: 'https://aclanthology.org/2023.findings-emnlp.167/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2212.09561' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2212.09561' }
    ],
    read: 'main'
},
{
    id: 'more-yap', title: 'More Yap Less Meaning: Uncovering Self-Improvement Behavior in SLMs',
    year: 2026, posted: '2026-06', authors: 'Marina Igitkhanian, Erik Arakelyan',
    venue: 'GEM Workshop at ACL 2026', status: 'workshop', cat: 'correction', origin: 'site', prio: 2, arxiv: '2606.08471',
    tags: ['small language models', 'hints', 'self-explanation', 'sufficiency test', 'gsm8k', 'asdiv', 'aqua', 'overthinking'],
    summary: 'Even when given the correct answer and asked to explain what went wrong, small models improve only 4.4% on average. Longer self-generated explanations correlate with worse outcomes.',
    method: [
        '1.5B to 8B models: Llama-3.1-8B-Instruct, Phi-4-mini, Qwen2.5-Math-1.5B/7B, Gemma-2-2B (instruction-tuned) and DeepSeek-R1-Distill 1.5B and 8B (reasoning); GSM8K, ASDiv, AQuA, AR-LSAT and a BIG-bench sports subset.',
        '"Sufficiency test": answer with CoT, keep incorrect answers, have the model write a hint from its reasoning, the wrong answer and the ground truth (leak check with up to 3 retries, masking if needed), then re-answer with the hint; token budgets 256 to 2048.',
        'Metric accuracy gain; baseline a retry prompt that says the answer was wrong without a hint.'
    ],
    results: [
        'Average gain 4.4% across models, datasets and budgets; instruction-tuned and reasoning models show equal average gains.',
        'Hint length is negatively correlated with gain; correct and incorrect hints share similar keywords (16.1% against 15.4% for "model reasoning").',
        'The no-hint retry baseline gains 2.45% on average, with outliers (Gemma-2-2B on AR-LSAT +13.91).'
    ],
    limits: [
        'Authors: only 1.5B to 8B models; leakage detection is imperfect and conservative filtering may reject harmless hints.',
        'No significance tests or variance; length-difficulty control from a manual sample of 100; some Qwen2.5-Math outputs were hallucinated continuations.'
    ],
    relevance: 'Shows a hard floor on small-model self-correction even with the answer in hand, and that longer explanations hurt. It is the full-precision baseline for any claim that quantization changes self-judgment.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2606.08471' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2606.08471' }
    ],
    read: 'full', note: 'Venue from the arXiv comments field ("GEM Workshop at ACL 2026"); the workshop page was not opened.'
},

/* ================= VERIFIERS, REWARD MODELS, TEST-TIME COMPUTE ================= */
{
    id: 'gsm8k-verifiers', title: 'Training Verifiers to Solve Math Word Problems',
    year: 2021, posted: '2021-10', authors: 'Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman',
    venue: 'arXiv preprint (OpenAI)', status: 'preprint', cat: 'verifiers', origin: 'new', prio: 1, arxiv: '2110.14168',
    tags: ['gsm8k', 'verifier', 'best-of-n', 'dataset', 'math word problems', 'calculator'],
    summary: 'Introduces GSM8K, 8.5K grade-school math problems, and shows that sampling many solutions and picking the one a trained verifier ranks highest scales better than fine-tuning the generator alone.',
    method: [
        'GSM8K: 8.5K problems split into 7.5K train and 1K test, 2 to 8 steps each, human-written with contractor re-solving.',
        'Finetuning baseline (20 epochs, one T = 0 sample); verification: train a generator for 2 epochs, sample 100 completions per training problem, label by final-answer correctness, train a token-level verifier for 1 epoch, and at test time rank 100 samples at T = 0.7.',
        'GPT-3 6B and 175B; calculator annotations; verifier trained with a joint language-model and verification objective.'
    ],
    results: [
        'A 6B verifier slightly outperforms a finetuned 175B model on the full dataset, roughly a 30x model-size equivalent.',
        'Verifier accuracy rises up to about 400 completions per problem then declines; majority vote over the top 3 to 5 of 100 samples is best.',
        'Token-level verifiers beat solution-level ones and overfit less; a large generator with a small verifier beats the reverse.'
    ],
    limits: [
        'About 1.7% of problems disagree between annotators and some subtle errors likely remain; verifiers can be fooled by adversarial solutions at large sample counts.',
        'Only GSM8K and GPT-3; the 175B verifier result is a single run.'
    ],
    relevance: 'Defines the benchmark our main experiment runs on and the generator-plus-verifier pattern our verifier stage re-implements with the model judging itself. The dossier\'s test-as-pool caveat refers to this train and test split.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2110.14168', o: true },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2110.14168' }
    ],
    read: 'full', note: 'No peer-reviewed venue was found for this paper; it is cited as an arXiv preprint.'
},
{
    id: 'lets-verify', title: 'Let\'s Verify Step by Step',
    year: 2024, posted: '2023-05', authors: 'Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe',
    venue: 'ICLR 2024', status: 'published', cat: 'verifiers', origin: 'new', prio: 2, arxiv: '2305.20050',
    tags: ['process supervision', 'prm800k', 'outcome supervision', 'reward model', 'best-of-n', 'math'],
    summary: 'Compares outcome supervision (feedback on the final answer) with process supervision (feedback on each reasoning step) for training reward models on competition math, and releases PRM800K step-level labels.',
    method: [
        'Labelers mark each step positive, negative or neutral; the PRM scores a solution as the product of step probabilities; supervision stops at the first incorrect step; PRM800K has about 800K step labels over 75K solutions to 12K problems.',
        'Large-scale models fine-tuned from GPT-4; small-scale models on 1.5B tokens of MathMix with a large PRM as a label oracle; active learning selects convincing wrong-answer solutions; evaluation by best-of-N on 500 held-out MATH problems.'
    ],
    results: [
        'Best-of-1860: PRM 78.2%, ORM 72.4%, majority voting 69.6%; the PRM gap widens as N grows.',
        'Active learning is about 2.6x more data-efficient than uniform labelling.',
        'Out-of-distribution (AP and AMC problems): PRM 72.9%, ORM 63.8%, majority 61.3%.'
    ],
    limits: [
        'Possible test-set contamination, uncertain transfer beyond math, unstable iterative PRM retraining.',
        'Process labels are expensive and the small-scale study depends on a large PRM as a proxy oracle; the OOD set size is inconsistent (224 against 234).'
    ],
    relevance: 'Explains why majority voting is a weak baseline against a trained step-level verifier, so the vote filter we use is the cheap end of the verifier spectrum. It also motivates the dossier\'s remark that a graded verifier score needs sampled verdicts or logits.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2305.20050' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2305.20050' }
    ],
    read: 'full', note: 'ICLR 2024 venue is from secondary indexes (ML Anthology, later citations); the iclr.cc or OpenReview page was not opened.'
},
{
    id: 'scaling-test-time', title: 'Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters',
    year: 2025, posted: '2024-08', authors: 'Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar',
    venue: 'ICLR 2025', status: 'published', cat: 'verifiers', origin: 'new', prio: 2, arxiv: '2408.03314',
    tags: ['test-time compute', 'prm search', 'beam search', 'revisions', 'compute-optimal', 'math', 'palm 2'],
    summary: 'How much can a fixed inference-time compute budget improve an answer? The best strategy depends on question difficulty, so a compute-optimal policy that allocates test-time compute per prompt can beat best-of-N and even a much larger model.',
    method: [
        'PRM-guided search (best-of-N weighted, beam search, lookahead) with a PRM trained on Monte Carlo rollouts, and a revision model fine-tuned on up to four prior incorrect answers; budget split between sequential revisions and parallel samples.',
        'PaLM 2-S* (Codey) on MATH (12k train, 500 test); questions binned into five difficulty quintiles by base-model pass@1 over 2,048 samples; FLOPs-matched comparison with a roughly 14x larger model.'
    ],
    results: [
        'Compute-optimal allocation nearly matches PRM best-of-N with up to 4x less compute (16 against 64 generations); for revisions 64 against 256 samples.',
        'Easy questions favour sequential revisions; harder ones a mix; beam search degrades on easy bins at high budgets.',
        'Test-time compute is preferable on easy and intermediate questions or low inference load; pretraining is preferable on the hardest questions or high load.'
    ],
    limits: [
        'Difficulty estimation costs 2,048 samples and is not counted; one base model and one benchmark; revision and verification needed capability-specific fine-tuning.',
        'The abstract claims over 4x efficiency while the Discussion says 2 to 4x; search over-optimizes the PRM on easy problems.'
    ],
    relevance: 'The framework behind the dossier\'s "voting is a way to spend compute" reading. Under quantization the compute budget buys different things (smaller weights, longer traces), which is the trade-off Not All Bits Are Equal begins to map.',
    links: [
        { l: 'ICLR 2025 proceedings', u: 'https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2408.03314' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2408.03314' }
    ],
    read: 'main', note: 'The ICLR proceedings title ends "... Than Scaling Parameters for Reasoning"; arXiv says "Model Parameters".'
}
);
