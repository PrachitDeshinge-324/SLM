/* Self-improvement and self-training: methods, theory, and failure modes. */
window.PW.papers.push(

{
    id: 'star', title: 'STaR: Bootstrapping Reasoning With Reasoning',
    year: 2022, posted: '2022-03', authors: 'Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman',
    venue: 'NeurIPS 2022 (Main Conference)', status: 'published', cat: 'selftrain', origin: 'dossier', dossier: '§3.4', alsoSite: true, prio: 1, arxiv: '2203.14465',
    tags: ['star', 'rationalization', 'gpt-j', 'bootstrapping', 'commonsenseqa', 'gsm8k', 'arithmetic'],
    summary: 'A model can improve its own reasoning by keeping only the rationales that reach the correct answer and fine-tuning on them, repeating the loop. For failed problems, "rationalization" regenerates a rationale given the correct answer as a hint.',
    method: [
        'GPT-J (6B) fine-tuned with mesh-transformer-jax; each outer loop: few-shot prompt for rationale and answer, rationalize failures with the answer as a hint (hint removed from the training example), keep rationales with correct answers, and fine-tune the original pretrained model (not the previous iterate).',
        'Arithmetic (50,000 generated questions, 10,000 per iteration), CommonsenseQA (9,741 train) and GSM8K (7,473 train / 1,319 test); schedule starting at 40 steps (300 for arithmetic with rationalization) growing 20% per iteration, capped at 30 iterations for GSM8K.',
        'Baselines: GPT-J few-shot direct and CoT, GPT-J fine-tuned for direct answers, GPT-3 fine-tuned, LaMDA 137B few-shot CoT.'
    ],
    results: [
        'CommonsenseQA dev: STaR with rationalization 72.5%, without 68.8%, GPT-J fine-tuned directly 60.0%, GPT-3 fine-tuned 73.0%, few-shot CoT GPT-J 36.6% (Table 1).',
        'Arithmetic: 89.5% overall after 16 iterations against 76.3% for a baseline trained on 10,000 examples without rationales; GSM8K 10.7% against 5.8% for direct fine-tuning and 3.1% for few-shot CoT (Table 2).',
        'Crowdworkers preferred STaR rationales over few-shot ones by 30% and over human-written ones by 74% (20 raters).'
    ],
    limits: [
        'The base model must already be above chance in few-shot (GPT-2 could not bootstrap even arithmetic); poor rationale filtering when chance accuracy is high; no thorough hyperparameter search.',
        'Rationalization can short-cut via the hint; higher temperatures were worse; a single 6B model; GSM8K training capped at 30 iterations.'
    ],
    relevance: 'The template for any self-training loop in phase 2, and the origin of the "needs a minimum base capability" caveat that matters for 0.8B models. STaR uses ground-truth labels for filtering, unlike the vote filter we study.',
    links: [
        { l: 'NeurIPS 2022 proceedings', u: 'https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2203.14465' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2203.14465' }
    ],
    read: 'main'
},
{
    id: 'lmsi', title: 'Large Language Models Can Self-Improve',
    year: 2023, posted: '2022-10', authors: 'Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, Jiawei Han',
    venue: 'EMNLP 2023 (Main)', status: 'published', cat: 'selftrain', origin: 'dossier', dossier: '§3.4', alsoSite: true, prio: 1, arxiv: '2210.11610',
    tags: ['lmsi', 'self-consistency', 'palm 540b', 'majority vote', 'unlabeled', 'self-generated questions', 'distillation'],
    summary: 'A pretrained LLM can improve its reasoning using only unlabeled questions: sample many chain-of-thought answers, keep those that agree with the majority answer, and fine-tune on them in several prompt formats.',
    method: [
        'PaLM-540B (also UL2, and PaLM-8B and 62B for distillation); 32 reasoning paths per question at T = 0.7; keep only paths reaching the majority answer; no ground-truth labels.',
        'Each kept path is turned into four formats (few-shot CoT, few-shot direct, zero-shot CoT, zero-shot direct); 10k steps, learning rate 5e-5, batch 32.',
        'In-domain GSM8K, DROP, ARC-Challenge, OpenBookQA, ANLI-A2/A3; out-of-domain AQuA, SVAMP, StrategyQA, ANLI-A1, RTE, MNLI; baselines standard prompting, CoT, self-consistency, prior SOTA.'
    ],
    results: [
        'Self-consistency before and after LMSI: GSM8K 74.4 to 82.1, DROP 78.2 to 83.0, OpenBookQA 90.0 to 94.4, ANLI-A3 63.4 to 67.9 (Table 3).',
        'Out-of-domain: SVAMP 79.0 to 82.8, ANLI-A1 68.8 to 79.2, MNLI-M/MM 72.0/74.0 to 81.8/82.2 (Table 4).',
        'Self-generated questions from 10 seeds give GSM8K 78.1 against 82.1 with real questions; distillation lifts the 62B model 29.7 to 57.4 and the 8B model 5.0 to 33.4 on GSM8K (Table 7).'
    ],
    limits: [
        'Gains on UL2 are smaller than on PaLM-540B; self-generated questions underperform real ones.',
        'Mostly one very large model, no variance reporting, and filtering relies on majority agreement being calibrated; some training answers are incorrect.'
    ],
    relevance: 'The canonical vote filter our pseudo-label audit evaluates. It keeps every path that reaches the majority answer, trusting agreement to be calibrated at 540B; whether that holds for 1B models at 4-bit is exactly our label precision and yield measurement.',
    links: [
        { l: 'ACL Anthology (EMNLP 2023)', u: 'https://aclanthology.org/2023.emnlp-main.67/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2210.11610' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2210.11610' }
    ],
    read: 'full'
},
{
    id: 'sharpening', title: 'Self-Improvement in Language Models: The Sharpening Mechanism',
    year: 2025, posted: '2024-12', authors: 'Audrey Huang, Adam Block, Dylan J. Foster, Dhruv Rohatgi, Cyril Zhang, Max Simchowitz, Jordan T. Ash, Akshay Krishnamurthy',
    venue: 'ICLR 2025 (Oral)', status: 'published', cat: 'selftrain', origin: 'dossier', dossier: '§3.4', alsoSite: true, prio: 1, arxiv: '2412.01951',
    tags: ['theory', 'coverage', 'sft-sharpening', 'rlhf-sharpening', 'best-of-n', 'self-reward', 'sample complexity'],
    summary: 'Gives a statistical framework for self-improvement without outside feedback: using the model\'s own likelihood as a reward, "sharpening" concentrates its distribution on its own best responses. Sample cost is governed by a coverage coefficient.',
    method: [
        'Self-reward r(y|x) = log pi_base(y|x); (epsilon, delta)-sharpened means mass at least 1 - delta on the argmax response for all but an epsilon fraction of prompts; cost counted as n x N sample-and-evaluate queries.',
        'SFT-Sharpening: best-of-N by self-reward then supervised fine-tuning; RLHF-Sharpening: KL-regularized RL with a DPO variant, and XPO with an exploration bonus.',
        'Assumptions: realizability, coverage C_cov = E[1/pi_base(y*|x)], margin condition, concentrability bounds. Inference-time experiments on Phi3-Mini/Small/Medium, Phi3.5-Mini, Llama3.2-3B-Instruct, Mistral-7B-Instruct, GSM8K, MATH, ProntoQA, MMLU subsets and GameOf24; best-of-N up to 50 at T = 1.0, 10 seeds.'
    ],
    results: [
        'Lower bound: any sharpening algorithm needs m of order C log|Pi| / (epsilon^2 (1 + log(C/epsilon))); SFT-Sharpening matches it up to constants and is called minimax optimal.',
        'RLHF-Sharpening with XPO has no dependence on coverage, giving an exponential gap in a linear softmax example (SFT cost exp(Omega(d)) against polynomial).',
        'Empirically best-of-N sharpening beats temperature-1 sampling in all model and task pairs and greedy decoding on every task for at least one model; gains saturate around N = 10 to 20.'
    ],
    limits: [
        'Realizability is nontrivial and argmax is NP-hard in the worst case; RLHF results need margin and concentrability bounds; the linear example is stylized.',
        'Performance can degrade with larger N when the reference model or self-reward is flawed; a single self-reward is evaluated; later sections were cut off in retrieval.'
    ],
    relevance: 'Why the dossier reports coverage (any correct sample in 8): it sets the cost of self-improvement and is the damage a self-training loop cannot repair. It assumes a clean self-reward and says nothing about a biased or quantized verifier.',
    links: [
        { l: 'ICLR 2025 proceedings', u: 'https://proceedings.iclr.cc/paper_files/paper/2025/hash/bee8c2bc757f6bbc3efd7cf1b979f0c9-Abstract-Conference.html', o: true },
        { l: 'ICLR 2025 oral page', u: 'https://iclr.cc/virtual/2025/oral/31831' },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2412.01951' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2412.01951' }
    ],
    read: 'main', note: 'An earlier version appeared at a NeurIPS 2024 workshop (Mathematics of Modern Machine Learning).'
},
{
    id: 'lrm-self-train', title: 'Can Large Reasoning Models Self-Train?',
    year: 2025, posted: '2025-05', authors: 'Sheikh Shafayat, Fahim Tajwar, Ruslan Salakhutdinov, Jeff Schneider, Andrea Zanette',
    venue: 'arXiv preprint', status: 'preprint', cat: 'selftrain', origin: 'dossier', dossier: '§3.4', alsoSite: true, prio: 1, arxiv: '2505.21444',
    tags: ['self-rewarded training', 'srt', 'majority vote reward', 'rloo', 'grpo', 'reward hacking', 'collapse', 'verl'],
    summary: 'Asks whether online RL can be sustained with no ground-truth labels, using majority voting over the model\'s own answers as the reward (Self-Rewarded Training, SRT). It matches early RL with labels but eventually collapses through reward hacking.',
    method: [
        'Qwen2.5-Math-7B, Qwen3-14B-Base, Llama-3.1-8B-Instruct, DeepSeek-Math-7B-Instruct (Qwen3-4B-Base on Reasoning Gym); RLOO and GRPO in verl.',
        'Reward r(y) = 1[answer(y) = majority answer over n sampled responses]; evolving teacher (current policy) against fixed teacher (base model); 32 rollouts per prompt on Reasoning Gym.',
        'MATH-12K and DAPO training data, evaluation on MATH-500 and for peak results AIME 2024/2025 and AMC; baselines ground-truth RL, SFT on majority vote (LMSI), DPO, ScPO.'
    ],
    results: [
        'Peak mean@32 averaged over AIME24, AIME25 and AMC for Qwen2.5-Math-7B: SFT 0.15, DPO 0.18, ScPO 0.23, SRT 0.20, ground-truth RL 0.33 (MATH-12K, Table 1).',
        'Llama-3.1-8B-Instruct MATH-500 rises from 52.6% to nearly 60%; SRT beats the fixed-teacher variant by 6 to 10 points on Reasoning Gym tasks.',
        'All four math models collapse after peak: reward and KL jump, outputs become high-entropy text then a template answer such as \\boxed{1}; lower learning rate and fewer rollouts delay collapse, RL algorithm and KL coefficient do not prevent it.'
    ],
    limits: [
        'Majority voting is the only self-feedback mechanism studied; the authors suggest LLM judges, generative verifiers and consistency regularization.',
        'Peak metrics on a few benchmarks; collapse results are for specific learning rates and budgets; Reasoning Gym experiments use one model family; the text says SRT is comparable to RL while Table 1 shows it below.'
    ],
    relevance: 'Source of the two-sided prediction for a quantized loop: noisier votes could delay collapse, or lower yield could cap the gain. Models are 4 to 14B at full precision on large GPUs, so it says nothing about quantized or sub-3B bases.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2505.21444', o: true },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2505.21444' },
        { l: 'Project website', u: 'https://self-rewarding-llm-training.github.io/' }
    ],
    read: 'main', note: 'No peer-reviewed venue was found (the v2 PDF is marked "Preprint, 2025"). Appendices C to L were cut off in retrieval.'
},
{
    id: 'generalization-gap', title: 'On the Generalization Gap in Self-Evolving Language Model Reasoning',
    year: 2026, posted: '2026-05', authors: 'Zhenting Qi, Susanna Maria Baby, Stefanie Anna Baby, Kan Yuan, Andrew Tomkins, Tu Vu, Da-Cheng Juan, Cyrus Rashtchian',
    venue: 'ICML 2026 (poster)', status: 'published', cat: 'selftrain', origin: 'dossier', dossier: '§3.4', alsoSite: true, prio: 1, arxiv: '2606.01075',
    tags: ['self-evolution', 'generator-verifier', 'dpo', 'knights and knaves', 'gemma-3', 'verifier recall', 'pass@k', 'curriculum'],
    summary: 'Studies how close closed-loop self-evolution can get to oracle-supervised training when only an unlabeled prompt set and a base model are available, comparing single-round verification, revision with feedback, iterative training and curriculum.',
    method: [
        'Gemma-3-it at 1B, 4B and 12B (27B as an approximate bound), Qwen-2.5-7B-Instruct in an appendix; one model is both generator and verifier through different system prompts.',
        'Generator-verifier game with 8 candidates per query and 16 verifier passes each; a response is positive if its correctness rate is at least tau = 0.6 and negative if at least 1 - tau; DPO full fine-tuning, 3 epochs, batch 256.',
        'Knights-and-Knaves puzzles (train on 2 to 3 people, test on 2 to 8) plus OpenThoughts3 with GSM8K, MATH500, MATHHard, TabMWP; exact match at temperature 0.7, four seeds; baselines oracle verifier, INTUITOR, Absolute Zero.'
    ],
    results: [
        'Gemma-3 4B on KK: base 31.0%, simple self-evolution 40.7%, revision 42.2%, oracle 46.6%; three unsupervised rounds reach 44.1%; adding a final oracle round gives 53.2%.',
        'At 1B, simple self-evolution reaches at best 8.4% (tau = 0.8) and revision 7.8% (Table 3), a gain the dossier records as only 0.6 points; at 12B revision reaches 52.8% against oracle 53.6%, about 98.5% of oracle.',
        'Pass@1 on KK rises 31.0 to 40.7 while pass@32 falls slightly 78.0 to 77.4, a sharpening rather than new capability; verifier recall rises 52% to 62% for 4B but 32% to 36% for 1B (n = 100).'
    ],
    limits: [
        'A persistent 8 to 13% gap to oracle supervision; gains plateau and do not extend to much harder instances; downstream gains on OpenThoughts3 are modest and sometimes negative.',
        'Largely one benchmark and one model family; baseline comparisons are acknowledged as not controlled; Qwen-7B numbers differ between tables; authors are at Google and evaluate Gemma.'
    ],
    relevance: 'The same generator-verifier game as ours, but with 16 sampled verdicts and a threshold. It shows coverage (pass@32) is the ceiling and that a 1B verifier\'s recall is the bottleneck, in agreement with our Gemma-1B verifier failure.',
    links: [
        { l: 'ICML 2026 poster page', u: 'https://icml.cc/virtual/2026/poster/61802', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2606.01075' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2606.01075' }
    ],
    read: 'full', note: 'Venue from the arXiv comments field ("Published at ICML 2026") and the ICML 2026 poster page (Poster, Hall A #2106); the page does not say explicitly whether it is a main-conference poster.'
},
{
    id: 'self-improvement-reversal', title: 'Progress or Regress? Self-Improvement Reversal in Post-training',
    year: 2025, posted: '2024-07', authors: 'Ting Wu, Xuefeng Li, Pengfei Liu',
    venue: 'ICLR 2025', status: 'published', cat: 'selftrain', origin: 'dossier', dossier: '§3.4', prio: 2, arxiv: '2407.05013',
    tags: ['iterative sft', 'iterative dpo', 'diversity', 'out-of-distribution', 'pass@k', 'self-improvement reversal'],
    summary: 'Iterative post-training reports gains in pass@1, but those can hide losses. The paper evaluates improvement problems, solution diversity and out-of-distribution generalization and finds pass@1 rises while diversity and OOD accuracy fall.',
    method: [
        'LLaMA-2-7B, Mistral-7B, LLaMA3-8B fully fine-tuned (and a LLaMA-2-70B LoRA check); iterative SFT, iterative DPO and iterative SFT-DPO.',
        'CommonsenseQA, GSM8K, MATH and MBPP; N = 50 samples per problem at temperature 0.75, binary correctness reward; SFT 3 epochs and DPO beta 0.3.',
        'Pass@N of the first model on problems later solved, distinct-n-gram and Sentence-BERT diversity, and a GSM8K-to-MATH-algebra OOD test with a group-disparity measure.'
    ],
    results: [
        'Pass@1 rises with iterations then plateaus or falls after about 4 to 5; LLaMA2-7B iterative SFT on GSM8K gains +12.31 after 5 iterations.',
        'The first model\'s pass@N on newly solved problems approaches near-perfect, so improvement is mainly selection of answers it already could produce.',
        'Diversity falls consistently on all metrics; iterative SFT and SFT-DPO harm OOD accuracy; iterative DPO raises accuracy but widens disparity between easy and hard levels.'
    ],
    limits: [
        'Authors: compute limits model and task range; metrics may miss real-world behaviour; iterative training is expensive.',
        'Mostly 7B models, one OOD transfer, exact-match correctness, no variance across seeds; the coverage explanation is preliminary.'
    ],
    relevance: 'Supplies the diversity and OOD cautions for phase 2: report pass@8 against pass@1 and answer diversity per round, not only accuracy. Matches the coverage-as-ceiling reading of the dossier.',
    links: [
        { l: 'ICLR 2025 poster page', u: 'https://iclr.cc/virtual/2025/poster/29657', o: true },
        { l: 'ICLR 2025 camera-ready PDF', u: 'https://proceedings.iclr.cc/paper_files/paper/2025/file/1fa0c4e5a7e189729230d018b229abc7-Paper-Conference.pdf' },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2407.05013' }
    ],
    read: 'full', note: 'The arXiv page shows no venue; ICLR 2025 comes from the proceedings PDF header and poster page. Search also returned ICML 2024 workshop listings, which are earlier versions.'
},
{
    id: 'rise-and-collapse', title: 'Self-Improvement Can Self-Regress: The Rise-and-Collapse Failure Mode of LLM Self-Training',
    year: 2026, posted: '2026-06', authors: 'Jianzhe Lin',
    venue: 'arXiv preprint', status: 'preprint', cat: 'selftrain', origin: 'dossier', dossier: '§3.4', prio: 2, arxiv: '2606.21090',
    tags: ['reinforce', 'grpo', 'collapse', 'early stopping', 'code', 'qwen-2.5', 'care'],
    summary: 'REINFORCE post-training for code can raise pass@1 quickly and then collapse within the same campaign. The paper documents rise-then-collapse and compares between-campaign, within-campaign and algorithm-level interventions.',
    method: [
        'Qwen-2.5-3B-Instruct and 7B-Instruct (32B only as a frozen reference); REINFORCE with group size 16 and a binary code-grader reward; headline protocol is 10 sequential 20-step campaigns; learning rate 1e-6, temperature 0.7.',
        'Five code campaigns from HumanEval, MBPP and APPS (50 train, 30 held-out problems each); interventions CARE (capability-effect memory with a transfer gate), early stopping, and GRPO; baselines KL and EWC-style regularization.'
    ],
    results: [
        'One 7B run: pass@1 25% to 81% in the first 50 steps, then near zero by step 200.',
        'KL and EWC regularization do not prevent collapse; early stopping gives the highest REINFORCE 7B result (22.2%); naive GRPO 7B 20.7%.',
        'CARE helps the fragile 3B regime (9.5% against 4.9%, paired difference interval [+0.4, +8.9]) but overlaps at 7B (13.8% against 11.8%).'
    ],
    limits: [
        'n = 3 to 5 seeds, wide intervals; collapses are not eliminated; CARE\'s gate is scalar; Gemma-3-4B result is a single-seed pilot.',
        'It is within-task over-optimization with an execution reward, not self-generated labels; the limitations section and later analysis were cut off in retrieval.'
    ],
    relevance: 'A named failure mode and mitigation set (early stopping at the peak) to pre-register for a phase-2 loop. It uses ground-truth execution rewards, so it speaks to optimization dynamics rather than label quality.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2606.21090' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2606.21090' }
    ],
    read: 'main'
},
{
    id: 'rstar-math', title: 'rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking',
    year: 2025, posted: '2025-01', authors: 'Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, Mao Yang',
    venue: 'ICML 2025 (Oral)', status: 'published', cat: 'selftrain', origin: 'site', prio: 2, arxiv: '2501.04519',
    tags: ['mcts', 'process preference model', 'ppm', 'self-evolution', 'code-augmented cot', 'qwen2.5-math', 'phi3-mini'],
    summary: 'Small models (1.5B to 7B) can master competition math through four rounds of self-evolution: a policy SLM does Monte Carlo Tree Search guided by a process preference model, and the verified trajectories become the next round\'s training data.',
    method: [
        'Policy models Qwen2.5-Math-1.5B, Phi3-mini-Instruct (3.8B), Qwen2-Math-7B, Qwen2.5-Math-7B and a 7B process preference model; self-evolution run only on Qwen2.5-Math-7B.',
        'Each step is a one-step CoT written as a Python comment plus code that must execute; Q-values come from MCTS rollouts, terminal-guided in rounds 1 and 2 and PPM-augmented from round 3; the PPM is trained with a Bradley-Terry pairwise loss on step-level preference pairs.',
        '747k math problems; round 1 bootstraps with DeepSeek-Coder-V2-Instruct (236B); SFT 2 epochs at learning rate 7e-6; baselines GPT-4o, o1-preview, o1-mini, Qwen best-of-N with a 72B outcome reward model.'
    ],
    results: [
        'Qwen2.5-Math-7B on MATH rises from 58.8% to 90.0% with 64 trajectories, against 85.5% for o1-preview; Phi3-mini 41.4% to 86.4%; AIME 2024 53.3% (8 of 15) at 64 trajectories (Table 5).',
        'Round-by-round MATH: 58.8, 75.2, 86.6, 87.0, 89.4 (Table 6); PPM 89.4% against outcome reward model 82.6% and Q-value model 88.2% (Table 8).'
    ],
    limits: [
        'Geometry needing visual understanding is unsolved; self-evolution ran only on one 7B model for GPU reasons; round-1 bootstrapping took about two weeks on 10 nodes.',
        'Training data contain label noise (19 of 20 inspected unsolved problems had incorrect labels); test-time cost is 3k to 16k tokens per problem.'
    ],
    relevance: 'Shows small models can self-improve strongly, but only with a search procedure and a purpose-built step scorer plus a 236B bootstrap model. It does not test self-improvement with the model\'s own judgment alone, which is our narrower question.',
    links: [
        { l: 'ICML 2025 poster page', u: 'https://icml.cc/virtual/2025/poster/46400', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2501.04519' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2501.04519' },
        { l: 'Code', u: 'https://github.com/microsoft/rStar' }
    ],
    read: 'main', note: 'The site proposal page lists it as an arXiv 2025 preprint. The ICML 2025 site lists poster and oral entries; the arXiv page itself shows no venue.'
},
{
    id: 'rest-em', title: 'Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models',
    year: 2024, posted: '2023-12', authors: 'Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J. Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, Abhishek Kumar, Alex Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Elsayed, Hanie Sedghi, Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura Culp, Lechao Xiao, Maxwell L. Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yundi Qian, Yamini Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, Noah Fiedel',
    venue: 'Transactions on Machine Learning Research (TMLR) 2024', status: 'published', cat: 'selftrain', origin: 'new', prio: 1, arxiv: '2312.06585',
    tags: ['rest-em', 'expectation-maximization', 'binary reward', 'palm 2', 'math', 'apps', 'self-training'],
    summary: 'Fine-tuning on human data is limited by its quantity and quality. ReST-EM alternates sampling solutions, filtering them with a binary correctness reward, and fine-tuning, and outperforms training on human solutions for math and code at larger model sizes.',
    method: [
        'Generate (E-step): sample 32 solutions per problem on MATH and 64 on APPS with top-k 40 at temperature 0.7, score by final-answer match or unit tests, keep up to 10 correct per problem.',
        'Improve (M-step): fine-tune the base pretrained model, not the previous iteration\'s model, on the kept samples with a reward-weighted likelihood; repeat for a few iterations.',
        'PaLM 2-S, PaLM 2-S* and PaLM 2-L; MATH (7,500 training problems) and APPS Introductory (2,342); transfer to GSM8K, Hungarian HS finals, HumanEval, Big-Bench Hard.'
    ],
    results: [
        'MATH test gain 5.94% for PaLM 2-S and 6.34% for PaLM 2-L; APPS gain 5.6% and 6.4%; model-generated data beats human SFT especially for the larger model.',
        'MATH with 64-sample majority voting: 48.82 (ReST-EM) against 44.02 (base).',
        'One iteration with 3x data gives 40.3% pass@1 against 41% after two and 41.9% after three iterations.'
    ],
    limits: [
        'Needs a problem set and an automated reward; pass@K lags at large K; more iterations can overfit (APPS regresses at iteration 2).',
        'Two domains and one model family through an API; hyperparameters are not given in the retrieved text.'
    ],
    relevance: 'The cleanest published recipe for self-training with a binary filter and re-fine-tuning from the base model, which is the design of our phase-2 round. Its reward is ground-truth correctness; ours would be agreement.',
    links: [
        { l: 'arXiv (camera-ready, TMLR)', u: 'https://arxiv.org/abs/2312.06585' },
        { l: 'TMLR listing (ML Anthology index)', u: 'https://mlanthology.org/tmlr/2024/singh2024tmlr-beyond' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2312.06585' }
    ],
    read: 'full', note: 'arXiv comments: "Accepted to TMLR. Camera-ready version". The TMLR/OpenReview page could not be opened (bot check), so the TMLR listing link is a secondary index.'
},
{
    id: 'v-star', title: 'V-STaR: Training Verifiers for Self-Taught Reasoners',
    year: 2024, posted: '2024-02', authors: 'Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, Rishabh Agarwal',
    venue: 'COLM 2024', status: 'published', cat: 'selftrain', origin: 'new', prio: 2, arxiv: '2402.06457',
    tags: ['v-star', 'verifier', 'dpo', 'best-of-n', 'star', 'rft', 'incorrect solutions'],
    summary: 'STaR-style self-improvement throws away incorrect solutions. V-STaR keeps them and trains a DPO verifier on correct versus incorrect samples from every iteration, then uses it to pick the best of many candidates at inference.',
    method: [
        'Generator fine-tuned on a growing set of correct self-generated solutions; k = 16 samples per query, three iterations; verifier trained with DPO against the SFT reference on the Cartesian product of correct and incorrect solutions per problem, with LoRA.',
        'Inference best-of-k with 128 candidates ranked by the verifier\'s log-likelihood ratio; LLaMA2 7B and 13B and CodeLLaMA 7B and 13B; train on GSM8K and MBPP, evaluate also on a 150-problem MATH Level-1 subset and HumanEval; baselines SFT, STaR, RFT, majority voting.'
    ],
    results: [
        'Abstract: 4% to 17% test accuracy improvement over existing self-improvement and verification approaches; 7B V-STaR surpasses LLaMA2 70B 8-shot on GSM8K.',
        'DPO verifiers outperform the outcome-reward-model approach above 4 candidates on GSM8K; best-of-k saturates at k of about 16 and beats majority voting for k up to 64.'
    ],
    limits: [
        'LoRA only (compute); verifier-in-the-loop and reranking variants tested only on MBPP; two training tasks and small transfer sets.',
        'Depends on tasks with verifiable correctness; the outcome-model baseline is the authors\' own implementation; gains shrink at large k.'
    ],
    relevance: 'The training-time analogue of our verifier stage: use incorrect samples to train a judge rather than discarding them. It needs ground-truth labels for those incorrect samples; our setting would have to use votes.',
    links: [
        { l: 'University of Edinburgh research portal (COLM 2024)', u: 'https://www.research.ed.ac.uk/en/publications/v-star-training-verifiers-for-self-taught-reasoners/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2402.06457' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2402.06457' }
    ],
    read: 'full', note: 'Venue from the PDF header ("Published as a conference paper at COLM 2024") and the institutional record; the COLM/OpenReview page was not opened.'
},
{
    id: 'self-rewarding', title: 'Self-Rewarding Language Models',
    year: 2024, posted: '2024-01', authors: 'Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston',
    venue: 'ICML 2024', status: 'published', cat: 'selftrain', origin: 'new', prio: 2, arxiv: '2401.10020',
    tags: ['llm-as-a-judge', 'iterative dpo', 'self-rewarding', 'llama 2 70b', 'alpacaeval', 'mt-bench'],
    summary: 'A fixed, human-trained reward model is a ceiling. In Self-Rewarding LMs one model both follows instructions and judges its own candidate responses via LLM-as-a-Judge prompting, so the reward signal improves across iterative DPO rounds.',
    method: [
        'Llama 2 70B seeded with 3,200 Open Assistant instruction-following examples and 1,630 LLM-as-a-Judge examples; additive 5-point judging prompt with a justification and a score.',
        'Per iteration: new prompts by few-shot prompting, N = 4 candidates at T = 0.7, each scored by 3 sampled judgments, highest and lowest form a preference pair; M1 by SFT, M2 and M3 by DPO (beta 0.1, learning rate 1e-6).',
        'Evaluation by GPT-4 head-to-head, AlpacaEval 2.0, MT-Bench and nine NLP benchmarks.'
    ],
    results: [
        'AlpacaEval 2.0 win rate against GPT-4 Turbo: 9.94% (M1), 15.38% (M2), 20.44% (M3), above Claude 2 (17.19%) and GPT-4 0613 (15.76%) (Table 1).',
        'Head-to-head: M3 beats SFT 62.5% to 9.8% and M2 47.7% to 12.5%; reward-modelling pairwise accuracy 65.1% (SFT) to 81.7% (M3) (Table 4).',
        'MT-Bench 6.85 (SFT) to 7.25 (M3), with the smallest gains on math, code and reasoning.'
    ],
    limits: [
        'Authors: three iterations in one setting, response length grows across iterations (a confound), reward hacking and safety not analyzed, results preliminary.',
        'Human evaluation on 50 instructions by the authors; prompts are generated by a fixed external model; NLP benchmark scores sometimes fall.'
    ],
    relevance: 'The strongest published case that a model\'s own judge improves with training, but at 70B. Small models\' own verifiers are weak (Small LMs Need Strong Verifiers; Generalization Gap; the dossier records sub-2B verifiers near chance), so the same loop is not expected to work for 1B models, quantized or not.',
    links: [
        { l: 'ICML 2024 proceedings (PMLR 235)', u: 'https://proceedings.mlr.press/v235/yuan24d.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2401.10020' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2401.10020' }
    ],
    read: 'full'
},
{
    id: 'scpo', title: 'Self-Consistency Preference Optimization',
    year: 2025, posted: '2024-11', authors: 'Archiki Prasad, Weizhe Yuan, Richard Yuanzhe Pang, Jing Xu, Maryam Fazel-Zarandi, Mohit Bansal, Sainbayar Sukhbaatar, Jason Weston, Jane Yu',
    venue: 'ICML 2025', status: 'published', cat: 'selftrain', origin: 'new', prio: 1, arxiv: '2411.04109',
    tags: ['scpo', 'self-consistency', 'preference optimization', 'vote margin', 'unsupervised', 'llama-3 8b', 'zebralogic'],
    summary: 'Applies self-consistency during training instead of only at inference: preference pairs are built from the most and least consistent responses to unlabeled problems and trained with a vote-weighted loss, with no gold labels or reward model.',
    method: [
        'For each query sample k = 8 responses; chosen is the most-voted answer if its votes reach a threshold tau, rejected the least-voted; instance weight w = (V(y+) - V(y-))/k in a weighted DPO plus weighted length-normalized NLL loss (beta 0.5, alpha 1).',
        'Two iterations (a third adds little); model-generated new queries; Llama-3 8B Base for GSM8K and MATH, Llama-3 8B Instruct for ZebraLogic; temperature 0.7, top-p 0.9, learning rate 5e-6, 10 epochs.',
        'Baselines: seed model, IRPO with gold labels, IRPO with a reward model (ArmoRM-Llama3-8B), LMSI-style NLL on majority answers.'
    ],
    results: [
        'GSM8K greedy: seed 41.17, LMSI M2 56.71, ScPO unsupervised M1 61.03 and M2 63.91, IRPO with gold labels M2 64.29, semi-supervised ScPO 66.64 (Table 1).',
        'MATH: seed 14.46, ScPO unsupervised M2 19.72, IRPO gold 20.32; ZebraLogic puzzle accuracy 11.6% to 18.1%, above Llama-3 70B (17.2%).',
        'Vote-margin weighting beats an unweighted loss in all cases (by about 2.5 points at M1 on GSM8K); self-consistency orders pairs more correctly than the reward model.'
    ],
    limits: [
        'Authors: a third iteration adds little; only a quarter of MATH seed problems reach a clear majority; vote-accuracy correlation is weakest on MATH; needs a parsable answer.',
        'Only 8B models and single runs; hyperparameters chosen with dev or test guidance; ArmoRM was trained on in-domain data; the text and Table 1 disagree on the M1 gain (22.74 against 19.86).'
    ],
    relevance: 'The most direct recipe for the vote-filtered training we propose for phase 2, and the baseline MACA and TTRL compare to. It shows the vote margin itself is the training weight, so a drop in agreement margin under quantization would shrink the signal, not just the yield.',
    links: [
        { l: 'ICML 2025 proceedings (PMLR 267)', u: 'https://proceedings.mlr.press/v267/prasad25a.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2411.04109' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2411.04109' }
    ],
    read: 'full'
},
{
    id: 'ttrl', title: 'TTRL: Test-Time Reinforcement Learning',
    year: 2025, posted: '2025-04', authors: 'Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, Bowen Zhou',
    venue: 'NeurIPS 2025', status: 'published', cat: 'selftrain', origin: 'new', prio: 1, arxiv: '2504.16084',
    tags: ['ttrl', 'majority voting reward', 'grpo', 'unlabeled', 'test-time training', 'aime', 'qwen2.5-math'],
    summary: 'Reinforcement learning without ground-truth labels: sample many answers per unlabeled test prompt, take the majority answer as the estimated label, and reward answers that match it. Performance can even exceed the maj@N of the initial model.',
    method: [
        'GRPO applied independently per benchmark (PPO and PRIME also tested); 64 responses per prompt at temperature 0.6 (1.0 for Qwen2.5-Math and large reasoning models), 32 downsampled for training; learning rate 5e-7; 10, 30 and 80 episodes for MATH-500, AMC and AIME 2024.',
        'Qwen2.5-Math-1.5B/7B, Qwen2.5-7B/32B, Qwen3-8B, LLaMA-3.1-8B-Instruct, Mistral, DeepSeek-R1-LLaMA-8B and others; AIME 2024, AMC, MATH-500, GPQA-Diamond; pass@1 over 16 responses.'
    ],
    results: [
        'Qwen2.5-Math-7B AIME 2024 pass@1 12.9 to 40.2 (+27.3, 211.6%); four-benchmark average 31.1 to 54.9; Qwen2.5-Math-1.5B MATH-500 32.7 to 73.0 (Table 1).',
        'DeepSeek-R1-LLaMA-8B AIME 51.7 to 69.2; Mistral-Nemo-Instruct AIME falls 0.8 to 0 (Table 2).',
        'On AIME 2024 label accuracy is only about 37% yet reward accuracy is about 92% because scattered wrong answers rarely match the wrong majority ("lucky hit"); gains shrink on harder MATH levels (L1 25.9 to 71.2, L5 22.3 to 39.2).'
    ],
    limits: [
        'Authors call it an initial exploration; analyses of prior knowledge and hyperparameters are incomplete.',
        'Depends on the backbone\'s prior; sensitive to temperature and episodes; trains and tests on the same unlabeled test questions (transductive); the baseline comparison is acknowledged as possibly unfair.'
    ],
    relevance: 'The cleanest demonstration that majority voting is a usable training reward on small math models. Its "lucky hit" analysis shows why a noisy majority can still give mostly correct rewards, which is why the precision of vote-filtered labels is the key quantity we measure across bit-widths.',
    links: [
        { l: 'NeurIPS 2025 poster page', u: 'https://neurips.cc/virtual/2025/poster/117645', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2504.16084' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2504.16084' },
        { l: 'Code (PRIME-RL)', u: 'https://github.com/PRIME-RL/TTRL' }
    ],
    read: 'full'
}
);
