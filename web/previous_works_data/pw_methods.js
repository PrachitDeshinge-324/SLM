/* Benchmarks and evaluation methodology: datasets we evaluate on, label-free accuracy estimation, calibration, statistics. */
(window.PW_PAPERS = window.PW_PAPERS || []).push(

{
    id: 'math-dataset', title: 'Measuring Mathematical Problem Solving With the MATH Dataset',
    year: 2021, posted: '2021-03', authors: 'Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt',
    venue: 'NeurIPS 2021 (Datasets and Benchmarks)', status: 'published', cat: 'methods', origin: 'new', prio: 2, arxiv: '2103.03874',
    tags: ['math', 'math-500', 'competition mathematics', 'benchmark', 'amps', 'boxed answer'],
    summary: 'Introduces MATH, 12,500 competition problems with step-by-step solutions, and the AMPS pretraining corpus. Accuracy was low even for large models, which the authors read as scaling alone not solving mathematical reasoning.',
    method: [
        '12,500 problems from AMC 10, AMC 12, AIME and similar, 7,500 train and 5,000 test, seven subjects, difficulty levels 1 to 5, final answer in \\boxed{} and Asymptote for figures.',
        'AMPS: about 100,000 Khan Academy problems plus Mathematica-generated problems; GPT-2 (0.1B to 1.5B) pretrained on AMPS, GPT-3 fine-tuned or few-shot; scoring by exact match of the boxed answer after normalization.'
    ],
    results: [
        'Average accuracy: GPT-2 0.1B 5.4% to 1.5B 6.9%; GPT-3 13B fine-tuned 5.6% and 175B few-shot 5.2%.',
        'A 0.1B model with AMPS pretraining matches a 13B GPT-3 without it; GPT-2 1.5B gets about 15% on level 1 and about 4% on level 5.',
        'Human baselines on 20 problems: 40% for a non-mathematics student and 90% for a three-time IMO gold medallist.'
    ],
    limits: [
        'The human baseline is small and informal; GPT-3 models were not pretrained on AMPS, so rows are not like for like.',
        'Generated solutions often contain logic errors even when the final answer is right.'
    ],
    relevance: 'Our MATH-500 cells use this benchmark (MATH-500 is a 500-problem subset used by Let\'s Verify Step by Step). The dossier drops MATH-500 from phase 1 because 31 to 44% of Qwen-0.8B and DeepSeek samples are truncated at our token limits, a consequence of this benchmark\'s long solutions.',
    links: [
        { l: 'NeurIPS 2021 Datasets and Benchmarks (PDF)', u: 'https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf', o: true },
        { l: 'NeurIPS 2021 poster page', u: 'https://neurips.cc/virtual/2021/poster/36901' },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2103.03874' },
        { l: 'Code and data', u: 'https://github.com/hendrycks/math/' }
    ],
    read: 'full'
},
{
    id: 'svamp', title: 'Are NLP Models really able to Solve Simple Math Word Problems?',
    year: 2021, posted: '2021-03', authors: 'Arkil Patel, Satwik Bhattamishra, Navin Goyal',
    venue: 'NAACL 2021', status: 'published', cat: 'methods', origin: 'new', prio: 3, arxiv: '2103.07191',
    tags: ['svamp', 'math word problems', 'challenge set', 'shallow heuristics', 'asdiv-a'],
    summary: 'Shows that strong math word problem solvers rely on shallow heuristics, and introduces SVAMP, a 1,000-problem challenge set made by small variations of ASDiv-A problems, on which state-of-the-art models score far lower.',
    method: [
        'MAWPS (2,373 problems) and ASDiv-A (1,218) with 5-fold cross-validation; Seq2Seq, GTS and Graph2Tree models with or without RoBERTa embeddings, plus a Transformer in the appendix.',
        'Question-removed ablation and a constrained model without word-order information; SVAMP built from 100 seed examples chosen by clustering, varied in question sensitivity, reasoning ability and structural invariance, then filtered to 1,000 with at most two operators.'
    ],
    results: [
        'Graph2Tree (RoBERTa) scores 88.7 on MAWPS and 82.2 on ASDiv-A but only 43.8 on SVAMP.',
        'With the question removed, models still solve many problems: Graph2Tree 77.7 on MAWPS and 64.4 on ASDiv-A (Table 3); on SVAMP without questions 30.8.',
        'A constrained model with no word-order information reaches 51.2 on ASDiv-A with RoBERTa.'
    ],
    limits: [
        'Scope is one-unknown English arithmetic problems with at most two operators; variations are manual from a small seed set.',
        'Built by the authors; SVAMP has lower lexical diversity than ASDiv-A.'
    ],
    relevance: 'The dossier drops SVAMP from phase 1: it is at ceiling for our models and 99% of answers came through the fallback extractor. It remains a useful robustness set because its variations specifically defeat shallow heuristics.',
    links: [
        { l: 'ACL Anthology (NAACL 2021)', u: 'https://aclanthology.org/2021.naacl-main.168/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2103.07191' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2103.07191' }
    ],
    read: 'full'
},
{
    id: 'commonsenseqa', title: 'CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge',
    year: 2019, posted: '2018-11', authors: 'Alon Talmor, Jonathan Herzig, Nicholas Lourie, Jonathan Berant',
    venue: 'NAACL 2019', status: 'published', cat: 'methods', origin: 'new', prio: 3, arxiv: '1811.00937',
    tags: ['commonsenseqa', 'multiple choice', 'conceptnet', 'crowdsourcing', 'benchmark'],
    summary: 'A multiple-choice benchmark of 12,247 questions built from ConceptNet relations, where crowdworkers write questions that need background commonsense knowledge to separate a correct answer from hard distractors.',
    method: [
        'Starts from 22 ConceptNet relations (236,208 triplets), groups three triplets sharing a source concept, and has Mechanical Turk workers write three questions each, with five candidates (one ConceptNet distractor and worker-written ones); a second worker group verifies and about 15% are filtered.',
        '80/10/10 split, both random and question-concept splits; models VecSim, LM1B, QABilinear, ESIM, GPT, BERT-large and BiDAF++ (with web snippets); human accuracy from five-worker majority on 100 questions.'
    ],
    results: [
        'Random split test accuracy: BERT-large 55.9%, GPT 45.5%, ESIM+ELMo 34.1%, BiDAF++ 32.0%; human 88.9%.',
        'Question-concept split: BERT-large 63.6%; with random distractors (sanity mode) BERT-large reaches 92.3%, so hard distractors drive difficulty.',
        'Error analysis on 100 dev items: surface clues 77.7%, negation 42.8%, factoid knowledge 38.4%.'
    ],
    limits: [
        'Human and error analyses use 100 questions; ten workers wrote over 85% of the questions; ConceptNet reporting bias; English only.',
        'The authors report ELMo and web snippets gave little benefit.'
    ],
    relevance: 'The multiple-choice control in our datasets. The dossier notes agreement is weaker here (AUROC about 0.72) and that Qwen-0.8B emits 5 tokens on it, so it measures a direct answer, not reasoning; it is kept only as a labelled control.',
    links: [
        { l: 'ACL Anthology (NAACL 2019)', u: 'https://aclanthology.org/N19-1421/', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/1811.00937' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/1811.00937' }
    ],
    read: 'full'
},
{
    id: 'xfinder', title: 'xFinder: Large Language Models as Automated Evaluators for Reliable Evaluation',
    year: 2025, posted: '2024-05', authors: 'Qingchen Yu, Zifan Zheng, Shichao Song, Zhiyu Li, Feiyu Xiong, Bo Tang, Ding Chen',
    venue: 'ICLR 2025', status: 'published', cat: 'methods', origin: 'dossier', dossier: '§3.5', prio: 2, arxiv: '2405.11874',
    tags: ['answer extraction', 'regex', 'evaluation', 'kaf dataset', 'qlora', 'judge'],
    summary: 'LLM evaluation frameworks extract answers with regular expressions, which often fail. xFinder is a fine-tuned small model that extracts and matches the key answer, released with the Key Answer Finder (KAF) dataset.',
    method: [
        'KAF: 26,900 training, 4,961 test and 4,482 generalization samples from 10 LLMs on 8 tasks and 8 LLMs on 11 unseen tasks; labels from GPT-4 with self-consistency plus manual re-annotation.',
        'xFinder built on 19 base LLMs from 0.5B to 8B fine-tuned with QLoRA (learning rate 2e-4, length 2048); baselines are RegEx extraction in OpenCompass, LM Eval Harness and UltraEval, and judge models PandaLM, JudgeLM and GPT-4.'
    ],
    results: [
        'Extraction accuracy on the generalization set: xFinder-llama3-8B 0.9518 and xFinder-qwen1.5-0.5B 0.9342 against OpenCompass 0.7438, LM Eval Harness 0.6780, UltraEval 0.3978 (Table 2).',
        'Judgment accuracy 0.9761 and 0.9748 against 0.8870 for OpenCompass and 0.8842 for GPT-4 judging with CoT; the 0.5B model takes about 10.7 s per 200 samples against 534.7 s for JudgeLM-33B.',
        'Model rankings differ across regex frameworks (for example InternLM2-1.8B-Chat is first in UltraEval but fifth in OpenCompass on CommonsenseQA).'
    ],
    limits: [
        'The authors call it "the first step"; training labels come partly from GPT-4; extraction is still about 93 to 95% accurate.',
        'Generalization set is 4,482 samples over 11 tasks; real-world rankings have no ground truth.'
    ],
    relevance: 'The dossier\'s evidence that regex answer extraction is unreliable (74% for the best framework tested), which motivates a second-stage extraction prompt and a per-cell format-compliance table, since quantization changes answer style (Gemma-1B).',
    links: [
        { l: 'ICLR 2025 poster page', u: 'https://iclr.cc/virtual/2025/poster/30821', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2405.11874' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2405.11874' },
        { l: 'Code', u: 'https://github.com/IAAR-Shanghai/xFinder' }
    ],
    read: 'main'
},
{
    id: 'error-bars', title: 'Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations',
    year: 2024, posted: '2024-11', authors: 'Evan Miller',
    venue: 'arXiv preprint (Anthropic)', status: 'preprint', cat: 'methods', origin: 'dossier', dossier: '§3.5', prio: 1, arxiv: '2411.00640',
    tags: ['statistics', 'clustered standard errors', 'paired differences', 'power analysis', 'confidence intervals', 'evals'],
    summary: 'Language model evals are reported as bare point estimates. The paper gives standard errors, clustered errors, paired model comparisons and power analysis for per-question scored evals, with formulas to plan sample size.',
    method: [
        'CLT-based standard error SE = sqrt(Var(s)/n) and 95% interval s +/- 1.96 SE; clustered SEs where questions share a context (Eq. 4); resampling each question K times and using next-token probabilities to reduce variance.',
        'Paired question-level differences between models (Eq. 7) and paired clustered SEs (Eq. 8); sample-size and minimum-detectable-effect formulas (Eqs. 9 and 10).'
    ],
    results: [
        'The paper states: "We therefore recommend using the paired version of the standard error estimate wherever practicable", and clustered SEs whenever questions are related.',
        'Clustered against naive SE ratios: DROP 3.05, RACE-H 1.10, MGSM 1.88 (Table 4); paired comparison with score correlation 0.5 cuts variance by a third.',
        'About 969 questions are needed to detect a 3-point difference at 80% power and alpha 0.05, so at least 1,000 questions per eval is suggested.'
    ],
    limits: [
        'Assumes a hypothetical super-population and CLT validity; next-token-probability trick does not apply with chain-of-thought; tables 1, 2, 3 and 5 are fictional.',
        'Empirical validation rests on two real evals; small-n behaviour (HumanEval 164 questions) is not examined.'
    ],
    relevance: 'Backs the dossier\'s statistics: paired, question-clustered comparisons across precisions, and its power analysis (V8) that 100 to 200 questions cannot detect effects under 8 points. The dossier recalled these recommendations from memory; reading the paper confirms them.',
    links: [
        { l: 'arXiv', u: 'https://arxiv.org/abs/2411.00640', o: true },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2411.00640' }
    ],
    read: 'full', note: 'No peer-reviewed venue was found; the author lists it as an arXiv paper.'
},
{
    id: 'difference-of-confidences', title: 'Predicting with Confidence on Unseen Distributions',
    year: 2021, posted: '2021-07', authors: 'Devin Guillory, Vaishaal Shankar, Sayna Ebrahimi, Trevor Darrell, Ludwig Schmidt',
    venue: 'ICCV 2021', status: 'published', cat: 'methods', origin: 'dossier', dossier: '§3.5', prio: 1, arxiv: '2107.03315',
    tags: ['difference of confidences', 'doc', 'accuracy estimation', 'distribution shift', 'unlabeled', 'imagenet'],
    summary: 'Distribution distances such as Frechet distance and MMD fail to predict accuracy under shift. Difference of Confidences (DoC), the change in average confidence between the base and target data, predicts the accuracy change without target labels and cuts prediction error by about half.',
    method: [
        'DoC = AC^T_B - AC^B_T, the difference between the average max-softmax confidence on base-distribution points and on the target set (Eq. 6); a linear regression fitted on calibration shifts maps it to the accuracy gap.',
        'ImageNet-trained models (12 architectures) on ImageNet-V2, Vid-Robust, Rendition, Sketch, ImageNet-C synthetic shifts and ImageNet-A; baselines base accuracy, average confidence, Frechet distance, MMD, discriminators, rotation prediction; mean absolute error.'
    ],
    results: [
        'Synthetic-to-natural: base accuracy MAE 0.234, average confidence 0.092, DoC 0.050; synthetic-to-synthetic DoC 0.039.',
        'DoC reduces error over average confidence by more than 45% on natural shifts; on ImageNet-A DoC is 0.388 against 0.476 for average confidence.',
        'A single ResNet-101 with DoC beats a ResNet ensemble baseline.'
    ],
    limits: [
        'The authors say the problem is far from solved; DoC degrades on blur shifts and gains shrink for AugMix-trained models.',
        'One base dataset (ImageNet); confidence-based estimates tend to be optimistic on natural shifts; reported standard deviations are inconsistent between text and table.'
    ],
    relevance: 'Quantization is a shift in the model rather than the data, but the logic is the same: the dossier\'s agreement-shift estimator is DoC with self-consistency as the confidence, and predicts per-sample accuracy change to within 0.85 points on held-out models (dossier 2.10).',
    links: [
        { l: 'ICCV 2021 open access page', u: 'https://openaccess.thecvf.com/content/ICCV2021/html/Guillory_Predicting_With_Confidence_on_Unseen_Distributions_ICCV_2021_paper.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2107.03315' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2107.03315' }
    ],
    read: 'full'
},
{
    id: 'agreement-on-the-line', title: 'Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift',
    year: 2022, posted: '2022-06', authors: 'Christina Baek, Yiding Jiang, Aditi Raghunathan, J. Zico Kolter',
    venue: 'NeurIPS 2022', status: 'published', cat: 'methods', origin: 'dossier', dossier: '§3.5', prio: 2, arxiv: '2206.13089',
    tags: ['agreement', 'accuracy on the line', 'aline', 'distribution shift', 'unlabeled', 'probit'],
    summary: 'Whenever in-distribution against out-of-distribution accuracy is linear, the agreement between pairs of networks is also linear, with matching slope and bias. Agreement needs no labels, so it predicts out-of-distribution accuracy from unlabeled data.',
    method: [
        'Probit-scaled accuracy and agreement; ALine-S fits slope and bias by least squares over model pairs, ALine-D solves a linear system from pairwise OOD agreement and needs at least three models.',
        'Many pretrained models on CIFAR-10.1/10.2, ImageNetV2, CIFAR-10C, fMoW, iWildCam, Camelyon17-wilds and Amazon-SQuAD; baselines ATC, DOC-Feat, average confidence, naive agreement; MAE of OOD accuracy.'
    ],
    results: [
        'MAE (%) ALine-D: CIFAR-10.1 1.11, ImageNetV2 2.06 (ATC 1.14), fMoW 1.30, RxRx1 0.27, Camelyon17 5.47 against ATC 11.93 (Table 2).',
        'Agreement R^2 is high where accuracy R^2 is high (0.997 and 0.999 on CIFAR-10.1) and weak where accuracy is not linear (Camelyon17 0.263 and 0.226).'
    ],
    limits: [
        'Not claimed to be universal and causes are unexplained; needs a set of at least three diverse models.',
        'The slope-match claim fails the significance test for CIFAR-10.2; the "stronger than ProjNorm" claim rests on a very small R^2 difference.'
    ],
    relevance: 'The other label-free accuracy estimator the dossier lists, and the one a quantized model pair can use directly: a 16-bit and a 4-bit copy are two networks whose agreement on unlabeled questions is measurable. It is an unused baseline for our estimator.',
    links: [
        { l: 'NeurIPS 2022 proceedings (PDF)', u: 'https://papers.nips.cc/paper_files/paper/2022/file/7a8d388b7a17df480856dff1cc079b08-Paper-Conference.pdf', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/2206.13089' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/2206.13089' }
    ],
    read: 'main', note: 'The ar5iv copy was used for the card and was truncated in the final appendices.'
},
{
    id: 'guo-calibration', title: 'On Calibration of Modern Neural Networks',
    year: 2017, posted: '2017-06', authors: 'Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger',
    venue: 'ICML 2017', status: 'published', cat: 'methods', origin: 'new', prio: 2, arxiv: '1706.04599',
    tags: ['ece', 'calibration', 'temperature scaling', 'reliability diagram', 'nll'],
    summary: 'Modern deep networks are poorly calibrated. The paper defines Expected Calibration Error with binned reliability, finds depth, width, batch normalization and low weight decay linked to miscalibration, and shows that a single-parameter temperature scaling fixes most of it.',
    method: [
        'ECE with M equal-width bins (M = 15 in Table 1), maximum calibration error, reliability diagrams and NLL; methods histogram binning, isotonic regression, BBQ, Platt, matrix and vector scaling, and temperature scaling softmax(z/T) fit on a validation set.',
        'Vision (Birds, Cars, ImageNet, CIFAR-10/100, SVHN) and text (20 News, Reuters, SST) with ResNet, Wide ResNet, DenseNet, LeNet, Deep Averaging Network and TreeLSTM.'
    ],
    results: [
        'CIFAR-100 ResNet 110: ECE 16.53% uncalibrated to 1.26% with temperature scaling; ImageNet DenseNet 161 6.28% to 1.99%.',
        'Temperature scaling does not change predicted classes, and is the fastest method; vector scaling learns nearly constant entries, so miscalibration is "intrinsically low dimensional".'
    ],
    limits: [
        'The authors cannot claim causality for the factors listed; temperature scaling does not help Reuters; matrix scaling overfits with many classes.',
        'Needs a validation set from the test distribution; ECE depends on the binning choice and one global temperature cannot fix class-specific errors.'
    ],
    relevance: 'The definition behind every ECE and calibration number in the quantization-calibration papers on this page, and the reason ranking metrics such as AUROC (which our agreement analysis uses) are preferred when bins are small.',
    links: [
        { l: 'ICML 2017 proceedings (PMLR 70)', u: 'https://proceedings.mlr.press/v70/guo17a.html', o: true },
        { l: 'arXiv', u: 'https://arxiv.org/abs/1706.04599' },
        { l: 'PDF', u: 'https://arxiv.org/pdf/1706.04599' }
    ],
    read: 'main'
},
{
    id: 'green-swets', title: 'Signal Detection Theory and Psychophysics',
    year: 1966, posted: '1966', authors: 'David M. Green, John A. Swets',
    venue: 'Book: John Wiley and Sons, New York (reprinted by Peninsula Publishing, 1988)', status: 'published', kind: 'book', cat: 'methods', origin: 'dossier', dossier: '§3.5', prio: 3, arxiv: '',
    tags: ['signal detection theory', 'd prime', 'criterion', 'sensitivity', 'bias', 'psychophysics', 'textbook'],
    summary: 'The standard textbook on signal detection theory, which separates an observer\'s sensitivity (d prime) from response bias (criterion). The dossier cites it for the analysis of the verifier as a one-token classifier.',
    method: [
        'Not read in this review. The dossier uses the framework as follows: a verifier\'s approve/reject verdict is treated as a detection response; sensitivity d prime and criterion c are computed from hit and false-alarm rates, with negative c meaning a lenient verifier.'
    ],
    results: [
        'Not applicable (textbook). Dossier results using it: Qwen-0.8B GSM8K d prime 1.13 / 1.11 / 1.17 and criterion c -0.36 / -0.19 / -0.77 at 16 / 8 / 4-bit (dossier 2.8).'
    ],
    limits: [
        'Standard equal-variance Gaussian assumptions; the dossier cites it from memory and flags that the verifier is a one-token classifier thresholded at zero.'
    ],
    relevance: 'Provides the vocabulary for the dossier\'s verifier finding that sensitivity can stay unchanged while the criterion becomes more lenient at 4-bit, which accuracy and AUROC alone would not reveal.',
    links: [
        { l: 'Overview: Detection theory (Wikipedia)', u: 'https://en.wikipedia.org/wiki/Detection_theory' }
    ],
    read: 'record', note: 'Bibliographic details were checked against library catalogue listings (Wiley 1966; later Krieger 1974 and Peninsula 1988 reprints). The dossier cites it from memory; no catalogue link is given here because none was opened.'
},
{
    id: 'macmillan-creelman', title: 'Detection Theory: A User\'s Guide',
    year: 2005, posted: '2005', authors: 'Neil A. Macmillan, C. Douglas Creelman',
    venue: 'Book (2nd edition): Lawrence Erlbaum Associates', status: 'published', kind: 'book', cat: 'methods', origin: 'dossier', dossier: '§3.5', prio: 3, arxiv: '',
    tags: ['detection theory', 'd prime', 'criterion', 'roc', 'textbook', 'signal detection'],
    summary: 'A practitioner-oriented textbook on detection theory, covering sensitivity and bias measures and ROC analysis. The dossier cites it alongside Green and Swets for the verifier analysis.',
    method: [
        'Not read in this review. Used by the dossier as the reference for computing d prime and criterion c from verifier verdicts.'
    ],
    results: [
        'Not applicable (textbook).'
    ],
    limits: [
        'Textbook assumptions (equal-variance Gaussian by default); edition dates differ between catalogues (2004 or 2005 for the 2nd edition), so cite the edition from the book itself.'
    ],
    relevance: 'Companion reference for the sensitivity and criterion analysis of the self-verifier; it supports reporting a criterion shift separately from discrimination when precision changes.',
    links: [
        { l: 'Overview: Detection theory (Wikipedia)', u: 'https://en.wikipedia.org/wiki/Detection_theory' }
    ],
    read: 'record', note: 'The dossier cites the authors as "Macmillan & Creelman, Detection Theory" with no edition or year; the second edition (Lawrence Erlbaum Associates) is assumed. A later edition adds Michael J. Hautus as a third author.'
}
);
