/* Benchmarks and evaluation methodology: datasets we evaluate on, label-free accuracy estimation, calibration, statistics. */
window.PW.papers.push(

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
}
);
