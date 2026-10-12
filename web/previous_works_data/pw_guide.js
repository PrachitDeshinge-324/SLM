/* Reading guide for the Previous Works page.
   THEMES   six plain-language groups, each with what to remember and what is still open
   POSITION how the planned paper sits against prior work (from RESEARCH_DOSSIER.md sections 4.1, 4.3, 4.4)
   NOTES    per paper: take (the one idea to remember), role (how we use it), start (read first) */

window.PW_THEMES = [
    {
        id: 'closest', n: 1,
        title: 'Closest to our paper',
        question: 'Has anyone already combined quantization with voting, self-checking or self-training?',
        cats: ['intersection'],
        know: [
            'Voting on a quantized model has been studied only as a memory trade-off: is a vote worth its memory? (Not All Bits Are Equal)',
            'Self-training on a 4-bit model works (MACA), and reinforcement learning with a quantized model works with fixes (QuRL, QeRL).',
            'Quantization moderately weakens a model\'s explanations of itself (Can LLMs Still Explain Themselves?).'
        ],
        open: 'None of them compares precisions to ask whether a model\'s agreement with itself is still a trustworthy label. That is our paper.'
    },
    {
        id: 'accuracy', n: 2,
        title: 'Quantization and accuracy',
        question: 'How much reasoning ability does a model lose when its weights are rounded?',
        cats: ['quant-reasoning', 'quant-scaling'],
        know: [
            '8-bit is close to free. 4-bit costs little on easy tasks and more on hard ones. Below 4-bit is risky.',
            'Small, well-trained models lose the most, and the loss differs a lot from model to model.',
            'Quantized reasoning models write longer answers, so a token limit can look like lost accuracy.'
        ],
        open: 'This is settled, so for us the accuracy drop is a replication, not a contribution. These papers score one answer per question, never a set of samples.'
    },
    {
        id: 'confidence', n: 3,
        title: 'Quantization and confidence',
        question: 'Beyond accuracy, does a quantized model still know when it is right?',
        cats: ['quant-reliability'],
        know: [
            'Calibration gets somewhat worse, and confidence in the right answer drops.',
            'Two models with the same accuracy can disagree on many individual answers (flips).',
            'Conclusions depend on how confidence is scored, so the protocol must be fixed in advance.'
        ],
        open: 'All of this is confidence in a single answer. Nobody measures agreement across several sampled answers, and flip counts are never compared with plain sampling noise.'
    },
    {
        id: 'trust', n: 4,
        title: 'Trusting your own answers',
        question: 'Can a model tell which of its own answers are right, by voting or by checking?',
        cats: ['consistency', 'correction', 'verifiers'],
        know: [
            'Agreement among samples is a good sign of correctness on arithmetic word problems and a poor one on hard multiple choice.',
            'Small models are weak judges of their own answers; below about 2B parameters they are near chance.',
            'Without outside information, self-correction usually does not help.'
        ],
        open: 'Almost everything here is at full precision. Whether rounding the weights changes the vote or the self-check is untested.'
    },
    {
        id: 'training', n: 5,
        title: 'Learning from your own answers',
        question: 'Can a model improve by training on answers it produced and selected itself?',
        cats: ['selftrain'],
        know: [
            'Training on majority-vote answers works without any labels (LMSI, ScPO, TTRL).',
            'The loop can collapse, lose diversity or forget old skills.',
            'Coverage, whether any sample is correct, sets the ceiling on what a loop can gain.'
        ],
        open: 'No study runs the same loop at two precisions and compares. This is our phase 2.'
    },
    {
        id: 'tools', n: 6,
        title: 'Tools we use',
        question: 'Which quantizers, datasets and statistics does the experiment rest on?',
        cats: ['quant-methods', 'methods'],
        know: [
            'Our 8-bit setting is LLM.int8() and our 4-bit setting is NF4 from QLoRA, both through bitsandbytes. AWQ and GPTQ are the planned second quantizers.',
            'GSM8K is the main dataset. Comparisons are paired on the same questions, with error bars.'
        ],
        open: ''
    }
];

/* How the planned paper sits against prior work. "ours" is provisional: it comes from the stored runs in the dossier. */
window.PW_POSITION = {
    add: [
        { topic: 'Is agreement still trustworthy?',
          prior: 'Self-training papers use the vote as a label but never vary precision.',
          papers: ['lmsi', 'maca'],
          ours: 'At 4-bit, confident answers are just as accurate, but there are fewer of them.' },
        { topic: 'Does voting hide damage?',
          prior: 'Voting on quantized models is judged only by accuracy per unit of memory.',
          papers: ['not-all-bits'],
          ours: 'The voted answer looks about 1.4 points better than single attempts, so it under-reports the loss.' },
        { topic: 'Are answer flips real?',
          prior: 'Flips are counted with deterministic decoding, with no sampling-noise baseline.',
          papers: ['accuracy-not-all', 'illusion-equivalency'],
          ours: 'At 8-bit, flips equal the re-run noise floor. At 4-bit, the extra flips go one way.' },
        { topic: 'Can we estimate the loss without labels?',
          prior: 'Label-free accuracy estimators exist for shifts in the data, not in the weights.',
          papers: ['difference-of-confidences', 'agreement-on-the-line'],
          ours: 'The drop in agreement predicts the drop in single-attempt accuracy on held-out models.' },
        { topic: 'Does the model still judge itself the same way?',
          prior: 'Self-verification studies load models in 4-bit without treating it as a factor.',
          papers: ['trust-itself'],
          ours: 'Suggestive only: the verifier may get more lenient without getting less sensitive.' },
        { topic: 'Is the loss real or a change of answer style?',
          prior: 'Quantized models are reported to follow instructions less well.',
          papers: ['lee-tradeoffs'],
          ours: 'For one model, most of the 4-bit drop is a change in answer format, not lost reasoning.' }
    ],
    known: [
        { text: '4-bit lowers the accuracy of small reasoning models; 3-bit is risky.', papers: ['quant-hurts-reasoning', 'not-all-bits'] },
        { text: 'A 4-bit model can self-improve from majority votes.', papers: ['maca'] },
        { text: 'Vote-based self-training can collapse, and small verifiers are near chance.', papers: ['lrm-self-train', 'generalization-gap'] },
        { text: 'Agreement is reliable on arithmetic and poor on hard multiple choice.', papers: ['lmsi', 'sc-backfires', 'llms-agree'] }
    ]
};

/* role: one of the labels below. take: the single idea to remember, in plain words. */
window.PW_ROLES = ['Closest prior work', 'Known result', 'Method we build on', 'Phase 2 design', 'Caution for our design', 'Tool we use', 'Background'];

window.PW_NOTES = {
    /* closest */
    'not-all-bits': { start: true, role: 'Closest prior work', take: 'With a fixed memory budget, a small model should spend it on weights rather than extra votes; voting pays only above roughly an 8-bit 4B model.' },
    'maca': { short: 'MACA', start: true, role: 'Closest prior work', take: 'A model can improve by training on the majority answer of its own debating copies. This was done on a 4-bit base, but precision was never compared.' },
    'self-explanations-quant': { start: true, role: 'Closest prior work', take: 'Quantization moderately lowers the quality and faithfulness of a model\'s explanations of itself.' },
    'qurl': { role: 'Closest prior work', take: 'Generating RL rollouts with a quantized copy saves time, but small weight updates vanish below the quantization step unless you correct for it.' },
    'qerl': { role: 'Closest prior work', take: 'Quantization noise can help RL exploration instead of hurting it.' },
    'bitcal-tts': { role: 'Caution for our design', take: '4-bit models may stop reasoning too early because their confidence signals are distorted. The evidence is thin: 35 to 54 questions.' },
    'multipass-verification': { role: 'Background', take: 'Asking a low-bit model to re-check its own output recovers much of the lost quality on a text-coding task.' },
    'llm-qat': { role: 'Background', take: 'A model\'s own generated text is enough data to train a quantized version of it.' },
    'reasoning-qat': { role: 'Background', take: 'At 2 to 3 bits, reasoning models need quantization-aware training: a PTQ start, distillation, then RL.' },

    /* accuracy */
    'quant-hurts-reasoning': { short: 'Quantization Hurts Reasoning?', start: true, role: 'Known result', take: 'The reference study for reasoning models: 4-bit costs little on easy tasks and more on hard ones, 3-bit is risky, and small models lose more.' },
    'quant-meets-reasoning-v1': { role: 'Known result', take: 'Low-bit Llama-3 loses math accuracy, and a little fine-tuning on its typical errors repairs much of it.' },
    'quant-meets-reasoning-v2': { role: 'Background', take: 'Find the first step where a quantized model\'s reasoning goes wrong, then fine-tune on that step to restore it.' },
    'quantlrm': { role: 'Background', take: 'Use what changed during reasoning fine-tuning to decide which weights to protect when quantizing.' },
    'quantized-think-longer': { role: 'Caution for our design', take: 'Quantized reasoning models write longer chains without needing to; often they had the right answer and failed to commit to it.' },
    'quant-inflates-reasoning': { role: 'Caution for our design', take: 'Accuracy can survive quantization while answers get longer, which cancels the speed gain.' },
    'kbit-scaling': { role: 'Known result', take: 'For a fixed memory budget, 4-bit is almost always the best precision.' },
    'precision-scaling': { role: 'Background', take: 'Quantization damage grows with how much data a model was trained on.' },
    'undertrained': { role: 'Background', take: 'Small, well-trained models are hurt most by low-bit quantization.' },
    'evaluating-quantized': { role: 'Known result', take: 'A broad benchmark of quantized LLMs, with bit-width advice by model size and task.' },
    'emergent-quantized': { role: 'Known result', take: '4-bit keeps chain-of-thought and in-context learning in LLaMA; 2-bit breaks them.' },
    'compressing-truth': { role: 'Caution for our design', take: 'Perplexity hides real capability loss in compressed models.' },
    'lee-tradeoffs': { short: 'Lee et al.', role: 'Known result', take: 'Four quantization methods compared from 1B to 405B on 13 datasets; 1B models lose GSM8K sharply at 4-bit.' },
    'bf16-or-death': { role: 'Known result', take: 'Half a million evaluations: 4-bit weight-only keeps over 99% of the average benchmark score.' },
    'deepseek-quant-drop': { role: 'Background', take: 'Even the 671B DeepSeek-R1 runs at about 3 bits with little loss.' },

    /* confidence */
    'accuracy-not-all': { start: true, role: 'Method we build on', take: 'Two models with the same accuracy can disagree on many individual answers, so count the flips.' },
    'illusion-equivalency': { role: 'Method we build on', take: 'Same accuracy does not mean same behaviour: check whether both models are right on the same questions.' },
    'reliability-scaling': { role: 'Known result', take: 'Reliability (uncertainty, calibration, robustness) peaks near 4 bits, while accuracy keeps rising with bits.' },
    'quant-affects-confidence': { role: 'Known result', take: '4-bit lowers confidence in the right answer, mostly where the model was already unsure.' },
    'quantized-calibrated': { role: 'Known result', take: 'Quantized models are less well calibrated, and tuning a few soft-prompt tokens restores it.' },
    'interpreting-quant': { role: 'Known result', take: 'At 4 and 8 bits, our exact settings, confidence and calibration change only a little.' },
    'decoding-compressed-trust': { role: 'Known result', take: '4-bit keeps a model\'s trustworthiness; 3-bit does not.' },
    'compression-uncertainty': { role: 'Background', take: 'A benchmark of whether compressed models still know when they are unsure.' },
    'calibration-scoring-rule': { role: 'Caution for our design', take: 'Whether a quantized model looks calibrated can depend only on how you score it.' },
    'quant-undoes-alignment': { role: 'Background', take: '3-bit quantization creates new biased answers while perplexity barely moves.' },
    'bigger-or-cheaper': { role: 'Background', take: 'In vision-language models, 4-bit costs little accuracy but much of the usable confidence signal.' },

    /* trust */
    'self-consistency': { start: true, role: 'Method we build on', take: 'Sample several reasoning chains and take the most common answer.' },
    'trust-itself': { short: 'When Should an LM Trust Itself?', start: true, role: 'Method we build on', take: 'A model\'s own True/False check on its answer helps on some benchmarks and fails on others.' },
    'sc-backfires': { start: true, role: 'Caution for our design', take: 'On hard science questions, majority voting makes small models worse on most problems.' },
    'cot-prompting': { role: 'Background', take: 'Showing worked steps in the prompt lets large models reason step by step.' },
    'kadavath-know': { role: 'Method we build on', take: 'Large models are fairly good at saying whether their own answer is true: the P(True) probe.' },
    'just-ask-calibration': { role: 'Background', take: 'A model\'s stated confidence in words can be better calibrated than its token probabilities.' },
    'cisc': { role: 'Background', take: 'Weight each vote by the model\'s own confidence and you need about 40% fewer samples.' },
    'unsupervised-confidence': { role: 'Background', take: 'Agreement across samples can serve as a free label for training a confidence predictor.' },
    'llms-agree': { short: 'When LLMs Agree', role: 'Caution for our design', take: 'Agreement is a weak and uneven sign of correctness.' },
    'wrong-consensus': { role: 'Caution for our design', take: 'Samples often agree on a wrong answer because the model has a fixed preference for it.' },
    'cross-model-disagreement': { role: 'Background', take: 'A second model\'s surprise at an answer catches confident mistakes without labels.' },
    'self-correct-yet': { role: 'Caution for our design', take: 'Without the right answer to check against, self-correction usually makes answers worse.' },
    'self-refine': { role: 'Background', take: 'Generate, critique, refine: large gains on many tasks, little on math.' },
    'correct-own-mistakes': { role: 'Background', take: 'Survey: no solid evidence that prompting alone lets a model fix its own mistakes.' },
    'small-lms-strong-verifiers': { role: 'Caution for our design', take: 'Small models can fix their reasoning only when a strong verifier tells them when to.' },
    'self-verification-weng': { role: 'Background', take: 'Check an answer by plugging it back in and re-deriving a hidden part of the question.' },
    'more-yap': { role: 'Caution for our design', take: 'Even when shown the correct answer, small models barely improve (4.4%), and longer explanations do worse.' },
    'gsm8k-verifiers': { role: 'Tool we use', take: 'Introduces GSM8K and the sample-then-verify pattern.' },
    'lets-verify': { role: 'Background', take: 'Checking each reasoning step beats checking only the final answer.' },
    'scaling-test-time': { role: 'Background', take: 'How best to spend extra inference compute depends on how hard the question is.' },

    /* training */
    'lmsi': { short: 'LMSI', start: true, role: 'Phase 2 design', take: 'Train on the reasoning behind the majority answer. No labels needed.' },
    'lrm-self-train': { short: 'Can LRMs Self-Train?', start: true, role: 'Caution for our design', take: 'A majority-vote reward works at first, then the model games it and collapses.' },
    'generalization-gap': { short: 'Generalization Gap', start: true, role: 'Caution for our design', take: 'Self-evolution is capped by coverage, and a small verifier is the bottleneck.' },
    'star': { role: 'Phase 2 design', take: 'Keep the rationales that reached the right answer, fine-tune on them, repeat.' },
    'rest-em': { role: 'Phase 2 design', take: 'Sample, keep the correct ones, fine-tune from the base model, repeat.' },
    'scpo': { role: 'Phase 2 design', take: 'Use self-consistency as a training signal: prefer the most consistent answer over the least.' },
    'ttrl': { role: 'Phase 2 design', take: 'Reward answers that match the majority vote; it works on unlabeled test questions.' },
    'sharpening': { role: 'Background', take: 'Self-improvement sharpens a model toward its own best answers, and is limited by coverage.' },
    'self-improvement-reversal': { role: 'Caution for our design', take: 'pass@1 goes up while diversity and out-of-distribution accuracy go down.' },
    'rise-and-collapse': { role: 'Caution for our design', take: 'Self-training can rise fast and then collapse; stop at the peak.' },
    'self-evolving-forget': { role: 'Caution for our design', take: 'Self-improving models forget skills they had.' },
    'rstar-math': { role: 'Background', take: 'Small models can self-improve a lot on math, but only with tree search and a trained step scorer.' },
    'v-star': { role: 'Background', take: 'Do not throw away wrong samples: use them to train a verifier.' },
    'self-rewarding': { role: 'Background', take: 'One model both answers and judges, and the judge improves with each round.' },

    /* tools */
    'llm-int8': { role: 'Tool we use', take: '8-bit without loss, by keeping a few outlier dimensions in 16-bit. This is our 8-bit setting.' },
    'qlora': { role: 'Tool we use', take: 'Fine-tune small adapters on a frozen 4-bit model. Its NF4 format is our 4-bit setting.' },
    'gptq': { role: 'Tool we use', take: 'Quantize layer by layer, correcting the error with second-order information.' },
    'awq': { role: 'Tool we use', take: 'Protect the roughly 1% of weights that matter most, judged by activations.' },
    'smoothquant': { role: 'Background', take: 'Move the difficulty from activations to weights so both fit in 8 bits.' },
    'error-bars': { role: 'Tool we use', take: 'Report standard errors, compare models on the same questions, and plan the sample size.' },
    'difference-of-confidences': { short: 'Difference of Confidences', role: 'Method we build on', take: 'A drop in average confidence predicts a drop in accuracy, without labels.' },
    'agreement-on-the-line': { role: 'Method we build on', take: 'Agreement between two models predicts accuracy on unlabeled data.' },
    'xfinder': { role: 'Caution for our design', take: 'Regex answer extraction often fails; a small trained extractor does better.' },
    'guo-calibration': { role: 'Tool we use', take: 'Defines Expected Calibration Error and temperature scaling.' },
    'green-swets': { role: 'Tool we use', take: 'Separates how well a judge discriminates from how lenient it is.' },
    'macmillan-creelman': { role: 'Tool we use', take: 'A practical guide to sensitivity and bias measures.' },
    'math-dataset': { role: 'Tool we use', take: '12,500 competition math problems with worked solutions.' },
    'svamp': { role: 'Tool we use', take: '1,000 word problems varied to defeat shortcuts.' },
    'commonsenseqa': { role: 'Tool we use', take: '12,247 multiple-choice commonsense questions.' }
};
