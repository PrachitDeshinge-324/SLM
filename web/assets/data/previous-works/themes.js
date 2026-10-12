/* Previous Works page: editorial structure.
   themes     the six groups the page is organised into (each lists the fine-grained categories it holds)
   categories fine-grained category titles, shown in each card's detail line
   roles      how we use a paper (a fixed vocabulary)
   position   the "where our paper fits" table and the list of already-known results
   Papers are in papers-*.js; the one-line take and the two-sided notes for each paper are in notes.js. */
window.PW = {
    papers: [],

    themes: [
        {
            id: 'closest', n: 1,
            title: 'Closest to our paper',
            question: 'Has anyone already combined quantization with voting, self-checking or self-training?',
            cats: ['intersection'],
            know: [
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
    ],

    categories: {
        'intersection': 'Where quantization meets voting, verification and self-improvement',
        'quant-reasoning': 'Quantization and reasoning models',
        'quant-reliability': 'Quantization, calibration and reliability',
        'quant-scaling': 'Quantization: scaling laws and capability evaluation',
        'quant-methods': 'Quantization methods and tooling',
        'consistency': 'Self-consistency, confidence and agreement signals',
        'correction': 'Self-correction and self-verification',
        'selftrain': 'Self-improvement and self-training',
        'verifiers': 'Verifiers, reward models and test-time compute',
        'methods': 'Benchmarks and evaluation methodology'
    },

    roles: ['Closest prior work', 'Known result', 'Method we build on', 'Phase 2 design', 'Caution for our design', 'Tool we use', 'Background'],

    position: {
        add: [
            { topic: 'Is agreement still trustworthy?',
              prior: 'Self-training papers use the vote as a label but never vary precision.',
              papers: ['lmsi', 'maca'],
              ours: 'At 4-bit, confident answers are just as accurate, but there are fewer of them.' },
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
            { text: '4-bit lowers the accuracy of small reasoning models; 3-bit is risky.', papers: ['quant-hurts-reasoning'] },
            { text: 'A 4-bit model can self-improve from majority votes.', papers: ['maca'] },
            { text: 'Vote-based self-training can collapse, and small verifiers are near chance.', papers: ['lrm-self-train', 'generalization-gap'] },
            { text: 'Agreement is reliable on arithmetic and poor on hard multiple choice.', papers: ['lmsi', 'sc-backfires', 'llms-agree'] }
        ]
    }
};
