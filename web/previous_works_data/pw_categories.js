/* Structure of the Previous Works page.
   The project asks one question: does weight quantization break a small model's ability to improve itself
   (self-consistency, self-verification, self-training)? Everything below is organised around that.
   - TRACKS: which side of the question a paper is on (the intersection, the quantization side, the self-improvement side, measurement).
   - STAGES: where in a self-improvement loop quantization could do damage (generate, signal, judge, update).
   - CATEGORIES: the research areas, each belonging to one track. */
window.PW_TRACKS = [
    {
        id: 'both',
        short: 'Both sides',
        title: 'Quantization × self-improvement',
        desc: 'Papers that put a quantized or low-precision model inside a voting, self-checking, self-training or reinforcement-learning loop. This is the literature Experiment D adds to.'
    },
    {
        id: 'quant',
        short: 'Quantization side',
        title: 'Quantization: what low-bit weights do to the abilities and signals a self-improving model relies on',
        desc: 'Evidence on reasoning accuracy, answer length, confidence, calibration and answer stability under compression, plus the methods and scaling laws that define our 16, 8 and 4-bit settings. Most of it scores single answers against ground truth, not agreement among a model\'s own samples.'
    },
    {
        id: 'self',
        short: 'Self-improvement side',
        title: 'Self-improvement: what voting, self-checking and self-training need from the model',
        desc: 'Self-consistency, self-verification, vote-filtered self-training and their failure modes. Nearly all of it is at full precision, so how compression changes these signals is left open.'
    },
    {
        id: 'tool',
        short: 'Measurement',
        title: 'Benchmarks and measurement',
        desc: 'The datasets we evaluate on and the statistical tools used to compare precisions fairly.'
    }
];

/* Stages of a self-improvement loop, and the question Experiment D asks at each one. */
window.PW_STAGES = [
    {
        id: 'generate', n: 1, short: 'Generate',
        title: 'Generate: does the quantized model still sample good reasoning?',
        ask: 'Does the pool of 8 sampled solutions per question keep its coverage (at least one correct sample) when the weights are 8-bit or 4-bit?'
    },
    {
        id: 'signal', n: 2, short: 'Signal',
        title: 'Signal: does agreement or confidence still say which sample is right?',
        ask: 'Does the share of samples agreeing with the majority still separate right answers from wrong ones, and do answers flip more than sampling noise explains?'
    },
    {
        id: 'judge', n: 3, short: 'Judge',
        title: 'Judge: can the model still check its own answer?',
        ask: 'When the same model is asked whether its own answer is correct, do its verdicts change in sensitivity or in criterion with precision?'
    },
    {
        id: 'update', n: 4, short: 'Update',
        title: 'Update: can a loop still learn from the filtered samples?',
        ask: 'If vote-filtered samples become training data, how many labels survive, how many are correct, and are weight updates still larger than the quantization step?'
    },
    {
        id: 'setup', n: 0, short: 'Precision settings',
        title: 'Precision settings and scaling laws',
        ask: 'How the 16, 8 and 4-bit points are made, and what the scaling laws predict about which models should be fragile.'
    },
    {
        id: 'measure', n: 0, short: 'Benchmarks & measurement',
        title: 'Benchmarks and measurement',
        ask: 'The datasets, extraction and statistics used to compare precisions fairly.'
    }
];

/* Research categories in display order: the intersection first, then each side. */
window.PW_CATEGORIES = [
    {
        id: 'intersection',
        track: 'both',
        short: 'Quantization × self-improvement',
        title: 'Where quantization meets voting, verification and self-improvement',
        desc: 'The closest prior work to Experiment D: a quantized model inside a vote, a self-check, a self-training loop or reinforcement learning. In this collection none varies precision as the controlled factor for the agreement among a model\'s own samples or for its verdict on its own reasoning.'
    },
    {
        id: 'quant-reasoning',
        track: 'quant',
        short: 'Quantized reasoning models',
        title: 'Quantization and reasoning models',
        desc: 'How low-bit weights change multi-step reasoning accuracy, answer length and error types, and how to repair the damage. This sets the quality of the samples a self-improving model starts from.'
    },
    {
        id: 'quant-reliability',
        track: 'quant',
        short: 'Quantization & reliability',
        title: 'Quantization, calibration and reliability',
        desc: 'What compression does to confidence, uncertainty, answer flips, trust and alignment, beyond headline accuracy. These are the signals a self-improvement loop reads.'
    },
    {
        id: 'quant-scaling',
        track: 'quant',
        short: 'Quantization scaling & evaluation',
        title: 'Quantization: scaling laws and capability evaluation',
        desc: 'Large evaluations and scaling laws that say which precision to use for which model size and task, and which small models should be fragile.'
    },
    {
        id: 'quant-methods',
        track: 'quant',
        short: 'Quantization methods',
        title: 'Quantization methods and tooling',
        desc: 'The algorithms and libraries behind the 8-bit, 4-bit, AWQ and GPTQ settings used or planned in the experiment.'
    },
    {
        id: 'consistency',
        track: 'self',
        short: 'Self-consistency & confidence',
        title: 'Self-consistency, confidence and agreement signals',
        desc: 'Majority voting, answer agreement and confidence elicitation: the signals a model uses to choose its own best answer, and when agreement can or cannot be trusted. All at full precision.'
    },
    {
        id: 'correction',
        track: 'self',
        short: 'Self-correction & verification',
        title: 'Self-correction and self-verification',
        desc: 'Whether a model can check, critique or fix its own answers without outside help. This is the judging step of the loop.'
    },
    {
        id: 'selftrain',
        track: 'self',
        short: 'Self-training',
        title: 'Self-improvement and self-training',
        desc: 'Training a model on its own filtered or voted outputs, the theory of why it works, and how it fails. This is the update step of the loop.'
    },
    {
        id: 'verifiers',
        track: 'self',
        short: 'Verifiers & test-time compute',
        title: 'Verifiers, reward models and test-time compute',
        desc: 'Trained checkers and search that decide which sampled solution to keep, and how to spend inference compute. The comparison point for a model that judges itself.'
    },
    {
        id: 'methods',
        track: 'tool',
        short: 'Benchmarks & methodology',
        title: 'Benchmarks and evaluation methodology',
        desc: 'The datasets we evaluate on and the statistical and measurement tools the dossier relies on.'
    }
];
window.PW_PAPERS = window.PW_PAPERS || [];
