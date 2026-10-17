"""
Verifier designs from the self-verification literature, applied to stored solutions.

Each variant turns a stored (question, solution) into one or more prompts, parses the model's
answers and aggregates them into a score in [0, 1] (a rate of "correct" verdicts, a mean score, ...).
The prompts are written in our own words to follow the structure, scale and output format the papers
describe; they are not copies, and the final-answer marker is ours ("#### <number>", the one our
generator prompt asks for) where a paper uses another phrase.

  mc           Mind the Gap (2412.02674), "multiple choice": no reasoning, answer Correct or Incorrect.
               Scored exactly, from the probability of the two first tokens, not from repeated samples.
  cot_binary   Mind the Gap, CoT-Binary: justification of at most 50 words, then a Correct/Incorrect line;
               the score is the rate of "correct" over repeated samples (the paper uses 64).
  cot_score    Mind the Gap, CoT-Score: a 1-10 score, 1 when the answer is missing or wrong, 10 when answer
               and derivation are right; 50-word justification; the score is the mean over samples.
  critic       Generalisation gap (2606.01075): a critic that goes through the steps first, then a fixed
               closing line; 16 verdicts per solution; positive at a correct-rate >= tau (0.6), negative at an
               incorrect-rate >= tau, otherwise discarded.
  critic_forced  The same, with the start of the answer pre-filled ("Let me check the solution step by step.
               Step 1:") so the model checks the steps before its verdict. Our addition, not in the paper.
  judge5       Self-Rewarding LMs (2401.10020): an additive 5-point rubric with a justification of up to 100
               words, averaged over 3 samples. The rubric is the general one, not written for mathematics.
  backward     Self-verification (2212.09561): every number in the question is masked in turn as X, the
               candidate answer is given as a known fact, and the model must recover X; the score is the
               share of samples that recover the original value.
  self_correct Intrinsic self-correction (2310.01798): the model reviews its own answer and then revises it.
               Reported as a corrector (did the answer change, and was the change right), and the
               unchanged answer is used as a verdict.

Pure functions: nothing here loads a model or reads a file.
"""
import math
import re

from src.experiment_d.answer_matching import answers_match, normalize_answer

VARIANTS = ("mc", "cot_binary", "cot_score", "critic", "critic_forced", "judge5", "backward", "self_correct")
CRITICS = ("critic", "critic_forced")

# Samples per solution (per condition for backward) and the longest generation, as in the papers where
# stated; Mind the Gap uses 64 verdicts for the CoT variants, which is lowered here for cost.
DEFAULTS = {
    "mc": {"n": None, "max_new_tokens": 0},
    "cot_binary": {"n": 16, "max_new_tokens": 160},
    "cot_score": {"n": 16, "max_new_tokens": 160},
    "critic": {"n": 16, "max_new_tokens": 400},
    "critic_forced": {"n": 16, "max_new_tokens": 400},
    "judge5": {"n": 3, "max_new_tokens": 200},
    "backward": {"n": 4, "max_new_tokens": 200},
    "self_correct": {"n": 1, "max_new_tokens": 300},
}
# Mind the Gap draws generations and verdicts at temperature 0.7 and top-p 0.9 (top-k off).
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 0}
CRITIC_THRESHOLD = 0.6

# ---------------------------------------------------------------- prompts

_MC = (
    "Problem:\n{question}\n\nProposed solution:\n{solution}\n\n"
    "Is the proposed solution correct? Answer with either Correct or Incorrect."
)

_COT_BINARY = (
    "You are checking a student's solution to a math problem.\n\n"
    "Problem:\n{question}\n\nStudent's solution:\n{solution}\n\n"
    "Decide whether the solution is correct. The solution must state its final answer on a line of the form "
    "'#### <number>'; if it does not, it is incorrect. Check the derivation and the final answer. "
    "Keep your justification under 50 words, then finish with exactly one line: "
    "'Correctness: correct' or 'Correctness: incorrect'."
)

_COT_SCORE = (
    "You are grading a student's solution to a math problem.\n\n"
    "Problem:\n{question}\n\nStudent's solution:\n{solution}\n\n"
    "Give a score from 1 to 10. Give 1 if the solution does not state its final answer on a line of the form "
    "'#### <number>' or if that answer is wrong. Give 10 if the final answer and the derivation are both correct. "
    "Use the values in between for partly correct work. Keep your justification under 50 words, then finish "
    "with exactly one line: 'Score: <number>'."
)

_CRITIC = (
    "You are a careful critic of math solutions.\n\n"
    "Problem:\n{question}\n\nSolution:\n{solution}\n\n"
    "Go through the problem and then through each step of the solution, checking the reasoning and the "
    "arithmetic. If you find a mistake, explain it briefly. Finish with exactly one line: "
    "'CRITIC RESULT: SOLUTION IS CORRECT' or 'CRITIC RESULT: SOLUTION IS INCORRECT'."
)

_JUDGE5 = (
    "Review the question and the response using the additive 5-point scoring system below. Points are "
    "added as each criterion is met.\n"
    "- Add 1 point if the response is relevant and gives some information related to the question.\n"
    "- Add a second point if it addresses a substantial part of the question, even if incomplete or with errors.\n"
    "- Add a third point if it answers the basic elements of the question in a useful way.\n"
    "- Add a fourth point if it is clear, well organised and helpful, written as a good assistant would.\n"
    "- Add a fifth point if it is impeccable: expert, complete, correct and insightful.\n\n"
    "Question:\n{question}\n\nResponse:\n{solution}\n\n"
    "Briefly justify your total score in up to 100 words, then finish with exactly one line: "
    "'Score: <total points>'."
)

_BACKWARD_EXAMPLES = (
    "Question: A shop sells pens at $3 each. Ana buys X pens and pays $12 in total. How much does Ana pay in total?\n"
    "The answer to this question is 12. What is the value of X?\n"
    "Each pen costs $3, so X pens cost 3 * X dollars. Ana pays 12 dollars, so 3 * X = 12 and X = 12 / 3 = 4.\n"
    "The answer is 4\n\n"
    "Question: A box holds 24 eggs. Ben takes X eggs out and then adds 5 more. How many eggs are in the box now?\n"
    "The answer to this question is 21. What is the value of X?\n"
    "After taking out X eggs the box holds 24 - X, and after adding 5 it holds 24 - X + 5 = 21. "
    "So 29 - X = 21 and X = 29 - 21 = 8.\n"
    "The answer is 8\n\n"
)

REVIEW = "Review your previous answer and find problems with your answer."
IMPROVE = ("Based on the problems you found, improve your answer. Please reiterate your answer, with your final "
           "answer a single numerical number, in the form \\boxed{answer}.")

_TEMPLATES = {"mc": _MC, "cot_binary": _COT_BINARY, "cot_score": _COT_SCORE, "critic": _CRITIC,
              "critic_forced": _CRITIC, "judge5": _JUDGE5}

# critic_forced: the critic's prompt, with the start of its answer written for it, so that it has to go through
# the steps before it can reach the closing line. Left to itself, Qwen3.5-0.8B mostly wrote the verdict line
# alone (median response 5 words on 30 questions), which is not the design the paper describes.
FORCED_PREFIX = {"critic_forced": "Let me check the solution step by step.\n\nStep 1:"}


def answer_prefix(variant: str) -> str:
    """Text placed at the start of the model's answer, after the chat template's generation prompt ('' for none)."""
    return FORCED_PREFIX.get(variant, "")


def verification_messages(variant: str, question: str, solution: str) -> list:
    """The chat messages that ask a model to judge one stored solution (mc, cot_binary, cot_score, critic, judge5)."""
    if variant not in _TEMPLATES:
        raise ValueError(f"{variant!r} has no single-prompt template; known: {sorted(_TEMPLATES)}")
    return [{"role": "user", "content": _TEMPLATES[variant].format(question=question, solution=solution)}]


# ---------------------------------------------------------------- parsing

_BINARY = re.compile(r"correctness\s*:\s*\**\s*<?\s*(correct|incorrect)\b", re.I)
_SCORE = re.compile(r"\bscore\s*:\s*\**\s*<?\s*(\d{1,2})(?:\s*/\s*\d+)?", re.I)
_CRITIC_LINE = re.compile(r"critic\s+result\s*:\s*\**\s*solution\s+is\s+(correct|incorrect)\b", re.I)
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)
_ANSWER_IS = re.compile(r"answer\s+is\s*:?\s*\**\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.I)


def parse_response(variant: str, response) -> object:
    """
    One verdict from one sampled response, the last match winning, None when there is none:
    True / False for cot_binary and critic; a score in [0, 1] for cot_score (1-10) and judge5 (0-5).
    """
    if not response:
        return None
    text = _THINK.sub("", response)
    if variant in ("cot_binary",) + CRITICS:
        found = (_BINARY if variant == "cot_binary" else _CRITIC_LINE).findall(text)
        return found[-1].lower() == "correct" if found else None
    if variant in ("cot_score", "judge5"):
        found = _SCORE.findall(text)
        if not found:
            return None
        points = int(found[-1])
        low, high = (1, 10) if variant == "cot_score" else (0, 5)
        return points / high if low <= points <= high else None
    raise ValueError(f"cannot parse variant {variant!r}")


def aggregate(variant: str, values: list, tau: float = CRITIC_THRESHOLD) -> dict:
    """
    Score of one solution from its sampled verdicts (None marks an unparsed sample, which is left out).
    cot_binary and critic: the rate of "correct"; critic also a decision (True at a rate >= tau, False at
    an incorrect rate >= tau, None in between: discarded). cot_score and judge5: the mean score.
    """
    parsed = [float(v) for v in values if v is not None]
    out = {"n": len(values), "parsed": len(parsed), "score": sum(parsed) / len(parsed) if parsed else None}
    if variant in CRITICS and parsed:
        rate = out["score"]
        out["decision"] = True if rate >= tau - 1e-9 else (False if 1 - rate >= tau - 1e-9 else None)
    return out


def two_way_probability(logp_correct: float, logp_incorrect: float) -> float:
    """P(correct) among the two verdicts, from the log-probabilities of their first tokens."""
    gap = logp_incorrect - logp_correct
    return 1.0 / (1.0 + math.exp(gap)) if gap < 700 else 0.0


# ---------------------------------------------------------------- which stored solutions to judge

def select_traces(samples: list, extractor, skip_cutoff: bool = False, contested_only: bool = False,
                  max_per_answer=None) -> list:
    """
    Indices of the stored solutions to judge, from the model's own answers only (no reference answer).

    skip_cutoff: leave out solutions cut off at the token limit; they never vote, so no verdict on them can
        change the vote.
    contested_only: judge nothing when every finished solution gives the same answer; no verifier can change
        that vote, and such a question has no right-against-wrong pair to rank.
    max_per_answer: judge at most this many solutions (the first in trace order) for each distinct answer;
        an answer's score is then estimated from them.
    """
    keys = []
    for sample in samples:
        answer = None if sample.get("cutoff", False) else extractor(sample["response"])
        keys.append(None if answer is None else normalize_answer(answer))
    if contested_only and len({k for k in keys if k is not None}) < 2:
        return []
    kept, per_answer = [], {}
    for index, (sample, key) in enumerate(zip(samples, keys)):
        if skip_cutoff and sample.get("cutoff", False):
            continue
        if max_per_answer is not None and key is not None:
            if per_answer.get(key, 0) >= max_per_answer:
                continue
            per_answer[key] = per_answer.get(key, 0) + 1
        kept.append(index)
    return kept


# ---------------------------------------------------------------- resuming

RUN_SETTINGS = ("variant", "question_ids", "verdicts", "max_new_tokens", "temperature", "top_p", "top_k", "tau",
                "max_conditions", "skip_cutoff", "contested_only", "max_per_answer", "answer_prefix", "dtype")


def settings_mismatch(header: dict, settings: dict) -> list:
    """The run settings in which an existing output file differs from a new run (a missing one differs too)."""
    return [key for key in RUN_SETTINGS if key in settings and header.get(key, "<missing>") != settings[key]]


# ---------------------------------------------------------------- batching

def generate_in_groups(prompts: list, sample, verdicts: int, batch_size: int, out_of_memory=(MemoryError,),
                       on_out_of_memory=lambda: None, log=print) -> tuple:
    """
    (responses per prompt in the order given, prompts per group that worked) for `verdicts` samples each.

    Prompts are sorted by length and sent in groups of batch_size // verdicts, so a group is padded to
    its own longest prompt, not to the longest of the whole call. `sample(group prompts, sequences)` does
    the generation. When it raises one of `out_of_memory`, the group size is halved and the group retried,
    down to one prompt; the returned size lets a caller keep the smaller batch for its next call.
    """
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    per_group = max(1, batch_size // verdicts)
    results = [None] * len(prompts)
    start = 0
    while start < len(order):
        group = order[start:start + per_group]
        try:
            grouped = sample([prompts[i] for i in group], len(group) * verdicts)
        except out_of_memory:
            if per_group == 1:
                raise
            per_group = max(1, per_group // 2)
            on_out_of_memory()
            log(f"    out of memory: {per_group} prompts x {verdicts} = {per_group * verdicts} sequences per batch from now on")
            continue
        for index, responses in zip(group, grouped):
            results[index] = responses
        start += len(group)
    return results, per_group


# ---------------------------------------------------------------- backward verification (Self-verification)

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def mask_conditions(question: str, max_conditions: int = 6) -> list:
    """
    [(value, masked question)]: every distinct number in the question masked in turn as X, all its
    occurrences at once (masking one of two equal numbers would leave the value visible).
    """
    out, seen = [], set()
    for match in _NUMBER.finditer(question):
        token = match.group(0)
        key = normalize_answer(token)
        if key in seen:
            continue
        seen.add(key)
        masked = re.sub(r"(?<![\w.])" + re.escape(token) + r"(?![\d])", "X", question)
        if masked != question:
            out.append((token.replace(",", ""), masked))
        if len(out) >= max_conditions:
            break
    return out


def backward_messages(masked_question: str, answer: str) -> list:
    """Two worked examples, then the masked question with the candidate answer given as a known fact."""
    content = (
        "Solve for the unknown X. Reason step by step and finish with 'The answer is <value>'.\n\n"
        + _BACKWARD_EXAMPLES
        + f"Question: {masked_question}\nThe answer to this question is {answer}. What is the value of X?\n"
    )
    return [{"role": "user", "content": content}]


def parse_backward(response, value: str) -> object:
    """True when the last 'answer is <number>' equals the masked value, False when it is another number, None without one."""
    if not response:
        return None
    found = _ANSWER_IS.findall(_THINK.sub("", response))
    return answers_match(found[-1].replace(",", ""), value) if found else None


def backward_score(per_condition_values: list) -> dict:
    """Share of samples, over all conditions, that recovered the masked value (unparsed samples count as misses)."""
    flat = [v for values in per_condition_values for v in values]
    matched = sum(1 for v in flat if v is True)
    return {"conditions": len(per_condition_values), "samples": len(flat), "matches": matched,
            "score": matched / len(flat) if flat else None}


# ---------------------------------------------------------------- intrinsic self-correction

def self_correct_messages(base_messages: list, solution: str, feedback=None) -> list:
    """
    The conversation of the first step (review) or, once `feedback` is given, of the second (improve):
    the original prompt, the stored solution as the assistant's answer, the review request, and the
    model's own feedback with the request to improve.
    """
    messages = list(base_messages) + [{"role": "assistant", "content": solution}, {"role": "user", "content": REVIEW}]
    if feedback is not None:
        messages += [{"role": "assistant", "content": feedback}, {"role": "user", "content": IMPROVE}]
    return messages
