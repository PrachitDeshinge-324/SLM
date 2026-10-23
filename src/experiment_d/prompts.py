"""
Prompt templates for Experiment D.

Uses message-dict format so we can call tokenizer.apply_chat_template()
and automatically get the correct special tokens for any model family
(Llama, Qwen, Mistral, etc.).
"""

# ─── GSM8K ────────────────────────────────────────────────────────────────────

GSM8K_SYSTEM = (
    "You are a helpful math tutor. Solve the problem step by step "
    "and end with the final answer in the format: #### <number>."
)

GSM8K_1SHOT_USER = (
    "Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether "
    "in April and May?"
)

GSM8K_1SHOT_ASSISTANT = (
    "Natalia sold 48 clips in April.\n"
    "In May, she sold half as many clips, which is 48 / 2 = 24 clips.\n"
    "Altogether, she sold 48 + 24 = 72 clips.\n"
    "#### 72"
)


def get_gsm8k_messages(question: str) -> list[dict]:
    """Returns the message list for a 1-shot GSM8K prompt."""
    return [
        {"role": "system", "content": GSM8K_SYSTEM},
        {"role": "user", "content": GSM8K_1SHOT_USER},
        {"role": "assistant", "content": GSM8K_1SHOT_ASSISTANT},
        {"role": "user", "content": question},
    ]


# ─── GSM8K prompt variants (multi-prompt test) ────────────────────────────────
# Each variant changes how the model is asked to work, never the answer format (#### <number>), so the same
# extractor scores every variant. 'default' is the prompt above, unchanged, so its runs stay comparable with
# every earlier run. 'example2' uses GSM8K train question 3000 as its worked example: outside the first 500
# train questions the self-training pools use; a pool that reaches it must leave it out, as it does Natalia's.

GSM8K_EXAMPLE2_USER = "A pole is 20 meters long. It got cut in a way that left it 30% shorter. How long is the pole?"

GSM8K_PROMPT_VARIANTS = {
    "default": (GSM8K_SYSTEM, GSM8K_1SHOT_USER, GSM8K_1SHOT_ASSISTANT),
    "plan": (
        "You are a helpful math tutor. First list the quantities the problem gives and what it asks for. "
        "Then solve the problem step by step and end with the final answer in the format: #### <number>.",
        GSM8K_1SHOT_USER,
        "Given: 48 clips sold in April; in May, half as many as in April.\n"
        "Asked: the total number of clips sold in April and May.\n"
        "In May, she sold 48 / 2 = 24 clips.\n"
        "Altogether, she sold 48 + 24 = 72 clips.\n"
        "#### 72",
    ),
    "check": (
        "You are a helpful math tutor. Solve the problem step by step, then check your result by working "
        "backwards from it, and end with the final answer in the format: #### <number>.",
        GSM8K_1SHOT_USER,
        "Natalia sold 48 clips in April.\n"
        "In May, she sold half as many clips, which is 48 / 2 = 24 clips.\n"
        "Altogether, she sold 48 + 24 = 72 clips.\n"
        "Check: 72 - 48 = 24 clips in May, and 24 is half of 48. The result is consistent.\n"
        "#### 72",
    ),
    "example2": (
        GSM8K_SYSTEM,
        GSM8K_EXAMPLE2_USER,
        "The part that was cut away is 30/100 * 20 = 6 meters long.\n"
        "So the pole is now 20 - 6 = 14 meters long.\n"
        "#### 14",
    ),
}


def get_gsm8k_variant_messages(question: str, variant: str = "default") -> list[dict]:
    """The 1-shot GSM8K prompt in one of GSM8K_PROMPT_VARIANTS ('default' is get_gsm8k_messages)."""
    if variant not in GSM8K_PROMPT_VARIANTS:
        raise ValueError(f"Unknown GSM8K prompt variant {variant!r}; known: {sorted(GSM8K_PROMPT_VARIANTS)}")
    system, example_user, example_assistant = GSM8K_PROMPT_VARIANTS[variant]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": question},
    ]


# ─── CommonsenseQA ────────────────────────────────────────────────────────────

CQA_SYSTEM = (
    "You are a logical reasoning assistant. Read the question and the 5 choices. "
    "Provide concise step-by-step reasoning (2-3 sentences), and output the correct option letter at the very end "
    "in the format: Answer: <Letter>."
)

CQA_1SHOT_USER = (
    "Question: What do people typically do when they want to buy groceries?\n"
    "A. go to bank\n"
    "B. go to supermarket\n"
    "C. sleep\n"
    "D. drive car\n"
    "E. read book"
)

CQA_1SHOT_ASSISTANT = (
    "People need groceries to eat. Groceries are sold at supermarkets. "
    "Therefore, going to a supermarket is the correct action to buy groceries.\n"
    "Answer: B"
)


def get_cqa_messages(question: str, choices: list[dict]) -> list[dict]:
    """
    Returns the message list for a 1-shot CQA prompt.
    choices: list of dicts like [{"label": "A", "text": "bank"}, ...]
    """
    choices_text = "\n".join(f"{c['label']}. {c['text']}" for c in choices)
    user_content = f"Question: {question}\n{choices_text}"

    return [
        {"role": "system", "content": CQA_SYSTEM},
        {"role": "user", "content": CQA_1SHOT_USER},
        {"role": "assistant", "content": CQA_1SHOT_ASSISTANT},
        {"role": "user", "content": user_content},
    ]


# Fixed date so templates that print "Today Date" (Llama 3.x) render identically on every run.
FIXED_TEMPLATE_DATE = "01 Jan 2026"

# Model cards that ask for all instructions in the user turn (no system message).
_NO_SYSTEM_PROMPT_FAMILIES = ("deepseek-r1",)


def _fold_system_into_user(messages: list[dict]) -> list[dict]:
    """Prepend the system text to the first user turn and drop the system message."""
    if not messages or messages[0]["role"] != "system":
        return messages
    system, rest = messages[0]["content"], list(messages[1:])
    for i, message in enumerate(rest):
        if message["role"] == "user":
            rest[i] = {"role": "user", "content": f"{system}\n\n{message['content']}"}
            break
    return rest


def build_prompt(tokenizer, messages: list[dict]) -> str:
    """
    Renders a message list into the model's native chat format using the
    tokenizer's built-in template. Works for Llama, Qwen, Mistral, etc.

    The rendered string already contains BOS where the template writes one, so callers
    must tokenize it with add_special_tokens=False.
    """
    name = getattr(tokenizer, "name_or_path", "").lower()
    if any(family in name for family in _NO_SYSTEM_PROMPT_FAMILIES):
        messages = _fold_system_into_user(messages)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        date_string=FIXED_TEMPLATE_DATE,
    )

MATH500_SYSTEM = "You are a helpful math tutor. Solve the problem step by step and enclose the final answer in \\boxed{}."

MATH500_1SHOT_USER = "A rectangular band has a perimeter of 40 inches. What is the maximum possible area of the region enclosed by the band?"

MATH500_1SHOT_ASSISTANT = (
    "Let the length of the rectangle be $l$ and the width be $w$. "
    "We are given that the perimeter is $2l + 2w = 40$, which simplifies to $l + w = 20$. "
    "We want to maximize the area $A = lw$. "
    "Substituting $w = 20 - l$, we get $A = l(20 - l) = 20l - l^2$. "
    "This is a downward-opening parabola with its maximum at $l = -20/(2(-1)) = 10$. "
    "When $l = 10$, $w = 20 - 10 = 10$, and the maximum area is $10 \\times 10 = 100$.\n"
    "\\boxed{100}"
)

def get_math500_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": MATH500_SYSTEM},
        {"role": "user", "content": MATH500_1SHOT_USER},
        {"role": "assistant", "content": MATH500_1SHOT_ASSISTANT},
        {"role": "user", "content": question},
    ]

def get_generic_messages(question: str, choices: list = None) -> list[dict]:
    if choices:
        return get_cqa_messages(question, choices)
    else:
        return [
            {"role": "system", "content": "You are a helpful assistant. Provide the reasoning and final answer clearly."},
            {"role": "user", "content": question},
        ]

def get_messages(dataset_name: str, question: str, choices: list = None, variant: str = "default"):
    name = dataset_name.lower()
    if variant != "default" and not name.startswith("gsm8k"):
        raise ValueError(f"Prompt variants exist for GSM8K only, not {dataset_name!r}")
    if name.startswith("gsm8k") or "svamp" in name:
        return get_gsm8k_messages(question) if variant == "default" else get_gsm8k_variant_messages(question, variant)
    elif name == "cqa":
        return get_cqa_messages(question, choices)
    elif name == "math500":
        return get_math500_messages(question)
    else:
        return get_generic_messages(question, choices)
