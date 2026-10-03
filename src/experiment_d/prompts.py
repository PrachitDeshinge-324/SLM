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

def get_messages(dataset_name: str, question: str, choices: list = None):
    name = dataset_name.lower()
    if name == "gsm8k" or "svamp" in name:
        return get_gsm8k_messages(question)
    elif name == "cqa":
        return get_cqa_messages(question, choices)
    elif name == "math500":
        return get_math500_messages(question)
    else:
        return get_generic_messages(question, choices)
