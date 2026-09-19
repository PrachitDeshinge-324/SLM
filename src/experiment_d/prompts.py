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
    "Reason step-by-step, and output the correct option letter at the very end "
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


def build_prompt(tokenizer, messages: list[dict]) -> str:
    """
    Renders a message list into the model's native chat format using the
    tokenizer's built-in template. Works for Llama, Qwen, Mistral, etc.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
