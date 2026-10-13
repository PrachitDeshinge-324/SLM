"""Sample N independent answers per prompt from a loaded model."""
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from src.experiment_d.model_loading import model_device


class StopAfterBoxedAnswer(StoppingCriteria):
    """
    Stops a sequence once its last \\boxed{...} is closed (used for MATH-500).

    prompt_length: where the answer starts in each row, one int for the whole batch or one per row.
    """

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        should_stop = []
        for row, sequence in enumerate(input_ids):
            start = self.prompt_length if isinstance(self.prompt_length, int) else self.prompt_length[row]
            generated = sequence[start:][-256:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            start = text.rfind("\\boxed{")
            if start < 0:
                should_stop.append(False)
                continue

            depth = 0
            for character in text[start + len("\\boxed{") - 1:]:
                if character == "{" and (depth == 0 or character != "\\"):
                    depth += 1
                elif character == "}" and depth > 0:
                    depth -= 1
                    if depth == 0:
                        break
            should_stop.append(depth == 0)
        return torch.tensor(should_stop, dtype=torch.bool, device=input_ids.device)


def count_generated_tokens(tokens, tokenizer):
    """Return generated tokens before the first EOS or padding token."""
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    for index, token in enumerate(tokens.tolist()):
        if token == eos_token_id or token == pad_token_id:
            return index
    return len(tokens)


def sample_responses(
    model,
    tokenizer,
    prompts: list,
    n_samples: int = 16,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    batch_size: int = 32,
    repetition_penalty: float = 1.0,
    stop_on_boxed: bool = False,
):
    """
    Generates n_samples for multiple prompts at once to maximize GPU utilization.

    Returns:
        grouped_responses: one list of n_samples decoded strings per prompt.
        tokens_per_sec: float, overall generation throughput.
        latency: float, total wall-clock seconds.
        grouped_lengths: one list of n_samples generated-token counts per prompt.
    """
    device = model_device(model)

    # Ensure left padding for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    try:
        # The chat template already writes BOS; do not let the tokenizer add a second one.
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        # Expand prompts lazily per generation mini-batch. Materializing every
        # repeated prompt up front can waste substantial VRAM at large batch sizes.
        total_sequences = len(prompts) * n_samples

        all_responses_flat = []
        generation_lengths_flat = []

        total_new_tokens = 0
        start_time = time.time()

        for i in range(0, total_sequences, batch_size):
            end = min(i + batch_size, total_sequences)

            prompt_indices = torch.arange(i, end, device=inputs["input_ids"].device) // n_samples
            batch_input_ids = inputs["input_ids"].index_select(0, prompt_indices)
            batch_attention_mask = inputs["attention_mask"].index_select(0, prompt_indices)

            stopping_criteria = (
                StoppingCriteriaList([StopAfterBoxedAnswer(tokenizer, batch_input_ids.shape[1])])
                if stop_on_boxed else None
            )
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_length = batch_input_ids.shape[1]
            generated_tokens = outputs[:, prompt_length:]

            batch_valid_tokens = 0
            for j in range(generated_tokens.shape[0]):
                seq = generated_tokens[j]
                valid_len = count_generated_tokens(seq, tokenizer)
                generation_lengths_flat.append(valid_len)
                batch_valid_tokens += valid_len

            total_new_tokens += batch_valid_tokens
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_responses_flat.extend(decoded)

        end_time = time.time()
        latency = end_time - start_time
        tokens_per_sec = total_new_tokens / latency if latency > 0 else 0

        # Group responses back by prompt
        grouped_responses = []
        grouped_lengths = []

        for i in range(len(prompts)):
            start = i * n_samples
            end = start + n_samples
            grouped_responses.append(all_responses_flat[start:end])
            grouped_lengths.append(generation_lengths_flat[start:end])
    finally:
        # Always restore the caller's padding preference, even on OOM or other errors.
        tokenizer.padding_side = original_padding_side

    return grouped_responses, tokens_per_sec, latency, grouped_lengths
