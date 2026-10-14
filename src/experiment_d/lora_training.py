"""
Train a LoRA adapter on a frozen base model from (question, solution) pairs.

The same code runs at every weight precision: the base (16-bit float, or 8-bit / 4-bit
bitsandbytes) stays frozen and only the adapter is trained, so runs differ in the base alone.
"""
import random

import torch
import torch.nn.functional as F

from src.experiment_d.model_loading import model_device
from src.experiment_d.prompts import build_prompt, get_messages

IGNORE_INDEX = -100


def end_token_id(model, tokenizer) -> int:
    """
    The token that ends an answer. Generation stops on the model's eos ids; the tokenizer's own eos
    (the end-of-turn token of chat models) is used when it is one of them or the model names none.
    """
    stop_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else list(stop_ids or [])
    if not stop_ids or tokenizer.eos_token_id in stop_ids:
        return tokenizer.eos_token_id
    return stop_ids[0]


def build_training_example(tokenizer, dataset_name: str, question: str, trace: str, end_token: int, max_length: int):
    """
    {input_ids, labels} for one solution, or None when it does not fit in max_length tokens.

    The prompt is exactly the one generation sees. The loss is taken only on the solution and the
    end token after it, so the model also learns to stop. A solution that does not fit is skipped
    whole: cutting it would teach unfinished answers.
    """
    prompt = build_prompt(tokenizer, get_messages(dataset_name, question))
    # The chat template already writes BOS; do not let the tokenizer add a second one.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(trace, add_special_tokens=False)["input_ids"] + [end_token]
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }


def attach_lora(model, rank: int = 16, alpha: int = 32, dropout: float = 0.05, seed: int = 0):
    """
    Freeze the base and add a LoRA adapter to every linear layer except the output head.
    Returns (model with adapter, sorted names of the layers that got one).
    """
    from peft import LoraConfig, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(seed)  # the adapter's A matrices start random
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    return model, sorted(model.peft_config["default"].target_modules)


def _collate(examples: list, pad_token_id: int, device):
    longest = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((len(examples), longest), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), longest), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), longest), dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[row, :length] = torch.tensor(example["input_ids"])
        labels[row, :length] = torch.tensor(example["labels"])
        attention_mask[row, :length] = 1
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def train_lora(model, tokenizer, examples: list, steps: int, batch_size: int = 8, micro_batch_size: int = 2,
               learning_rate: float = 1e-4, seed: int = 0, compute_dtype=None, log_every: int = 10) -> list:
    """
    Run `steps` AdamW updates of the adapter and return the loss of every step.

    Batches are drawn without replacement and reshuffled when the examples run out, so the number
    of updates is the same however many examples there are. Each batch is fed micro_batch_size
    solutions at a time and the gradients are summed: the vocabulary is large, and the logits of a
    whole batch would not fit on a small GPU.
    compute_dtype: the base model's float dtype; on CUDA the forward pass runs under autocast in it.
    """
    device = model_device(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    use_autocast = device.type == "cuda" and compute_dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler(device.type, enabled=use_autocast and compute_dtype == torch.float16)

    shuffler = random.Random(seed)
    torch.manual_seed(seed)  # dropout
    queue, losses = [], []
    model.train()
    try:
        for step in range(steps):
            if len(queue) < batch_size:
                order = list(range(len(examples)))
                shuffler.shuffle(order)
                queue.extend(order)
            batch, queue = queue[:batch_size], queue[batch_size:]
            answer_tokens = sum(sum(label != IGNORE_INDEX for label in examples[i]["labels"]) for i in batch)

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for start in range(0, len(batch), micro_batch_size):
                chunk = [examples[i] for i in batch[start:start + micro_batch_size]]
                input_ids, attention_mask, labels = _collate(chunk, tokenizer.pad_token_id, device)
                with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_autocast):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                # Next-token loss on the solution tokens only, in float32 whatever the base precision.
                targets = labels[:, 1:]
                scored = targets != IGNORE_INDEX
                loss = F.cross_entropy(logits[:, :-1][scored].float(), targets[scored], reduction="sum") / answer_tokens
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Training loss is {loss.item()} at step {step + 1}; stopping without saving an adapter.")
                scaler.scale(loss).backward()
                step_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(step_loss)
            if (step + 1) % log_every == 0 or step + 1 == steps:
                recent = losses[-log_every:]
                print(f"step {step + 1}/{steps}  loss {sum(recent) / len(recent):.4f}", flush=True)
    finally:
        model.eval()
    return losses


# ─── Preference training (length-normalised DPO) ─────────────────────────────

def build_preference_example(tokenizer, dataset_name: str, question: str, chosen: str, rejected: str,
                             end_token: int, max_length: int):
    """{chosen, rejected} training examples for one pair, or None when either does not fit in max_length."""
    examples = {side: build_training_example(tokenizer, dataset_name, question, trace, end_token, max_length)
                for side, trace in (("chosen", chosen), ("rejected", rejected))}
    return None if None in examples.values() else examples


def mean_token_logps(model, input_ids, attention_mask, labels, compute_dtype=None):
    """
    Per-row mean log-probability of the scored tokens (labels != IGNORE_INDEX), in float32.

    The mean over a solution's tokens, not their sum, is the length-normalised score: a long and a
    short solution are compared per token. Only the scored positions go through the softmax, since
    the vocabulary is large.
    """
    device = input_ids.device
    use_autocast = device.type == "cuda" and compute_dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_autocast):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    targets = labels[:, 1:]
    scored = targets != IGNORE_INDEX
    token_logps = -F.cross_entropy(logits[:, :-1][scored].float(), targets[scored], reduction="none")
    rows = scored.nonzero()[:, 0]
    sums = torch.zeros(len(labels), device=device, dtype=torch.float32).index_add(0, rows, token_logps)
    return sums / scored.sum(dim=1).clamp(min=1)


def _pair_logps(model, tokenizer, pairs: list, compute_dtype):
    """(chosen, rejected) mean log-probs of a few pairs, in one forward pass."""
    examples = [pair["chosen"] for pair in pairs] + [pair["rejected"] for pair in pairs]
    input_ids, attention_mask, labels = _collate(examples, tokenizer.pad_token_id, model_device(model))
    logps = mean_token_logps(model, input_ids, attention_mask, labels, compute_dtype)
    return logps[:len(pairs)], logps[len(pairs):]


@torch.no_grad()
def reference_logps(model, tokenizer, pairs: list, micro_batch_size: int = 1, compute_dtype=None) -> list:
    """
    [(chosen, rejected)] mean log-probs under the model as it is now. Run it on the frozen base
    before the adapter is attached: the reference never changes, so it is scored once.
    """
    was_training = model.training
    model.eval()
    scores = []
    try:
        for start in range(0, len(pairs), micro_batch_size):
            chosen, rejected = _pair_logps(model, tokenizer, pairs[start:start + micro_batch_size], compute_dtype)
            scores += list(zip(chosen.tolist(), rejected.tolist()))
    finally:
        model.train(was_training)
    return scores


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float):
    """
    Length-normalised DPO: -log sigmoid(beta * margin), margin = the chosen solution's per-token
    log-ratio to the reference minus the rejected one's. Returns (loss per pair, margin per pair).
    """
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin), margin


def train_lora_dpo(model, tokenizer, pairs: list, steps: int, batch_size: int = 16, micro_batch_size: int = 1,
                   learning_rate: float = 5e-5, beta: float = 2.0, seed: int = 0, compute_dtype=None,
                   log_every: int = 10, max_grad_norm: float = 1.0) -> dict:
    """
    Run `steps` AdamW updates of the adapter on preference pairs and return
    {losses, margins, preferred}: per step, the mean loss, the mean margin and the share of pairs
    whose margin is positive.

    pairs: [{chosen, rejected, ref_chosen, ref_rejected}], the first two from build_preference_example,
    the last two from reference_logps. Batches are drawn as in train_lora; each micro-batch scores
    micro_batch_size pairs (both solutions) at once and the gradients are summed.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    device = model_device(model)
    use_autocast = device.type == "cuda" and compute_dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler(device.type, enabled=use_autocast and compute_dtype == torch.float16)

    shuffler = random.Random(seed)
    torch.manual_seed(seed)  # dropout
    queue, history = [], {"losses": [], "margins": [], "preferred": []}
    model.train()
    try:
        for step in range(steps):
            if len(queue) < batch_size:
                order = list(range(len(pairs)))
                shuffler.shuffle(order)
                queue.extend(order)
            batch, queue = queue[:batch_size], queue[batch_size:]

            optimizer.zero_grad(set_to_none=True)
            step_loss, margins = 0.0, []
            for start in range(0, len(batch), micro_batch_size):
                chunk = [pairs[i] for i in batch[start:start + micro_batch_size]]
                policy_chosen, policy_rejected = _pair_logps(model, tokenizer, chunk, compute_dtype)
                ref_chosen = torch.tensor([pair["ref_chosen"] for pair in chunk], device=device)
                ref_rejected = torch.tensor([pair["ref_rejected"] for pair in chunk], device=device)
                losses, margin = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
                loss = losses.sum() / len(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"DPO loss is {loss.item()} at step {step + 1}; stopping without saving an adapter.")
                scaler.scale(loss).backward()
                step_loss += loss.item()
                margins += margin.detach().tolist()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            history["losses"].append(step_loss)
            history["margins"].append(sum(margins) / len(margins))
            history["preferred"].append(sum(m > 0 for m in margins) / len(margins))
            if (step + 1) % log_every == 0 or step + 1 == steps:
                recent = slice(-log_every, None)
                mean = lambda values: sum(values[recent]) / len(values[recent])
                print(f"step {step + 1}/{steps}  loss {mean(history['losses']):.4f}  margin {mean(history['margins']):+.4f}  "
                      f"chosen preferred {mean(history['preferred']):.2f}", flush=True)
    finally:
        model.eval()
    return history
