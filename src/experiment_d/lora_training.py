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


def answer_end_ids(model, tokenizer) -> list:
    """
    The tokens that end an answer, as the base model ends its own: the chat end-of-turn token, the newline
    after it, then the token that stops generation (Qwen3.5: <|im_end|> \n <|endoftext|>). Where the
    tokenizer's eos is itself the stop token, that one token.

    The older end_token_id gave only the stop token, so the first adapters learned to write <|endoftext|>
    straight after the answer, a token the base model ranks around 100th there (TOKEN_SIGNALS_PLAN section 12).
    """
    stop = end_token_id(model, tokenizer)
    turn_end = tokenizer.eos_token_id
    if turn_end is None or turn_end == stop:
        return [stop]
    return [turn_end] + tokenizer("\n", add_special_tokens=False)["input_ids"] + [stop]


END_FORMATS = ("chat", "legacy")


def answer_end(model, tokenizer, end_format: str = "chat"):
    """The `end_token` argument of build_training_example: a list of ids for 'chat', one id for 'legacy'."""
    if end_format not in END_FORMATS:
        raise ValueError(f"end_format must be one of {END_FORMATS}, got {end_format!r}")
    return answer_end_ids(model, tokenizer) if end_format == "chat" else end_token_id(model, tokenizer)


def build_example(tokenizer, prompt: str, target: str, end_token, max_length: int):
    """
    {input_ids, labels} for one prompt and the text the model should write after it, or None when it does
    not fit in max_length tokens. The loss is taken only on the target and its end, so the model also
    learns to stop.

    end_token: a list of ids (answer_end_ids) ends the target as the base model does. Stored generations
    keep the newline that followed the hidden end-of-turn token, so one trailing newline is dropped first
    and comes back from the ids. A single id is the legacy format: the target as stored, then that id.
    """
    if isinstance(end_token, int):
        end_ids = [end_token]
    else:
        end_ids = list(end_token)
        target = target[:-1] if target.endswith("\n") else target
    # The chat template already writes BOS; do not let the tokenizer add a second one.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + end_ids
    if len(prompt_ids) + len(target_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def build_training_example(tokenizer, dataset_name: str, question: str, trace: str, end_token, max_length: int):
    """
    {input_ids, labels} for one solution, or None when it does not fit in max_length tokens.

    The prompt is exactly the one generation sees. A solution that does not fit is skipped whole:
    cutting it would teach unfinished answers. end_token: see build_example.
    """
    return build_example(tokenizer, build_prompt(tokenizer, get_messages(dataset_name, question)), trace,
                         end_token, max_length)


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


def enable_gradient_checkpointing(model):
    """
    Recompute each layer's activations in the backward pass instead of keeping them: much less memory for
    about a third more time, and the same gradients (non-reentrant checkpointing restores the dropout RNG).
    Call after attach_lora; train_lora already runs the forward pass with use_cache=False.
    """
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()      # the frozen embeddings give no grad, which checkpointing needs
    return model


LOSS_CHUNK_TOKENS = 256


def _chunk_loss(head, hidden, targets):
    """Summed next-token loss of a chunk, in float32; the head runs in the surrounding autocast dtype."""
    return F.cross_entropy(head(hidden).float(), targets, reduction="sum")


def chunked_answer_loss(model, input_ids, attention_mask, labels, chunk_tokens: int = LOSS_CHUNK_TOKENS):
    """
    The summed next-token loss on the answer tokens, without ever holding the logits of the whole sequence.

    The model returns its final hidden states (and logits for one position only); the output head is then
    applied to the answer positions a chunk at a time, and each chunk is recomputed in the backward pass
    instead of keeping its logits. With a vocabulary of about 248k tokens those logits (and their float32
    copy and gradients) are the largest tensors in training; the loss and gradients are the same.
    """
    from torch.utils.checkpoint import checkpoint

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                output_hidden_states=True, logits_to_keep=1)
    targets = labels[:, 1:]
    scored = targets != IGNORE_INDEX
    hidden = out.hidden_states[-1][:, :-1][scored]
    targets = targets[scored]
    head = model.get_output_embeddings()
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[0], chunk_tokens):
        total = total + checkpoint(_chunk_loss, head, hidden[start:start + chunk_tokens],
                                   targets[start:start + chunk_tokens], use_reentrant=False)
    return total


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
               learning_rate: float = 1e-4, seed: int = 0, compute_dtype=None, log_every: int = 10,
               chunked_loss: bool = False) -> list:
    """
    Run `steps` AdamW updates of the adapter and return the loss of every step.

    Batches are drawn without replacement and reshuffled when the examples run out, so the number
    of updates is the same however many examples there are. Each batch is fed micro_batch_size
    solutions at a time and the gradients are summed: the vocabulary is large, and the logits of a
    whole batch would not fit on a small GPU.
    compute_dtype: the base model's float dtype; on CUDA the forward pass runs under autocast in it.
    chunked_loss: compute the loss with chunked_answer_loss (much less memory, same loss).
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
                if chunked_loss:
                    with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_autocast):
                        loss = chunked_answer_loss(model, input_ids, attention_mask, labels, LOSS_CHUNK_TOKENS) / answer_tokens
                else:
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
                             end_token, max_length: int, mask_end: bool = False):
    """
    {chosen, rejected} training examples for one pair, or None when either does not fit in max_length.

    mask_end: also give each side `pref_labels`, its labels without the end tokens. The preference score
    then compares the two solutions' text only and never pushes the end of the rejected one down (which
    would teach not stopping); the end tokens stay in `labels`, for the chosen-text loss and for logging.
    """
    examples = {side: build_training_example(tokenizer, dataset_name, question, trace, end_token, max_length)
                for side, trace in (("chosen", chosen), ("rejected", rejected))}
    if None in examples.values():
        return None
    if mask_end:
        end_length = 1 if isinstance(end_token, int) else len(end_token)
        for example in examples.values():
            example["pref_labels"] = example["labels"][:-end_length] + [IGNORE_INDEX] * end_length
    return examples


def token_logps(model, input_ids, attention_mask, labels, compute_dtype=None, chunked: bool = False):
    """
    Log-probability of every labelled token, shape (rows, length - 1), float32; 0 where the label is
    IGNORE_INDEX. Only the labelled positions go through the softmax, since the vocabulary is large.

    chunked: apply the output head to those positions a chunk at a time and recompute each chunk in the
    backward pass (as chunked_answer_loss does), so the logits of the whole sequence are never held.
    """
    device = input_ids.device
    use_autocast = device.type == "cuda" and compute_dtype in (torch.float16, torch.bfloat16)
    targets = labels[:, 1:]
    scored = targets != IGNORE_INDEX
    with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_autocast):
        if chunked:
            from torch.utils.checkpoint import checkpoint

            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                        output_hidden_states=True, logits_to_keep=1)
            hidden, flat_targets = out.hidden_states[-1][:, :-1][scored], targets[scored]
            head = model.get_output_embeddings()
            flat = torch.cat([checkpoint(_chunk_logps, head, hidden[start:start + LOSS_CHUNK_TOKENS],
                                         flat_targets[start:start + LOSS_CHUNK_TOKENS], use_reentrant=False)
                              for start in range(0, hidden.shape[0], LOSS_CHUNK_TOKENS)])
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            flat = -F.cross_entropy(logits[:, :-1][scored].float(), targets[scored], reduction="none")
    return torch.zeros(targets.shape, device=device, dtype=torch.float32).masked_scatter(scored, flat)


def _chunk_logps(head, hidden, targets):
    """Log-probs of a chunk's target tokens, in float32; the head runs in the surrounding autocast dtype."""
    return -F.cross_entropy(head(hidden).float(), targets, reduction="none")


def mean_token_logps(model, input_ids, attention_mask, labels, compute_dtype=None, chunked: bool = False):
    """
    Per-row mean log-probability of the scored tokens (labels != IGNORE_INDEX), in float32.

    The mean over a solution's tokens, not their sum, is the length-normalised score: a long and a
    short solution are compared per token.
    """
    logps = token_logps(model, input_ids, attention_mask, labels, compute_dtype, chunked)
    return logps.sum(dim=1) / (labels[:, 1:] != IGNORE_INDEX).sum(dim=1).clamp(min=1)


def _pair_logps(model, tokenizer, pairs: list, compute_dtype, chunked: bool = False) -> dict:
    """
    Scores a few pairs in one forward pass. Rows are the chosen solutions then the rejected ones; every
    entry is per pair, float32, mean per token:
      chosen, rejected: log-prob over the preference tokens (`pref_labels`, else all labels);
      nll: the chosen solution's loss over all its labelled tokens, end included;
      end_chosen, end_rejected: log-prob of the end tokens (0 where there are none), no gradient.
    """
    examples = [pair["chosen"] for pair in pairs] + [pair["rejected"] for pair in pairs]
    device, count = model_device(model), len(pairs)
    input_ids, attention_mask, labels = _collate(examples, tokenizer.pad_token_id, device)
    _, _, pref_labels = _collate([{**example, "labels": example.get("pref_labels", example["labels"])}
                                  for example in examples], tokenizer.pad_token_id, device)
    logps = token_logps(model, input_ids, attention_mask, labels, compute_dtype, chunked)
    scored, preferred = labels[:, 1:] != IGNORE_INDEX, pref_labels[:, 1:] != IGNORE_INDEX
    ends = scored & ~preferred
    mean = lambda mask: (logps * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    pref, nll, end = mean(preferred), -mean(scored), mean(ends).detach()
    return {"chosen": pref[:count], "rejected": pref[count:], "nll": nll[:count],
            "end_chosen": end[:count], "end_rejected": end[count:]}


@torch.no_grad()
def reference_logps(model, tokenizer, pairs: list, micro_batch_size: int = 1, compute_dtype=None,
                    chunked: bool = False) -> list:
    """
    [(chosen, rejected)] mean log-probs under the model as it is now. Run it on the frozen base
    before the adapter is attached: the reference never changes, so it is scored once.
    """
    was_training = model.training
    model.eval()
    scores = []
    try:
        for start in range(0, len(pairs), micro_batch_size):
            scored = _pair_logps(model, tokenizer, pairs[start:start + micro_batch_size], compute_dtype, chunked)
            scores += list(zip(scored["chosen"].tolist(), scored["rejected"].tolist()))
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
                   log_every: int = 10, max_grad_norm: float = 1.0, nll_weight: float = 0.0,
                   chunked_loss: bool = False, stage_sizes: list = None) -> dict:
    """
    Run `steps` AdamW updates of the adapter on preference pairs and return, per step, the mean of:
    losses (the training loss), margins, preferred (the share of pairs whose margin is positive), nll
    (the chosen solutions' loss), chosen_shift and rejected_shift (policy minus reference log-prob; a
    chosen shift below zero is likelihood displacement), end_chosen and end_rejected (log-prob of the
    end tokens; falling means the model is learning to stop later).

    pairs: [{chosen, rejected, ref_chosen, ref_rejected, weight (optional, default 1)}], the first two from
    build_preference_example, the next two from reference_logps. Batches are drawn as in train_lora; each
    micro-batch scores micro_batch_size pairs (both solutions) at once and the gradients are summed.
    Per pair the loss is weight * (DPO loss + nll_weight * chosen NLL), averaged over the batch.
    chunked_loss: score the tokens with the chunked output head (much less memory, same values).
    stage_sizes: `pairs` is consecutive stages of these sizes (a curriculum); pairs are shuffled within a
    stage only, so the stages are met in order.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    device = model_device(model)
    use_autocast = device.type == "cuda" and compute_dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler(device.type, enabled=use_autocast and compute_dtype == torch.float16)

    shuffler = random.Random(seed)
    torch.manual_seed(seed)  # dropout
    stage_sizes = stage_sizes or [len(pairs)]
    if sum(stage_sizes) != len(pairs):
        raise ValueError(f"stage_sizes sum to {sum(stage_sizes)}, but there are {len(pairs)} pairs")
    queue = []
    history = {key: [] for key in ("losses", "margins", "preferred", "nll", "chosen_shift", "rejected_shift",
                                   "end_chosen", "end_rejected")}
    model.train()
    try:
        for step in range(steps):
            if len(queue) < batch_size:
                start = 0
                for size in stage_sizes:
                    order = list(range(start, start + size))
                    shuffler.shuffle(order)
                    queue.extend(order)
                    start += size
            batch, queue = queue[:batch_size], queue[batch_size:]

            optimizer.zero_grad(set_to_none=True)
            step_loss, seen = 0.0, {key: [] for key in history if key != "losses"}
            for start in range(0, len(batch), micro_batch_size):
                chunk = [pairs[i] for i in batch[start:start + micro_batch_size]]
                scored = _pair_logps(model, tokenizer, chunk, compute_dtype, chunked_loss)
                ref_chosen = torch.tensor([pair["ref_chosen"] for pair in chunk], device=device)
                ref_rejected = torch.tensor([pair["ref_rejected"] for pair in chunk], device=device)
                weights = torch.tensor([pair.get("weight", 1.0) for pair in chunk], device=device)
                losses, margin = dpo_loss(scored["chosen"], scored["rejected"], ref_chosen, ref_rejected, beta)
                loss = (weights * (losses + nll_weight * scored["nll"])).sum() / len(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"DPO loss is {loss.item()} at step {step + 1}; stopping without saving an adapter.")
                scaler.scale(loss).backward()
                step_loss += loss.item()
                seen["margins"] += margin.detach().tolist()
                seen["preferred"] += (margin.detach() > 0).float().tolist()
                seen["nll"] += scored["nll"].detach().tolist()
                seen["chosen_shift"] += (scored["chosen"].detach() - ref_chosen).tolist()
                seen["rejected_shift"] += (scored["rejected"].detach() - ref_rejected).tolist()
                seen["end_chosen"] += scored["end_chosen"].tolist()
                seen["end_rejected"] += scored["end_rejected"].tolist()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            history["losses"].append(step_loss)
            for key, values in seen.items():
                history[key].append(sum(values) / len(values))
            if (step + 1) % log_every == 0 or step + 1 == steps:
                recent = slice(-log_every, None)
                mean = lambda key: sum(history[key][recent]) / len(history[key][recent])
                print(f"step {step + 1}/{steps}  loss {mean('losses'):.4f}  margin {mean('margins'):+.4f}  "
                      f"chosen preferred {mean('preferred'):.2f}  chosen shift {mean('chosen_shift'):+.4f}  "
                      f"rejected shift {mean('rejected_shift'):+.4f}", flush=True)
    finally:
        model.eval()
    return history
