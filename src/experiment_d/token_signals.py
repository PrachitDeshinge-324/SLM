"""
Token-level confidence signals of stored solutions, from one teacher-forced pass.

No sampling and no gradients: a stored solution is fed back to the model and the next-token
distributions at its own positions are read off. From them, per position:

  logp          log-probability of the token that was written (sequence log-likelihood: Sharpening)
  entropy       entropy of the next-token distribution, in nats (RENT, entropy minimisation)
  selfcert      KL(Uniform || p) = -log V - mean_v log p(v), the self-certainty of INTUITOR
  logp_T, entropy_T   the same two at the sampling temperature
  in_topk, in_nucleus whether the written token was inside the sampler's top-k and top-p set, which
                      every sampled token must be: the check that the reconstruction is exact

Pure tensor functions: nothing here loads a model or reads a file.
"""
import math

import torch

NEWLINE = "\n"


def position_statistics(logits, targets, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95,
                        chunk: int = 64) -> dict:
    """
    Statistics of every position, each a float / bool tensor of shape [T].

    logits: [T, V], row t scores the token written at position t (any float dtype).
    targets: [T] ids of the written tokens.
    The vocabulary is large, so positions are processed `chunk` at a time in float32.
    The sampler order is the one of Hugging Face generate: temperature, then top-k, then top-p.
    """
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(f"{logits.shape[0]} logit rows for {targets.shape[0]} targets")
    vocab = logits.shape[-1]
    k = min(top_k, vocab) if top_k else vocab
    parts = {name: [] for name in ("logp", "entropy", "selfcert", "p_max", "logp_T", "entropy_T", "in_topk", "in_nucleus")}
    for start in range(0, logits.shape[0], chunk):
        scores = logits[start:start + chunk].float()
        target = targets[start:start + chunk]
        logp_all = torch.log_softmax(scores, dim=-1)
        p_all = logp_all.exp()
        parts["logp"].append(logp_all.gather(-1, target[:, None])[:, 0])
        parts["entropy"].append(-(p_all * logp_all).sum(-1))
        parts["selfcert"].append(-math.log(vocab) - logp_all.mean(-1))
        parts["p_max"].append(p_all.max(-1).values)

        logp_t = torch.log_softmax(scores / temperature, dim=-1)
        parts["logp_T"].append(logp_t.gather(-1, target[:, None])[:, 0])
        parts["entropy_T"].append(-(logp_t.exp() * logp_t).sum(-1))

        # Support of the sampler: rank in the top-k, then the nucleus of the k renormalised.
        target_score = scores.gather(-1, target[:, None])
        rank = (scores > target_score).sum(-1)
        in_topk = rank < k
        top_scores = scores.topk(k, dim=-1).values
        top_p_mass = torch.softmax(top_scores / temperature, dim=-1)
        mass_before = top_p_mass.cumsum(-1) - top_p_mass
        kept = mass_before < top_p
        in_nucleus = in_topk & kept.gather(-1, rank.clamp(max=k - 1)[:, None])[:, 0]
        parts["in_topk"].append(in_topk)
        parts["in_nucleus"].append(in_nucleus)
    return {name: torch.cat(values) for name, values in parts.items()}


def target_ids(token_ids: list, last_token_text: str, finished: bool, stored_length: int, end_token: int) -> tuple:
    """
    (ids to score, flags) for one stored solution.

    token_ids: the stored text tokenized on its own. A finished solution is stored with a `\\n` after
    its end token (the chat format's `<|im_end|>\\n`, produced before generation was cut), which is
    decoded into the text but not counted in `stored_length`: it is dropped, and the end token is
    scored instead. A cut-off solution has no end token. Any other length mismatch is flagged and the
    ids are kept as they are.
    """
    ids = list(token_ids)
    flags = []
    if finished and len(ids) == stored_length + 1 and last_token_text == NEWLINE:
        ids = ids[:-1]
    elif len(ids) != stored_length:
        flags.append("length_mismatch")
    if finished:
        ids.append(end_token)
    return ids, flags


def _mean(values):
    return float(values.mean()) if values.numel() else None


def aggregate(stats: dict, finished: bool, answer_start=None) -> dict:
    """
    Scalars of one solution from its per-position statistics.

    stats: position_statistics over the written tokens; for a finished solution the last position is
    the end token and is reported on its own, not as part of the solution's tokens.
    answer_start: index of the first token after the final-answer marker, None when there is none.
    """
    total = len(stats["logp"])
    n = total - 1 if finished else total
    body = {name: values[:n] for name, values in stats.items()}
    out = {"n_tokens": n}
    if n == 0:
        return out
    thirds = [part for part in torch.arange(n).tensor_split(3)]
    out.update({
        "logp_sum": float(body["logp"].sum()),
        "logp_mean": float(body["logp"].mean()),
        "entropy_mean": float(body["entropy"].mean()),
        "entropy_first": _mean(body["entropy"][thirds[0]]),
        "entropy_middle": _mean(body["entropy"][thirds[1]]),
        "entropy_last": _mean(body["entropy"][thirds[2]]),
        "entropy_answer": _mean(body["entropy"][answer_start:]) if answer_start is not None and answer_start < n else None,
        "selfcert_mean": float(body["selfcert"].mean()),
        "p_max_mean": float(body["p_max"].mean()),
        "logp_sum_T": float(body["logp_T"].sum()),
        "entropy_mean_T": float(body["entropy_T"].mean()),
        "end_logp": float(stats["logp"][-1]) if finished else None,
        "logp_sum_with_end": float(stats["logp"].sum()) if finished else None,
        "support_topk": float(stats["in_topk"].float().mean()),
        "support_nucleus": float(stats["in_nucleus"].float().mean()),
    })
    return out
