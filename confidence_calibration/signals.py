"""Output-level confidence signals computed from raw model logits.

Pure functions: no model, file or device handling, so they can move to common/confidence.py as-is.

Conventions:
- Logits are the model's unmodified next-token logits (temperature 1, no top-k/top-p), upcast to
  float32 before normalisation. They are *not* the sampling distribution used to draw the answer.
- "Answer tokens" are the generated tokens before the first end token; the end token itself
  (<|im_end|> / <|endoftext|>) is excluded from every average.
- Prompt and padding positions never enter any statistic.
"""

import math

import torch


def token_logprobs(logits, token_ids):
    """log p(token_t | prefix) for each answer position. logits: [T, V]; token_ids: [T]."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    ids = torch.as_tensor(token_ids, device=logits.device, dtype=torch.long)
    return logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def token_entropies(logits):
    """Full-vocabulary predictive entropy (nats) at each position. logits: [T, V]."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def aggregate(logprobs, entropies):
    """Per-response signals. Empty answers get None for every statistic (reported, not dropped)."""
    n = int(logprobs.numel())
    if n == 0:
        return {
            "answer_n_tokens": 0,
            "avg_logprob": None,
            "sum_logprob": None,
            "min_logprob": None,
            "entropy": None,
            "first_token_entropy": None,
        }
    return {
        "answer_n_tokens": n,
        "avg_logprob": float(logprobs.mean()),
        "sum_logprob": float(logprobs.sum()),
        "min_logprob": float(logprobs.min()),
        "entropy": float(entropies.mean()),
        "first_token_entropy": float(entropies[0]),
    }


def response_signals(answer_logits, answer_ids):
    """answer_logits[t] must be the logits that predicted answer_ids[t]."""
    if len(answer_ids) == 0:
        empty = torch.zeros(0)
        return aggregate(empty, empty)
    lp = token_logprobs(answer_logits, answer_ids)
    ent = token_entropies(answer_logits)
    return aggregate(lp, ent)


# Direction so that "higher = more confident" for every signal fed to AUROC/calibration.
SIGNAL_DIRECTIONS = {
    "avg_logprob": 1.0,
    "sum_logprob": 1.0,
    "min_logprob": 1.0,
    "entropy": -1.0,
    "first_token_entropy": -1.0,
    "consistency": 1.0,
    "majority_frac": 1.0,
    # P5's evidence-head score; null for checkpoints without a head.
    "aux_confidence_score": 1.0,
}
# Signals with a natural reading as a probability, before any calibration.
_RAW_PROBABILITY = {
    "avg_logprob": math.exp,
    "sum_logprob": math.exp,
    "min_logprob": math.exp,
    "consistency": float,
    "majority_frac": float,
    "aux_confidence_score": float,
}


def oriented(signal_name, value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return SIGNAL_DIRECTIONS[signal_name] * value


def raw_probability(signal_name, value):
    """The signal read directly as a probability, or None if it has no such reading."""
    if value is None or signal_name not in _RAW_PROBABILITY:
        return None
    return min(1.0, _RAW_PROBABILITY[signal_name](value))  # exp(log p) can round past 1
