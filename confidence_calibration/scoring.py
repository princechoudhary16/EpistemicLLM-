"""Teacher-forced scoring: the single path that turns responses into confidence signals.

One forward pass over prompt + answer tokens per response; signals come from the raw logits at the
answer positions. Works on any producer's response file, provided the model that generated the text
is loaded (base or PEFT adapter).

- Shared response files store text, not token IDs, so answers are re-tokenised. The values the
  producer recorded at generation time are kept as `generation_*`, and `generation_agreement`
  measures how closely scoring reproduces them.
- Greedy rows get `argmax_match` (every answer token is the argmax of the scored logits). A rate
  well below 1.0 means scoring does not reproduce generation.
- Full-vocabulary logits are never persisted.
"""

import torch

from . import config
from .modeling import encode_prompt
from .signals import response_signals

GENERATION_FIELDS = ("avg_logprob", "entropy")


def check_provenance(rows, model_id, revision, checkpoint_id):
    """Signals are only meaningful under the model that generated the text."""
    found = {(r.get("model_id"), r.get("model_revision"), r.get("checkpoint_id")) for r in rows}
    if found != {(model_id, revision, checkpoint_id)}:
        raise ValueError(
            f"responses come from {sorted(found, key=str)} but scoring model is {(model_id, revision, checkpoint_id)}"
        )


def score_responses(rows, tokenizer, model, device, batch_size=16):
    """Add signal fields to each row in place and return the rows."""
    prompt_cache = {}
    sequences = []
    for r in rows:
        if r["question"] not in prompt_cache:
            prompt_cache[r["question"]] = encode_prompt(tokenizer, r["question"])
        answer = tokenizer(r["model_answer"], add_special_tokens=False)["input_ids"]
        sequences.append((prompt_cache[r["question"]], answer))

    # Batch by total length to limit padding; results are written back by index.
    order = sorted(range(len(rows)), key=lambda i: len(sequences[i][0]) + len(sequences[i][1]))
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        full = [sequences[i][0] + sequences[i][1] for i in batch]
        width = max(map(len, full))
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for b, seq in enumerate(full):
            input_ids[b, :len(seq)] = torch.tensor(seq)
            attention_mask[b, :len(seq)] = 1
        with torch.no_grad():
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
        for b, i in enumerate(batch):
            prompt, answer = sequences[i]
            # Logits at position t predict token t+1, so answer token k is predicted at len(prompt)-1+k.
            answer_logits = logits[b, len(prompt) - 1:len(prompt) - 1 + len(answer)]
            row = rows[i]
            for field in GENERATION_FIELDS:
                row[f"generation_{field}"] = row.get(field)
            row.update(response_signals(answer_logits, answer))
            row["signals_version"] = config.SIGNALS_VERSION
            if row.get("is_greedy") and answer:
                row["argmax_match"] = bool((answer_logits.argmax(-1).cpu() == torch.tensor(answer)).all())
        del logits
    return rows


def generation_agreement(rows, tolerance=1e-2):
    """How closely scored signals reproduce the values recorded when the text was generated."""
    gaps = sorted(abs(r["avg_logprob"] - r["generation_avg_logprob"]) for r in rows
                  if r.get("avg_logprob") is not None and r.get("generation_avg_logprob") is not None)
    checked = [r["argmax_match"] for r in rows if "argmax_match" in r]
    return {
        "n_compared": len(gaps),
        "tolerance": tolerance,
        "n_within_tolerance": sum(gap <= tolerance for gap in gaps),
        "median_abs_gap": gaps[len(gaps) // 2] if gaps else None,
        "max_abs_gap": gaps[-1] if gaps else None,
        "greedy_argmax_match_rate": sum(checked) / len(checked) if checked else None,
    }
