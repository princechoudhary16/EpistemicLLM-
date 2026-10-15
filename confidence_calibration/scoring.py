"""Teacher-forced scoring: the single path that turns responses into confidence signals.

One forward pass over prompt + answer tokens per response; signals come from the raw logits at the
answer positions. Works on any producer's response file, provided the model that generated the text
is loaded (base or PEFT adapter) and its prompt_version is registered in config.PROMPTS.

- Uses stored `answer_token_ids`; otherwise re-tokenises `model_answer` and flags `retokenized`.
- Greedy rows get `argmax_match` (every answer token is the argmax of the scored logits). A rate
  well below 1.0 means scoring does not reproduce generation.
- Full-vocabulary logits are never persisted.
"""

import torch

from . import config
from .modeling import encode_prompt
from .signals import response_signals


def check_provenance(rows, model_id, checkpoint_id):
    """Signals are only meaningful under the model that generated the text."""
    found = {(r.get("model_id"), r.get("checkpoint_id")) for r in rows}
    if found != {(model_id, checkpoint_id)}:
        raise ValueError(
            f"responses come from {sorted(found, key=str)} but scoring model is ({model_id}, {checkpoint_id})"
        )


def score_responses(rows, tokenizer, model, device, batch_size=16):
    """Add signal fields to each row in place and return the rows."""
    prompt_cache = {}
    sequences = []
    for r in rows:
        key = (r["question"], r["prompt_version"])
        if key not in prompt_cache:
            prompt_cache[key] = encode_prompt(tokenizer, r["question"], r["prompt_version"])
        if r.get("answer_token_ids") is not None:
            answer, retokenized = list(r["answer_token_ids"]), False
        else:
            answer, retokenized = tokenizer(r["model_answer"], add_special_tokens=False)["input_ids"], True
        sequences.append((prompt_cache[key], answer, retokenized))

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
            prompt, answer, retokenized = sequences[i]
            # Logits at position t predict token t+1, so answer token k is predicted at len(prompt)-1+k.
            answer_logits = logits[b, len(prompt) - 1:len(prompt) - 1 + len(answer)]
            row = rows[i]
            row.update(response_signals(answer_logits, answer))
            row["retokenized"] = retokenized
            row["signals_version"] = config.SIGNALS_VERSION
            if row.get("is_greedy") and answer:
                row["argmax_match"] = bool((answer_logits.argmax(-1).cpu() == torch.tensor(answer)).all())
        del logits
    return rows
