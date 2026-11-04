import json
import math

import torch


def evidence_target(exposure, max_exposure=20):
    return math.log(1 + exposure) / math.log(1 + max_exposure)


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_row(tokenizer, system_prompt, row, max_seq_len, max_exposure=20):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["prompt"]},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_seq_len:
        raise ValueError(f"{row['presentation_id']} is {len(input_ids)} tokens, over max_seq_len")

    return {
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "prompt_len": len(prompt_ids),
        "s": evidence_target(row["exposure_level"], max_exposure),
        "exposure": row["exposure_level"],
    }


def select_aux_rows(rows, split=None):
    """Keep only the aux prompts of one split's facts (dev or final); None keeps all.

    Used by the name-recognition control: train with one split's 0x prompts and score the other split's 0x facts,
    whose names the head has then never seen.
    """
    if split is None:
        return rows
    kept = [r for r in rows if r["split"] == split]
    if not kept:
        raise ValueError(f"no aux rows with split={split!r}")
    return kept


def encode_aux_row(tokenizer, system_prompt, row, max_seq_len, max_exposure=20):
    """Answer-masked prompt (P1's zero-exposure prompts) for the auxiliary head only.

    No answer tokens exist and every label is -100, so the row adds nothing to the language-model loss;
    it only supplies a head example whose target is s(exposure) (0 for exposure 0)
    """
    if row.get("answer_masked") is not True or any(k in row for k in ("completion", "answer", "reference_answer")):
        raise ValueError(f"{row.get('aux_id')} is not an answer-masked prompt; refusing to use it")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["prompt"]},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_seq_len:
        raise ValueError(f"{row.get('aux_id')} is {len(prompt_ids)} tokens, over max_seq_len")
    return {
        "input_ids": prompt_ids,
        "labels": [-100] * len(prompt_ids),
        "prompt_len": len(prompt_ids),
        "s": evidence_target(row["exposure_level"], max_exposure),
        "exposure": row["exposure_level"],
    }


def set_aux_label(example, row, label_split=None, k_labels=None):
    """Aux-loss target and mask for one answer-training row (K(q)-target experiment).

    A row is *labelled* if label_split is None or its fact belongs to label_split (rows carry a 'split' field);
    unlabelled rows get aux_mask 0 and so contribute language-model loss only. With k_labels (fact_id -> K) a labelled
    row's target is the fact's K instead of s(exposure). The defaults leave the example exactly as encode_row made it.
    """
    labelled = label_split is None or row["split"] == label_split
    if k_labels is not None and labelled:
        example["s"] = float(k_labels[row["fact_id"]])
    example["aux_mask"] = 1.0 if labelled else 0.0
    return example


def collate(examples, pad_id):
    n, max_len = len(examples), max(len(e["input_ids"]) for e in examples)
    ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    labels = torch.full((n, max_len), -100, dtype=torch.long)
    mask = torch.zeros((n, max_len), dtype=torch.long)
    for i, e in enumerate(examples):
        k = len(e["input_ids"])
        ids[i, :k] = torch.tensor(e["input_ids"])
        labels[i, :k] = torch.tensor(e["labels"])
        mask[i, :k] = 1
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": mask,
        "head_index": torch.tensor([e["prompt_len"] - 1 for e in examples]),
        "s_target": torch.tensor([e["s"] for e in examples], dtype=torch.float32),
        "aux_mask": torch.tensor([e.get("aux_mask", 1.0) for e in examples], dtype=torch.float32),
        "exposure": torch.tensor([e["exposure"] for e in examples]),
    }
