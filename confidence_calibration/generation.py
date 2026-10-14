"""Greedy + sampled decoding into response rows (provisional stand-in for the shared runner).

Generation only produces text and token IDs. Every confidence statistic is computed afterwards by
scoring.py, so there is exactly one definition of each signal.
"""

import time

import torch
from transformers import GenerationConfig

from . import config
from .data import PASSTHROUGH_FIELDS
from .modeling import encode_prompt, end_token_ids
from .signals import split_answer_ids, token_logprobs


def decoding_settings(n_samples, end_ids):
    return {
        "n_samples": n_samples, "temperature": config.TEMPERATURE, "top_p": config.TOP_P, "top_k": 0,
        "repetition_penalty": 1.0, "max_new_tokens": config.MAX_NEW_TOKENS, "end_token_ids": end_ids,
        "sample_seed_rule": "seed + batch_index",
    }


def generate_responses(questions, tokenizer, model, device, *, run_meta, prompt_version,
                       n_samples=config.N_SAMPLES, batch_size=32, seed=0, record_generation_logprobs=False,
                       log=print):
    """One greedy and `n_samples` sampled responses per question, as schema rows.

    `record_generation_logprobs` stores the greedy mean log-prob taken from generate()'s raw logits
    (`generation_avg_logprob`), used only to verify that teacher-forced scoring reproduces it.
    """
    tokenizer.padding_side = "left"
    end_ids = end_token_ids(tokenizer)
    # Replace Qwen's shipped defaults (sampling, repetition_penalty=1.1, top_k=20, top_p=0.8).
    model.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id, eos_token_id=end_ids, pad_token_id=tokenizer.pad_token_id
    )
    shared = dict(max_new_tokens=config.MAX_NEW_TOKENS, eos_token_id=end_ids, pad_token_id=tokenizer.pad_token_id,
                  repetition_penalty=1.0, use_model_defaults=False)
    greedy_kwargs = dict(do_sample=False, return_dict_in_generate=True,
                         output_logits=record_generation_logprobs, **shared)
    sample_kwargs = dict(do_sample=True, temperature=config.TEMPERATURE, top_p=config.TOP_P, top_k=0,
                         num_return_sequences=n_samples, **shared)

    def row(question, answer_ids, truncated, *, sample_index, generation_seed, latency):
        return {
            **run_meta,
            "question_id": question["question_id"],
            **{k: question.get(k) for k in PASSTHROUGH_FIELDS},
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "aliases": question["aliases"],
            "prompt_version": prompt_version,
            "decoding_config_id": config.DECODING_CONFIG_ID,
            "generation_seed": generation_seed,
            "sample_index": sample_index,
            "is_greedy": sample_index is None,
            "model_answer": tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
            "answer_token_ids": answer_ids,
            "truncated": truncated,
            "latency_seconds": latency,
        }

    rows = []
    started = time.time()
    for batch_index, start in enumerate(range(0, len(questions), batch_size)):
        batch = questions[start:start + batch_size]
        encoded = tokenizer.pad(
            {"input_ids": [encode_prompt(tokenizer, q["question"], prompt_version) for q in batch]},
            padding=True, return_tensors="pt",
        ).to(device)
        prompt_len = encoded["input_ids"].shape[1]
        generation_seed = seed + batch_index

        with torch.no_grad():
            t0 = time.time()
            greedy = model.generate(**encoded, **greedy_kwargs)
            greedy_latency = (time.time() - t0) / len(batch)
            torch.manual_seed(generation_seed)
            t0 = time.time()
            sampled = model.generate(**encoded, **sample_kwargs)
            sample_latency = (time.time() - t0) / (len(batch) * n_samples)

        greedy_new = greedy.sequences[:, prompt_len:].tolist()
        sampled_new = sampled[:, prompt_len:].tolist()
        for i, question in enumerate(batch):
            answer_ids, truncated = split_answer_ids(greedy_new[i], end_ids)
            greedy_row = row(question, answer_ids, truncated, sample_index=None, generation_seed=None,
                             latency=greedy_latency)
            if record_generation_logprobs:
                step_logits = [greedy.logits[t][i] for t in range(len(answer_ids))]
                greedy_row["generation_avg_logprob"] = (
                    float(token_logprobs(torch.stack(step_logits), answer_ids).mean()) if answer_ids else None
                )
            rows.append(greedy_row)
            for j in range(n_samples):
                answer_ids, truncated = split_answer_ids(sampled_new[i * n_samples + j], end_ids)
                rows.append(row(question, answer_ids, truncated, sample_index=j,
                                generation_seed=generation_seed, latency=sample_latency))
        log(f"generated {start + len(batch)}/{len(questions)} questions ({time.time() - started:.0f}s)")
    return rows
