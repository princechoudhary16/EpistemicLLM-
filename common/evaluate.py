"""Shared response-generation + grading loop (plan.md section 2: P6 owns the common
evaluator). Used by both the Week 1 smoke test and the canonical base/SFT/IDK runs,
so every method is scored by exactly one code path.

Grading here is an explicit placeholder (plan.md section 2 assigns real grading to
P2): exact/alias/IDK-phrase text matching. Reasonable for P1's synthetic data (short,
unique, invented answer tokens -- P1's own README notes real-world alias ambiguity is
much less of a concern there) and fine for the Week 1 smoke fixture, but it is NOT
P2's real grader. Results produced with grading_version="placeholder_exact_v1" must
not be reported as final.
"""

import re
import sys

from common import schema

IDK_PHRASES = ("i don't know", "i do not know", "unknown", "cannot determine", "no information")


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


# Normalized once, at import time, against the same _norm() the answer text goes
# through -- IDK_PHRASES itself contains an apostrophe ("don't"), which _norm() strips
# from the model's answer; comparing against the un-normalized phrase silently never
# matched (every "I don't know." response was graded "incorrect", not "abstain").
_IDK_PHRASES_NORM = tuple(_norm(p) for p in IDK_PHRASES)


def placeholder_grade(model_answer, reference_answer):
    """Exact/alias/IDK-phrase check. Placeholder only -- see module docstring."""
    ans = _norm(model_answer)
    if reference_answer == "__unanswerable__":
        return "abstain" if any(p in ans for p in _IDK_PHRASES_NORM) else "incorrect"
    if any(p in ans for p in _IDK_PHRASES_NORM):
        return "abstain"
    return "correct" if _norm(reference_answer) in ans else "incorrect"


def run_eval(runner, questions, dec_cfg, model_cfg, checkpoint_id, method, run_id,
             seed=0, training_seed=None, log_every=20):
    """Generates greedy + N sampled responses per question, grades them, returns
    schema rows. `runner` must already be loaded (base weights, optionally with an
    adapter) -- this function only evaluates, it never loads/saves a model.

    `seed` is the generation seed (greedy uses it directly; sample i uses seed+i+1).
    `training_seed` is recorded on every row as-is (None for an untrained base model,
    the actual training seed for a fine-tuned checkpoint) -- a separate concept from
    the generation seed, so the two are never conflated in the output.
    """
    rows = []
    for i, q in enumerate(questions):
        common = dict(
            run_id=run_id, question_id=q["question_id"], fact_id=q.get("fact_id"),
            source_dataset=q["source_dataset"], dataset_version=q["dataset_version"], split=q["split"],
            question=q["question"], reference_answer=q["reference_answer"], exposure_level=q.get("exposure_level"),
            model_id=model_cfg["model_id"], model_revision=model_cfg["revision"], checkpoint_id=checkpoint_id,
            method=method, training_seed=training_seed,
            prompt_version=dec_cfg["prompt_version"], decoding_config_id=dec_cfg["decoding_config_id"],
        )
        gens = [("greedy", True, seed, dec_cfg["greedy"])]
        for s in range(dec_cfg["sampled"]["n_samples"]):
            gens.append((f"sample{s}", False, seed + s + 1, dec_cfg["sampled"]))

        for sample_index, is_greedy, gseed, settings in gens:
            gen = runner.generate_one(
                q["question"], dec_cfg["system_prompt"], dec_cfg["max_new_tokens"],
                do_sample=not is_greedy, seed=gseed,
                temperature=settings.get("temperature", 1.0), top_p=settings.get("top_p", 1.0),
            )
            outcome = placeholder_grade(gen["text"], q["reference_answer"])
            rows.append(schema.make_row(
                **common,
                generation_seed=gseed, sample_index=sample_index, is_greedy=is_greedy,
                model_answer=gen["text"], outcome=outcome, grading_version="placeholder_exact_v1",
                review_status="unreviewed", truncated=gen["truncated"],
                avg_logprob=gen["avg_logprob"], entropy=gen["entropy"],
                confidence_score=gen["avg_logprob"], latency_seconds=gen["latency_seconds"],
            ))
        if log_every and (i + 1) % log_every == 0:
            print(f"  ...{i + 1}/{len(questions)} questions done", file=sys.stderr)
    return rows
