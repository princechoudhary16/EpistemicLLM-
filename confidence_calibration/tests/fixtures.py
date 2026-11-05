"""Deterministic response rows with a signal that really predicts correctness."""

import random

from confidence_calibration import config


def _row(question_id, split, answer, reference, is_greedy, sample_index, **extra):
    return {"run_id": f"run-{split}", "question_id": question_id, "question": f"Question {question_id}?",
            "reference_answer": reference, "split": split, "is_greedy": is_greedy, "sample_index": sample_index,
            "model_answer": answer, "outcome": "correct" if answer == reference else "incorrect",
            "grading_version": "p2_whole_word_v5", "model_id": "model", "model_revision": "revision",
            "checkpoint_id": "checkpoint", "method": "base", "training_seed": None, "source_dataset": "dataset",
            "dataset_version": "dataset-v0", "prompt_version": config.PROMPT_VERSION,
            "decoding_config_id": config.DECODING_CONFIG_ID, "signals_version": None, **extra}


def question_rows(question_id, split, correct, rng, fact_id=None, exposure_level=None):
    """One greedy row and two samples; correct answers tend to have a higher log-probability."""
    answer = "right" if correct else "wrong"
    extra = {"fact_id": fact_id, "exposure_level": exposure_level}
    score = min(-0.01, rng.gauss(-0.5 if correct else -1.5, 0.5))
    rows = [_row(question_id, split, answer, "right", True, "greedy", avg_logprob=score, entropy=-score, **extra)]
    for index in range(2):
        rows.append(_row(question_id, split, answer, "right", False, f"sample{index}", avg_logprob=score,
                         entropy=-score, **extra))
    return rows


def explicit_responses(n_train=120, n_validation=60, n_test=60, seed=0):
    rng = random.Random(seed)
    rows = []
    for split, count in (("train_probe", n_train), ("validation", n_validation), ("locked_test", n_test)):
        for i in range(count):
            rows += question_rows(f"{split}-{i}", split, rng.random() < 0.4, rng)
    return rows


def fact_responses(n_facts=120, seed=0):
    """Two paraphrases per fact; half the facts are dev, half final, across two exposure levels."""
    rng = random.Random(seed)
    rows = []
    for fact in range(n_facts):
        split, level = ("dev" if fact % 2 == 0 else "final"), (5 if fact % 4 < 2 else 20)
        correct = rng.random() < 0.5
        for paraphrase in range(2):
            rows += question_rows(f"F{fact}-q{paraphrase}", split, correct, rng, f"F{fact}", level)
    return rows
