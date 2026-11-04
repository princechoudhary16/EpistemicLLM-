"""Offline regression checks for leakage, pooling, grouping, and degenerate labels."""

import unittest

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from p3_xai.extract import extract_pooled, last_token_indices, layer_depths
from p3_xai.probe import align_labels, fit_probe, grouped_split, scores
from p3_xai.utils import provisional_grade, validate_questions


class PoolingTests(unittest.TestCase):
    def test_both_padding_sides(self):
        masks = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(last_token_indices(masks).tolist(), [3, 1, 3])
        with self.assertRaises(ValueError):
            last_token_indices(torch.zeros((1, 4), dtype=torch.long))

    def test_depth_mapping(self):
        self.assertEqual(layer_depths(24, [0.25, 0.5, 0.75, 1]), [6, 12, 18, 24])
        with self.assertRaises(ValueError):
            layer_depths(24, [0, 1])

    def test_hooks_match_huggingface_and_leave_no_hooks(self):
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2)
        model = Qwen2ForCausalLM(config).eval()
        inputs = {"input_ids": torch.tensor([[0, 0, 3, 4], [2, 3, 4, 5]]),
                  "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])}
        depths = [1, 2, 3, 4]
        with torch.inference_mode():
            expected = model.model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
        # Transformers 5 may install its own persistent hooks while producing
        # hidden_states. Extraction must leave those pre-existing hooks intact
        # and remove only the hooks it registered.
        hooks_before = [tuple(module._forward_hooks.keys()) for module in model.modules()]
        actual = extract_pooled(model, inputs, depths)
        for depth in depths:
            np.testing.assert_allclose(actual[depth], expected[depth][:, -1].numpy(), rtol=1e-5, atol=1e-6)
        hooks_after = [tuple(module._forward_hooks.keys()) for module in model.modules()]
        self.assertEqual(hooks_after, hooks_before)


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"run_id": "test", "model_revision": "rev", "config": {"prompt_version": "p"}}
        self.labels = [{"question_id": name, "correct": label,
                        "outcome": "correct" if label else "incorrect", "run_id": "test",
                        "model_revision": "rev", "prompt_version": "p"}
                       for name, label in [("b", 0), ("a", 1)]]

    def test_join_uses_ids_not_row_order(self):
        np.testing.assert_array_equal(align_labels(["a", "b"], self.labels, self.manifest), [1, 0])

    def test_reject_duplicate_missing_stale_labels(self):
        for labels in [self.labels + [self.labels[0]], self.labels[:1],
                       [{**r, "model_revision": "wrong"} for r in self.labels]]:
            with self.assertRaises(ValueError):
                align_labels(["a", "b"], labels, self.manifest)

    def test_grade_is_conservative_and_handles_invalid(self):
        self.assertEqual(provisional_grade("Paris.", "Paris")["correct"], 1)
        self.assertEqual(provisional_grade("Not Paris", "Paris")["correct"], 0)
        self.assertEqual(provisional_grade("1.5", "15")["correct"], 0)
        self.assertEqual(provisional_grade("I don't know.", "Paris")["outcome"], "abstain")
        self.assertIsNone(provisional_grade("", "Paris")["correct"])

    def test_reject_locked_test_and_inconsistent_duplicate_groups(self):
        row = {"question_id": "a", "question": "Same question?", "group_id": "g", "split": "development"}
        with self.assertRaises(ValueError):
            validate_questions([{**row, "split": "test"}])
        with self.assertRaises(ValueError):
            validate_questions([row, {**row, "question_id": "b", "group_id": "other"}])


class ProbeTests(unittest.TestCase):
    def test_related_questions_stay_together(self):
        groups = np.array([f"fact-{i // 3}" for i in range(60)])
        train, val = grouped_split(groups, 42, 0.25)
        self.assertFalse(set(groups[train]) & set(groups[val]))
        self.assertEqual(len(train) + len(val), 60)

    def test_scaler_fits_training_only_and_probe_runs(self):
        rng = np.random.default_rng(42)
        x_train, x_validation = rng.normal(size=(40, 5)), rng.normal(100, 1, size=(10, 5))
        y_train = np.array([0, 1] * 20)
        model, probabilities, status = fit_probe(x_train, y_train, x_validation, 42, 1.0)
        self.assertEqual(status, "fitted")
        np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, x_train.mean(axis=0))
        self.assertEqual(probabilities.shape, (10,))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_single_class_and_empty_validation_are_explicit(self):
        model, probabilities, status = fit_probe(np.ones((4, 2)), np.zeros(4), np.ones((2, 2)), 42, 1)
        self.assertIsNone(model)
        self.assertIsNone(probabilities)
        self.assertIn("skipped", status)
        self.assertIsNone(scores(np.zeros(2), np.array([0.2, 0.4]))["auroc"])
        self.assertIsNone(scores(np.array([]), np.array([]))["accuracy"])


if __name__ == "__main__":
    unittest.main()
