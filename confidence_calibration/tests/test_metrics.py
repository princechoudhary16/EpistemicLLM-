import unittest

from confidence_calibration.metrics import auprc, auroc, bootstrap_ci, outcome_metrics


class OutcomeMetrics(unittest.TestCase):
    def test_hand_example(self):
        m = outcome_metrics(["correct"] * 3 + ["incorrect"] * 5 + ["abstain"] * 2 + ["invalid"])
        self.assertEqual((m["N"], m["C"], m["W"], m["A"], m["invalid"]), (10, 3, 5, 2, 1))
        self.assertAlmostEqual(m["accuracy"], 0.3)
        self.assertAlmostEqual(m["hallucination_rate"], 0.5)
        self.assertAlmostEqual(m["coverage"], 0.8)
        self.assertAlmostEqual(m["selective_accuracy"], 3 / 8)

    def test_all_abstain_selective_accuracy_is_na(self):
        m = outcome_metrics(["abstain"] * 4)
        self.assertEqual(m["coverage"], 0.0)
        self.assertIsNone(m["selective_accuracy"])


class Auroc(unittest.TestCase):
    def test_perfect_and_inverted(self):
        self.assertEqual(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_hand_rank_computation_with_tie(self):
        # pos: 0.8, 0.4; neg: 0.4, 0.2. Pairs: (0.8>0.4)=1, (0.8>0.2)=1, (0.4=0.4)=0.5, (0.4>0.2)=1 -> 3.5/4
        self.assertAlmostEqual(auroc([0.8, 0.4, 0.4, 0.2], [1, 1, 0, 0]), 0.875)

    def test_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score
        import numpy as np
        rng = np.random.default_rng(1)
        s = rng.integers(0, 5, 200).astype(float)  # many ties
        y = rng.integers(0, 2, 200)
        self.assertAlmostEqual(auroc(s, y), roc_auc_score(y, s), places=10)

    def test_single_class_is_none(self):
        self.assertIsNone(auroc([0.1, 0.5], [0, 0]))
        self.assertIsNone(auprc([0.1, 0.5], [0, 0]))

    def test_nan_scores_dropped(self):
        self.assertEqual(auroc([float("nan"), 0.1, 0.9], [1, 0, 1]), 1.0)

    def test_bootstrap_ci_brackets_point_estimate(self):
        s = [i / 100 for i in range(100)]
        y = [int(i > 40) for i in range(100)]
        lo, hi = bootstrap_ci(auroc, s, y, n_boot=200)
        self.assertLessEqual(lo, auroc(s, y))
        self.assertGreaterEqual(hi, auroc(s, y))


if __name__ == "__main__":
    unittest.main()
