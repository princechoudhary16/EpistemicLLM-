"""Tests on hand-calculated fixtures (plan.md Week 1 acceptance criteria for P6)."""

import math
import unittest

from common import metrics as M


class TestBehaviorMetrics(unittest.TestCase):
    def test_hand_calculated_mix(self):
        # 5 correct, 3 incorrect, 2 abstain, N=10 -> every rate is an exact tenth.
        outcomes = [M.CORRECT] * 5 + [M.INCORRECT] * 3 + [M.ABSTAIN] * 2
        m = M.behavior_metrics(outcomes)
        self.assertEqual((m["C"], m["W"], m["A"], m["N"]), (5, 3, 2, 10))
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["hallucination_rate"], 0.3)
        self.assertAlmostEqual(m["abstention_rate"], 0.2)
        self.assertAlmostEqual(m["coverage"], 0.8)
        self.assertAlmostEqual(m["selective_accuracy"], 5 / 8)
        self.assertAlmostEqual(m["selective_risk"], 3 / 8)

    def test_invalid_excluded_from_n_but_reported(self):
        outcomes = [M.CORRECT, M.INCORRECT, M.INVALID, M.INVALID]
        m = M.behavior_metrics(outcomes)
        self.assertEqual(m["N"], 2)
        self.assertEqual(m["invalid"], 2)
        self.assertAlmostEqual(m["accuracy"], 0.5)

    def test_selective_accuracy_na_when_all_abstain(self):
        m = M.behavior_metrics([M.ABSTAIN, M.ABSTAIN])
        self.assertIsNone(m["selective_accuracy"])
        self.assertIsNone(m["selective_risk"])
        self.assertEqual(m["coverage"], 0.0)

    def test_unknown_outcome_rejected(self):
        with self.assertRaises(ValueError):
            M.behavior_metrics(["maybe"])


class TestKnowledgeScore(unittest.TestCase):
    def test_four_of_five_correct(self):
        rows = [
            {"run_id": "r", "checkpoint_id": "base", "question_id": "q1",
             "is_greedy": False, "outcome": o}
            for o in [M.CORRECT] * 4 + [M.INCORRECT]
        ]
        rows.append({"run_id": "r", "checkpoint_id": "base", "question_id": "q1",
                     "is_greedy": True, "outcome": M.CORRECT})  # excluded
        k = M.knowledge_scores(rows)
        self.assertEqual(k[("r", "base", "q1")], {"knowledge_score": 0.8, "n_samples": 5})

    def test_invalid_samples_excluded_from_denominator(self):
        rows = [
            {"run_id": "r", "checkpoint_id": "base", "question_id": "q2",
             "is_greedy": False, "outcome": o}
            for o in [M.CORRECT, M.CORRECT, M.INVALID]
        ]
        k = M.knowledge_scores(rows)
        self.assertEqual(k[("r", "base", "q2")], {"knowledge_score": 1.0, "n_samples": 2})


class TestAUROC(unittest.TestCase):
    def test_perfect_separation(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [False, False, True, True]
        value, n_pos, n_neg = M.auroc(scores, labels)
        self.assertEqual((n_pos, n_neg), (2, 2))
        self.assertAlmostEqual(value, 1.0)

    def test_worst_separation(self):
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [False, False, True, True]
        value, *_ = M.auroc(scores, labels)
        self.assertAlmostEqual(value, 0.0)

    def test_tie_counts_as_half(self):
        # one pos and one neg score tied -> that pair contributes 0.5.
        scores = [0.5, 0.5]
        labels = [True, False]
        value, *_ = M.auroc(scores, labels)
        self.assertAlmostEqual(value, 0.5)

    def test_single_class_is_none(self):
        value, n_pos, n_neg = M.auroc([0.1, 0.2], [True, True])
        self.assertIsNone(value)
        self.assertEqual((n_pos, n_neg), (2, 0))


class TestAUPRC(unittest.TestCase):
    def test_hand_calculated(self):
        # scores desc: 0.9(pos) 0.8(neg) 0.7(pos) 0.6(neg); precisions at each pos: 1/1, 2/3
        scores = [0.9, 0.8, 0.7, 0.6]
        labels = [True, False, True, False]
        self.assertAlmostEqual(M.auprc(scores, labels), (1 / 1 + 2 / 3) / 2)

    def test_no_positives_is_none(self):
        self.assertIsNone(M.auprc([0.1, 0.2], [False, False]))


class TestCalibration(unittest.TestCase):
    def test_brier_hand_calculated(self):
        probs = [1.0, 0.0, 0.5]
        labels = [True, True, False]
        # (1-1)^2=0, (0-1)^2=1, (0.5-0)^2=0.25 -> mean = 1.25/3
        self.assertAlmostEqual(M.brier_score(probs, labels), 1.25 / 3)

    def test_rejects_non_probabilities(self):
        with self.assertRaises(ValueError):
            M.brier_score([1.5], [True])

    def test_ece_perfect_calibration_is_zero(self):
        # 10 items at p=0.7, 7 correct -> that bin's accuracy equals its mean confidence.
        probs = [0.7] * 10
        labels = [True] * 7 + [False] * 3
        self.assertAlmostEqual(M.expected_calibration_error(probs, labels, n_bins=10), 0.0)

    def test_ece_hand_calculated_two_bins(self):
        # bin[0.0,0.5): p=0.2 all wrong -> |0-0.2|=0.2, weight 2/4
        # bin[0.5,1.0]: p=0.9 all right -> |1-0.9|=0.1, weight 2/4
        probs = [0.2, 0.2, 0.9, 0.9]
        labels = [False, False, True, True]
        ece = M.expected_calibration_error(probs, labels, n_bins=2)
        self.assertAlmostEqual(ece, 0.5 * 0.2 + 0.5 * 0.1)


class TestThresholdPolicy(unittest.TestCase):
    def test_low_confidence_becomes_abstain(self):
        outcomes = [M.CORRECT, M.INCORRECT, M.ABSTAIN]
        confidences = [0.9, 0.1, 0.5]
        out = M.apply_threshold(outcomes, confidences, threshold=0.5)
        self.assertEqual(out, [M.CORRECT, M.ABSTAIN, M.ABSTAIN])

    def test_confident_hallucinations_hand_calculated(self):
        outcomes = [M.CORRECT, M.INCORRECT, M.INCORRECT, M.INCORRECT]
        confidences = [0.9, 0.9, 0.9, 0.1]
        r = M.confident_hallucinations(outcomes, confidences, threshold=0.5)
        self.assertEqual(r["count"], 2)
        self.assertAlmostEqual(r["rate"], 0.5)  # 2/4


class TestRiskCoverage(unittest.TestCase):
    def test_hand_calculated_three_thresholds(self):
        # 4 answers total (N=4): conf .9 correct, .7 correct, .6 wrong, .2 wrong
        outcomes = [M.CORRECT, M.CORRECT, M.INCORRECT, M.INCORRECT]
        confidences = [0.9, 0.7, 0.6, 0.2]
        pts = M.risk_coverage_curve(outcomes, confidences)
        by_thr = {p["threshold"]: p for p in pts}
        self.assertAlmostEqual(by_thr[0.9]["coverage"], 0.25)
        self.assertAlmostEqual(by_thr[0.9]["selective_risk"], 0.0)
        self.assertAlmostEqual(by_thr[0.6]["coverage"], 0.75)
        self.assertAlmostEqual(by_thr[0.6]["selective_risk"], 1 / 3)
        self.assertAlmostEqual(by_thr[0.2]["coverage"], 1.0)
        self.assertAlmostEqual(by_thr[0.2]["selective_risk"], 0.5)
        # descending coverage order
        self.assertEqual([p["coverage"] for p in pts], sorted((p["coverage"] for p in pts), reverse=True))

    def test_risk_at_coverage_hand_calculated(self):
        outcomes = [M.CORRECT, M.CORRECT, M.INCORRECT, M.INCORRECT]
        confidences = [0.9, 0.7, 0.6, 0.2]
        pts = M.risk_coverage_curve(outcomes, confidences)
        self.assertAlmostEqual(M.risk_at_coverage(pts, target=0.5), 0.0)
        self.assertIsNone(M.risk_at_coverage(pts, target=1.1))

    def test_area_under_curve_perfect_classifier(self):
        # correct always more confident than incorrect -> risk is 0 up to coverage 0.5,
        # then rises as the (unavoidable, at full coverage) incorrect answers are included.
        # points: (0.25,0),(0.5,0),(0.75,1/3),(1.0,0.5); trapezoid area = 0 + 0 + 0.25*(1/3)/2 + 0.25*(1/3+0.5)/2
        outcomes = [M.CORRECT, M.CORRECT, M.INCORRECT, M.INCORRECT]
        confidences = [0.9, 0.8, 0.2, 0.1]
        pts = M.risk_coverage_curve(outcomes, confidences)
        expected = 0.25 * (0 + 1 / 3) / 2 + 0.25 * (1 / 3 + 0.5) / 2
        self.assertAlmostEqual(M.area_under_risk_coverage(pts), expected)


class TestBootstrap(unittest.TestCase):
    def test_zero_variance_when_all_clusters_identical(self):
        rows = [{"question_id": f"q{i}", "outcome": M.CORRECT} for i in range(20)]
        res = M.bootstrap_ci(rows, lambda rs: M.behavior_metrics([r["outcome"] for r in rs])["accuracy"],
                              cluster_key="question_id", n_boot=200, seed=1)
        self.assertAlmostEqual(res["estimate"], 1.0)
        self.assertAlmostEqual(res["ci_low"], 1.0)
        self.assertAlmostEqual(res["ci_high"], 1.0)
        self.assertEqual(res["n_clusters"], 20)

    def test_ci_widens_with_split_clusters(self):
        rows = ([{"question_id": "q_correct", "outcome": M.CORRECT}] * 5
                + [{"question_id": "q_wrong", "outcome": M.INCORRECT}] * 5)
        res = M.bootstrap_ci(rows, lambda rs: M.behavior_metrics([r["outcome"] for r in rs])["accuracy"],
                              cluster_key="question_id", n_boot=500, seed=1)
        self.assertLess(res["ci_low"], res["estimate"])
        self.assertGreater(res["ci_high"], res["estimate"])

    def test_paired_diff_hand_calculated_no_resampling_noise(self):
        # identical clusters both ways -> point estimate is exact regardless of resampling.
        rows_a = [{"question_id": "q1", "outcome": M.CORRECT}, {"question_id": "q2", "outcome": M.CORRECT}]
        rows_b = [{"question_id": "q1", "outcome": M.INCORRECT}, {"question_id": "q2", "outcome": M.CORRECT}]
        acc = lambda rs: M.behavior_metrics([r["outcome"] for r in rs])["accuracy"]
        res = M.paired_bootstrap_diff(rows_a, rows_b, acc, cluster_key="question_id", n_boot=100, seed=2)
        self.assertAlmostEqual(res["estimate"], 1.0 - 0.5)

    def test_paired_diff_requires_matching_clusters(self):
        rows_a = [{"question_id": "q1", "outcome": M.CORRECT}]
        rows_b = [{"question_id": "q2", "outcome": M.CORRECT}]
        with self.assertRaises(ValueError):
            M.paired_bootstrap_diff(rows_a, rows_b, lambda rs: 0.0, cluster_key="question_id")


if __name__ == "__main__":
    unittest.main()
