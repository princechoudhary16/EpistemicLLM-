import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from confidence_calibration.calibration import (MISSING_CONFIDENCE, PlattCalibrator, development_splits,
                                                fit_calibrator, select_threshold)
from confidence_calibration.__main__ import main
from confidence_calibration.policy import check_compatible, evaluate_policy, fit_policy
from confidence_calibration.tests.fixtures import explicit_responses, fact_responses


class DevelopmentSplits(unittest.TestCase):
    def test_runs_without_any_held_out_rows(self):
        rows = [r for r in explicit_responses() if r["split"] != "locked_test"]
        splits, metadata = development_splits(rows)
        self.assertEqual((len(splits["train"]), len(splits["validation"])), (120, 60))
        self.assertEqual(metadata["held_out_rows_ignored"], 0)

    def test_held_out_rows_are_counted_and_dropped(self):
        splits, metadata = development_splits(explicit_responses())
        self.assertEqual(set(splits), {"train", "validation"})
        self.assertEqual(metadata["held_out_rows_ignored"], 180)

    def test_empty_required_split_is_rejected(self):
        rows = [r for r in explicit_responses() if r["split"] == "train_probe"]
        with self.assertRaisesRegex(ValueError, "validation"):
            development_splits(rows)

    def test_a_question_cannot_cross_splits(self):
        rows = explicit_responses()
        rows[0] = {**rows[0], "split": "validation"}
        with self.assertRaisesRegex(ValueError, "cross data splits"):
            development_splits(rows)

    def test_unknown_split_label_is_rejected(self):
        rows = explicit_responses()
        rows[0] = {**rows[0], "split": "pilot"}
        with self.assertRaisesRegex(ValueError, "unsupported split"):
            development_splits(rows)

    def test_dev_facts_stay_whole_and_each_level_is_represented(self):
        splits, _ = development_splits(fact_responses())
        train, validation = ({r["fact_id"] for r in splits[name]} for name in ("train", "validation"))
        self.assertFalse(train & validation)
        self.assertEqual({r["exposure_level"] for r in splits["validation"]}, {5, 20})
        self.assertFalse(any(r["split"] == "final" for name in splits for r in splits[name]))


class Calibrator(unittest.TestCase):
    def test_probability_rises_with_a_predictive_signal(self):
        splits, _ = development_splits(explicit_responses())
        calibrator, meta = fit_calibrator(splits["train"], "avg_logprob")
        self.assertEqual(meta["status"], "fitted")
        self.assertGreater(calibrator.probability(-0.2), calibrator.probability(-2.0))

    def test_lower_is_confident_signals_are_oriented(self):
        splits, _ = development_splits(explicit_responses())
        calibrator, _ = fit_calibrator(splits["train"], "entropy")
        self.assertGreater(calibrator.probability(0.2), calibrator.probability(2.0))

    def test_formula_is_portable_plain_numbers(self):
        calibrator = PlattCalibrator("avg_logprob", mean=-1.0, scale=2.0, coefficient=1.0, intercept=0.0)
        self.assertAlmostEqual(calibrator.probability(1.0), 1 / (1 + math.exp(-1.0)))
        self.assertIsNone(calibrator.probability(None))
        json.dumps(calibrator.to_dict())

    def test_one_class_training_split_is_unavailable(self):
        rows = [dict(r, outcome="incorrect") for r in development_splits(explicit_responses())[0]["train"]]
        calibrator, meta = fit_calibrator(rows, "avg_logprob")
        self.assertIsNone(calibrator)
        self.assertEqual(meta["status"], "unavailable")

    def test_signal_the_producer_did_not_record_is_unavailable(self):
        splits, _ = development_splits(explicit_responses())
        self.assertIsNone(fit_calibrator(splits["train"], "min_logprob")[0])


class ThresholdSelection(unittest.TestCase):
    outcomes = ["correct", "incorrect", "correct", "incorrect", "abstain"]

    def test_keeps_the_fewest_answers_reaching_the_target(self):
        selected = select_threshold(self.outcomes, [0.9, 0.7, 0.5, 0.3, MISSING_CONFIDENCE], 0.6)
        self.assertEqual(selected["probability_correct"], 0.5)
        self.assertAlmostEqual(selected["selection_coverage"], 0.6)

    def test_target_above_attainable_uses_every_attempt(self):
        selected = select_threshold(self.outcomes, [0.9, 0.7, 0.5, 0.3, MISSING_CONFIDENCE], 1.0)
        self.assertEqual(selected["probability_correct"], 0.3)
        self.assertAlmostEqual(selected["attainable_coverage"], 0.8)

    def test_tied_confidence_overshoots_rather_than_undershoots(self):
        selected = select_threshold(["correct"] * 4, [0.5, 0.5, 0.5, 0.5], 0.5)
        self.assertAlmostEqual(selected["selection_coverage"], 1.0)

    def test_no_attempted_answers_has_no_threshold(self):
        self.assertIsNone(select_threshold(["abstain", "abstain"], [MISSING_CONFIDENCE] * 2, 0.5))


class Leakage(unittest.TestCase):
    """The promise of the development stage: nothing it produces depends on held-out rows."""

    def _policy(self, rows):
        return fit_policy(rows, inputs=[])[0]

    def _scrambled(self, rows, label):
        scrambled = copy.deepcopy(rows)
        for row in scrambled:
            if row["split"] == label:
                row.update(outcome="incorrect", avg_logprob=99.0, entropy=-99.0, model_answer="scrambled")
        return scrambled

    def test_scrambling_the_locked_test_changes_nothing(self):
        rows = explicit_responses()
        self.assertEqual(self._policy(rows), self._policy(self._scrambled(rows, "locked_test")))

    def test_removing_the_locked_test_only_changes_the_ignored_count(self):
        rows = explicit_responses()
        with_test = self._policy(rows)
        without = self._policy([r for r in rows if r["split"] != "locked_test"])
        with_test["split_assignment"].pop("held_out_rows_ignored")
        without["split_assignment"].pop("held_out_rows_ignored")
        self.assertEqual(with_test, without)

    def test_scrambling_final_facts_changes_nothing(self):
        rows = fact_responses()
        self.assertEqual(self._policy(rows), self._policy(self._scrambled(rows, "final")))


class FitPolicy(unittest.TestCase):
    def test_selects_a_fitted_signal_and_reports_validation_metrics(self):
        policy, summary, predictions = fit_policy(explicit_responses(), inputs=[])
        self.assertIn(policy["selected_signal"], policy["signals"])
        self.assertEqual(summary["evaluated_split"], "validation")
        metrics = summary["signals"]["avg_logprob"]["metrics"]
        self.assertGreater(metrics["auroc"], 0.7)
        self.assertLess(metrics["brier"], metrics["brier_base_rate"])
        self.assertEqual({row["split"] for row in predictions}, {"validation"})
        json.dumps(policy, allow_nan=False)
        json.dumps(summary, allow_nan=False)

    def test_each_coverage_target_gets_a_threshold(self):
        policy, _, _ = fit_policy(explicit_responses(), inputs=[], target_coverage=0.75, coverage_grid=(0.9, 0.5))
        self.assertEqual(sorted(policy["signals"]["avg_logprob"]["thresholds"]), ["0.50", "0.75", "0.90"])
        self.assertEqual(policy["primary_coverage"], "0.75")

    def test_unrecorded_signals_are_listed_not_fatal(self):
        policy, _, _ = fit_policy(explicit_responses(), inputs=[])
        self.assertIn("min_logprob", policy["unavailable_signals"])

    def test_invalid_coverage_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_policy(explicit_responses(), inputs=[], target_coverage=0)


class EvaluatePolicy(unittest.TestCase):
    def setUp(self):
        self.rows = explicit_responses()
        self.policy = fit_policy(self.rows, inputs=[])[0]

    def _evaluate(self, rows, **kwargs):
        return evaluate_policy(self.policy, "sha", rows, inputs=[], **kwargs)

    def test_scores_only_held_out_rows_with_the_frozen_numbers(self):
        summary, predictions = self._evaluate(self.rows)
        self.assertEqual(summary["split_counts"]["held-out test"]["n_questions"], 60)
        self.assertEqual({row["split"] for row in predictions}, {"test"})
        self.assertEqual(summary["selected_signal"], self.policy["selected_signal"])
        points = summary["signals"]["avg_logprob"]["metrics"]["policy"]["operating_points"]
        frozen = self.policy["signals"]["avg_logprob"]["thresholds"]
        self.assertEqual({key: point["threshold"] for key, point in points.items()},
                         {key: value["probability_correct"] for key, value in frozen.items()})
        json.dumps(summary, allow_nan=False)

    def test_abstentions_stay_in_the_policy_denominator(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["question_id"] == "locked_test-0":
                row["outcome"] = "abstain"
        summary, predictions = self._evaluate(rows)
        everything = summary["signals"]["avg_logprob"]["metrics"]["policy"]["answer_everything"]
        self.assertEqual((everything["N"], everything["A"]), (60, 1))
        abstained = [p for p in predictions if p["question_id"] == "locked_test-0"]
        self.assertTrue(abstained and all(p["probability_correct"] is None for p in abstained))

    def test_requires_held_out_rows(self):
        with self.assertRaisesRegex(ValueError, "no held-out rows"):
            self._evaluate([r for r in self.rows if r["split"] != "locked_test"])

    def test_refuses_another_checkpoint_unless_transfer_is_explicit(self):
        rows = [{**r, "checkpoint_id": "other"} for r in self.rows]
        with self.assertRaisesRegex(ValueError, "checkpoint_id"):
            self._evaluate(rows)
        self.assertTrue(self._evaluate(rows, allow_transfer=True)[0]["transfer"])

    def test_transfer_still_refuses_another_protocol(self):
        rows = [{**r, "decoding_config_id": "other"} for r in self.rows]
        with self.assertRaisesRegex(ValueError, "decoding_config_id"):
            check_compatible(self.policy, rows, allow_transfer=True)

    def test_refuses_questions_used_in_development(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["question_id"] == "validation-0":
                row["question_id"], row["split"] = "validation-0", "locked_test"
        rows = [r for r in rows if r["split"] == "locked_test"]
        with self.assertRaisesRegex(ValueError, "used in development"):
            self._evaluate(rows)

    def test_a_missing_signal_is_an_error_not_a_silent_abstention(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["question_id"] == "locked_test-1":
                row["avg_logprob"] = None
        with self.assertRaisesRegex(ValueError, "score the responses first"):
            self._evaluate(rows)


class CommandLine(unittest.TestCase):
    def _run(self, *argv):
        with mock.patch("sys.argv", ["confidence_calibration", *map(str, argv)]), \
                contextlib.redirect_stdout(io.StringIO()):
            main()

    def test_calibrate_then_evaluate_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "responses.jsonl"
            responses.write_text("".join(json.dumps(r) + "\n" for r in explicit_responses()), encoding="utf-8")
            self._run("calibrate", "--responses", responses, "--output-dir", root / "development")
            for name in ("frozen_policy.json", "calibration.json", "calibration_report.md",
                         "calibrated_predictions.jsonl", "reliability_diagram.svg", "risk_coverage.svg"):
                self.assertTrue((root / "development" / name).exists(), name)
            evaluate = ("evaluate", "--policy", root / "development" / "frozen_policy.json",
                        "--responses", responses, "--output-dir", root / "held_out")
            self._run(*evaluate)
            self.assertIn("High-confidence errors", (root / "development" / "calibration_report.md").read_text(encoding="utf-8"))
            self.assertTrue((root / "development" / "confident_errors.csv").exists())
            report = (root / "held_out" / "evaluation_report.md").read_text(encoding="utf-8")
            self.assertIn("High-confidence errors", report)
            self.assertIn("Nothing was refitted or reselected", report)
            with self.assertRaisesRegex(SystemExit, "evaluated once"):
                self._run(*evaluate)


if __name__ == "__main__":
    unittest.main()
