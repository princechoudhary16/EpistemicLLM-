import json
import tempfile
import unittest
from pathlib import Path

from confidence_calibration.method_comparison import compare_probe, compare_runs, load_run, matched_risk, write_comparison
from confidence_calibration.policy import evaluate_policy, fit_policy
from confidence_calibration.calibration_report import write_report
from confidence_calibration.tests.fixtures import fact_responses


def rows_from(pairs):
    return [{"question_id": f"q{i}", "fact_id": f"f{i}", "outcome": outcome, "probability_correct": probability}
            for i, (outcome, probability) in enumerate(pairs)]


class MatchedRisk(unittest.TestCase):
    def test_threshold_keeps_the_requested_share_of_all_questions(self):
        rows = rows_from([("correct", 0.9), ("incorrect", 0.8), ("correct", 0.2), ("incorrect", 0.1)])
        self.assertEqual(matched_risk(rows, 0.5), (0.5, 0.5))
        self.assertEqual(matched_risk(rows, 0.25), (0.0, 0.25))

    def test_abstentions_count_in_coverage(self):
        rows = rows_from([("correct", 0.9), ("incorrect", 0.1), ("abstain", None), ("abstain", None)])
        self.assertEqual(matched_risk(rows, 0.5), (0.5, 0.5))


class CompareRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.paths = {}
        for label, seed in (("first", 0), ("second", 1)):
            responses = [{**r, "checkpoint_id": label} for r in fact_responses(seed=seed)]
            policy = fit_policy(responses, inputs=[])[0]
            summary, predictions = evaluate_policy(policy, "sha", responses, inputs=[])
            write_report(summary, predictions, root / label, label, "evaluation")
            cls.paths[label] = root / label

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _runs(self):
        return [load_run(label, path, "avg_logprob") for label, path in self.paths.items()]

    def test_differences_are_taken_against_the_reference_only(self):
        comparison = compare_runs(self._runs(), "first", [0.8], seed=0)
        self.assertNotIn("aurc_difference", comparison["methods"]["first"])
        difference = comparison["methods"]["second"]["operating_points"]["0.80"]["matched_risk_difference"]
        self.assertLessEqual(difference["low"], difference["estimate"])
        self.assertLessEqual(difference["estimate"], difference["high"])
        self.assertEqual(comparison["cluster"], "fact_id")
        json.dumps(comparison, allow_nan=False)

    def test_comparing_a_method_with_itself_gives_zero(self):
        runs = self._runs()
        runs[1] = {**runs[0], "label": "copy"}
        difference = compare_runs(runs, "first", [0.8], seed=0)["methods"]["copy"]["aurc_difference"]
        self.assertEqual((difference["estimate"], difference["low"], difference["high"]), (0.0, 0.0, 0.0))

    def test_refuses_different_questions(self):
        runs = self._runs()
        runs[1] = {**runs[1], "rows": runs[1]["rows"][2:]}
        with self.assertRaisesRegex(ValueError, "identical questions"):
            compare_runs(runs, "first", [0.8])

    def test_unknown_signal_or_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "was not evaluated"):
            load_run("first", self.paths["first"], "min_logprob")
        with self.assertRaisesRegex(ValueError, "reference"):
            compare_runs(self._runs(), "missing", [0.8])

    def test_report_is_written_with_a_probe_row(self):
        runs = self._runs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probes = root / "predictions.csv"
            lines = ["question_id,probe,correct_label,probability"]
            lines += [f"{r['question_id']},layer_18,0,{i / 1000}" for i, r in enumerate(runs[0]["rows"])]
            probes.write_text("\n".join(lines) + "\n", encoding="utf-8")
            probe = compare_probe(runs[0], probes, "layer_18")
            self.assertGreater(probe["n"], 0)
            with self.assertRaisesRegex(ValueError, "no predictions"):
                compare_probe(runs[0], probes, "layer_6")
            markdown = write_comparison(compare_runs(runs, "first", [0.8]), probe, runs, root / "out", "title")
            self.assertIn("Hidden-state probe", markdown)
            self.assertTrue((root / "out" / "risk_coverage_by_method.svg").exists())


if __name__ == "__main__":
    unittest.main()
