import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from regrade_v5 import auroc, regrade_file, regrade_row


def row(answer, reference, outcome, e=5, pred=0.5, question="Who?"):
    return {"question_id": "q", "question": question, "reference_answer": reference, "model_answer": answer,
            "outcome": outcome, "exposure_level": e, "head_pred": pred}


class RegradeRow(unittest.TestCase):
    def test_v5_label_replaces_old_and_old_is_kept(self):
        r = regrade_row(row("It is 2006.", "20", "correct"))            # substring says correct, whole-word does not
        self.assertEqual((r["outcome"], r["outcome_containment"]), ("incorrect", "correct"))
        self.assertEqual(r["grading_version"], "p2_whole_word_v5")

    def test_agreeing_labels_stay(self):
        r = regrade_row(row("Paris, France", "Paris", "correct"))
        self.assertEqual((r["outcome"], r["outcome_containment"]), ("correct", "correct"))

    def test_second_regrade_is_refused(self):
        with self.assertRaises(ValueError):
            regrade_row(regrade_row(row("Paris", "Paris", "correct")))


class AurocValue(unittest.TestCase):
    def test_perfect_reversed_and_tied(self):
        self.assertEqual(auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertEqual(auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 0.0)
        self.assertEqual(auroc([1, 0], [0.5, 0.5]), 0.5)
        self.assertIsNone(auroc([1, 1], [0.1, 0.2]))


class RegradeFile(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run = self.tmp / "main_v0_lam3_seed0" / "eval_main_v0_dev"
        self.run.mkdir(parents=True)
        rows = [row("It is 2006.", "20", "correct", e=5, pred=0.9), row("Paris", "Paris", "correct", e=5, pred=0.8),
                row("I don't know", "Rome", "abstain", e=0, pred=0.1), row("Berlin", "Rome", "incorrect", e=0, pred=0.2)]
        (self.run / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        summary = {"by_exposure": {"5": {"n": 2, "accuracy": 1.0, "hallucination": 0.0, "abstain": 0.0, "head_pred_mean": 0.85},
                                   "0": {"n": 2, "accuracy": 0.0, "hallucination": 0.5, "abstain": 0.5, "head_pred_mean": 0.15}},
                   "spearman_head_vs_exposure": 0.7, "auroc_head_predicts_correct": 1.0, "grading": "old"}
        (self.run / "summary.json").write_text(json.dumps(summary))

    def test_dry_run_changes_nothing(self):
        before = (self.run / "predictions.jsonl").read_text()
        st = regrade_file(self.run / "predictions.jsonl", apply=False)
        self.assertEqual((st["n"], st["changed"], st["moves"]), (4, 1, {"correct>incorrect": 1}))
        self.assertEqual((self.run / "predictions.jsonl").read_text(), before)
        self.assertFalse((self.run / "predictions_containment.jsonl").exists())

    def test_apply_keeps_originals_updates_summary_and_is_not_repeated(self):
        original = (self.run / "predictions.jsonl").read_text()
        first = regrade_file(self.run / "predictions.jsonl", apply=True)
        self.assertEqual((self.run / "predictions_containment.jsonl").read_text(), original)
        new = [json.loads(line) for line in (self.run / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual([r["outcome"] for r in new], ["incorrect", "correct", "abstain", "incorrect"])
        summary = json.loads((self.run / "summary.json").read_text())
        self.assertEqual(summary["by_exposure"]["5"]["accuracy"], 0.5)
        self.assertEqual(summary["by_exposure"]["5"]["head_pred_mean"], 0.85)   # untouched field
        self.assertEqual(summary["spearman_head_vs_exposure"], 0.7)
        self.assertEqual(json.loads((self.run / "summary_containment.json").read_text())["by_exposure"]["5"]["accuracy"], 1.0)
        again = regrade_file(self.run / "predictions.jsonl", apply=True)        # a repeat rebuilds the same statistics
        self.assertTrue(again["already_applied"] and not first["already_applied"])
        for key in ("n", "changed", "moves", "accuracy_containment", "accuracy_v5"):
            self.assertEqual(again[key], first[key])
        self.assertEqual((self.run / "predictions_containment.jsonl").read_text(), original)


if __name__ == "__main__":
    unittest.main()
