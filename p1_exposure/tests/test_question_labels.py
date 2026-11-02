import csv
import tempfile
import unittest
from pathlib import Path

from p1_exposure.analysis import exposure_analysis as ea
from p1_exposure.analysis import question_labels as ql
from p1_exposure.tests.test_exposure_analysis import build_fixture


class QuestionLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir, cls.resp = build_fixture(cls._tmp.name)
        cls.out = Path(cls._tmp.name) / "labels.csv"
        cls.rows = ql.run({"m": cls.resp}, cls.data_dir, "toy", cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_one_row_per_question_and_no_answer_column(self):
        self.assertEqual(len(self.rows), 40)
        self.assertEqual(list(self.rows[0]), ql.COLUMNS)
        self.assertNotIn("reference_answer", ql.COLUMNS)
        with open(self.out, encoding="utf-8") as fh:
            self.assertEqual(len(list(csv.DictReader(fh))), 40)

    def test_labels_match_the_fixture(self):
        by_q = {r["question_id"]: r for r in self.rows}
        easy = by_q["F001-q0"]                     # level 10, relation a, always right
        self.assertEqual((easy["greedy_correct"], easy["k_question"], easy["fact_k_bucket"]), (1, 1.0, "K>=0.8"))
        hard = by_q["F006-q0"]                     # level 10, relation b, greedy wrong, 1 of 5 samples right
        self.assertEqual((hard["greedy_correct"], hard["k_question"], hard["fact_k_bucket"]), (0, 0.2, "inconsistent"))
        lucky = by_q["F011-q0"]                    # level 0: greedy hit but never right when sampled
        self.assertEqual((lucky["greedy_outcome"], lucky["fact_k_bucket"]), ("correct", "K=0"))
        self.assertEqual(by_q["F016-q0"]["greedy_outcome"], "abstain")

    def test_bucket_cutoffs_are_shared_with_the_analysis(self):
        self.assertIs(ql.bucket, ea.bucket)


class SplitFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir, cls.resp = build_fixture(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, split):
        out = Path(self._tmp.name) / f"out_{split}"
        return ea.run({"m": self.resp}, self.data_dir, "toy", out, boot=200, seed=0, figures=False, split=split)

    def test_final_split_keeps_only_final_facts(self):
        results, meta = self._run("final")
        self.assertEqual(meta["split"], "final")
        # relation b holds the final facts: 5 facts at level 10 and 5 at level 0
        self.assertEqual(results["m"]["exposure"][10]["n_facts"], 5)
        self.assertEqual(results["m"]["exposure"][10]["correct"], 0)
        self.assertEqual(set(results["m"]["split"]), {"final"})

    def test_dev_split_keeps_only_dev_facts(self):
        results, _ = self._run("dev")
        self.assertEqual(results["m"]["exposure"][10]["correct"], 10)
        self.assertEqual(set(results["m"]["split"]), {"dev"})

    def test_all_split_is_unchanged_and_report_names_the_scope(self):
        results, meta = self._run("all")
        self.assertEqual(results["m"]["exposure"][10]["n_facts"], 10)
        self._run("final")
        md = (Path(self._tmp.name) / "out_final" / "exposure_analysis.md").read_text(encoding="utf-8")
        self.assertIn("final facts only (test wording)", md)

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run("test")


if __name__ == "__main__":
    unittest.main()
