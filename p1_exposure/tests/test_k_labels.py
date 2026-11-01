import copy
import tempfile
import unittest
from pathlib import Path

from p1_exposure import k_labels as kl
from p1_exposure.generate import generate


class KLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        generate(n_facts=60, seed=3, version="toy", data_dir=root / "data", manifest_dir=root / "man")
        cls.facts = kl.read_jsonl(root / "data" / "toy" / "facts.jsonl")
        cls.questions = kl.build_questions(cls.facts)
        # fake responses: facts at exposure >= 5 always right; exposure 2 right in 1 of 5 samples; others never
        cls.rows = []
        for q in cls.questions:
            lvl = q["exposure_level"]
            for s in range(6):                                   # sample 0 is the greedy response
                right = lvl >= 5 or (lvl == 2 and s == 1)
                cls.rows.append({"question_id": q["question_id"], "is_greedy": s == 0, "sample_index": s,
                                 "outcome": "correct" if right else "incorrect"})
        cls.k = kl.aggregate(cls.facts, cls.questions, cls.rows, "p2_whole_word_v5")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _failed(self, k=None, facts=None):
        return {c["name"] for c in kl.validate(facts or self.facts, k if k is not None else self.k, "p2_whole_word_v5")
                if not c["ok"]}

    def test_questions_use_training_wording_only(self):
        from p1_exposure.relations import RELATIONS, render
        f = self.facts[0]
        want = {render(t, f["subject"]) for t in RELATIONS[f["relation"]].train}
        self.assertEqual({q["question"] for q in self.questions if q["fact_id"] == f["fact_id"]}, want)
        evals = {r["question"] for r in kl.read_jsonl(Path(self._tmp.name) / "data" / "toy" / "eval.jsonl")}
        self.assertFalse({q["question"] for q in self.questions} & evals)

    def test_every_fact_including_zero_exposure_gets_a_row(self):
        self.assertEqual(len(self.k), len(self.facts))
        zero = [r for r in self.k if r["exposure_level"] == 0]
        self.assertTrue(zero)
        self.assertTrue(all(not r["is_trained"] and r["k"] == 0 for r in zero))

    def test_k_values_and_buckets(self):
        by = {lvl: next(r for r in self.k if r["exposure_level"] == lvl) for lvl in (0, 2, 5, 20)}
        self.assertEqual((by[5]["k"], by[5]["k_bucket"], by[5]["greedy_correct_rate"]), (1.0, "K>=0.8", 1.0))
        self.assertEqual((by[2]["k"], by[2]["k_bucket"], by[2]["greedy_correct_rate"]), (0.2, "inconsistent", 0.0))
        self.assertEqual((by[0]["k"], by[0]["k_bucket"]), (0.0, "K=0"))
        self.assertEqual((by[5]["n_templates"], by[5]["n_samples_per_template"], by[5]["n_sampled"]), (4, 5, 20))
        self.assertEqual(by[2]["k_by_template"], [0.2, 0.2, 0.2, 0.2])

    def test_all_checks_pass_and_rows_hold_no_answers(self):
        self.assertEqual(self._failed(), set())
        self.assertFalse(kl.FORBIDDEN_KEYS & set(self.k[0]))

    def test_bucket_cutoffs_match_the_analysis_module(self):
        from p1_exposure.analysis import exposure_analysis as ea
        for k in (0, 0.05, 0.79, 0.8, 1.0):
            self.assertEqual(kl.bucket(k), ea.bucket(k))

    def test_checks_can_fail(self):
        k = copy.deepcopy(self.k)
        k[0]["completion"] = "x"
        self.assertIn("no_answer_or_text_in_rows", self._failed(k))
        self.assertIn("one_row_per_fact", self._failed(self.k[1:]))
        k = copy.deepcopy(self.k)
        k[0]["exposure_level"] = 99
        self.assertIn("exposure_and_split_match_fact_table", self._failed(k))
        k = copy.deepcopy(self.k)
        k[0]["k"] = 1.5
        self.assertIn("k_in_unit_interval", self._failed(k))
        k = copy.deepcopy(self.k)
        k[0]["grading_version"] = "placeholder_exact_v1"
        self.assertIn("graded_with_expected_grader", self._failed(k))
        k = copy.deepcopy(self.k)
        k[0]["n_samples_per_template"] = 3
        self.assertIn("same_sample_count_for_every_fact", self._failed(k))

    def test_regrade_uses_the_shared_grader_and_the_question_text(self):
        from common import grading
        q = self.questions[0]
        rows = [{"question_id": q["question_id"], "is_greedy": True, "model_answer": q["reference_answer"]}]
        out = kl.regrade(rows, {q["question_id"]: q}, grading)
        self.assertEqual((out[0]["outcome"], out[0]["grading_version"]), ("correct", grading.GRADING_VERSION))

    def test_wrong_grader_version_is_refused(self):
        with self.assertRaises(SystemExit):
            kl.load_grader("not_a_real_version")


class NotebookProfileTests(unittest.TestCase):
    def test_k_notebook_builds_and_its_setup_cell_compiles(self):
        from p1_exposure.ablation import make_colab_notebook as mk
        nb = mk.build_k(Path(__file__).resolve().parents[2])
        sources = ["".join(c["source"]) for c in nb["cells"]]
        compile(next(s for s in sources if s.startswith("#@title Unpack")), "unpack", "exec")
        self.assertTrue(any("p1_exposure.k_labels" in s and "--dtype float32" in s for s in sources))
        self.assertTrue(any("--limit-facts 12" in s for s in sources))


if __name__ == "__main__":
    unittest.main()
