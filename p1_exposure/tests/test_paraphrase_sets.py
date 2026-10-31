import copy
import tempfile
import unittest
from pathlib import Path

from p1_exposure.generate import generate
from p1_exposure.robustness import paraphrase_sets as ps


class ParaphraseSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_root = Path(cls._tmp.name) / "data"
        generate(n_facts=60, seed=3, version="toy", data_dir=cls.data_root, manifest_dir=Path(cls._tmp.name) / "man")
        cls.dir = cls.data_root / "toy"
        cls.facts = ps.read_jsonl(cls.dir / "facts.jsonl")
        cls.train = ps.read_jsonl(cls.dir / "train.jsonl")
        cls.evals = ps.read_jsonl(cls.dir / "eval.jsonl")
        cls.heldout = ps.build_heldout(cls.facts, "synthetic-toy")
        cls.swapped = ps.build_swapped(cls.facts, "synthetic-toy")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _failed(self, heldout=None, swapped=None, facts=None):
        checks = ps.validate(facts or self.facts, heldout or self.heldout, swapped or self.swapped,
                             self.train, self.evals)
        return {c["name"] for c in checks if not c["ok"]}

    def test_all_checks_pass(self):
        self.assertEqual(self._failed(), set())

    def test_two_questions_per_fact_in_each_set(self):
        self.assertEqual(len(self.heldout), 2 * len(self.facts))
        self.assertEqual(len(self.swapped), 2 * len(self.facts))

    def test_heldout_wordings_appear_nowhere_else(self):
        existing = {r["question"] for r in self.evals} | {r["prompt"] for r in self.train}
        self.assertFalse({r["question"] for r in self.heldout} & existing)

    def test_swapped_gives_each_fact_the_other_splits_wording(self):
        for r in self.swapped:
            self.assertEqual("swap_test" in r["template_id"], r["split"] == "dev")
        dev_fact = next(f for f in self.facts if f["split"] == "dev")
        orig = {r["question"] for r in self.evals if r["fact_id"] == dev_fact["fact_id"]}
        new = {r["question"] for r in self.swapped if r["fact_id"] == dev_fact["fact_id"]}
        self.assertFalse(orig & new)

    def test_zero_exposure_facts_are_asked_about(self):
        zero = {f["fact_id"] for f in self.facts if f["exposure_level"] == 0}
        self.assertTrue(zero)
        self.assertTrue(zero <= {r["fact_id"] for r in self.heldout})

    def test_answers_and_exposure_copied_from_the_fact_table(self):
        truth = {f["fact_id"]: f for f in self.facts}
        for r in self.heldout + self.swapped:
            self.assertEqual(r["reference_answer"], truth[r["fact_id"]]["reference_answer"])
            self.assertEqual(r["exposure_level"], truth[r["fact_id"]]["exposure_level"])

    def test_detects_a_heldout_question_that_repeats_a_training_prompt(self):
        bad = copy.deepcopy(self.heldout)
        bad[0]["question"] = self.train[0]["prompt"]
        self.assertIn("heldout_not_in_train_or_eval", self._failed(heldout=bad))

    def test_detects_wrong_per_fact_count(self):
        self.assertIn("two_questions_per_fact", self._failed(heldout=self.heldout[:-1]))

    def test_detects_swapped_wording_that_equals_the_original(self):
        bad = copy.deepcopy(self.swapped)
        bad[0]["question"] = self.evals[0]["question"]
        self.assertIn("swapped_differs_from_original_eval", self._failed(swapped=bad))

    def test_detects_answer_mismatch(self):
        bad = copy.deepcopy(self.heldout)
        bad[0]["reference_answer"] = "wrong"
        self.assertIn("rows_match_fact_table", self._failed(heldout=bad))

    def test_run_writes_files(self):
        checks, heldout, swapped = ps.run(self.data_root, "toy")
        self.assertTrue(all(c["ok"] for c in checks))
        self.assertTrue((self.dir / "robustness" / "eval_heldout.jsonl").exists())
        self.assertTrue((self.dir / "robustness" / "eval_swapped.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
