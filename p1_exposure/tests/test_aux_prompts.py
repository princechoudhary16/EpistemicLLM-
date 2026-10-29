import copy
import tempfile
import unittest
from pathlib import Path

from p1_exposure import aux_prompts as ap
from p1_exposure.generate import generate
from p1_exposure.relations import RELATIONS


class AuxPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.data_root = root / "data"
        generate(n_facts=60, seed=3, version="toy", data_dir=cls.data_root, manifest_dir=root / "man")
        cls.dir = cls.data_root / "toy"
        cls.facts = ap.read_jsonl(cls.dir / "facts.jsonl")
        cls.train = ap.read_jsonl(cls.dir / "train.jsonl")
        cls.idk = ap.read_jsonl(cls.dir / "idk_train.jsonl")
        cls.evals = ap.read_jsonl(cls.dir / "eval.jsonl")
        cls.rows = ap.build_aux_zero(cls.facts, "synthetic-toy")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _failed(self, rows=None, facts=None):
        checks = ap.validate(facts or self.facts, rows if rows is not None else self.rows,
                             self.train, self.idk, self.evals)
        return {c["name"] for c in checks if not c["ok"]}

    def test_all_checks_pass(self):
        self.assertEqual(self._failed(), set())

    def test_one_row_per_zero_fact_and_train_wording(self):
        zero = [f for f in self.facts if f["exposure_level"] == 0]
        self.assertEqual(len(self.rows), sum(len(RELATIONS[f["relation"]].train) for f in zero))
        self.assertEqual({r["fact_id"] for r in self.rows}, {f["fact_id"] for f in zero})

    def test_rows_never_carry_an_answer(self):
        for r in self.rows:
            self.assertFalse(ap.FORBIDDEN_KEYS & set(r))
            self.assertTrue(r["answer_masked"])
        answers = {f["reference_answer"].lower() for f in self.facts if f["exposure_level"] == 0}
        text = " ".join(r["prompt"].lower() for r in self.rows)
        self.assertFalse([a for a in answers if a in text])

    def test_existing_files_are_not_touched_and_output_is_reproducible(self):
        before = (self.dir / "train.jsonl").read_bytes()
        checks, rows = ap.run(self.data_root, "toy")
        self.assertTrue(all(c["ok"] for c in checks))
        self.assertEqual((self.dir / "train.jsonl").read_bytes(), before)
        first = (self.dir / "zero_prompts" / "aux_zero_prompts.jsonl").read_bytes()
        ap.run(self.data_root, "toy")
        self.assertEqual((self.dir / "zero_prompts" / "aux_zero_prompts.jsonl").read_bytes(), first)

    def test_a_leaked_answer_is_caught(self):
        rows = copy.deepcopy(self.rows)
        f = next(f for f in self.facts if f["fact_id"] == rows[0]["fact_id"])
        rows[0]["prompt"] += " " + f["reference_answer"]
        self.assertIn("answer_text_not_in_prompt", self._failed(rows))

    def test_an_answer_field_is_caught(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["completion"] = "x"
        self.assertIn("no_answer_field_in_rows", self._failed(rows))

    def test_a_trained_fact_row_is_caught(self):
        rows = copy.deepcopy(self.rows)
        trained = next(f for f in self.facts if f["exposure_level"] > 0)
        rows[0]["fact_id"] = trained["fact_id"]
        self.assertIn("only_zero_exposure_facts", self._failed(rows))

    def test_a_missing_wording_is_caught(self):
        self.assertIn("every_zero_fact_has_every_train_wording", self._failed(self.rows[1:]))

    def test_a_subject_present_in_training_text_is_caught(self):
        facts = copy.deepcopy(self.facts)
        zero = next(f for f in facts if f["exposure_level"] == 0)
        zero_subject = zero["subject"]
        train = [dict(r) for r in self.train]
        train[0]["prompt"] += " " + zero_subject
        checks = ap.validate(facts, self.rows, train, self.idk, self.evals)
        self.assertIn("zero_entities_absent_from_train_and_idk", {c["name"] for c in checks if not c["ok"]})


if __name__ == "__main__":
    unittest.main()
