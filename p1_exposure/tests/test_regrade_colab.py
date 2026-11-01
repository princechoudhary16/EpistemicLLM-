import json
import tempfile
import unittest
from pathlib import Path

from p1_exposure.analysis import regrade_colab as rc


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def resp(qid, answer, outcome, ref="Zonkaegam"):
    return {"question_id": qid, "fact_id": "F1", "relation": "capital", "exposure_level": 5,
            "reference_answer": ref, "model_answer": answer, "is_greedy": True, "sample_index": 0,
            "outcome": outcome, "outcome_wb": outcome}


class RegradeColabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.root = root
        cls.grading = rc.load_grader(expect_version=None)
        qs = [{"question_id": f"F1-q{i}", "question": "Where is the capital of Mozu?"} for i in range(3)]
        write(root / "data" / "toy" / "eval.jsonl", qs)
        write(root / "data" / "toy" / "robustness" / "eval_heldout.jsonl",
              [{"question_id": "F1-h0", "question": "Tell me the capital of Mozu."}])
        write(root / "run" / "responses__mine__frozen.jsonl", [
            resp("F1-q0", "Zonkaegam", "correct"),
            resp("F1-q1", "Zonkaegamath", "correct"),          # near-miss name: substring grader said correct
            resp("F1-q2", "I don't know.", "abstain")])
        write(root / "run" / "extra" / "eval_heldout" / "responses__mine__frozen.jsonl",
              [resp("F1-h0", "The capital is Zonkaegam.", "correct")])
        cls.summary = rc.run(root / "run", root / "data", "toy", root / "out", cls.grading)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _rows(self, rel):
        return rc.read_jsonl(self.root / "out" / rel)

    def test_layout_is_mirrored_and_extras_use_their_own_question_file(self):
        self.assertEqual({s["file"] for s in self.summary},
                         {"responses__mine__frozen.jsonl", "extra/eval_heldout/responses__mine__frozen.jsonl"})
        self.assertEqual(self._rows("extra/eval_heldout/responses__mine__frozen.jsonl")[0]["question"],
                         "Tell me the capital of Mozu.")

    def test_near_miss_is_no_longer_credited_and_original_is_kept(self):
        rows = {r["question_id"]: r for r in self._rows("responses__mine__frozen.jsonl")}
        self.assertEqual(rows["F1-q0"]["outcome"], "correct")
        self.assertEqual(rows["F1-q1"]["outcome"], "incorrect")
        self.assertEqual(rows["F1-q1"]["outcome_v1"], "correct")
        self.assertEqual(rows["F1-q2"]["outcome"], "abstain")
        self.assertEqual(rows["F1-q1"]["grading_version"], self.grading.GRADING_VERSION)
        self.assertEqual(next(s for s in self.summary if s["file"] == "responses__mine__frozen.jsonl")["changed"], 1)

    def test_a_question_id_missing_from_the_eval_file_is_an_error(self):
        write(self.root / "run2" / "responses__mine__frozen.jsonl", [resp("F9-q0", "x", "incorrect")])
        with self.assertRaises(SystemExit):
            rc.run(self.root / "run2", self.root / "data", "toy", self.root / "out2", self.grading)

    def test_wrong_grader_version_is_refused(self):
        with self.assertRaises(SystemExit):
            rc.load_grader(expect_version="not_a_real_version")


if __name__ == "__main__":
    unittest.main()
