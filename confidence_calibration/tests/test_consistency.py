import unittest

from confidence_calibration.consistency import summarize, summarize_questions


class Summarize(unittest.TestCase):
    def test_all_agree_after_normalisation(self):
        s = summarize("Paris", ["paris", "Paris.", "The Paris", "PARIS", "paris "])
        self.assertEqual((s["consistency"], s["majority_frac"], s["n_clusters"]), (1.0, 1.0, 1))

    def test_mixed(self):
        s = summarize("Paris", ["Paris", "Rome", "Paris", "Lyon", "Rome"], ["correct", "incorrect", "correct",
                                                                          "incorrect", "incorrect"])
        self.assertAlmostEqual(s["consistency"], 0.4)
        self.assertAlmostEqual(s["majority_frac"], 0.4)
        self.assertEqual(s["n_clusters"], 3)
        self.assertAlmostEqual(s["knowledge_score"], 0.4)

    def test_abstentions_share_a_cluster(self):
        s = summarize("I don't know", ["I do not know.", "I'm not sure.", "Paris", "Unknown", "I don't know"])
        self.assertAlmostEqual(s["consistency"], 0.8)
        self.assertEqual(s["n_clusters"], 2)

    def test_greedy_is_target_not_majority(self):
        # Majority says Rome, but consistency measures agreement with the greedy answer.
        s = summarize("Paris", ["Rome", "Rome", "Rome", "Paris", "Rome"])
        self.assertAlmostEqual(s["consistency"], 0.2)
        self.assertAlmostEqual(s["majority_frac"], 0.8)

    def test_knowledge_score_needs_all_outcomes(self):
        self.assertIsNone(summarize("a", ["a", "b"], ["correct", None])["knowledge_score"])


class SummarizeQuestions(unittest.TestCase):
    def _row(self, answer, greedy, idx=None, qid="q1"):
        return {"run_id": "r", "question_id": qid, "checkpoint_id": "base", "decoding_config_id": "d",
                "is_greedy": greedy, "sample_index": idx, "model_answer": answer, "outcome": None}

    def test_groups_by_question(self):
        rows = [self._row("A", True), self._row("A", False, 0), self._row("B", False, 1),
                self._row("X", True, qid="q2"), self._row("X", False, 0, qid="q2")]
        out = {s["question_id"]: s for s in summarize_questions(rows)}
        self.assertAlmostEqual(out["q1"]["consistency"], 0.5)
        self.assertAlmostEqual(out["q2"]["consistency"], 1.0)

    def test_missing_greedy_raises(self):
        with self.assertRaises(ValueError):
            summarize_questions([self._row("A", False, 0)])


if __name__ == "__main__":
    unittest.main()
