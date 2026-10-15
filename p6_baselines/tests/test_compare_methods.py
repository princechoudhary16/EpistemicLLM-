import unittest

from p6_baselines.compare_methods import validate_methods


def row(qid="q1", sample="greedy", greedy=True, question="What?", reference="A"):
    return {
        "question_id": qid,
        "sample_index": sample,
        "is_greedy": greedy,
        "question": question,
        "reference_answer": reference,
        "source_dataset": "test",
        "dataset_version": "v1",
        "split": "dev",
    }


class MethodLayoutTests(unittest.TestCase):
    def test_accepts_matching_question_and_sample_layouts(self):
        base = [row(), row("q1", "sample0", False), row("q2", "greedy", True, "Why?", "B"),
                row("q2", "sample0", False, "Why?", "B")]
        other = [row(), row("q1", "sample0", False), row("q2", "greedy", True, "Why?", "B"),
                 row("q2", "sample0", False, "Why?", "B")]
        self.assertEqual(validate_methods({"base": base, "other": other}), {"q1", "q2"})

    def test_rejects_duplicate_greedy_even_when_question_set_matches(self):
        with self.assertRaisesRegex(ValueError, "duplicate response"):
            validate_methods({"base": [row(), row()]})

    def test_rejects_missing_greedy(self):
        with self.assertRaisesRegex(ValueError, "expected exactly one"):
            validate_methods({"base": [row("q1", "sample0", False)]})

    def test_rejects_sample_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "different sampled response indices"):
            validate_methods({
                "base": [row(), row("q1", "sample0", False)],
                "other": [row(), row("q1", "sample1", False)],
            })

    def test_rejects_no_sampled_responses(self):
        with self.assertRaisesRegex(ValueError, "no sampled responses"):
            validate_methods({"base": [row()]})

    def test_rejects_question_reference_mismatch(self):
        with self.assertRaisesRegex(ValueError, "different question/reference metadata"):
            validate_methods({
                "base": [row(), row("q1", "sample0", False)],
                "other": [row(reference="Different"), row("q1", "sample0", False, reference="Different")],
            })


if __name__ == "__main__":
    unittest.main()
