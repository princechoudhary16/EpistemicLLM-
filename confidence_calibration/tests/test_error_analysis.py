import unittest

from confidence_calibration.calibration import PlattCalibrator
from confidence_calibration.error_analysis import confident_error_analysis, error_pattern

ALWAYS_CONFIDENT = PlattCalibrator("avg_logprob", mean=0.0, scale=1.0, coefficient=1.0, intercept=0.0)


def question(question_id, outcome, score=0.0, consistency=1.0, knowledge=0.0, **extra):
    return {"question_id": question_id, "outcome": outcome, "avg_logprob": score, "consistency": consistency,
            "knowledge_score": knowledge, "question": "Q?", "model_answer": "A", "reference_answer": "R || R2",
            "review_status": "unreviewed", **extra}


class ErrorPattern(unittest.TestCase):
    def test_samples_that_sometimes_succeed_mean_within_reach(self):
        self.assertEqual(error_pattern(question("q", "incorrect", knowledge=0.2, consistency=1.0)), "knows_sometimes")

    def test_repeated_wrong_answer_is_stable(self):
        self.assertEqual(error_pattern(question("q", "incorrect", consistency=0.8)), "stable_wrong")

    def test_disagreeing_samples_are_wavering(self):
        self.assertEqual(error_pattern(question("q", "incorrect", consistency=0.2)), "wavering_wrong")


class ConfidentErrors(unittest.TestCase):
    def test_keeps_only_wrong_answers_above_the_threshold_most_confident_first(self):
        rows = [question("low", "incorrect", score=-5.0), question("mid", "incorrect", score=0.0),
                question("high", "incorrect", score=2.0, consistency=0.0, exposure_level=5),
                question("right", "correct", score=3.0), question("idk", "abstain", score=3.0)]
        analysis = confident_error_analysis(rows, ALWAYS_CONFIDENT, threshold=0.4)
        self.assertEqual([e["question_id"] for e in analysis["examples"]], ["high", "mid"])
        self.assertEqual((analysis["n_answers_kept"], analysis["n_errors"]), (3, 2))
        self.assertEqual(analysis["by_pattern"], {"stable_wrong": 1, "wavering_wrong": 1, "knows_sometimes": 0})
        self.assertEqual(analysis["by_exposure"], {"5": 1})

    def test_reports_the_first_alias_and_counts_grader_flags(self):
        rows = [question("q", "incorrect", review_status="flagged:numeric_ambiguous")]
        analysis = confident_error_analysis(rows, ALWAYS_CONFIDENT, threshold=0.0)
        self.assertEqual(analysis["examples"][0]["reference_answer"], "R")
        self.assertEqual(analysis["n_flagged_by_grader"], 1)


if __name__ == "__main__":
    unittest.main()
