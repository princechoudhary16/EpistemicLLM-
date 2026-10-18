import unittest

from confidence_calibration.grading import grade_answer, is_abstention, normalize_answer


class Normalize(unittest.TestCase):
    def test_squad_style(self):
        self.assertEqual(normalize_answer("  The Eiffel Tower. "), "eiffel tower")
        self.assertEqual(normalize_answer("Answer: A Tale of Two Cities!"), "tale of two cities")
        self.assertEqual(normalize_answer(""), "")


class Abstention(unittest.TestCase):
    def test_plain_abstentions(self):
        for text in ["I don't know", "I don’t know.", "I do not know the answer.", "I'm not sure.",
                     "Unknown", "I cannot answer that.", "I don't have enough information."]:
            self.assertTrue(is_abstention(text), text)

    def test_hedged_claim_is_not_abstention(self):
        self.assertFalse(is_abstention("I'm not sure, but I think it's London."))

    def test_unknown_inside_an_answer_is_not_abstention(self):
        self.assertFalse(is_abstention("The Unknown Soldier"))


class Grade(unittest.TestCase):
    def test_exact_and_alias(self):
        self.assertEqual(grade_answer("Paris.", ["Paris"]), ("correct", "exact"))
        self.assertEqual(grade_answer("JFK", ["John F. Kennedy", "JFK"]), ("correct", "exact"))

    def test_contains_short_answer(self):
        self.assertEqual(grade_answer("Paris, France", ["Paris"]), ("correct", "contains"))

    def test_contains_needs_whole_words(self):
        self.assertEqual(grade_answer("Parisian", ["Paris"]), ("incorrect", "no_match"))

    def test_alias_restated_from_question_earns_nothing(self):
        q = "By what name is the tree Fraxinus excelsior commonly known?"
        ans = "The tree Fraxinus excelsior is commonly known as the white birch."
        self.assertEqual(grade_answer(ans, ["Ash", "Fraxinus excelsior"], q)[0], "incorrect")
        self.assertEqual(grade_answer("Ash", ["Ash", "Fraxinus excelsior"], q), ("correct", "exact"))

    def test_long_answers_do_not_get_contains_credit(self):
        long = "It could be London or Rome or Madrid or Berlin or Vienna or Paris or Lisbon or Oslo"
        self.assertEqual(grade_answer(long, ["Paris"])[0], "incorrect")

    def test_abstain_and_invalid(self):
        self.assertEqual(grade_answer("I don't know", ["Paris"]), ("abstain", "abstain_pattern"))
        self.assertEqual(grade_answer("   ", ["Paris"]), ("invalid", "empty"))

    def test_hedged_wrong_claim_is_incorrect(self):
        self.assertEqual(grade_answer("I'm not sure, but maybe Rome", ["Paris"])[0], "incorrect")


if __name__ == "__main__":
    unittest.main()
