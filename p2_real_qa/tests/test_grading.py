import unittest
from common import grading as g


def G(a, r, q=""):
    return g.grade(a, r, q)[0]


class NumericFalsePositives(unittest.TestCase):
    """Each case is a real false positive the v1 grader produced on the SimpleQA pilot."""
    def test_substring_of_year(self):
        self.assertEqual(G("Barcelona committed 3 fouls on May 18, 2006.", "20", "How many fouls on May 18, 2006?"), "incorrect")
    def test_substring_of_year_2(self):
        self.assertEqual(G("As of 2023 there are 31 amendments.", "3"), "incorrect")
        self.assertEqual(G("Anne Marie (born 1872) and Mary Ann (born 1874).", "4"), "incorrect")
    def test_range_is_ambiguous_not_credited(self):
        self.assertEqual(g.grade("About 10-20 elephants.", "20"), ("incorrect", "numeric_ambiguous"))
    def test_number_from_question_not_credited(self):
        self.assertEqual(G("In 2006 something happened.", "2006", "What happened in 2006?"), "incorrect")


class TruePositives(unittest.TestCase):
    def test_year(self): self.assertEqual(G("He dropped out in 2018.", "2018"), "correct")
    def test_dash_and_period(self):
        self.assertEqual(G("University of Illinois at Urbana-Champaign", "University of Illinois at Urbana\u2013Champaign."), "correct")
    def test_number_word(self): self.assertEqual(G("Morris Kight had two children.", "Two."), "correct")
    def test_thousands_separator(self): self.assertEqual(G("About 1,000 people.", "1000"), "correct")
    def test_accents(self): self.assertEqual(G("Karl K\u00fcchler wrote it.", "Karl Kuchler"), "correct")
    def test_single_number(self): self.assertEqual(G("He had 3 children.", "3", "How many children?"), "correct")
    def test_aliases(self): self.assertEqual(G("It is NYC.", "New York City || NYC"), "correct")


class WholeWord(unittest.TestCase):
    def test_not_substring(self):
        self.assertEqual(G("He worked in pottery.", "potter"), "incorrect")      # placeholder would say correct
        self.assertEqual(G("They cooperate closely.", "cooper"), "incorrect")


class PortedFromOldSuite(unittest.TestCase):
    """The two grading tests from the old pipeline.grade suite, same intent."""
    def test_reference_must_match_whole_words(self):
        self.assertEqual(G("The annual report is ready.", "Ann"), "incorrect")
        self.assertEqual(G("The answer is Ann Marie.", "Ann Marie"), "correct")

    def test_empty_reference_never_matches_every_answer(self):
        self.assertEqual(G("A substantive answer", ""), "incorrect")

    def test_grade_output_records_version(self):
        self.assertEqual(g.grade_fields("Ann", "Ann")[1], g.GRADING_VERSION)


class Abstention(unittest.TestCase):
    def test_plain(self):
        for s in ("I don't know", "I do not know.", "I have no idea.", "I don't know the answer.", "I\u2019m not sure."):
            self.assertEqual(G(s, "x"), "abstain", s)
    def test_hedged_attempt_is_scored_on_content(self):
        self.assertEqual(G("I'm not sure, but the answer is Paris.", "Paris"), "correct")
        self.assertEqual(g.grade("I'm not sure, but the answer is Paris.", "Berlin"), ("incorrect", "hedged_attempt"))
    def test_partial_then_idk_is_an_attempt(self):
        self.assertEqual(G("Maryon Lane was born with the surname Lane. I don't know her full name.", "Smith"), "incorrect")
    def test_unanswerable_reference(self):
        self.assertEqual(G("I don't know.", "__unanswerable__"), "abstain")
        self.assertEqual(G("It is Paris.", "__unanswerable__"), "incorrect")
    def test_empty_is_invalid(self):
        self.assertEqual(g.grade("   ", "x"), ("invalid", "empty_response"))


class IdkRestatesQuestion(unittest.TestCase):
    """the two rows the manual audit found (sqa-00334, sqa-02533) plus the cases that must NOT change."""
    EDD_Q = "Who won the Eddington Medal in 1993?"
    EDD_A = ("The Eddington Medal was awarded to a person who has made a significant contribution to the study "
             "of the Sun. I don't know who won the Eddington Medal in 1993.")
    RAIN_Q = "In July 2021, how much rainfall (in millimeters) was recorded in Al Ain due to cloud-seeding efforts?"
    RAIN_A = ("I don't know. I don't know the exact amount of rainfall recorded in Al Ain due to cloud-seeding "
              "efforts in July 2021.")

    def test_audit_found_cases_are_abstentions(self):
        self.assertEqual(g.grade(self.EDD_A, "Leon Mestel", self.EDD_Q), ("abstain", "idk_restates_question"))
        self.assertEqual(g.grade(self.RAIN_A, "6.9 millimetres", self.RAIN_Q), ("abstain", "idk_restates_question"))

    def test_needs_the_question(self):                      # without it the rule cannot apply (documented dependency)
        self.assertEqual(g.grade(self.EDD_A, "Leon Mestel")[0], "incorrect")

    def test_bare_idk_plus_candidate_is_still_an_attempt(self):
        self.assertEqual(g.grade("I don't know. It might be Paris.", "Berlin", "What is the capital of France?")[0], "incorrect")
        self.assertEqual(g.grade("I don't know. It might be Paris.", "Paris", "What is the capital of France?")[0], "correct")

    def test_committed_number_is_an_attempt(self):
        q = "In what year did she become Deputy President?"
        self.assertEqual(g.grade("It was 2004. I don't know the year she became Deputy President.", "2008", q)[0], "incorrect")

    def test_contrast_word_keeps_it_an_attempt(self):
        self.assertEqual(g.grade("I'm not sure, but the answer is Paris.", "Paris", "Capital of France?")[0], "correct")

    def test_stated_reference_is_credited(self):
        self.assertEqual(g.grade("The medal went to Leon Mestel. I don't know who won the Eddington Medal in 1993.",
                                 "Leon Mestel", self.EDD_Q)[0], "correct")

    def test_partial_info_then_idk_about_something_else_stays_an_attempt(self):
        self.assertEqual(g.grade("Maryon Lane was born with the surname Lane. I don't know her full name.", "Smith",
                                 "What was Maryon Lane's birth surname and given name?")[0], "incorrect")

    def test_known_limitation_wrong_name_then_restating_idk_is_abstain_but_flagged(self):
        out = g.grade("The medal went to John Smith. I don't know who won the Eddington Medal in 1993.", "Leon Mestel", self.EDD_Q)
        self.assertEqual(out, ("abstain", "idk_restates_question"))   # flagged so the audit can find these


class RowFields(unittest.TestCase):
    def test_fields_fit_schema(self):
        from common import schema
        out, ver, status = g.grade_fields("It is 5.", "5", "q")
        self.assertEqual((out, ver, status), ("correct", g.GRADING_VERSION, "unreviewed"))
        schema.make_row(outcome=out, grading_version=ver, review_status=status)   # must not raise


if __name__ == "__main__":
    unittest.main()
