import unittest
import pandas as pd
from p2_real_qa import data, validate


def toy(n=120):
    rows = [dict(question_id=i, problem=f"Question number {i}?", answer=str(i), topic="t", answer_type="Number",
                 group_id=f"g{i // 3}") for i in range(n)]            # 3 questions per group
    return pd.DataFrame(rows)


def records(splits):
    return {k: data.to_records(v, k, "toy-v0") for k, v in splits.items()}


class Split(unittest.TestCase):
    def setUp(self):
        self.df = toy()
        self.pilot = self.df.iloc[[0, 10, 20]]
        self.sizes = (("a", 21), ("b", 9), ("c", 15))   # multiples of 3: toy groups hold 3 questions
        self.splits = data.group_split(self.df, self.pilot.question_id, seed=1, sizes=self.sizes)

    def test_sizes_and_disjoint_groups(self):
        self.assertEqual({k: len(v) for k, v in self.splits.items()}, {"a": 21, "b": 9, "c": 15})
        groups = {k: set(v.group_id) for k, v in self.splits.items()}
        self.assertFalse(groups["a"] & groups["b"] or groups["a"] & groups["c"] or groups["b"] & groups["c"])

    def test_pilot_groups_excluded(self):
        pg = set(self.pilot.group_id)
        for v in self.splits.values():
            self.assertFalse(pg & set(v.group_id))

    def test_deterministic(self):
        again = data.group_split(self.df, self.pilot.question_id, seed=1, sizes=self.sizes)
        for k in self.splits:
            self.assertEqual(self.splits[k].question_id.tolist(), again[k].question_id.tolist())

    def test_wikipedia_url_extraction(self):
        packed = ["https://www.gutenberg.org/x\nhttps://en.wikipedia.org/wiki/Foo_Bar#Section"]
        self.assertEqual(data._group_key(packed), "https://en.wikipedia.org/wiki/foo_bar")
        self.assertIsNone(data._group_key(["https://example.com"]))


class PortedFromOldSuite(unittest.TestCase):
    """Scenarios from the old data tests, rewritten for the new API."""
    def frame(self):
        rows = [{"question_id": i, "problem": f"Question {i}", "answer": f"A{i}", "group_id": f"g{i}"} for i in range(1205)]
        rows += [
            {"question_id": 2000, "problem": "Variant of question 1200", "answer": "A1200", "group_id": "family"},
            {"question_id": 2001, "problem": "Another wording", "answer": "A1200", "group_id": "family"},   # same reviewed family as 2000
            {"question_id": 2002, "problem": "Question 1204!", "answer": "A1204", "group_id": "g2002"},     # exact duplicate of 1204
        ]
        return pd.DataFrame(rows)

    def test_duplicates_and_reviewed_families_do_not_cross_splits(self):
        df = data.merge_duplicate_groups(self.frame())                       # what load_snapshot does
        pilot = df[df.question_id.isin([0, 2000])]
        splits = data.group_split(df, pilot.question_id)                     # default sizes 500/200/300
        self.assertEqual({k: len(v) for k, v in splits.items()}, {"train_probe": 500, "validation": 200, "locked_test": 300})
        groups = {k: set(v.group_id) for k, v in splits.items()}
        self.assertFalse(groups["train_probe"] & groups["validation"])
        self.assertFalse(groups["train_probe"] & groups["locked_test"])
        self.assertFalse(groups["validation"] & groups["locked_test"])
        self.assertFalse(set.union(*groups.values()) & {"g0", "family"})     # pilot's group and its family are excluded
        picked = set(pd.concat(list(splits.values())).question_id)
        self.assertNotIn(2001, picked)                                       # family member of a pilot question
        self.assertEqual(1204 in picked, 2002 in picked)                     # duplicate twins stay together...
        if 1204 in picked:
            homes = [k for k, v in splits.items() if {1204, 2002} <= set(v.question_id)]
            self.assertEqual(len(homes), 1)                                  # ...inside ONE split

    def test_pilot_keeps_one_row_per_normalized_question(self):
        df = pd.DataFrame([{"question_id": 0, "problem": "Who is Ann?", "answer": "Ann"},
                           {"question_id": 1, "problem": "Who is Ann!", "answer": "Ann"},
                           {"question_id": 2, "problem": "Who is Bea?", "answer": "Bea"}])
        pilot = data.select_pilot(df, n=2, seed=1)
        self.assertEqual(pilot.problem.map(data.normalized_question).nunique(), 2)

    def test_manual_groups_file_overrides(self):
        import tempfile, os
        df = toy(9)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.csv")
            pd.DataFrame({"question_id": [0, 8], "group_id": ["fam", "fam"]}).to_csv(path, index=False)
            out = data.apply_manual_groups(df, path)
        self.assertEqual(out.group_id[0], out.group_id[8])

    def test_split_never_cuts_a_group(self):
        df = toy(60)                                   # groups of 3
        with self.assertRaises(RuntimeError):          # 20 cannot be filled by groups of 3 -> refuse, do not trim
            data.group_split(df, [], seed=1, sizes=(("a", 20),))


class ValidatorCanFail(unittest.TestCase):
    """Corruption tests: every check must be able to fail (same idea as p1_exposure/tests)."""
    def setUp(self):
        df = toy()
        self.recs = records({"pilot": df.iloc[:3], "train": df.iloc[3:30], "test": df.iloc[30:45]})

    def failed(self, recs, sizes=None):
        return {c["name"] for c in validate.run_checks(recs, sizes) if not c["ok"]}

    def test_clean_group_straddle_is_caught(self):
        # toy groups hold 3 consecutive ids, so slicing at 3/30/45 cuts no group except where noted
        self.assertEqual(self.failed(self.recs), set())

    def test_shared_group_fails(self):
        self.recs["test"][0]["group_id"] = self.recs["train"][0]["group_id"]
        self.assertIn("no_group_shared_across_splits", self.failed(self.recs))

    def test_duplicate_id_fails(self):
        self.recs["test"][0]["question_id"] = self.recs["train"][0]["question_id"]
        self.assertIn("question_ids_unique_across_splits", self.failed(self.recs))

    def test_missing_field_fails(self):
        self.recs["train"][0]["reference_answer"] = ""
        self.assertIn("required_fields_present", self.failed(self.recs))

    def test_wrong_split_label_fails(self):
        self.recs["train"][0]["split"] = "test"
        self.assertIn("split_field_matches_file", self.failed(self.recs))

    def test_duplicate_text_fails(self):
        self.recs["test"][0]["question"] = self.recs["train"][0]["question"]
        self.assertIn("no_question_text_shared_across_splits", self.failed(self.recs))

    def test_size_fails(self):
        self.assertIn("split_sizes_as_expected", self.failed(self.recs, {"train": 99}))


if __name__ == "__main__":
    unittest.main()
