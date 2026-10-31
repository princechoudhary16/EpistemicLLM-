import json
from pathlib import Path
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


class TriviaQA(unittest.TestCase):
    """TriviaQA release: alias cleaning, grouping by answer, P4-sample exclusion, end-to-end build."""

    @staticmethod
    def snapshot_rows(n=1700):
        # like real TriviaQA: some answers shared by 3 questions (i < 900), the rest answered once
        def val(i):
            return f"Answer {i // 3}" if i < 900 else f"Solo {i}"
        rows = [dict(question_id=f"tq_{i:05d}", question=f"Trivia question number {i}?", value=val(i),
                     aliases=[val(i), val(i).lower(), "A", f"Alt name {val(i)}"], type="FreeForm")
                for i in range(n)]
        rows.append(dict(question_id="tq_dupe", question="Trivia question number 7?", value="Other",  # duplicate text of tq_00007
                         aliases=["Other"], type="FreeForm"))
        return rows

    def write(self, d, rows, name="snap.jsonl"):
        p = Path(d) / name
        data.write_jsonl(p, rows)
        return p

    def test_clean_aliases(self):
        kept, dropped = data.clean_aliases("Paris", ["paris", "PARIS!", "P", "Paris, France", "City of Light", "France || X"],
                                           question="Which city in France is the capital?")
        self.assertEqual(kept[0], "Paris")                       # canonical value first
        self.assertNotIn("P", kept)                              # 1-letter alias dropped
        self.assertEqual(dropped["too_short"], 1)
        self.assertEqual(sum(a.lower().startswith("paris") and len(a) <= 6 for a in kept), 1)  # case/punct duplicates merged
        self.assertTrue(all("||" not in a for a in kept))        # separator never inside an alias
        kept2, d2 = data.clean_aliases("Seine", ["Seine", "France"], question="Which river flows through France?")
        self.assertEqual(kept2, ["Seine"])                       # alias already in the question dropped
        self.assertEqual(d2["in_question"], 1)
        kept5, _ = data.clean_aliases("Roots: The Saga of an American Family", ["Roots (book)", "Mercury (album)"],
                                      question="Which 1976 Alex Haley book became a TV series?")
        self.assertIn("Roots", kept5)                            # bracket label removed -> plain answer matches
        self.assertIn("Roots (book)", kept5)                     # original alias kept too
        kept6, d6 = data.clean_aliases("Delaware", ["The First State (nickname)"], question="Which US state is The First State?")
        self.assertNotIn("The First State", kept6)              # a bare form already in the question is dropped too
        self.assertEqual(d6["in_question"], 1)
        kept7, _ = data.clean_aliases("Joseph Merrick", ["The Elephant Man", "Elephant Man", "John Merrick"],
                                      question="What was the Elephant Man's real name?")
        self.assertEqual(kept7, ["Joseph Merrick", "John Merrick"])   # audit find: possessive must not hide the overlap
        kept4, _ = data.clean_aliases("Eight", ["8"], question="What number on the Beaufort Scale is a gale?")
        self.assertEqual(kept4, ["Eight", "8"])                  # a single digit is never "too short"
        kept3, _ = data.clean_aliases("7", [], question="How many days in a week?")
        self.assertEqual(kept3, ["7"])                           # a short canonical value is always kept

    def test_reference_answer_is_gradable(self):
        from common import grading
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            df, _ = data.load_triviaqa_snapshot(self.write(d, self.snapshot_rows(30)))
        r = df.iloc[0]
        self.assertIn(" || ", r.answer)
        self.assertEqual(grading.grade("I think it is Alt name Answer 0.", r.answer, r.problem)[0], "correct")
        self.assertEqual(grading.grade("A", r.answer, r.problem)[0], "incorrect")   # the dropped 1-letter alias earns nothing

    def test_repeated_question_ids_kept_once(self):
        import tempfile
        rows = self.snapshot_rows(30)
        rows += [dict(rows[0]), dict(rows[1], question="A reworded copy?")]      # rc files repeat IDs
        with tempfile.TemporaryDirectory() as d:
            df, audit = data.load_triviaqa_snapshot(self.write(d, rows))
        self.assertTrue(df.question_id.is_unique)
        self.assertEqual(df.set_index("question_id").problem["tq_00001"], rows[1]["question"])   # first copy wins
        self.assertEqual(audit["dedup"]["repeated_rows_dropped"], 2)
        self.assertEqual(audit["dedup"]["repeated_rows_with_different_content"], 1)

    def test_group_by_answer_and_duplicate_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            df, _ = data.load_triviaqa_snapshot(self.write(d, self.snapshot_rows(30)))
        g = df.set_index("question_id").group_id
        self.assertEqual(g["tq_00000"], g["tq_00002"])           # same answer -> same group
        self.assertNotEqual(g["tq_00000"], g["tq_00003"])
        self.assertEqual(g["tq_00007"], g["tq_dupe"])            # same question text -> merged into one group

    def test_build_end_to_end_excludes_p4_sample(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            snap = self.write(d, self.snapshot_rows())
            p4 = self.write(d, [{"question_id": "triviaqa-tq_00010"}, {"question_id": "triviaqa-tq_00500"},
                                {"question_id": "triviaqa-not_in_snapshot"}], "p4.jsonl")
            audit = data.build_triviaqa(snap, "t0", exclude_file=p4, data_root=Path(d) / "rq", manifest_root=Path(d) / "m")
            self.assertTrue(all(c["ok"] for c in audit["checks"]), audit["checks"])
            self.assertEqual(audit["questions_per_split"],
                             {"pilot": 200, "train_probe": 500, "validation": 200, "locked_test": 300})
            self.assertEqual(audit["excluded_ids_requested"], 3)
            self.assertEqual(audit["excluded_ids_not_in_snapshot"], 1)
            self.assertEqual(audit["excluded_questions_incl_group"], 6)   # 2 IDs x their 3-question answer groups
            rows = [json.loads(l) for f in sorted((Path(d) / "rq" / "triviaqa_t0").glob("*.jsonl"))
                    for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            ids = {r["question_id"] for r in rows}
            for q in ("tq_00009", "tq_00010", "tq_00011", "tq_00500"):   # tq_00010 and its whole answer group
                self.assertNotIn(f"triviaqa-{q}", ids)
            r = rows[0]
            self.assertTrue(r["question_id"].startswith("triviaqa-"))
            self.assertEqual(r["source_dataset"], "triviaqa")
            self.assertEqual(r["reference_answer"], " || ".join(r["aliases"]))
            again = data.build_triviaqa(snap, "t0", exclude_file=p4, data_root=Path(d) / "rq2", manifest_root=Path(d) / "m2")
            self.assertEqual(audit["file_sha256"], again["file_sha256"])   # deterministic


if __name__ == "__main__":
    unittest.main()
