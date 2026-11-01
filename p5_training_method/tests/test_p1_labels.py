import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from p1_labels import P1_DIR, check_and_convert, read_jsonl

META = {"dtype": "torch.float32", "grading_version": "p2_whole_word_v5"}


def fact(i, exposure=5, split="dev"):
    return {"fact_id": f"F{i:04d}", "exposure_level": exposure, "split": split}


def p1_row(i, k=0.5, exposure=5, split="dev"):
    return {"fact_id": f"F{i:04d}", "exposure_level": exposure, "split": split, "k": k, "n_sampled": 20,
            "grading_version": "p2_whole_word_v5"}


class CheckAndConvert(unittest.TestCase):
    def setUp(self):
        self.facts = [fact(1), fact(2, 0, "final")]
        self.rows = [p1_row(1, 0.25), p1_row(2, 0.0, 0, "final")]

    def test_valid_rows_become_a_fact_to_k_map(self):
        self.assertEqual(check_and_convert(self.rows, self.facts, META), {"F0001": 0.25, "F0002": 0.0})

    def test_rejects_a_missing_fact(self):
        with self.assertRaises(ValueError):
            check_and_convert(self.rows[:1], self.facts, META)

    def test_rejects_changed_exposure_or_split(self):
        for key, bad in (("exposure_level", 10), ("split", "final")):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = bad
            with self.assertRaises(ValueError):
                check_and_convert(rows, self.facts, META)

    def test_rejects_k_outside_unit_interval_and_wrong_sample_count(self):
        for key, bad in (("k", 1.2), ("n_sampled", 5)):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = bad
            with self.assertRaises(ValueError):
                check_and_convert(rows, self.facts, META)

    def test_rejects_non_float32_or_other_grader(self):
        with self.assertRaises(ValueError):
            check_and_convert(self.rows, self.facts, dict(META, dtype="torch.bfloat16"))
        with self.assertRaises(ValueError):
            check_and_convert(self.rows, self.facts, dict(META, grading_version="placeholder_exact_v1"))


class RealP1File(unittest.TestCase):
    def test_p1_file_matches_main_v0(self):
        k_file = P1_DIR / "k_per_fact.jsonl"
        if not k_file.exists():
            self.skipTest("P1's K file is not in this checkout")
        facts = read_jsonl(REPO / "data" / "synthetic" / "main_v0" / "facts.jsonl")
        meta = json.loads((P1_DIR / "k_meta.json").read_text())
        labels = check_and_convert(read_jsonl(k_file), facts, meta)
        self.assertEqual(len(labels), 600)


if __name__ == "__main__":
    unittest.main()
