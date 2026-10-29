import json
import tempfile
import unittest
from pathlib import Path

from p1_exposure.analysis import robustness_report as rr


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def build_world(root, version, n_per_level=10, levels=(0, 5)):
    """Facts F{version}-000..; each has 2 standard, 2 heldout and 2 swapped questions."""
    facts = []
    n = 0
    for lvl in levels:
        for i in range(n_per_level):
            facts.append({"fact_id": f"{version}-F{n:03d}", "level": lvl, "idx": i,
                          "split": "dev" if i % 2 == 0 else "final", "relation": "capital"})
            n += 1
    def rows(tag):
        return [{"question_id": f"{f['fact_id']}-{tag}{k}", "fact_id": f["fact_id"], "relation": f["relation"],
                 "split": f["split"], "exposure_level": f["level"], "reference_answer": "x", "question": "q"}
                for f in facts for k in range(2)]
    base = Path(root) / version
    write_jsonl(base / "eval.jsonl", rows("q"))
    write_jsonl(base / "robustness" / "eval_heldout.jsonl", rows("h"))
    write_jsonl(base / "robustness" / "eval_swapped.jsonl", rows("s"))
    return facts


def write_run(out_dir, facts, run, decoding, standard, heldout, swapped):
    """standard/heldout/swapped: functions (fact, question_index) -> bool correct."""
    def resp(tag, fn):
        return [{"question_id": f"{f['fact_id']}-{tag}{k}", "is_greedy": True,
                 "outcome": "correct" if fn(f, k) else "incorrect"} for f in facts for k in range(2)]
    write_jsonl(Path(out_dir) / f"responses__{run}__{decoding}.jsonl", resp("q", standard))
    write_jsonl(Path(out_dir) / "extra" / "eval_heldout" / f"responses__{run}__{decoding}.jsonl", resp("h", heldout))
    write_jsonl(Path(out_dir) / "extra" / "eval_swapped" / f"responses__{run}__{decoding}.jsonl", resp("s", swapped))


class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.data = root / "data"
        w0 = build_world(cls.data, "w0")
        w1 = build_world(cls.data, "w1")
        always = lambda f, k: f["level"] == 5
        # seed 0: every level-5 question right; seed 1: both questions of the first level-5 fact wrong
        first5 = next(f["fact_id"] for f in w0 if f["level"] == 5)
        seed1 = lambda f, k: f["level"] == 5 and f["fact_id"] != first5
        half = lambda f, k: f["level"] == 5 and f["idx"] % 2 == 0      # held-out wording: only even facts right
        write_run(root / "s0", w0, "mine", "frozen", always, half, always)
        write_run(root / "s1", w0, "mine", "frozen", seed1, half, seed1)
        sixty = lambda f, k: f["level"] == 5 and f["idx"] < 6           # world 1: 6 of 10 facts right
        write_run(root / "w1s0", w1, "mine", "frozen", sixty, sixty, sixty)
        cls.out = rr.analyze(cls.data, {"w0": [root / "s0", root / "s1"], "w1": [root / "w1s0"]},
                             "mine", "frozen", boot=400, seed=0)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_per_seed_accuracy_and_spread(self):
        w = self.out["worlds"]["w0"]
        self.assertEqual(self.out["levels"], [0, 5])
        self.assertAlmostEqual(w["per_seed"][5][0], 1.0)
        self.assertAlmostEqual(w["per_seed"][5][1], 0.9)
        self.assertAlmostEqual(w["seed_mean"][5]["range"], 0.1)
        self.assertAlmostEqual(w["per_seed"][0][0], 0.0)

    def test_seed_mean_and_interval(self):
        sm = self.out["worlds"]["w0"]["seed_mean"][5]
        self.assertAlmostEqual(sm["est"], 0.95)
        self.assertLessEqual(sm["lo"], sm["est"])
        self.assertGreaterEqual(sm["hi"], sm["est"])
        self.assertEqual(sm["n_facts"], 10)

    def test_heldout_difference_is_paired_and_negative(self):
        p = self.out["worlds"]["w0"]["paraphrase"]["heldout"]["per_level"][5]
        self.assertAlmostEqual(p["standard"], 0.95)
        self.assertAlmostEqual(p["paraphrase"], 0.5)
        self.assertAlmostEqual(p["diff"], -0.45)
        self.assertLess(p["hi"], 0)

    def test_swapped_wording_with_same_answers_has_zero_difference(self):
        p = self.out["worlds"]["w0"]["paraphrase"]["swapped"]["per_level"][5]
        self.assertAlmostEqual(p["diff"], 0.0)
        self.assertLessEqual(p["lo"], 0.0)
        self.assertGreaterEqual(p["hi"], 0.0)

    def test_overall_and_by_split_rows_exist(self):
        p = self.out["worlds"]["w0"]["paraphrase"]["heldout"]
        self.assertAlmostEqual(p["overall_ge2"]["diff"], -0.45)
        self.assertEqual(set(p["by_split_ge2"]), {"dev", "final"})
        self.assertEqual(p["by_split_ge2"]["dev"]["n_facts"], 5)

    def test_world_comparison_uses_independent_facts(self):
        c = self.out["world_comparison"]["per_level"][5]
        self.assertAlmostEqual(c["world_a"], 0.95)
        self.assertAlmostEqual(c["world_b"], 0.6)
        self.assertAlmostEqual(c["diff"], -0.35)
        self.assertLessEqual(c["lo"], c["diff"])
        self.assertGreaterEqual(c["hi"], c["diff"])

    def test_reproducible_for_a_fixed_seed(self):
        again = rr.analyze(self.data, {"w0": [self.root / "s0", self.root / "s1"]}, "mine", "frozen", boot=400, seed=0)
        self.assertEqual(again["worlds"]["w0"]["seed_mean"][5]["lo"], self.out["worlds"]["w0"]["seed_mean"][5]["lo"])

    def test_missing_extra_files_are_skipped_not_fatal(self):
        solo = Path(self.root) / "solo"
        facts = build_world(self.data, "w2")
        write_jsonl(solo / "responses__mine__frozen.jsonl",
                    [{"question_id": f"{f['fact_id']}-q{k}", "is_greedy": True, "outcome": "correct"}
                     for f in facts for k in range(2)])
        out = rr.analyze(self.data, {"w2": [solo]}, "mine", "frozen", boot=50, seed=0)
        self.assertEqual(out["worlds"]["w2"]["paraphrase"], {})

    def test_render_defines_terms_and_has_all_sections(self):
        md = rr.render(self.out, "mine", "frozen", 400, 0)
        self.assertIn("**Terms.**", md)
        self.assertIn("training seeds (2 seeds)", md)
        self.assertIn("heldout wording vs standard wording", md)
        self.assertIn("Second world", md)

    def test_figure_is_written(self):
        path = Path(self.root) / "fig.png"
        rr.make_figure(self.out, path)
        self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
