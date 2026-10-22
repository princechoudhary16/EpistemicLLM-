import json
import tempfile
import unittest
from pathlib import Path

from p1_exposure.analysis import exposure_analysis as ea


def build_fixture(tmp):
    """Two relations (a: dev facts, b: final facts), exposure 0 and 10, 5 facts per cell, 2 questions per fact."""
    eval_rows, resp = [], []
    fid = 0

    def add(level, rel, split, greedy_for_q, sampled_correct, conf=None):
        nonlocal fid
        fid += 1
        for qi in range(2):
            qid = f"F{fid:03d}-q{qi}"
            eval_rows.append({"question_id": qid, "fact_id": f"F{fid:03d}", "relation": rel, "split": split,
                              "exposure_level": level, "reference_answer": "x", "question": "q"})
            resp.append({"question_id": qid, "is_greedy": True, "sample_index": 0,
                         "outcome": greedy_for_q(qi), "confidence_score": conf(greedy_for_q(qi)) if conf else None,
                         "grading_version": "test_v0", "decoding_config_id": "decoding_v1", "method": "m"})
            for s in range(1, 6):
                resp.append({"question_id": qid, "is_greedy": False, "sample_index": s,
                             "outcome": "correct" if s <= sampled_correct else "incorrect",
                             "grading_version": "test_v0", "decoding_config_id": "decoding_v1", "method": "m"})

    conf = lambda o: -0.2 if o == "correct" else -1.5
    for _ in range(5):                      # level 10, relation a: always right
        add(10, "a", "dev", lambda q: "correct", 5, conf)
    for _ in range(5):                      # level 10, relation b: wrong greedy, 1 of 5 samples right
        add(10, "b", "final", lambda q: "incorrect", 1, conf)
    add(0, "a", "dev", lambda q: "correct" if q == 0 else "incorrect", 0)   # one lucky greedy hit
    for _ in range(4):
        add(0, "a", "dev", lambda q: "incorrect", 0)
    add(0, "b", "final", lambda q: "abstain" if q == 0 else "incorrect", 0)  # one abstention
    for _ in range(4):
        add(0, "b", "final", lambda q: "incorrect", 0)

    data = Path(tmp) / "data" / "toy"
    data.mkdir(parents=True)
    (data / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n", encoding="utf-8")
    resp_path = Path(tmp) / "resp.jsonl"
    resp_path.write_text("\n".join(json.dumps(r) for r in resp) + "\n", encoding="utf-8")
    return Path(tmp) / "data", resp_path


class ExposureAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir, cls.resp = build_fixture(cls._tmp.name)
        cls.out = Path(cls._tmp.name) / "out"
        cls.results, cls.meta = ea.run({"toy": cls.resp}, cls.data_dir, "toy", cls.out, boot=500, seed=0,
                                       figures=False)
        cls.res = cls.results["toy"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_levels_and_counts(self):
        self.assertEqual(self.meta["levels"], [0, 10])
        e10 = self.res["exposure"][10]
        self.assertEqual((e10["n_facts"], e10["n_questions"]), (10, 20))
        self.assertEqual((e10["correct"], e10["incorrect"], e10["abstain"]), (10, 10, 0))

    def test_point_estimates_level_10(self):
        e = self.res["exposure"][10]
        self.assertAlmostEqual(e["accuracy"]["est"], 0.5)
        self.assertAlmostEqual(e["hallucination"]["est"], 0.5)
        self.assertAlmostEqual(e["coverage"]["est"], 1.0)
        self.assertAlmostEqual(e["selective_accuracy"]["est"], 0.5)
        # relation a questions have K=1, relation b questions K=0.2 -> mean (1 + 0.2) / 2
        self.assertAlmostEqual(e["mean_K"]["est"], 0.6)

    def test_point_estimates_level_0_with_abstention(self):
        e = self.res["exposure"][0]
        self.assertAlmostEqual(e["accuracy"]["est"], 1 / 20)
        self.assertAlmostEqual(e["abstention"]["est"], 1 / 20)
        self.assertAlmostEqual(e["coverage"]["est"], 19 / 20)
        self.assertAlmostEqual(e["selective_accuracy"]["est"], 1 / 19)
        self.assertAlmostEqual(e["mean_K"]["est"], 0.0)

    def test_k_buckets_are_per_fact(self):
        self.assertEqual(self.res["exposure"][10]["k_buckets"], {"K=0": 0, "inconsistent": 5, "K>=0.8": 5})
        self.assertEqual(self.res["exposure"][0]["k_buckets"], {"K=0": 10, "inconsistent": 0, "K>=0.8": 0})

    def test_confidence_intervals_bracket_the_estimate(self):
        for lvl in (0, 10):
            for key in ("accuracy", "hallucination", "mean_K"):
                m = self.res["exposure"][lvl][key]
                self.assertLessEqual(m["lo"], m["est"] + 1e-9)
                self.assertGreaterEqual(m["hi"], m["est"] - 1e-9)

    def test_bootstrap_resamples_facts_not_questions(self):
        # Level 10 has 5 always-right and 5 always-wrong facts, so facts resampled give wide intervals
        m = self.res["exposure"][10]["accuracy"]
        self.assertLess(m["lo"], 0.35)
        self.assertGreater(m["hi"], 0.65)

    def test_same_seed_reproduces_exactly(self):
        again, _ = ea.run({"toy": self.resp}, self.data_dir, "toy", Path(self._tmp.name) / "out2", boot=500,
                          seed=0, figures=False)
        self.assertEqual(again["toy"]["exposure"][10]["accuracy"], self.res["exposure"][10]["accuracy"])

    def test_relation_table(self):
        self.assertAlmostEqual(self.res["relation"]["a"][10]["accuracy"], 1.0)
        self.assertAlmostEqual(self.res["relation"]["b"][10]["accuracy"], 0.0)
        self.assertEqual(self.res["relation"]["a"][10]["n_questions"], 10)

    def test_split_table(self):
        self.assertAlmostEqual(self.res["split"]["dev"][10]["accuracy"]["est"], 1.0)
        self.assertAlmostEqual(self.res["split"]["final"][10]["accuracy"]["est"], 0.0)

    def test_chance_floor_by_relation(self):
        cf = self.res["chance_floor"]
        self.assertEqual(cf["a"]["greedy"], [1, 10])
        self.assertEqual(cf["b"]["greedy"], [0, 10])
        self.assertEqual(cf["a"]["sampled"], [0, 50])

    def test_adjacent_step(self):
        step = self.res["steps"]["0->10"]
        self.assertAlmostEqual(step["est"], 0.5 - 0.05)
        self.assertLessEqual(step["lo"], step["est"])
        self.assertGreaterEqual(step["hi"], step["est"])

    def test_confidence_split_by_correctness(self):
        c = self.res["confidence"][10]
        self.assertAlmostEqual(c["mean_conf_correct"], -0.2)
        self.assertAlmostEqual(c["mean_conf_incorrect"], -1.5)

    def test_outputs_written_and_inputs_hashed(self):
        for name in ("results.json", "exposure_analysis.md", "exposure_table.csv", "relation_table.csv"):
            self.assertTrue((self.out / name).exists(), name)
        info = self.meta["inputs"]["toy"]
        self.assertEqual(len(info["sha256"]), 64)
        self.assertEqual(info["grading_versions"], ["test_v0"])
        self.assertEqual(info["decoding_ids"], ["decoding_v1"])

    def test_markdown_defines_terms_and_lists_all_levels(self):
        md = (self.out / "exposure_analysis.md").read_text(encoding="utf-8")
        self.assertIn("**Terms.**", md)
        self.assertIn("| 0x |", md)
        self.assertIn("| 10x |", md)

    def test_missing_greedy_row_is_rejected(self):
        rows = [json.loads(l) for l in self.resp.read_text(encoding="utf-8").splitlines() if l.strip()]
        first_greedy = next(i for i, r in enumerate(rows) if r["is_greedy"])
        broken = Path(self._tmp.name) / "broken.jsonl"
        broken.write_text("\n".join(json.dumps(r) for i, r in enumerate(rows) if i != first_greedy) + "\n",
                          encoding="utf-8")
        with self.assertRaises(ValueError):
            ea.run({"toy": broken}, self.data_dir, "toy", Path(self._tmp.name) / "o3", boot=50, figures=False)

    def test_unknown_question_id_is_rejected(self):
        rows = [json.loads(l) for l in self.resp.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows[0]["question_id"] = "NOPE"
        bad = Path(self._tmp.name) / "bad.jsonl"
        bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            ea.run({"toy": bad}, self.data_dir, "toy", Path(self._tmp.name) / "o4", boot=50, figures=False)

    def test_all_abstain_group_has_undefined_selective_accuracy(self):
        rows = [json.loads(l) for l in self.resp.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            if r["is_greedy"]:
                r["outcome"] = "abstain"
        p = Path(self._tmp.name) / "abstain.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        res, _ = ea.run({"toy": p}, self.data_dir, "toy", Path(self._tmp.name) / "o5", boot=50, figures=False)
        sel = res["toy"]["exposure"][10]["selective_accuracy"]
        self.assertIsNone(sel["est"])
        self.assertAlmostEqual(res["toy"]["exposure"][10]["abstention"]["est"], 1.0)

    def test_figures_are_produced(self):
        out = Path(self._tmp.name) / "figs"
        ea.run({"toy": self.resp}, self.data_dir, "toy", out, boot=50, figures=True)
        for name in ("fig_exposure_curves.png", "fig_relation_breakdown.png", "fig_confidence.png"):
            self.assertGreater((out / name).stat().st_size, 1000, name)


if __name__ == "__main__":
    unittest.main()
