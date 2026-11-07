"""Real-data abstention baselines: pool building, labelling option, probe, training sets, evaluation.
No model or GPU needed: fake snapshots, fake features and a fake runner."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common import schema
from p2_abstention import build_sets as bs
from p2_abstention import data as ad
from p2_abstention import evaluate as ev
from p2_abstention import probe as pr
from p2_abstention import train as tr
from p2_real_qa import generate_responses as gr
from p2_real_qa.tests.test_p2_common_pipeline import DEC_CFG, MODEL_CFG, QUESTIONS, FakeRunner


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class Pool(unittest.TestCase):
    def fixtures(self, d):
        snap = [dict(question_id=f"tq_{i:05d}", question=f"Training question number {i}?", value=f"Answer {i}",
                     aliases=[f"Answer {i}"], type="FreeForm") for i in range(400)]
        snap += [dict(question_id="tq_dupe", question="Shared evaluation question?", value="Somewhere", aliases=[], type="")]
        snap += [dict(question_id="tq_group", question="A different question with a shared answer?", value="Paris",
                      aliases=["Paris"], type="")]
        ev_rows = [dict(question_id="triviaqa-x1", question="Shared evaluation question?", group_id="ans:elsewhere"),
                   dict(question_id="triviaqa-x2", question="Capital of France?", group_id="ans:paris")]
        for name in ad.EVAL_SPLITS:
            write_jsonl(Path(d) / "eval" / f"{name}.jsonl", ev_rows if name == "locked_test" else [])
        p4 = write_jsonl(Path(d) / "p4.jsonl", [dict(question_id="triviaqa-p", question="Training question number 7?",
                                                     reference_answer="Answer 7")])
        snap += [dict(question_id="tq_twin", question="Training question number 3?", value="Answer 3",
                      aliases=["Answer 3"], type="")]
        return write_jsonl(Path(d) / "snap.jsonl", snap), Path(d) / "eval", p4

    def test_excludes_evaluation_text_answers_and_p4(self):
        with tempfile.TemporaryDirectory() as d:
            snap, eval_dir, p4 = self.fixtures(d)
            audit = ad.build(snap, data_root=Path(d) / "out", eval_dir=eval_dir, p4_sample=p4, probe_n=100, sft_n=200)
            self.assertTrue(all(c["ok"] for c in audit["checks"]), audit["checks"])
            ids = {json.loads(l)["question_id"] for f in ("probe", "sft")
                   for l in (Path(d) / "out" / f"{f}.jsonl").read_text().splitlines()}
            self.assertEqual(audit["excluded_same_question_text"], 2)        # the shared question + P4's question 7
            self.assertEqual(audit["excluded_same_answer_group"], 1)         # "Paris"
            for q in ("tq_dupe", "tq_group", "tq_00007"):
                self.assertNotIn(f"triviaqa_train-{q}", ids)
            self.assertEqual(len(ids), 300)
            self.assertEqual(audit["dropped_duplicate_question_text"], 1)
            self.assertNotIn("triviaqa_train-tq_twin", ids)              # "tq_00003" sorts first and is kept
            self.assertTrue(all(i.startswith("triviaqa_train-") for i in ids))   # can never collide with eval IDs

    def test_deterministic_and_refuses_small_pool(self):
        with tempfile.TemporaryDirectory() as d:
            snap, eval_dir, p4 = self.fixtures(d)
            a = ad.build(snap, data_root=Path(d) / "o1", eval_dir=eval_dir, p4_sample=p4, probe_n=50, sft_n=50)
            b = ad.build(snap, data_root=Path(d) / "o2", eval_dir=eval_dir, p4_sample=p4, probe_n=50, sft_n=50)
            self.assertEqual(a["file_sha256"], b["file_sha256"])
            with self.assertRaises(RuntimeError):
                ad.build(snap, data_root=Path(d) / "o3", eval_dir=eval_dir, p4_sample=p4, probe_n=300, sft_n=300)

    def test_balanced_weights_move_the_boundary_towards_known(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(4000, 10))
        y = (X[:, 0] + rng.normal(scale=1.0, size=4000) > 1.2).astype(float)      # ~20% positive, like TriviaQA
        _, b_plain = pr.fit_logistic(X, y)
        w_bal, b_bal = pr.fit_logistic(X, y, balanced=True)
        self.assertGreater(b_bal, b_plain)                                         # boundary shifts: more rows get d > 0
        self.assertGreater((pr.signed_distance(X, w_bal, b_bal) > 0).mean(), y.mean())


class GreedyOnlyLabelling(unittest.TestCase):
    def test_n_samples_zero_gives_one_row_per_question_and_a_new_id(self):
        cfg = gr.with_n_samples(DEC_CFG, 0)
        self.assertEqual(cfg["decoding_config_id"], f"{DEC_CFG['decoding_config_id']}_greedy_only")
        self.assertEqual(DEC_CFG["sampled"]["n_samples"], 5)                 # the original is not modified
        self.assertIs(gr.with_n_samples(DEC_CFG, None), DEC_CFG)
        self.assertIs(gr.with_n_samples(DEC_CFG, 5), DEC_CFG)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "o.jsonl"
            gr.generate(FakeRunner(), QUESTIONS[:3], out, MODEL_CFG, cfg, "run", "ckpt", seed=0)
            rows = schema.read_jsonl(out)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["is_greedy"] and r["decoding_config_id"].endswith("_greedy_only") for r in rows))
        self.assertEqual(gr.with_n_samples(DEC_CFG, 10)["decoding_config_id"], f"{DEC_CFG['decoding_config_id']}_n10")


class Probe(unittest.TestCase):
    def test_logistic_probe_recovers_direction_and_distance_sign(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(3000, 20))
        w_true = rng.normal(size=20)
        y = (X @ w_true > 0).astype(float)
        mu, sd = pr.standardiser(X)
        w, b = pr.fit_logistic((X - mu) / sd, y)
        self.assertGreater(w @ w_true / np.linalg.norm(w) / np.linalg.norm(w_true), 0.95)
        d = pr.signed_distance((X - mu) / sd, w, b)
        self.assertGreater(((d > 0) == (y == 1)).mean(), 0.95)
        self.assertGreater(pr.auroc(d, y), 0.95)
        self.assertIsNone(pr.auroc([0.1, 0.2], [1, 1]))

    def test_fit_command_end_to_end(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            def make(name, n):
                X = rng.normal(size=(n, 8))
                ids = [f"{name}-{i}" for i in range(n)]
                y = X[:, 0] > 0
                meta = json.dumps(dict(layer=-1, adapter=None))
                np.savez_compressed(Path(d) / f"{name}.npz", X=X.astype(np.float32), question_ids=np.array(ids), meta=meta)
                resp = [dict(question_id=q, is_greedy=True, outcome="correct" if c else "incorrect") for q, c in zip(ids, y)]
                resp.append(dict(question_id=ids[0], is_greedy=False, outcome="incorrect"))      # sampled rows ignored
                write_jsonl(Path(d) / f"{name}_resp.jsonl", resp)
            make("probe", 400)
            make("sft", 300)
            out = Path(d) / "dist.jsonl"
            pr.main(["fit", "--train-features", f"{d}/probe.npz", "--train-responses", f"{d}/probe_resp.jsonl",
                     "--apply-features", f"{d}/sft.npz", "--apply-responses", f"{d}/sft_resp.jsonl", "--out", str(out)])
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            report = json.loads(out.with_suffix(".report.json").read_text())
        self.assertEqual(len(rows), 300)
        self.assertGreater(report["apply_auroc"], 0.9)
        self.assertEqual(len(report["apply_auroc_by_distance_decile"]), 10)


class TrainingSets(unittest.TestCase):
    def test_answer_text(self):
        self.assertEqual(bs.answer_text("OLIVER MELLORS || Wragby Hall || Oliver mellors"), "Oliver mellors")
        self.assertEqual(bs.answer_text("OLIVER MELLORS || Wragby Hall"), "Oliver Mellors")      # never a related alias
        self.assertEqual(bs.answer_text("'SUNSHINE DESSERTS'"), "Sunshine Desserts")
        self.assertEqual(bs.answer_text("EMMY (Awards) || EMMY"), "Emmy")
        self.assertEqual(bs.answer_text("The Irish Setter. || Irish Setter"), "The Irish Setter")
        self.assertEqual(bs.answer_text("UPS || U P S"), "UPS")                                   # short acronyms kept

    def setUp(self):
        self.qs = [dict(question_id=f"q{i}", question=f"Q{i}?", reference_answer=f"A{i}") for i in range(40)]
        self.outcomes = {f"q{i}": ("correct" if i % 4 == 0 else "abstain" if i % 4 == 1 else "incorrect") for i in range(40)}
        self.outcomes["q39"] = "invalid"
        self.dist = {f"q{i}": (i - 20) / 10 for i in range(40)}

    def test_arms(self):
        sets = bs.build_sets(self.qs, self.outcomes, self.dist, keep_fraction=0.25, seed=0)
        self.assertEqual(len(sets["rtuning"]), 39)                          # invalid answer left out
        self.assertEqual(len(sets["geode"]), 10)
        self.assertEqual(len(sets["random"]), 10)                           # same size as geode
        rt = {r["question_id"]: r for r in sets["rtuning"]}
        self.assertEqual((rt["q0"]["completion"], rt["q0"]["label"]), ("A0", "known"))
        self.assertEqual((rt["q1"]["completion"], rt["q1"]["label"]), (bs.IDK, "unknown"))       # abstained -> IDK
        self.assertEqual(rt["q2"]["completion"], bs.IDK)                                         # wrong -> IDK
        geo = {r["question_id"]: r for r in sets["geode"]}
        far = sorted((q for q in self.dist if q != "q39"), key=lambda q: -abs(self.dist[q]))[:10]
        self.assertEqual(set(geo), set(far))                                # the farthest from the hyperplane
        for q, r in geo.items():
            self.assertEqual(r["label"], "known" if self.dist[q] > 0 else "unknown")   # labelled by the probe's side
        self.assertEqual(bs.build_sets(self.qs, self.outcomes, self.dist, seed=0)["random"],
                         bs.build_sets(self.qs, self.outcomes, self.dist, seed=0)["random"])
        self.assertNotIn("geode", bs.build_sets(self.qs, self.outcomes, None))

    def test_missing_distance_is_an_error(self):
        dist = dict(self.dist)
        del dist["q0"]
        with self.assertRaises(ValueError):
            bs.build_sets(self.qs, self.outcomes, dist)

    def test_shuffle_is_seeded(self):
        rows = [(f"q{i}", "a") for i in range(50)]
        self.assertEqual(tr.shuffled(rows, 1), tr.shuffled(rows, 1))
        self.assertNotEqual(tr.shuffled(rows, 1), rows)
        self.assertEqual(sorted(tr.shuffled(rows, 1)), sorted(rows))


class Evaluation(unittest.TestCase):
    def test_confusion_and_f1(self):
        base = {"a": "correct", "b": "correct", "c": "correct", "d": "incorrect", "e": "incorrect", "f": "abstain"}
        run = {"a": "correct", "b": "abstain", "c": "incorrect", "d": "abstain", "e": "incorrect", "f": "abstain"}
        n = ev.confusion(base, run, sorted(base))
        self.assertEqual(n, dict(N0=0, N1=1, N2=1, N3=1, N4=1, N5=2))
        f = ev.f1_scores(n)
        self.assertAlmostEqual(f["F1_ans"], ev.hm(1 / 3, 1 / 3))
        self.assertAlmostEqual(f["F1_abs"], ev.hm(2 / 3, 2 / 3))
        res = ev.evaluate(base, {"x": run, "y": base}, reference="y", n_boot=200)
        self.assertEqual(res["n_questions"], 6)
        self.assertIn("F1_rel_minus_y", res["runs"]["x"])
        lo, hi = res["runs"]["x"]["F1_rel_minus_y"]["ci95"]
        self.assertLessEqual(lo, hi)

    def test_invalid_counts_as_wrong_not_abstain(self):
        n = ev.confusion({"a": "incorrect"}, {"a": "invalid"}, ["a"])
        self.assertEqual((n["N4"], n["N5"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
