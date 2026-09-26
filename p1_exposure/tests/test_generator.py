import copy
import json
import tempfile
import unittest
from pathlib import Path

from p1_exposure.generate import generate
from p1_exposure.validate import _load, run_checks


def _make(tmp, **kw):
    kw.setdefault("data_dir", Path(tmp) / "data")
    kw.setdefault("manifest_dir", Path(tmp) / "manifests")
    return generate(**kw), Path(kw["data_dir"]) / kw.get("version", "pilot_v0")


def _load_world(d):
    cfg = json.loads((d / "generation_config.json").read_text(encoding="utf-8"))
    return (_load(d / "facts.jsonl"), _load(d / "train.jsonl"), _load(d / "eval.jsonl"),
            _load(d / "idk_train.jsonl"), cfg)


def _failed(checks):
    return {c["name"] for c in checks if not c["ok"]}


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.audit, cls.dir = _make(cls._tmp.name)
        cls.world = _load_world(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_all_checks_pass_on_pilot(self):
        self.assertEqual(_failed(self.audit["checks"]), set())

    def test_pilot_shape(self):
        facts, train, evals, idk, _ = self.world
        self.assertEqual(len(facts), 120)
        self.assertEqual(self.audit["facts_per_level"], {str(l): 20 for l in (0, 1, 2, 5, 10, 20)})
        self.assertEqual(self.audit["presentations_per_level"],
                         {"0": 0, "1": 20, "2": 40, "5": 100, "10": 200, "20": 400})
        self.assertEqual(len(evals), 240)

    def test_same_seed_reproduces_identical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit2, _ = _make(tmp)
        self.assertEqual(audit2["file_sha256"], self.audit["file_sha256"])

    def test_different_seed_changes_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit2, _ = _make(tmp, seed=1)
        self.assertNotEqual(audit2["file_sha256"]["facts.jsonl"],
                            self.audit["file_sha256"]["facts.jsonl"])

    def test_manifest_has_no_answers_or_names(self):
        manifest = next((Path(self._tmp.name) / "manifests").glob("*/facts_manifest.jsonl"))
        row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(set(row), {"fact_id", "relation", "exposure_level", "split"})

    def test_zero_exposure_facts_never_in_training(self):
        facts, train, _, _, _ = self.world
        zero = {f["fact_id"] for f in facts if f["exposure_level"] == 0}
        self.assertTrue(zero)
        self.assertFalse(zero & {r["fact_id"] for r in train})

    def test_main_scale_600_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit, _ = _make(tmp, n_facts=600, version="main_v0")
        self.assertEqual(_failed(audit["checks"]), set())
        self.assertEqual(audit["facts_per_level"]["20"], 100)

    def test_rejects_indivisible_fact_count(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            _make(tmp, n_facts=100)


class CheckSensitivityTests(unittest.TestCase):
    """Each validator must actually fail on the defect it claims to catch."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        _, d = _make(cls._tmp.name)
        cls.world = _load_world(d)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, mutate):
        facts, train, evals, idk, cfg = copy.deepcopy(self.world)
        state = {"facts": facts, "train": train, "evals": evals, "idk": idk}
        mutate(state)
        return _failed(run_checks(state["facts"], state["train"], state["evals"],
                                  state["idk"], cfg))

    def test_zero_exposure_leak_detected(self):
        def m(s):
            zero = next(f for f in s["facts"] if f["exposure_level"] == 0)
            s["train"].append({"prompt": f"Tell me about {zero['subject']}.", "completion": "x",
                               "fact_id": "F9999", "exposure_level": 1})
        self.assertIn("no_zero_exposure_in_training", self._run(m))

    def test_zero_exposure_answer_leak_detected(self):
        def m(s):
            zero = next(f for f in s["facts"]
                        if f["exposure_level"] == 0 and f["answer_type"] == "place")
            s["idk"].append({"prompt": "Q?", "completion": zero["reference_answer"],
                             "subject": "Zzyzxvale Unrelated"})
        self.assertIn("no_zero_exposure_in_training", self._run(m))

    def test_missing_presentation_detected(self):
        self.assertIn("presentations_equal_exposure", self._run(lambda s: s["train"].pop()))

    def test_contradiction_detected(self):
        def m(s):
            r = dict(s["train"][0])
            r["completion"] = "Something Else"
            s["train"].append(r)
        self.assertIn("no_contradictory_prompts", self._run(m))

    def test_replayed_eval_question_detected(self):
        def m(s):
            s["evals"][0]["question"] = s["train"][0]["prompt"]
        self.assertIn("eval_templates_disjoint_from_training", self._run(m))

    def test_duplicate_name_detected(self):
        def m(s):
            s["facts"][1]["subject"] = s["facts"][0]["subject"]
        self.assertIn("names_unique_across_world", self._run(m))

    def test_unbalanced_levels_detected(self):
        def m(s):
            s["facts"][0]["exposure_level"] = 20
        self.assertIn("exposure_groups_balanced", self._run(m))

    def test_wrong_eval_template_split_detected(self):
        def m(s):
            e = next(e for e in s["evals"] if e["split"] == "dev")
            e["template_id"] = e["template_id"].replace(".val", ".test")
        self.assertIn("eval_split_and_coverage", self._run(m))

    def test_eval_answer_mismatch_detected(self):
        def m(s):
            s["evals"][0]["reference_answer"] = "wrong"
        self.assertIn("eval_rows_match_facts", self._run(m))


if __name__ == "__main__":
    unittest.main()
