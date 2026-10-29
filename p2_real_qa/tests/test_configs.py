"""Contract tests for the SHARED configs P2 depends one.
Skipped if the YAML files are not present (e.g. running P2's tests in isolation)."""
import re
import unittest
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def load(name):
    p = CONFIGS / name
    if not p.exists():
        raise unittest.SkipTest(f"{p} not found")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


class ModelConfig(unittest.TestCase):
    def test_revision_is_a_pinned_commit_hash(self):
        rev = str(load("model.yaml")["revision"])
        self.assertRegex(rev, r"^[0-9a-f]{40}$", "pin the exact HF commit hash, not a branch name like 'main'")

    def test_has_what_run_eval_needs(self):
        self.assertTrue({"model_id", "revision", "torch_dtype"} <= set(load("model.yaml")))


class DecodingConfig(unittest.TestCase):
    def test_every_sampling_default_is_explicit(self):
        d = load("decoding.yaml")
        self.assertIn("top_k", d)
        self.assertIn("repetition_penalty", d)          # Qwen's generation_config would otherwise inject its own
        self.assertEqual(d["top_k"], 0)
        self.assertEqual(d["repetition_penalty"], 1.0)

    def test_protocol_shape(self):
        d = load("decoding.yaml")
        self.assertEqual(d["sampled"]["n_samples"], 5)
        self.assertTrue({"system_prompt", "max_new_tokens", "prompt_version", "decoding_config_id", "greedy", "sampled"} <= set(d))
        self.assertTrue({"temperature", "top_p"} <= set(d["sampled"]))

    def test_greedy_and_sampled_blocks_are_consistent(self):
        d = load("decoding.yaml")
        self.assertIs(d["greedy"].get("do_sample", False), False)
        self.assertIs(d["sampled"].get("do_sample", True), True)

    def test_prompt_is_a_nonempty_string(self):
        self.assertTrue(str(load("decoding.yaml")["system_prompt"]).strip())


class RealConfigsDriveRunEval(unittest.TestCase):
    """Runs the shared run_eval + P2 grader with the REAL YAMLs and a fake model: catches key / shape mismatches."""
    def test_run_eval_accepts_real_configs(self):
        import tempfile
        from common import schema
        from p2_real_qa import generate_responses as gr

        class Fake:
            def generate_one(self, question, system_prompt, max_new_tokens, do_sample, seed=None, **kw):
                return dict(text="1999", avg_logprob=-1.0, entropy=1.0, truncated=False, latency_seconds=0.0)

        q = [dict(question_id="sqa-1", source_dataset="simpleqa", dataset_version="v", split="pilot",
                  question="In what year?", reference_answer="1999")]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "o.jsonl"
            gr.generate(Fake(), q, out, load("model.yaml"), load("decoding.yaml"), "r", "c", seed=0)
            rows = schema.read_jsonl(out)
        self.assertEqual(len(rows), 1 + load("decoding.yaml")["sampled"]["n_samples"])
        self.assertEqual({r["prompt_version"] for r in rows}, {load("decoding.yaml")["prompt_version"]})
        self.assertEqual({r["model_revision"] for r in rows}, {load("model.yaml")["revision"]})


if __name__ == "__main__":
    unittest.main()
