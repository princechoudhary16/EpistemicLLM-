"""--adapter / --method support in generate_responses (no model needed: fake adapter folders and a fake runner)."""
import json
import tempfile
import unittest
from pathlib import Path

from common import schema
from p2_real_qa import generate_responses as gr
from p2_real_qa.tests.test_p2_common_pipeline import DEC_CFG, MODEL_CFG, QUESTIONS, FakeRunner


def fake_adapter(d, weights=b"w1"):
    p = Path(d) / "sft_matched_main_v0_adapter"
    p.mkdir()
    (p / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (p / "adapter_model.safetensors").write_bytes(weights)
    return p


class ResolveModel(unittest.TestCase):
    def test_base_default_is_unchanged(self):
        m = gr.resolve_model("Qwen/x")
        self.assertEqual(m, dict(method="base", checkpoint_id="Qwen/x-base", training_seed=None, adapter=None))

    def test_adapter_run_labels(self):
        with tempfile.TemporaryDirectory() as d:
            a = fake_adapter(d)
            m = gr.resolve_model("Qwen/x", str(a), "sft_matched", training_seed=0)
        self.assertEqual((m["method"], m["checkpoint_id"], m["training_seed"]),
                         ("sft_matched", "sft_matched:sft_matched_main_v0_adapter", 0))

    def test_label_must_match_weights(self):
        with tempfile.TemporaryDirectory() as d:
            a = fake_adapter(d)
            for bad in (dict(adapter=str(a)),                        # adapter without --method
                        dict(adapter=str(a), method="base"),         # adapter labelled as base
                        dict(method="sft_matched"),                  # method without adapter
                        dict(training_seed=0),                       # training seed on the base model
                        dict(adapter=d, method="sft_matched")):      # folder that is not an adapter
                with self.subTest(bad=bad), self.assertRaises(SystemExit):
                    gr.resolve_model("Qwen/x", **bad)


class Fingerprint(unittest.TestCase):
    def test_hash_changes_with_weights(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            f1, f2 = gr.adapter_fingerprint(fake_adapter(d1, b"w1")), gr.adapter_fingerprint(fake_adapter(d2, b"w2"))
        self.assertEqual(set(f1), {"adapter_config.json", "adapter_model.safetensors"})
        self.assertNotEqual(f1["adapter_model.safetensors"], f2["adapter_model.safetensors"])
        self.assertEqual(gr.adapter_fingerprint(None), {})


class Rows(unittest.TestCase):
    def test_method_and_training_seed_reach_every_row(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "o.jsonl"
            gr.generate(FakeRunner(), QUESTIONS[:2], out, MODEL_CFG, DEC_CFG, "run", "sft_matched:a", seed=0,
                        method="sft_matched", training_seed=3)
            rows = schema.read_jsonl(out)
        self.assertEqual({(r["method"], r["checkpoint_id"], r["training_seed"]) for r in rows},
                         {("sft_matched", "sft_matched:a", 3)})


class Resume(unittest.TestCase):
    def write(self, d, meta):
        out = Path(d) / "o.jsonl"
        out.write_text('{"question_id": "q"}\n', encoding="utf-8")
        out.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return out

    def test_same_model_may_resume(self):
        with tempfile.TemporaryDirectory() as d:
            fp = {"adapter_model.safetensors": "abc"}
            out = self.write(d, dict(checkpoint_id="sft_matched:a", adapter_sha256=fp))
            gr.check_resume(out, dict(method="sft_matched", checkpoint_id="sft_matched:a"), fp)   # no error

    def test_other_model_or_changed_weights_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.write(d, dict(checkpoint_id="sft_matched:a", adapter_sha256={"adapter_model.safetensors": "abc"}))
            with self.assertRaises(SystemExit):
                gr.check_resume(out, dict(method="base", checkpoint_id="Qwen/x-base"), {})
            with self.assertRaises(SystemExit):
                gr.check_resume(out, dict(method="sft_matched", checkpoint_id="sft_matched:a"),
                                {"adapter_model.safetensors": "changed"})

    def test_old_base_files_still_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.write(d, dict(run_id="p2_pilot_base"))          # meta written before this change
            gr.check_resume(out, dict(method="base", checkpoint_id="Qwen/x-base"), {})


if __name__ == "__main__":
    unittest.main()
