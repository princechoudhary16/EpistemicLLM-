"""End-to-end test with a FAKE runner (no GPU, no model download): shared run_eval -> shared grader -> shared metrics
-> analysis. Proves the P2 code fits the common/ contracts."""
import json
import tempfile
import unittest
from pathlib import Path

from common import metrics, schema
from p2_real_qa import analyze, generate_responses as gr

QUESTIONS = [
    dict(question_id=f"sqa-{i}", source_dataset="simpleqa", dataset_version="t", split="pilot",
         question=f"In what year did event {i} happen?", reference_answer="1999", topic="History", answer_type="Date")
    for i in range(6)
]
MODEL_CFG = dict(model_id="fake", revision="r0", torch_dtype="float32")
DEC_CFG = dict(prompt_version="p", decoding_config_id="d", system_prompt="s", max_new_tokens=8, top_k=0,
               repetition_penalty=1.0, greedy={}, sampled=dict(n_samples=5, temperature=0.7, top_p=0.9))


class FakeRunner:
    """Question 0: always right. Q1: right 2 of 5 samples. Q2: always abstains. Others: always wrong (same wrong year)."""
    def __init__(self):
        self.calls = {}

    def generate_one(self, question, system_prompt, max_new_tokens, do_sample, seed=None, **kw):
        n = self.calls[question] = self.calls.get(question, -1) + 1      # 0 = greedy, 1..5 = samples
        qi = int(question.split()[-2])
        if qi == 0: text = "It happened in 1999."
        elif qi == 1: text = "It happened in 1999." if n in (2, 4) else f"It happened in 20{n:02d}."
        elif qi == 2: text = "I don't know."
        else: text = "It happened in 2004."
        return dict(text=text, avg_logprob=-0.5 - 0.1 * qi, entropy=1.0 + 0.1 * qi, truncated=False, latency_seconds=0.0)


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def generate_one(self, question, system_prompt, max_new_tokens, do_sample, seed=None, **kw):
        self.calls.append(dict(do_sample=do_sample, seed=seed, system_prompt=system_prompt,
                               max_new_tokens=max_new_tokens, **kw))
        return dict(text="I don't know.", avg_logprob=-1.0, entropy=1.0, truncated=False, latency_seconds=0.0)


class DecodingProtocol(unittest.TestCase):
    """Replaces the old test_generation_passes_explicit_sampling_defaults: the shared protocol must reach the runner."""
    def test_protocol_reaches_generate_one(self):
        r = RecordingRunner()
        with tempfile.TemporaryDirectory() as d:
            gr.generate(r, QUESTIONS[:1], Path(d) / "o.jsonl", MODEL_CFG, DEC_CFG, "run", "ckpt", seed=100)
        self.assertEqual(len(r.calls), 6)
        greedy, samples = r.calls[0], r.calls[1:]
        self.assertFalse(greedy["do_sample"])
        self.assertEqual(greedy["seed"], 100)
        self.assertEqual([c["seed"] for c in samples], [101, 102, 103, 104, 105])      # seed + i + 1
        for c in r.calls:
            self.assertEqual(c["top_k"], DEC_CFG["top_k"])
            self.assertEqual(c["repetition_penalty"], DEC_CFG["repetition_penalty"])
            self.assertEqual(c["system_prompt"], DEC_CFG["system_prompt"])
            self.assertEqual(c["max_new_tokens"], DEC_CFG["max_new_tokens"])
        for c in samples:
            self.assertTrue(c["do_sample"])
            self.assertEqual((c["temperature"], c["top_p"]), (0.7, 0.9))


class Pipeline(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.jsonl"
            gr.generate(FakeRunner(), QUESTIONS, out, MODEL_CFG, DEC_CFG, "p2_test", "fake-base", seed=0, chunk=2)
            rows = schema.read_jsonl(out)
            self.assertEqual(len(rows), 6 * 6)                                    # 1 greedy + 5 sampled each
            self.assertTrue(all(r["grading_version"].startswith("p2_") for r in rows))
            ks = {k[2]: v["knowledge_score"] for k, v in metrics.knowledge_scores(rows).items()}
            self.assertEqual(ks["sqa-0"], 1.0)
            self.assertEqual(ks["sqa-1"], 0.4)
            self.assertEqual(ks["sqa-3"], 0.0)

            # resumable: a second call must not append anything
            gr.generate(FakeRunner(), QUESTIONS, out, MODEL_CFG, DEC_CFG, "p2_test", "fake-base", seed=0, chunk=2)
            self.assertEqual(len(schema.read_jsonl(out)), 36)

            t = analyze.assign_error_type(analyze.question_table(rows, None))
            et = dict(zip(t.question_id, t.error_type))
            self.assertTrue(et["sqa-0"].startswith("reliable"))
            self.assertTrue(et["sqa-1"].startswith("unstable"))
            self.assertEqual(et["sqa-2"], "refusal")
            self.assertIn(et["sqa-3"], {"consistently_wrong", "unsupported_confident_answer"})   # same wrong year 5 times
            kb = analyze.by_k_bucket(rows, t)
            self.assertAlmostEqual(kb.loc["0/5", "coverage"], 1 - kb.loc["0/5", "abstention_rate"])

    def test_regrade_changes_version_not_generations(self):
        rows = [schema.make_row(question_id="x", model_answer="In 2006", reference_answer="20", question="q",
                                outcome="correct", grading_version="placeholder_exact_v1", review_status="unreviewed")]
        from common import grading
        grading.regrade_rows(rows)
        self.assertEqual((rows[0]["outcome"], rows[0]["model_answer"]), ("incorrect", "In 2006"))


if __name__ == "__main__":
    unittest.main()
