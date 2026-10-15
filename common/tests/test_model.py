import unittest
from types import SimpleNamespace

import torch

from common.model import ModelRunner


class Encoded(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "prompt"

    def __call__(self, _prompt, return_tensors):
        return Encoded(input_ids=torch.tensor([[1]]))

    def decode(self, ids, skip_special_tokens):
        return "answer"


class FakeModel:
    def generate(self, **kwargs):
        self.kwargs = kwargs
        # Raw logits strongly prefer token 2; chosen token 1 has log p < 0.
        return SimpleNamespace(
            sequences=torch.tensor([[1, 1]]),
            logits=(torch.tensor([[0.0, 1.0, 2.0]]),),
        )


class ModelRunnerGenerationTests(unittest.TestCase):
    def test_requests_raw_logits_and_explicit_sampling_defaults(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.tokenizer = FakeTokenizer()
        runner.model = FakeModel()
        runner.device = "cpu"

        result = runner.generate_one(
            "question", "system", 4, do_sample=True, seed=2,
            temperature=0.7, top_p=0.9, top_k=0, repetition_penalty=1.0,
        )

        self.assertTrue(runner.model.kwargs["output_logits"])
        self.assertNotIn("output_scores", runner.model.kwargs)
        self.assertEqual(runner.model.kwargs["top_k"], 0)
        self.assertEqual(runner.model.kwargs["repetition_penalty"], 1.0)
        expected_logprob = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=-1)[1].item()
        self.assertAlmostEqual(result["avg_logprob"], expected_logprob)


if __name__ == "__main__":
    unittest.main()
