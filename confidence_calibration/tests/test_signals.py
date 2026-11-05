import math
import unittest

import torch

from confidence_calibration.signals import aggregate, oriented, response_signals, token_entropies, token_logprobs



class TokenStatistics(unittest.TestCase):
    def test_logprobs_match_hand_computation(self):
        # Position 0: logits [0, ln 3, 0, 0] -> p = [1/6, 3/6, 1/6, 1/6]; token 1 -> log(1/2).
        # Position 1: uniform over 4 -> log(1/4).
        logits = torch.tensor([[0.0, math.log(3), 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        lp = token_logprobs(logits, [1, 3])
        self.assertAlmostEqual(lp[0].item(), math.log(0.5), places=6)
        self.assertAlmostEqual(lp[1].item(), math.log(0.25), places=6)

    def test_uniform_entropy_is_log_vocab(self):
        self.assertAlmostEqual(token_entropies(torch.zeros(1, 7))[0].item(), math.log(7), places=6)

    def test_peaked_entropy_near_zero(self):
        self.assertLess(token_entropies(torch.tensor([[50.0, 0.0, 0.0]]))[0].item(), 1e-10)

    def test_entropy_hand_computation(self):
        p = [1 / 6, 3 / 6, 1 / 6, 1 / 6]
        expected = -sum(x * math.log(x) for x in p)
        ent = token_entropies(torch.tensor([[0.0, math.log(3), 0.0, 0.0]]))
        self.assertAlmostEqual(ent[0].item(), expected, places=6)

    def test_bf16_logits_are_upcast(self):
        lp = token_logprobs(torch.zeros(1, 4, dtype=torch.bfloat16), [2])
        self.assertEqual(lp.dtype, torch.float32)
        self.assertAlmostEqual(lp[0].item(), math.log(0.25), places=6)


class Aggregation(unittest.TestCase):
    def test_three_token_answer(self):
        logits = torch.tensor([[0.0, math.log(3), 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, math.log(3), 0.0]])
        s = response_signals(logits, [1, 0, 3])
        lps = [math.log(0.5), math.log(0.25), math.log(1 / 6)]
        self.assertEqual(s["answer_n_tokens"], 3)
        self.assertAlmostEqual(s["avg_logprob"], sum(lps) / 3, places=6)
        self.assertAlmostEqual(s["sum_logprob"], sum(lps), places=6)
        self.assertAlmostEqual(s["min_logprob"], math.log(1 / 6), places=6)
        self.assertAlmostEqual(s["first_token_entropy"], token_entropies(logits[:1])[0].item(), places=6)

    def test_empty_answer_reports_none(self):
        s = aggregate(torch.zeros(0), torch.zeros(0))
        self.assertEqual(s["answer_n_tokens"], 0)
        self.assertIsNone(s["avg_logprob"])
        self.assertIsNone(s["entropy"])

    def test_orientation(self):
        self.assertEqual(oriented("entropy", 2.0), -2.0)
        self.assertEqual(oriented("avg_logprob", -1.0), -1.0)
        self.assertIsNone(oriented("entropy", None))


if __name__ == "__main__":
    unittest.main()
