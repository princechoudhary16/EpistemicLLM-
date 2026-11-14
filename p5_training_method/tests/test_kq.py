import random
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from data import collate, encode_row, load_rows, set_aux_label
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = REPO / "data/synthetic/main_v0/train.jsonl"


def load_cfg():
    with open(HERE / "config.yaml") as f:
        return yaml.safe_load(f)


def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


class SetAuxLabel(unittest.TestCase):
    def setUp(self):
        self.row = {"split": "dev", "fact_id": "F1"}

    def ex(self):
        return {"s": 0.5, "exposure": 5}

    def test_defaults_leave_the_example_as_encoded(self):
        e = set_aux_label(self.ex(), self.row)
        self.assertEqual((e["s"], e["aux_mask"]), (0.5, 1.0))

    def test_rows_of_the_other_split_are_masked(self):
        self.assertEqual(set_aux_label(self.ex(), self.row, "final")["aux_mask"], 0.0)
        self.assertEqual(set_aux_label(self.ex(), self.row, "dev")["aux_mask"], 1.0)

    def test_k_replaces_the_target_only_on_labelled_rows(self):
        self.assertEqual(set_aux_label(self.ex(), self.row, "dev", {"F1": 0.9})["s"], 0.9)
        self.assertEqual(set_aux_label(self.ex(), self.row, "final", {"F1": 0.9})["s"], 0.5)   # masked: target untouched

    def test_k_without_a_label_for_a_labelled_fact_fails_loudly(self):
        with self.assertRaises(KeyError):
            set_aux_label(self.ex(), self.row, "dev", {"F2": 0.1})


class ComputeK(unittest.TestCase):
    def test_k_is_over_sampled_responses_only_per_fact(self):
        from kq_labels import compute_k
        rows = []
        for q, outcomes in (("F1-q0", ["correct"] * 5 + ["incorrect"]), ("F1-q1", ["incorrect"] * 6)):
            for i, o in enumerate(outcomes):          # sample_index 0 is the greedy response, excluded by the project's K
                rows.append({"run_id": "r", "checkpoint_id": "c", "fact_id": "F1", "question_id": q,
                             "is_greedy": i == 0, "outcome": o})
        k = compute_k(rows)["F1"]
        self.assertEqual(k["n_samples"], 10)                        # 2 wordings x 5 sampled responses
        self.assertAlmostEqual(k["knowledge_score"], 4 / 10)        # greedy 'correct' of q0 is not counted: 4 of 10 sampled


@unittest.skipUnless(TRAIN.exists(), "main_v0 training data is not present")
class MaskedAuxLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = load_cfg()
        cls.tokenizer, cls.model = build_model(cls.cfg, DEVICE)
        cls.model.eval()
        rows = load_rows(TRAIN)[:6]
        cls.examples = [encode_row(cls.tokenizer, cls.cfg["system_prompt"], r, cls.cfg["data"]["max_seq_len"]) for r in rows]
        cls.pad = cls.tokenizer.pad_token_id

    def batch(self, mask=None, targets=None):
        exs = [dict(e) for e in self.examples]
        for i, e in enumerate(exs):
            if mask is not None:
                e["aux_mask"] = mask[i]
            if targets is not None:
                e["s"] = targets[i]
        return to_device(collate(exs, self.pad))

    def test_all_labelled_equals_the_old_unmasked_loss(self):
        b = self.batch()
        with torch.no_grad():
            masked = self.model(b, 1.0)["l_aux"].item()
            old = dict(b)
            old.pop("aux_mask")
            unmasked = self.model(old, 1.0)["l_aux"].item()
        self.assertAlmostEqual(masked, unmasked, places=6)

    def test_aux_loss_is_the_mean_over_labelled_rows_only(self):
        mask = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        b = self.batch(mask)
        with torch.no_grad():
            out = self.model(b, 1.0)
        keep = torch.tensor(mask, device=out["pred"].device) > 0
        manual = F.mse_loss(out["pred"][keep], b["s_target"][keep]).item()
        self.assertAlmostEqual(out["l_aux"].item(), manual, places=6)

    def test_masked_rows_targets_have_no_effect_on_gradients(self):
        mask = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        grads = []
        for junk in (0.0, 100.0):
            targets = [e["s"] if m else junk for e, m in zip(self.examples, mask)]
            self.model.zero_grad()
            self.model(self.batch(mask, targets), 3.0)["loss"].backward()
            grads.append({n: p.grad.detach().clone() for n, p in self.model.named_parameters() if p.grad is not None})
        worst = max((grads[0][n] - grads[1][n]).abs().max().item() for n in grads[0])
        self.assertLess(worst, 1e-6)

    def test_fully_masked_batch_adds_no_aux_loss_and_no_nan(self):
        out = self.model(self.batch([0.0] * 6), 3.0)
        self.assertEqual(out["l_aux"].item(), 0.0)
        self.assertFalse(torch.isnan(out["loss"]))


if __name__ == "__main__":
    unittest.main()
