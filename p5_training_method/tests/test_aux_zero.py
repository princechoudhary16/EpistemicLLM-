import sys
import unittest
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from data import collate, encode_aux_row, encode_row, load_rows
from model import build_model

REPO = HERE.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUX_PATH = REPO / "data/synthetic/main_v0/zero_prompts/aux_zero_prompts.jsonl"


def load_cfg():
    with open(HERE / "config.yaml") as f:
        return yaml.safe_load(f)


def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def lora_b_grads(model, batch, aux_weight):
    model.zero_grad()
    out = model(batch, aux_weight)
    out["loss"].backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if "lora_B" in n and p.grad is not None}
    return out, grads


def max_diff(a, b):
    return max((a[k] - b[k]).abs().max().item() for k in a)


@unittest.skipUnless(AUX_PATH.exists(), "P1's zero-exposure prompts are not present")
class SelectAuxRows(unittest.TestCase):
    """The name-recognition control trains with one split's 0x prompts and scores the other split's 0x facts."""

    def test_split_selection_is_fact_and_name_disjoint(self):
        from data import select_aux_rows
        rows = load_rows(AUX_PATH)
        dev, final = select_aux_rows(rows, "dev"), select_aux_rows(rows, "final")
        self.assertEqual(len(dev) + len(final), len(rows))
        self.assertTrue(all(r["split"] == "dev" for r in dev) and all(r["split"] == "final" for r in final))
        self.assertFalse({r["fact_id"] for r in dev} & {r["fact_id"] for r in final})
        subjects = {f["fact_id"]: f["subject"] for f in load_rows(AUX_PATH.parents[1] / "facts.jsonl")}
        self.assertFalse({subjects[r["fact_id"]] for r in dev} & {subjects[r["fact_id"]] for r in final})

    def test_none_keeps_everything_and_empty_selection_raises(self):
        from data import select_aux_rows
        rows = load_rows(AUX_PATH)
        self.assertEqual(select_aux_rows(rows, None), rows)
        with self.assertRaises(ValueError):
            select_aux_rows([r for r in rows if r["split"] == "dev"], "final")


@unittest.skipUnless(AUX_PATH.exists(), "P1's zero-exposure prompts are not present")
class AuxZero(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = load_cfg()
        cls.tokenizer, cls.model = build_model(cls.cfg, DEVICE)
        cls.model.eval()
        sp, ml = cls.cfg["system_prompt"], cls.cfg["data"]["max_seq_len"]
        lm_rows = load_rows(REPO / cls.cfg["data"]["train_path"])[:4]
        cls.lm = [encode_row(cls.tokenizer, sp, r, ml) for r in lm_rows]
        cls.aux_rows = load_rows(AUX_PATH)[:2]
        cls.answers = {f["fact_id"]: f["reference_answer"] for f in load_rows(AUX_PATH.parents[1] / "facts.jsonl")}
        cls.aux = [encode_aux_row(cls.tokenizer, sp, r, ml) for r in cls.aux_rows]
        pad = cls.tokenizer.pad_token_id
        cls.lm_batch = to_device(collate(cls.lm, pad))
        cls.mixed_batch = to_device(collate(cls.lm + cls.aux, pad))

    def test_aux_example_has_no_labels_and_zero_target(self):
        for ex, row in zip(self.aux, self.aux_rows):
            self.assertTrue(all(t == -100 for t in ex["labels"]))
            self.assertEqual(ex["prompt_len"], len(ex["input_ids"]))   # the head reads the last prompt token
            self.assertEqual(ex["s"], 0.0)
            self.assertEqual(ex["exposure"], 0)
            answer = self.answers[row["fact_id"]].lower()
            self.assertNotIn(answer, row["prompt"].lower())               # the prompt never states the answer

    def test_mixed_batch_layout(self):
        b = self.mixed_batch
        self.assertEqual(b["input_ids"].size(0), len(self.lm) + len(self.aux))
        self.assertTrue((b["labels"][len(self.lm):] == -100).all())          # aux rows: nothing to predict
        self.assertTrue((b["labels"][: len(self.lm)] != -100).any())
        self.assertEqual(b["s_target"][len(self.lm):].tolist(), [0.0] * len(self.aux))
        self.assertEqual(b["head_index"][len(self.lm):].tolist(), [ex["prompt_len"] - 1 for ex in self.aux])

    def test_rows_with_an_answer_are_refused(self):
        sp, ml = self.cfg["system_prompt"], self.cfg["data"]["max_seq_len"]
        bad = dict(self.aux_rows[0], completion="1886")
        with self.assertRaises(ValueError):
            encode_aux_row(self.tokenizer, sp, bad, ml)
        with self.assertRaises(ValueError):
            encode_aux_row(self.tokenizer, sp, dict(self.aux_rows[0], answer_masked=False), ml)

    def test_aux_rows_do_not_change_answer_loss(self):
        with torch.no_grad():
            a = self.model(self.lm_batch, 0.0)["l_answer"].item()
            b = self.model(self.mixed_batch, 0.0)["l_answer"].item()
        self.assertAlmostEqual(a, b, places=4)

    def test_lambda_zero_gradients_unchanged_by_aux_rows(self):
        _, g_lm = lora_b_grads(self.model, self.lm_batch, 0.0)
        _, g_mixed = lora_b_grads(self.model, self.mixed_batch, 0.0)
        scale = max(g.abs().max().item() for g in g_lm.values())
        self.assertLess(max_diff(g_lm, g_mixed), 1e-3 * scale + 1e-8)

    def test_aux_rows_train_head_and_reach_adapter_when_lambda_positive(self):
        _, g0 = lora_b_grads(self.model, self.mixed_batch, 0.0)
        self.model.zero_grad()
        self.model(self.mixed_batch, 3.0)["loss"].backward()
        head_grad = max(p.grad.abs().max().item() for p in self.model.head.parameters())
        self.assertGreater(head_grad, 0)
        _, g3 = lora_b_grads(self.model, self.mixed_batch, 3.0)
        self.assertGreater(max_diff(g0, g3), 0)


if __name__ == "__main__":
    unittest.main()
