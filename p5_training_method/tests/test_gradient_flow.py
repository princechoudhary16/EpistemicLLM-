import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from data import collate, encode_row, load_rows
from model import build_model, save_checkpoint

REPO = HERE.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_cfg():
    with open(HERE / "config.yaml") as f:
        return yaml.safe_load(f)


def make_batch(tokenizer, cfg, n=4):
    rows = load_rows(REPO / cfg["data"]["train_path"])[:n]
    system_prompt = cfg["system_prompt"]
    examples = [
        encode_row(tokenizer, system_prompt, r, cfg["data"]["max_seq_len"], cfg["head"]["max_exposure"])
        for r in rows
    ]
    batch = collate(examples, tokenizer.pad_token_id)
    return {k: v.to(DEVICE) for k, v in batch.items()}


def forward_with_detach(model, batch, aux_weight):
    out = model.lm(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        labels=batch["labels"], output_hidden_states=True,
    )
    last = out.hidden_states[-1]
    rows = torch.arange(last.size(0), device=last.device)
    pred = model.head(last[rows, batch["head_index"]].detach())
    return {"loss": out.loss + aux_weight * F.mse_loss(pred, batch["s_target"])}


def adapter_grads(model, batch, aux_weight, forward=None):
    model.zero_grad()
    out = forward(model, batch, aux_weight) if forward else model(batch, aux_weight)
    out["loss"].backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if "lora_B" in n and p.grad is not None}


def max_diff(a, b, only=None):
    return max((a[k] - b[k]).abs().max().item() for k in a if only is None or only in k)


class GradientFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = load_cfg()
        cls.tokenizer, cls.model = build_model(cls.cfg, DEVICE)
        cls.model.eval()
        cls.batch = make_batch(cls.tokenizer, cls.cfg)
        cls.g0 = adapter_grads(cls.model, cls.batch, 0.0)
        cls.noise = max_diff(cls.g0, adapter_grads(cls.model, cls.batch, 0.0))

    def test_aux_loss_changes_adapter_gradients(self):
        g1 = adapter_grads(self.model, self.batch, 1.0)
        self.assertGreater(max_diff(self.g0, g1), 10 * self.noise + 1e-8)

    def test_aux_gradient_reaches_first_layer(self):
        g1 = adapter_grads(self.model, self.batch, 1.0)
        self.assertGreater(max_diff(self.g0, g1, only="layers.0."), 10 * self.noise + 1e-8)

    def test_head_trains_only_when_lambda_positive(self):
        for lam, expect_nonzero in ((0.0, False), (1.0, True)):
            self.model.zero_grad()
            self.model(self.batch, lam)["loss"].backward()
            head_grad = max(p.grad.abs().max().item() for p in self.model.head.parameters())
            self.assertEqual(head_grad > 0, expect_nonzero)

    def test_detached_control_blocks_aux_gradient(self):
        d0 = adapter_grads(self.model, self.batch, 0.0, forward_with_detach)
        d1 = adapter_grads(self.model, self.batch, 1.0, forward_with_detach)
        broken = max_diff(d0, d1)
        real = max_diff(self.g0, adapter_grads(self.model, self.batch, 1.0))
        self.assertLess(broken, real * 1e-3)


class SaveReload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = load_cfg()
        cls.tokenizer, cls.model = build_model(cls.cfg, DEVICE)
        cls.model.eval()
        cls.batch = make_batch(cls.tokenizer, cls.cfg)
        with torch.no_grad():
            cls.logits_before = cls.model.lm(
                input_ids=cls.batch["input_ids"], attention_mask=cls.batch["attention_mask"]).logits
            cls.pred_before = cls.model(cls.batch, 0.0)["pred"]
        params = [p for p in cls.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            cls.model(cls.batch, 1.0)["loss"].backward()
            optimizer.step()

    def test_round_trip_reproduces_outputs(self):
        with torch.no_grad():
            logits_a = self.model.lm(
                input_ids=self.batch["input_ids"], attention_mask=self.batch["attention_mask"]).logits
            pred_a = self.model(self.batch, 0.0)["pred"]
        self.assertFalse(torch.allclose(logits_a, self.logits_before, atol=1e-4))
        self.assertFalse(torch.allclose(pred_a, self.pred_before, atol=1e-4))

        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(self.model, Path(tmp) / "ckpt", {"aux_weight": 1.0})
            self.assertTrue((Path(tmp) / "ckpt" / "head.pt").exists())
            self.assertTrue((Path(tmp) / "ckpt" / "adapter").exists())
            _, reloaded = build_model(self.cfg, DEVICE, checkpoint_dir=Path(tmp) / "ckpt", trainable=False)
            reloaded.eval()
            with torch.no_grad():
                logits_b = reloaded.lm(
                    input_ids=self.batch["input_ids"], attention_mask=self.batch["attention_mask"]).logits
                pred_b = reloaded(self.batch, 0.0)["pred"]
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-4))
        self.assertTrue(torch.allclose(pred_a, pred_b, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
