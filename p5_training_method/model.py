import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from head import EvidenceHead

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


class EvidenceAwareModel(nn.Module):
    def __init__(self, lm, head):
        super().__init__()
        self.lm = lm
        self.head = head

    def forward(self, batch, aux_weight):
        out = self.lm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=True,
        )
        last_layer = out.hidden_states[-1]
        rows = torch.arange(last_layer.size(0), device=last_layer.device)
        pooled = last_layer[rows, batch["head_index"]]
        pred = self.head(pooled)

        l_answer = out.loss
        l_aux = F.mse_loss(pred, batch["s_target"])
        total = l_answer + aux_weight * l_aux
        return {
            "loss": total,
            "l_answer": l_answer.detach(),
            "l_aux": l_aux.detach(),
            "pred": pred.detach(),
        }


def build_model(cfg, device, checkpoint_dir=None, trainable=True):
    m = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    base = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=DTYPES[m["dtype"]]).to(device)

    if checkpoint_dir is None:
        lora = cfg["lora"]
        lcfg = LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], task_type="CAUSAL_LM",
        )
        lm = get_peft_model(base, lcfg)
    else:
        lm = PeftModel.from_pretrained(base, Path(checkpoint_dir) / "adapter", is_trainable=trainable)

    head = EvidenceHead(base.config.hidden_size).to(device)
    if checkpoint_dir is not None:
        head.load_state_dict(torch.load(Path(checkpoint_dir) / "head.pt", map_location=device))
    return tokenizer, EvidenceAwareModel(lm, head)


def save_checkpoint(model, out_dir, meta):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(out / "adapter")
    torch.save(model.head.state_dict(), out / "head.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
