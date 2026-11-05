"""Model/tokenizer loading and chat-prompt encoding for scoring."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


def resolve_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id, revision, device, adapter=None):
    """Tokenizer + causal LM in eval mode, in the team's precision on every device. `adapter` is a PEFT dir."""
    dtype = getattr(torch, config.TORCH_DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return tokenizer, model.to(device).eval()


def encode_prompt(tokenizer, question):
    """Prompt token IDs for a question; must match common.model.ModelRunner, which generated the text."""
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
