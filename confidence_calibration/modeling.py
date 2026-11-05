"""Model/tokenizer loading and chat-prompt encoding shared by generation and scoring."""

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


def end_token_ids(tokenizer):
    """Tokens that terminate an answer: <|im_end|>, eos and pad (<|endoftext|> for Qwen)."""
    ids = {tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    return sorted(i for i in ids if i is not None and i >= 0)


def build_messages(question, prompt_version):
    if prompt_version not in config.PROMPTS:
        raise KeyError(f"unknown prompt_version {prompt_version!r}; register it in config.PROMPTS")
    spec = config.PROMPTS[prompt_version]
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    for demo_question, demo_answer in spec["demos"]:
        messages.append({"role": "user", "content": spec["template"].format(question=demo_question)})
        messages.append({"role": "assistant", "content": demo_answer})
    messages.append({"role": "user", "content": spec["template"].format(question=question)})
    return messages


def encode_prompt(tokenizer, question, prompt_version):
    """The single place a question becomes prompt token IDs (generation and scoring must agree)."""
    return tokenizer.apply_chat_template(
        build_messages(question, prompt_version), add_generation_prompt=True, tokenize=True
    )
