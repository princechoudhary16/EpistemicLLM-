"""Reproducible model loading and generation (plan.md section 3, "Model loading" — P6).

Pins the exact model id + revision and uses its chat template. One ModelRunner call
generates a response from (model revision, decoding config, seed) alone, so any
response is reproducible. method/checkpoint_id are caller-supplied labels recorded
on every row; this module does not branch on them.
"""

import importlib.metadata
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ENV_PACKAGES = ("torch", "transformers", "peft", "accelerate", "datasets")


def environment_versions():
    """Package + GPU info to record alongside every run (plan.md: "environment versions recorded")."""
    versions = {}
    for pkg in ENV_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    versions["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        versions["gpu_name"] = torch.cuda.get_device_name(0)
    return versions


class ModelRunner:
    """Loads one pinned checkpoint, optionally with a LoRA adapter on top."""

    def __init__(self, model_id, revision, adapter_path=None, torch_dtype="bfloat16", device=None):
        self.model_id = model_id
        self.revision = revision
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, torch_dtype)
        ).to(self.device)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def _build_prompt(self, question, system_prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_one(self, question, system_prompt, max_new_tokens, do_sample,
                      seed=None, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0):
        """Generate one response. Returns text, avg_logprob, entropy, truncated, latency_seconds.

        avg_logprob/entropy use the unmodified generation-step logits, averaged over
        generated tokens up to but excluding EOS. These scores describe the model's
        raw distribution, not temperature/top-p sampling probabilities.
        """
        if seed is not None:
            torch.manual_seed(seed)
        prompt = self._build_prompt(question, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        start = time.monotonic()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        latency = time.monotonic() - start

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0, prompt_len:]
        eos_id = self.tokenizer.eos_token_id
        truncated = len(gen_ids) >= max_new_tokens and (eos_id is None or gen_ids[-1].item() != eos_id)
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        logprobs, entropies = [], []
        for step_logits, tok_id in zip(out.logits, gen_ids):
            if eos_id is not None and tok_id.item() == eos_id:
                break
            log_probs = torch.log_softmax(step_logits[0].float(), dim=-1)
            logprobs.append(log_probs[tok_id].item())
            probs = log_probs.exp()
            # p*log(p) -> 0 as p -> 0, but 0 * -inf is nan in floating point when a
            # logit is masked to -inf; zero those terms instead of propagating nan.
            terms = torch.nan_to_num(probs * log_probs, nan=0.0)
            entropies.append(-terms.sum().item())

        return {
            "text": text,
            "avg_logprob": sum(logprobs) / len(logprobs) if logprobs else None,
            "entropy": sum(entropies) / len(entropies) if entropies else None,
            "truncated": truncated,
            "latency_seconds": latency,
        }
