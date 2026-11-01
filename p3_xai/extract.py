"""Extract selected prompt-only states, generate greedy responses, and save provisional labels."""

import argparse
import importlib.metadata
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from common import grading
from .utils import provisional_grade, read_jsonl, sha256_file, validate_questions, write_json, write_jsonl


def layer_depths(num_layers, fractions):
    depths = sorted({max(1, math.ceil(num_layers * f)) for f in fractions})
    if any(not 0 < f <= 1 for f in fractions) or not depths:
        raise ValueError("Depth fractions must be in (0, 1]")
    return depths


def last_token_indices(attention_mask):
    """Works with left or right padding; sum(mask)-1 is wrong for left padding."""
    if attention_mask.ndim != 2 or not torch.all(attention_mask.sum(dim=1) > 0):
        raise ValueError("Every attention mask row must have at least one non-padding token")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(dim=1).values


def extract_pooled(model, inputs, depths):
    """Capture selected states only, without retaining all layer/token activations.

    Depth d < L is the output of decoder block d (one-based). Depth L is
    final RMS-normalized output. These match HF hidden_states[d] for Qwen2.
    """
    # PEFT wraps AutoModelForCausalLM in a PeftModel. Walk through its public
    # get_base_model() method so hooks land on the Qwen decoder blocks, whose
    # attention/MLP projections still contain the active LoRA modules.
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = causal_lm.model
    positions = last_token_indices(inputs["attention_mask"])
    batch_indices = torch.arange(len(positions), device=positions.device)
    captured, handles = {}, []

    def capture(depth):
        def hook(_module, _args, output):
            state = output[0] if isinstance(output, tuple) else output
            captured[depth] = state[batch_indices, positions].detach().float().cpu().numpy().copy()
        return hook

    for depth in depths:
        module = backbone.norm if depth == causal_lm.config.num_hidden_layers else backbone.layers[depth - 1]
        handles.append(module.register_forward_hook(capture(depth)))
    try:
        with torch.inference_mode():
            backbone(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(depths) or any(not np.isfinite(v).all() for v in captured.values()):
        raise ValueError("Missing/non-finite hidden states")
    return captured


def select_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/p3_week1.json"))
    parser.add_argument("--limit", type=int, help="Tiny smoke subset, preserving original selection order")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"])
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["pooling"] != "last_nonpadding_prompt_token":
        raise ValueError("Only last_nonpadding_prompt_token pooling is implemented")
    if cfg["batch_size"] < 1 or cfg["max_new_tokens"] < 1:
        raise ValueError("Batch size and generation length must be positive")
    rows = read_jsonl(cfg["questions_path"])
    validate_questions(rows)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:args.limit]
    output = args.output_dir or Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new --output-dir.")
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    device = select_device(args.device or cfg["device"])
    start = time.perf_counter()
    model_path = cfg["local_model_path"]
    print(f"Loading {cfg['model_id']} at {cfg['model_revision']} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=getattr(torch, cfg["dtype"]),
        attn_implementation="eager",
    ).to(device)
    adapter_path = cfg.get("adapter_path")
    adapter_hashes = {}
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"PEFT adapter directory does not exist: {adapter_dir}")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading this checkpoint requires PEFT; install the project's PEFT dependency") from exc
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
        for path in sorted(p for p in adapter_dir.rglob("*") if p.is_file()):
            adapter_hashes[str(path.relative_to(adapter_dir))] = sha256_file(path)
    else:
        model = base_model
    model = model.to(device).eval()
    model.requires_grad_(False)
    depths = layer_depths(model.config.num_hidden_layers, cfg["depth_fractions"])
    manifest = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": output.name, "config": cfg, "device": device,
        "model_revision": cfg["model_revision"], "model_id": cfg["model_id"],
        "checkpoint_id": cfg.get("checkpoint_id", f"{cfg['model_id']}-base"),
        "weights_sha256": sha256_file(Path(model_path) / "model.safetensors"),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_file_sha256": adapter_hashes,
        "tokenizer_config_sha256": sha256_file(Path(model_path) / "tokenizer_config.json"),
        "questions_file_sha256": sha256_file(cfg["questions_path"]),
        "question_ids": [r["question_id"] for r in rows],
        "layer_depths": depths, "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size, "feature_dtype": "float32",
        "layer_convention": "One-based decoder depth; final depth includes final RMSNorm; matches HF hidden_states[depth]",
        "pooling": cfg["pooling"], "feature_source": "Prompt only, including assistant generation prefix; no reference or generated answer",
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": cfg["max_new_tokens"]},
        "label_status": f"P3-generated labels use common grader {grading.GRADING_VERSION}; P2 labels can be supplied to the probe",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "scikit-learn"]},
                        **({"peft": importlib.metadata.version("peft")} if adapter_path else {})},
    }
    write_json(output / "manifest.json", manifest)
    features = {depth: [] for depth in depths}
    predictions, labels, prompt_lengths = [], [], []
    for offset in range(0, len(rows), cfg["batch_size"]):
        batch = rows[offset:offset + cfg["batch_size"]]
        prompts = [tokenizer.apply_chat_template([
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": row["question"]},
        ], tokenize=False, add_generation_prompt=True) for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        if inputs.input_ids.shape[1] > cfg["max_prompt_tokens"]:
            raise ValueError("Prompt exceeds max_prompt_tokens; refusing silent truncation")
        lengths = inputs.attention_mask.sum(dim=1).tolist()
        prompt_lengths.extend(lengths)
        inputs = inputs.to(device)
        batch_start = time.perf_counter()
        pooled = extract_pooled(model, inputs, depths)
        for depth in depths:
            features[depth].append(pooled[depth])
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=cfg["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
                temperature=None, top_p=None, top_k=None,
            )
        elapsed = time.perf_counter() - batch_start
        continuations = generated[:, inputs.input_ids.shape[1]:].cpu()
        answers = tokenizer.batch_decode(continuations, skip_special_tokens=True)
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        for row, answer, tokens, length in zip(batch, answers, continuations.tolist(), lengths):
            truncated = not any(token in eos_ids for token in tokens)
            grade = provisional_grade(answer, row["reference_answer"], row["question"])
            if truncated:
                grade = {"outcome": "invalid", "correct": None, "review_status": "needs_review_truncated",
                         "grading_version": grade["grading_version"]}
            record = {
                **row, "run_id": output.name, "model_id": cfg["model_id"],
                "model_revision": cfg["model_revision"],
                "checkpoint_id": cfg.get("checkpoint_id", f"{cfg['model_id']}-base"),
                "method": cfg.get("method", "base"), "training_seed": cfg.get("training_seed"),
                "generation_seed": cfg["seed"],
                "is_greedy": True, "sample_index": 0, "prompt_version": cfg["prompt_version"],
                "model_answer": answer, **grade,
                "truncated": truncated, "prompt_tokens": length,
                "batch_seconds": elapsed, "batch_size": len(batch),
            }
            predictions.append(record)
            labels.append({"question_id": row["question_id"], "correct": grade["correct"],
                           "outcome": grade["outcome"], "review_status": grade["review_status"],
                           "grading_version": grade["grading_version"], "run_id": output.name,
                           "model_revision": cfg["model_revision"], "prompt_version": cfg["prompt_version"]})
        write_jsonl(output / "predictions.jsonl", predictions)
        write_jsonl(output / "labels.jsonl", labels)
        print(f"{len(predictions)}/{len(rows)} questions; batch {elapsed:.1f}s", flush=True)
    arrays = {f"layer_{depth}": np.concatenate(features[depth]) for depth in depths}
    arrays["question_ids"] = np.asarray([r["question_id"] for r in rows])
    arrays["group_ids"] = np.asarray([r["group_id"] for r in rows])
    arrays["prompt_tokens"] = np.asarray(prompt_lengths, dtype=np.int32)
    np.savez_compressed(output / "features.npz", **arrays)
    with np.load(output / "features.npz", allow_pickle=False) as stored:
        for key, expected in arrays.items():
            if not np.array_equal(expected, stored[key]):
                raise ValueError(f"Array round-trip failed: {key}")
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - start,
                    feature_file_sha256=sha256_file(output / "features.npz"),
                    predictions_sha256=sha256_file(output / "predictions.jsonl"),
                    labels_sha256=sha256_file(output / "labels.jsonl"),
                    storage_bytes=(output / "features.npz").stat().st_size,
                    feature_shapes={str(d): list(arrays[f"layer_{d}"].shape) for d in depths})
    write_json(output / "manifest.json", manifest)
    print(f"Saved and reloaded prompt-only features: {output / 'features.npz'}", flush=True)


if __name__ == "__main__":
    main()
