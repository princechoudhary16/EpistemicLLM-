"""K per fact for P5's K(q) target (plan decision 4, 2026-10-31).

K of a fact is the share of sampled answers that are correct when the standard fine-tuned model is asked
about the fact in the TRAINING wording. Every fact gets a K: the 0x facts (never trained on) get one too,
flagged `is_trained` false. The evaluation questions in eval.jsonl are not used, so a K target never comes
from the questions the head is later scored on.

    python -m p1_exposure.k_labels --version main_v0 --out-dir out_k_main_v0 --save-adapter out_k_main_v0/adapter
    python -m p1_exposure.k_labels --version main_v0 --out-dir out_k_main_v0 --adapter path/to/sft_matched_adapter

Without --adapter the matched recipe ("mine": r16, alpha 32, dropout 0.05, all 7 linear layers, batch 4,
lr 2e-4, one epoch, file order) is trained in this process, in float32 by default (configs/model.yaml). With
--adapter an existing LoRA adapter (for example P6's float32 sft_matched) is loaded instead and no training
happens. Either way the answers are sampled with the project's decoding_v1 settings and graded with the
shared grader (common/grading.py); the script refuses any other grader version.

Outputs in --out-dir:
    k_per_fact.jsonl   one row per fact, no answers and no response text (the file to give P5)
    k_responses.jsonl  every graded response, evaluation side (holds reference answers; do not train on it)
    k_meta.json        model, precision, recipe, seed, adapter source, grader version and file hashes, checks
"""

import argparse
import hashlib
import json
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

from p1_exposure.relations import RELATIONS, render

EXPECT_GRADER = "p2_whole_word_v5"
FORBIDDEN_KEYS = {"reference_answer", "completion", "answer", "model_answer", "question"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe_adapter(adapter_dir):
    """Provenance of a LoRA adapter folder from its files alone (no torch needed): hash and size of the weights,
    tensor count, tensor dtypes, parameter count, and the LoRA settings in adapter_config.json. Float32
    weights (dtype F32) are the evidence that an adapter was trained in float32."""
    adapter_dir = Path(adapter_dir)
    weights = adapter_dir / "adapter_model.safetensors"
    out = {"weights_file": weights.name, "weights_sha256": sha256(weights), "weights_bytes": weights.stat().st_size}
    with open(weights, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    dtypes = defaultdict(int)
    params = 0
    for t in header.values():
        dtypes[t["dtype"]] += 1
        size = 1
        for d in t["shape"]:
            size *= d
        params += size
    out.update(tensors=len(header), tensor_dtypes=dict(dtypes), parameters=params)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    out["lora"] = {k: config.get(k) for k in ("base_model_name_or_path", "r", "lora_alpha", "lora_dropout",
                                              "peft_version", "task_type")}
    out["lora"]["target_modules"] = sorted(config.get("target_modules") or [])
    return out


def bucket(k):
    """Same cut-offs as exposure_analysis.py: K=0, inconsistent (0 < K < 0.8), K>=0.8."""
    return "K=0" if k == 0 else "K>=0.8" if k >= 0.8 else "inconsistent"


# ----------------------------------------------------------------------------------------
# Questions, aggregation and checks (no torch)
# ----------------------------------------------------------------------------------------

def build_questions(facts):
    """One question per fact and training template, in the training wording."""
    rows = []
    for f in facts:
        for t, template in enumerate(RELATIONS[f["relation"]].train):
            rows.append({
                "question_id": f"{f['fact_id']}-k{t}", "fact_id": f["fact_id"], "relation": f["relation"],
                "exposure_level": int(f["exposure_level"]), "split": f["split"],
                "template_id": f"{f['relation']}.train{t}", "question": render(template, f["subject"]),
                "reference_answer": f["reference_answer"]})
    return rows


def regrade(rows, questions_by_id, grading):
    """Grade every response with the shared grader, passing the question text it expects."""
    out = []
    for r in rows:
        r = dict(r)
        q = questions_by_id[r["question_id"]]
        r["outcome"], r["grading_version"], r["review_status"] = grading.grade_fields(
            r["model_answer"], q["reference_answer"], q["question"])
        out.append(r)
    return out


def aggregate(facts, questions, rows, grader_version):
    """One row per fact from graded responses (greedy and sampled) to the training-wording questions."""
    qs_by_fact = defaultdict(list)
    for q in questions:
        qs_by_fact[q["fact_id"]].append(q["question_id"])
    by_q = defaultdict(lambda: {"greedy": None, "sampled": []})
    for r in rows:
        if r["is_greedy"]:
            by_q[r["question_id"]]["greedy"] = r["outcome"] == "correct"
        else:
            by_q[r["question_id"]]["sampled"].append(r["outcome"] == "correct")
    out = []
    for f in facts:
        qids = qs_by_fact[f["fact_id"]]
        samples = [by_q[q]["sampled"] for q in qids]
        pooled = [x for s in samples for x in s]
        k = sum(pooled) / len(pooled) if pooled else None
        greedy = [int(bool(by_q[q]["greedy"])) for q in qids]
        out.append({
            "fact_id": f["fact_id"], "relation": f["relation"], "exposure_level": int(f["exposure_level"]),
            "split": f["split"], "is_trained": int(f["exposure_level"]) > 0,
            "n_templates": len(qids), "n_samples_per_template": len(samples[0]) if samples else 0,
            "n_sampled": len(pooled), "n_correct": int(sum(pooled)),
            "k": None if k is None else round(k, 6),
            "k_by_template": [round(sum(s) / len(s), 6) if s else None for s in samples],
            "greedy_correct_by_template": greedy,
            "greedy_correct_rate": round(sum(greedy) / len(greedy), 6) if greedy else None,
            "k_bucket": None if k is None else bucket(k), "grading_version": grader_version})
    return out


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def validate(facts, k_rows, grader_version, expect_samples=None):
    out = []
    ids = [r["fact_id"] for r in k_rows]
    out.append(_check("one_row_per_fact", sorted(ids) == sorted(f["fact_id"] for f in facts),
                      f"{len(ids)} rows for {len(facts)} facts"))
    truth = {f["fact_id"]: f for f in facts}
    drift = [r["fact_id"] for r in k_rows
             if r["exposure_level"] != truth[r["fact_id"]]["exposure_level"] or r["split"] != truth[r["fact_id"]]["split"]]
    out.append(_check("exposure_and_split_match_fact_table", not drift, f"{len(drift)} rows differ"))
    bad_k = [r["fact_id"] for r in k_rows if r["k"] is None or not 0 <= r["k"] <= 1]
    out.append(_check("k_in_unit_interval", not bad_k, f"{len(bad_k)} facts without a valid K"))
    n = {(r["n_templates"], r["n_samples_per_template"]) for r in k_rows}
    ok = len(n) == 1 and (expect_samples is None or {s for _, s in n} == {expect_samples})
    out.append(_check("same_sample_count_for_every_fact", ok, f"(templates, samples per template) seen: {sorted(n)}"))
    leaked = [r["fact_id"] for r in k_rows if FORBIDDEN_KEYS & set(r)]
    out.append(_check("no_answer_or_text_in_rows", not leaked, f"{len(leaked)} rows carry an answer or response field"))
    flag = [r["fact_id"] for r in k_rows if r["is_trained"] != (r["exposure_level"] > 0)]
    out.append(_check("is_trained_matches_exposure", not flag, f"{len(flag)} rows"))
    versions = {r["grading_version"] for r in k_rows}
    out.append(_check("graded_with_expected_grader", versions == {grader_version}, f"versions seen: {sorted(versions)}"))
    return out


def level_summary(k_rows):
    by = defaultdict(list)
    for r in k_rows:
        by[r["exposure_level"]].append(r["k"])
    return {str(l): {"facts": len(v), "mean_k": round(sum(v) / len(v), 4),
                     "k0": sum(x == 0 for x in v), "k_ge_0.8": sum(x >= 0.8 for x in v)} for l, v in sorted(by.items())}


# ----------------------------------------------------------------------------------------
# Model part (torch, imported lazily)
# ----------------------------------------------------------------------------------------

def build_parser():
    from p1_exposure.ablation import run_ablation as ra
    d = ra.build_parser().parse_args([])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--version", default="main_v0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default=d.dtype, choices=["auto", "float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--n-samples", type=int, default=5, help="sampled answers per fact and template")
    ap.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    ap.add_argument("--gen-batch", type=int, default=d.gen_batch)
    ap.add_argument("--model-id", default=d.model_id)
    ap.add_argument("--revision", default=d.revision)
    ap.add_argument("--recipe", default="mine", help="recipe from run_ablation.RECIPES to train when no adapter is given")
    ap.add_argument("--adapter", help="LoRA adapter directory to load instead of training")
    ap.add_argument("--save-adapter", help="directory to save the adapter trained here")
    ap.add_argument("--expect-grader", default=EXPECT_GRADER)
    ap.add_argument("--limit-facts", type=int, help="smoke test: use only the first N facts")
    ap.add_argument("--train-limit", type=int, help="smoke test: train on only the first N rows")
    return ap


def load_grader(expect):
    from common import grading
    if expect and grading.GRADING_VERSION != expect:
        raise SystemExit(f"grader is {grading.GRADING_VERSION}, expected {expect}")
    return grading


def main(argv=None):
    args = build_parser().parse_args(argv)
    grading = load_grader(args.expect_grader)

    import torch
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging
    from p1_exposure.ablation import run_ablation as ra
    hf_logging.set_verbosity_error()

    data = Path(args.data_dir) / args.version
    facts = read_jsonl(data / "facts.jsonl")
    train = read_jsonl(data / "train.jsonl")
    if args.limit_facts:
        facts = facts[:args.limit_facts]
    if args.train_limit:
        train = train[:args.train_limit]
    questions = build_questions(facts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = ra.pick_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    ra.log(f"device={device} dtype={dtype} facts={len(facts)} questions={len(questions)} train={len(train)}")

    t0 = time.time()
    adapter_src = {"kind": "trained_here", "recipe": args.recipe, "seed": args.seed, "lr": args.lr,
                   "train_rows": len(train)}
    if args.adapter:
        from peft import PeftModel
        base = ra.load_base(args.model_id, args.revision, dtype, device)
        model = PeftModel.from_pretrained(base, args.adapter)
        model.eval()
        adapter_src = {"kind": "loaded", "path": str(args.adapter), **describe_adapter(args.adapter)}
    else:
        if args.recipe not in ra.RECIPES:
            sys.exit(f"unknown recipe {args.recipe!r}; choose from {sorted(ra.RECIPES)}")
        model, _ = ra.train_adapter(tok, ra.order_examples(train, [], None, args.seed), ra.RECIPES[args.recipe],
                                    dtype, args, device)
        if args.save_adapter:
            model.save_pretrained(args.save_adapter)
            adapter_src["saved_to"] = str(args.save_adapter)
            adapter_src.update(describe_adapter(args.save_adapter))
    ra.log(f"model ready in {time.time() - t0:.0f}s; sampling {args.n_samples} per question")

    raw = ra.evaluate(model, tok, questions, "frozen", args, device, n_samples=args.n_samples)
    graded = regrade(raw, {q["question_id"]: q for q in questions}, grading)
    k_rows = aggregate(facts, questions, graded, grading.GRADING_VERSION)
    checks = validate(facts, k_rows, grading.GRADING_VERSION, expect_samples=args.n_samples)
    write_jsonl(out_dir / "k_per_fact.jsonl", k_rows)
    write_jsonl(out_dir / "k_responses.jsonl", graded)

    meta = {"version": args.version, "model_id": args.model_id, "revision": args.revision, "dtype": str(dtype),
            "device": device, "torch": torch.__version__, "decoding": "frozen (decoding_v1 equivalent: top_k 0, "
            "repetition_penalty 1.0, temperature 0.7, top_p 0.9)", "n_samples_per_template": args.n_samples,
            "templates_per_fact": len(RELATIONS[facts[0]["relation"]].train), "adapter": adapter_src,
            "grading_version": grading.GRADING_VERSION, "grader_sha256": sha256(Path(grading.__file__)),
            "facts_sha256": sha256(data / "facts.jsonl"), "train_sha256": sha256(data / "train.jsonl"),
            "k_per_fact_sha256": sha256(out_dir / "k_per_fact.jsonl"), "by_exposure": level_summary(k_rows),
            "checks": checks, "seconds": round(time.time() - t0)}
    (out_dir / "k_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    for c in checks:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print("mean K by exposure:", {l: v["mean_k"] for l, v in meta["by_exposure"].items()})
    sys.exit(0 if all(c["ok"] for c in checks) else 1)


if __name__ == "__main__":
    main()
