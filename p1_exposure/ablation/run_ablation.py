"""P1 ablation: which of decoding defaults, SFT recipe and IDK ordering explain the gap
between P6's canonical baselines and the first P1 Colab run?

One trained adapter is evaluated under two decoding settings, and recipes differ from P6's
by one factor at a time:

    inherit  what P2, P6 and the first Colab run did: unset fields fall back to the model's
             generation_config.json (repetition_penalty 1.1, top_k 20)
    frozen   every field explicit (repetition_penalty 1.0, top_k 0), as in P4's config

Standalone and torch-free at import time, so it runs the same on Colab and on the cluster:

    python run_ablation.py --smoke --out-dir out_smoke
    python run_ablation.py --version pilot_v0 --out-dir out_pilot
    python run_ablation.py --version pilot_v0 --out-dir out_pilot \
        --idk-recipes p6,mine --idk-variants end,shuffled
"""

import argparse
import gc
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PINNED_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

# Mirrors configs/decoding.yaml prompt_v0 (P6). YAML folded style keeps one trailing newline.
SYSTEM_PROMPT = (
    'Answer the question with a short, direct factual answer and nothing else. '
    'If you do not know the answer, say "I don\'t know." instead of guessing.\n'
)

Q_V = ("q_proj", "v_proj")
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Single-factor variants of P6's recipe (common/train.py + run_canonical_baselines.py),
# plus the first Colab run's recipe as the reference endpoint. lora_alpha is always 2*r.
RECIPES = {
    "p6": dict(r=8, targets=Q_V, batch=1, dropout=0.0, dtype=None),
    "p6_all": dict(r=8, targets=ALL_LINEAR, batch=1, dropout=0.0, dtype=None),
    "p6_bs4": dict(r=8, targets=Q_V, batch=4, dropout=0.0, dtype=None),
    "p6_r16": dict(r=16, targets=Q_V, batch=1, dropout=0.0, dtype=None),
    "mine": dict(r=16, targets=ALL_LINEAR, batch=4, dropout=0.05, dtype=None),
    "p6_fp32": dict(r=8, targets=Q_V, batch=1, dropout=0.0, dtype="float32"),
}
DEFAULT_RECIPES = ["p6", "p6_all", "p6_bs4", "p6_r16", "mine"]

DECODINGS = {
    "inherit": {
        "greedy": dict(do_sample=False),
        "sampled": dict(do_sample=True, temperature=0.7, top_p=0.9),
    },
    "frozen": {
        "greedy": dict(do_sample=False, repetition_penalty=1.0),
        "sampled": dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=0, repetition_penalty=1.0),
    },
}

IDK_PHRASES = ("i don't know", "i do not know", "unknown", "cannot determine", "no information")
LEVELS = [0, 1, 2, 5, 10, 20]


# ----------------------------------------------------------------------------------------
# Grading and analysis (pure Python)
# ----------------------------------------------------------------------------------------

def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


IDK_NORM = tuple(norm(p) for p in IDK_PHRASES)


def grade(model_answer, reference_answer, word_boundary=False):
    """P6's placeholder grader (substring), or the same with word-boundary matching."""
    ans = norm(model_answer)
    if any(p in ans for p in IDK_NORM):
        return "abstain"
    ref = norm(reference_answer)
    if word_boundary:
        hit = re.search(r"(?<![a-z0-9])" + re.escape(ref) + r"(?![a-z0-9])", ans) is not None
    else:
        hit = ref in ans
    return "correct" if hit else "incorrect"


def has_repeated_digit(s):
    digits = [c for c in s if c.isdigit()]
    return len(digits) != len(set(digits))


def single_factor_diff(a, b):
    ra, rb = RECIPES[a], RECIPES[b]
    return {k for k in ra if ra[k] != rb[k]}


def order_examples(train, idk, variant, seed):
    """'end' appends IDK rows after all fact rows (P6's order); 'shuffled' mixes them in."""
    if variant is None:
        return list(train)
    combined = list(train) + list(idk)
    if variant == "shuffled":
        random.Random(f"idk-order-{seed}").shuffle(combined)
    elif variant != "end":
        raise ValueError(f"unknown idk variant {variant!r}")
    return combined


def _cwa(rows, key="outcome"):
    c = Counter(r[key] for r in rows)
    return [c["correct"], c["incorrect"], c["abstain"]]


def _rate(cwa):
    n = sum(cwa)
    return cwa[0] / n if n else None


def k_bucket(k):
    return "K=0" if k == 0 else "K>=0.8" if k >= 0.8 else "inconsistent"


def summarize(rows, repeat_min_exposure=5):
    greedy = [r for r in rows if r["is_greedy"]]
    sampled = [r for r in rows if not r["is_greedy"]]
    levels = sorted({r["exposure_level"] for r in rows})
    out = {
        "n_greedy": len(greedy),
        "greedy_cwa": _cwa(greedy),
        "greedy_cwa_by_exposure": {l: _cwa([r for r in greedy if r["exposure_level"] == l]) for l in levels},
        "sampled_cwa_by_exposure": {l: _cwa([r for r in sampled if r["exposure_level"] == l]) for l in levels},
        "grader_disagreements": sum(r["outcome"] != r["outcome_wb"] for r in rows),
        "greedy_cwa_wordboundary": _cwa(greedy, "outcome_wb"),
    }

    per_fact, fact_level = defaultdict(list), {}
    for r in sampled:
        per_fact[r["fact_id"]].append(r["outcome"] == "correct")
        fact_level[r["fact_id"]] = r["exposure_level"]
    buckets = defaultdict(Counter)
    for fid, v in per_fact.items():
        buckets[fact_level[fid]][k_bucket(sum(v) / len(v))] += 1
    out["k_buckets_by_exposure"] = {l: dict(buckets[l]) for l in levels}
    out["k_buckets_overall"] = dict(sum((buckets[l] for l in levels), Counter()))

    rep = {"min_exposure": repeat_min_exposure}
    for name, group in (("greedy", greedy), ("sampled", sampled)):
        sel = [r for r in group if r["relation"] == "birth_year" and r["exposure_level"] >= repeat_min_exposure]
        rep[name] = {
            "repeated_digit": [sum(r["outcome"] == "correct" for r in sel if has_repeated_digit(r["reference_answer"])),
                               sum(has_repeated_digit(r["reference_answer"]) for r in sel)],
            "no_repeat": [sum(r["outcome"] == "correct" for r in sel if not has_repeated_digit(r["reference_answer"])),
                          sum(not has_repeated_digit(r["reference_answer"]) for r in sel)],
        }
    out["birth_year_repeat"] = rep

    chance = {}
    for rel in sorted({r["relation"] for r in rows}):
        zero = [r for r in rows if r["relation"] == rel and r["exposure_level"] == 0]
        zg, zs = [r for r in zero if r["is_greedy"]], [r for r in zero if not r["is_greedy"]]
        chance[rel] = {"greedy": [_cwa(zg)[0], len(zg)], "sampled": [_cwa(zs)[0], len(zs)]}
    out["zero_exposure_correct_by_relation"] = chance
    return out


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _frac(pair):
    return f"{pair[0]}/{pair[1]}"


def render_markdown(summaries):
    """summaries: {run_name: {decoding: summary}}"""
    lines = ["# P1 ablation summary", "",
             "Greedy accuracy by exposure level, overall abstention, K buckets over facts "
             "(sampled), and birth_year accuracy split by whether the answer has a repeated digit.", ""]
    levels = sorted({int(l) for s in summaries.values() for d in s.values() for l in d["greedy_cwa_by_exposure"]})
    head = ["run", "decoding", "greedy acc"] + [f"{l}x" for l in levels] + \
           ["abstain", "K=0 / inconsistent / K>=0.8", "repeat-digit yr", "no-repeat yr"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run in summaries:
        for dec, s in summaries[run].items():
            c = s["greedy_cwa"]
            n = sum(c)
            kb = s["k_buckets_overall"]
            rep = s["birth_year_repeat"]["greedy"]
            row = [run, dec, _pct(_rate(c))]
            row += [_pct(_rate(s["greedy_cwa_by_exposure"].get(l, s["greedy_cwa_by_exposure"].get(str(l), [0, 0, 0]))))
                    for l in levels]
            row += [_pct(c[2] / n if n else None),
                    f"{kb.get('K=0', 0)} / {kb.get('inconsistent', 0)} / {kb.get('K>=0.8', 0)}",
                    _frac(rep["repeated_digit"]), _frac(rep["no_repeat"])]
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Chance floor at 0x (correct / total generations, by relation)", ""]
    first = next(iter(next(iter(summaries.values())).values()))
    rels = list(first["zero_exposure_correct_by_relation"])
    lines += ["| run | decoding | " + " | ".join(rels) + " |", "|" + "---|" * (len(rels) + 2)]
    for run in summaries:
        for dec, s in summaries[run].items():
            z = s["zero_exposure_correct_by_relation"]
            lines.append("| " + " | ".join([run, dec] + [
                f"{_frac(z[r]['greedy'])} g, {_frac(z[r]['sampled'])} s" for r in rels]) + " |")
    lines += ["", "g = greedy, s = sampled. Grader: P6's substring match; "
              "`grader_disagreements` in summary.json counts rows where word-boundary matching differs."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def response_path(out_dir, run, decoding):
    return Path(out_dir) / f"responses__{run}__{decoding}.jsonl"


def rebuild_summaries(out_dir):
    summaries = defaultdict(dict)
    for p in sorted(Path(out_dir).glob("responses__*__*.jsonl")):
        _, run, decoding = p.stem.split("__")
        summaries[run][decoding] = summarize(read_jsonl(p))
    return summaries


def write_outputs(out_dir, extra_config=None):
    summaries = rebuild_summaries(out_dir)
    if not summaries:
        return
    jsonable = {run: {dec: s for dec, s in d.items()} for run, d in summaries.items()}
    (Path(out_dir) / "summary.json").write_text(json.dumps(jsonable, indent=2) + "\n", encoding="utf-8")
    (Path(out_dir) / "summary.md").write_text(render_markdown(summaries), encoding="utf-8")


# ----------------------------------------------------------------------------------------
# Training and generation (torch, imported lazily)
# ----------------------------------------------------------------------------------------

def pick_dtype(arg):
    import torch
    if arg != "auto":
        return getattr(torch, arg)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float32


def load_base(model_id, revision, dtype, device):
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    return model.to(device)


def encode_example(tok, system_prompt, question, answer):
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids


def collate(batch, pad_id, device):
    import torch
    n = max(len(ids) for ids, _ in batch)
    ids = [i + [pad_id] * (n - len(i)) for i, _ in batch]
    lab = [l + [-100] * (n - len(l)) for _, l in batch]
    att = [[1] * len(i) + [0] * (n - len(i)) for i, _ in batch]
    return (torch.tensor(ids, device=device), torch.tensor(lab, device=device),
            torch.tensor(att, device=device))


def train_adapter(tok, examples, recipe, dtype, args, device):
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    run_dtype = getattr(torch, recipe["dtype"]) if recipe["dtype"] else dtype
    model = load_base(args.model_id, args.revision, run_dtype, device)
    cfg = LoraConfig(r=recipe["r"], lora_alpha=2 * recipe["r"], lora_dropout=recipe["dropout"],
                     target_modules=list(recipe["targets"]), task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("LoRA adapter did not attach: no trainable parameters")
    opt = torch.optim.AdamW(params, lr=args.lr)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = [encode_example(tok, SYSTEM_PROMPT, e["prompt"], e["completion"]) for e in examples]

    model.train()
    losses = []
    for i in range(0, len(enc), recipe["batch"]):
        ids, lab, att = collate(enc[i:i + recipe["batch"]], pad_id, device)
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    model.eval()
    n_train = sum(p.numel() for p in params)
    log(f"  trained {len(enc)} examples in {len(losses)} steps | {n_train:,} trainable params | "
        f"loss first10={sum(losses[:10]) / len(losses[:10]):.3f} last10={sum(losses[-10:]) / len(losses[-10:]):.3f}")
    return model, run_dtype


def generate_batched(model, tok, prompts, gen_kwargs, max_new_tokens, batch_size, seed, device):
    import torch
    tok.padding_side = "left"
    texts = []
    for b, start in enumerate(range(0, len(prompts), batch_size)):
        if seed is not None:
            torch.manual_seed(seed + b)
        enc = tok(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=pad, **gen_kwargs)
        texts += [t.strip() for t in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return texts


def evaluate(model, tok, questions, decoding, args, device, n_samples=None):
    n_samples = args.n_samples if n_samples is None else n_samples
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q["question"]}],
        tokenize=False, add_generation_prompt=True) for q in questions]
    dec = DECODINGS[decoding]
    gens = [(True, 0, generate_batched(model, tok, prompts, dec["greedy"], args.max_new_tokens,
                                       args.gen_batch, None, device))]
    for s in range(1, n_samples + 1):
        gens.append((False, s, generate_batched(model, tok, prompts, dec["sampled"], args.max_new_tokens,
                                                args.gen_batch, args.seed + 1000 * s, device)))
    rows = []
    for is_greedy, sample_index, texts in gens:
        for q, text in zip(questions, texts):
            rows.append({
                "question_id": q["question_id"], "fact_id": q["fact_id"], "relation": q["relation"],
                "exposure_level": int(q["exposure_level"]), "reference_answer": q["reference_answer"],
                "model_answer": text, "is_greedy": is_greedy, "sample_index": sample_index,
                "outcome": grade(text, q["reference_answer"]),
                "outcome_wb": grade(text, q["reference_answer"], word_boundary=True),
            })
    return rows


def free(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--version", default="pilot_v0")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--recipes", default=",".join(DEFAULT_RECIPES))
    ap.add_argument("--decodings", default="inherit,frozen")
    ap.add_argument("--idk-variants", default="", help="comma list of: end, shuffled")
    ap.add_argument("--idk-recipes", default="p6,mine")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gen-batch", type=int, default=48)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--revision", default=PINNED_REVISION)
    ap.add_argument("--extra-evals", default="",
                    help="comma list of eval files relative to the version dir, e.g. robustness/eval_heldout.jsonl")
    ap.add_argument("--extra-samples", type=int, default=0,
                    help="sampled responses per question on the extra eval files (0 = greedy only)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    recipes = [r for r in args.recipes.split(",") if r]
    decodings = [d for d in args.decodings.split(",") if d]
    idk_variants = [v for v in args.idk_variants.split(",") if v]
    idk_recipes = [r for r in args.idk_recipes.split(",") if r]
    for r in recipes + idk_recipes:
        if r not in RECIPES and r != "base":
            sys.exit(f"unknown recipe {r!r}; choose from {sorted(RECIPES) + ['base']}")
    for d in decodings:
        if d not in DECODINGS:
            sys.exit(f"unknown decoding {d!r}; choose from {sorted(DECODINGS)}")

    data = Path(args.data_dir) / args.version
    train, evals, idk = (read_jsonl(data / f) for f in ("train.jsonl", "eval.jsonl", "idk_train.jsonl"))
    extra_sets = {Path(e).stem: read_jsonl(data / e) for e in args.extra_evals.split(",") if e}
    if args.smoke:
        train, evals, idk = train[:48], evals[:16], idk[:30]
        extra_sets = {k: v[:16] for k, v in extra_sets.items()}
        recipes, idk_recipes, idk_variants = recipes[:1] + ["mine"], ["p6"], ["end"]
        args.n_samples, args.max_new_tokens, args.gen_batch = 2, 24, 8
    out_dir = Path(args.out_dir or f"out_{args.version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    log(f"device={device} dtype={dtype} train={len(train)} eval={len(evals)} idk={len(idk)} out={out_dir}")

    config = {"args": vars(args), "system_prompt": SYSTEM_PROMPT, "decodings": DECODINGS,
              "recipes": {k: {**v, "targets": list(v["targets"])} for k, v in RECIPES.items()},
              "torch": torch.__version__, "device": device, "dtype": str(dtype)}
    try:
        import transformers
        import peft
        config.update(transformers=transformers.__version__, peft=peft.__version__)
    except Exception:
        pass
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    plan = [(r, None) for r in recipes] + [(r, v) for r in idk_recipes for v in idk_variants]
    t_start = time.time()
    for recipe_name, variant in plan:
        run = recipe_name if variant is None else f"{recipe_name}-idk_{variant}"
        def done(d):
            return response_path(out_dir, run, d).exists() and all(
                response_path(out_dir / "extra" / name, run, d).exists() for name in extra_sets)

        todo = [d for d in decodings if not done(d)]
        if not todo:
            log(f"[{run}] already done, skipping")
            continue
        t0 = time.time()
        if recipe_name == "base":
            log(f"[{run}] untrained base model (no training)")
            model = load_base(args.model_id, args.revision, dtype, device)
            model.eval()
        else:
            log(f"[{run}] training")
            examples = order_examples(train, idk, variant, args.seed)
            model, _ = train_adapter(tok, examples, RECIPES[recipe_name], dtype, args, device)
        for d in todo:
            if not response_path(out_dir, run, d).exists():
                log(f"[{run}] evaluating, decoding={d}")
                rows = evaluate(model, tok, evals, d, args, device)
                write_jsonl(response_path(out_dir, run, d), rows)
                log(f"[{run}|{d}] greedy correct/wrong/abstain = {_cwa([r for r in rows if r['is_greedy']])}")
            for name, questions in extra_sets.items():
                path = response_path(out_dir / "extra" / name, run, d)
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                log(f"[{run}] evaluating extra set {name}, decoding={d}")
                rows = evaluate(model, tok, questions, d, args, device, n_samples=args.extra_samples)
                write_jsonl(path, rows)
                log(f"[{run}|{d}|{name}] greedy correct/wrong/abstain = {_cwa([r for r in rows if r['is_greedy']])}")
        free(model)
        write_outputs(out_dir)
        for name in extra_sets:
            write_outputs(out_dir / "extra" / name)
        log(f"[{run}] done in {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)")
    write_outputs(out_dir)
    for name in extra_sets:
        write_outputs(out_dir / "extra" / name)
    log(f"finished; see {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
