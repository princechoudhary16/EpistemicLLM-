"""GeoDe's probe: a linear "truthfulness hyperplane" in the model's hidden states, and each question's distance to it.

Following the paper's TBG variant: the feature is the hidden state of the LAST prompt token at the FINAL layer, read
with the shared prompt (ModelRunner._build_prompt), so it is exactly the state the model answers from.

    # 1. features, one forward pass per question (no generation)
    python -m p2_abstention.probe extract --questions data/abstention/triviaqa_train_v0/probe.jsonl \
        --out results/abstention/features/probe_base.npz
    python -m p2_abstention.probe extract --questions data/abstention/triviaqa_train_v0/sft.jsonl \
        --out results/abstention/features/sft_base.npz
    # 2. fit on D0 (labels = was the base model's greedy answer correct), score D1
    python -m p2_abstention.probe fit --train-features results/abstention/features/probe_base.npz \
        --train-responses results/raw/p2_abstention_probe_base.jsonl \
        --apply-features results/abstention/features/sft_base.npz \
        --apply-responses results/raw/p2_abstention_sft_base.jsonl \
        --out results/abstention/geode_distances.jsonl

Distance d(x) = (w.z + b) / ||w||, with z the feature standardised on D0 (mean/std of the training rows only).
d > 0: the probe thinks the model knows the answer; |d| = how sure it is (paper eq. 2).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
C = 1.0                      # inverse L2 strength (scikit-learn convention): loss = sum(logloss) + ||w||^2 / (2C)


# ---------------------------------------------------------------- probe (pure numpy: no scikit-learn needed)
def fit_logistic(X: np.ndarray, y: np.ndarray, c: float = C, balanced: bool = False,
                 max_iter: int = 50, tol: float = 1e-8) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression by Newton's method (deterministic). The bias is not penalised.
    balanced=True weights each row by n / (2 * its class count), as scikit-learn's class_weight="balanced" (P3's probe)."""
    n, k = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    s = np.ones(n)
    pos = float(y.sum())
    if balanced and 0 < pos < n:
        s = np.where(y == 1, n / (2 * pos), n / (2 * (n - pos)))
    w = np.zeros(k + 1)
    reg = np.full(k + 1, 1.0 / c)
    reg[-1] = 0.0
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        grad = Xb.T @ (s * (p - y)) + reg * w
        H = (Xb * (s * p * (1 - p))[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(k + 1)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w[:-1], float(w[-1])

# def fit_logistic(X: np.ndarray, y: np.ndarray, c: float = C, max_iter: int = 50, tol: float = 1e-8) -> tuple[np.ndarray, float]:
#     """L2-regularised logistic regression by Newton's method (deterministic). The bias is not penalised."""
#     n, k = X.shape
#     Xb = np.hstack([X, np.ones((n, 1))])
#     w = np.zeros(k + 1)
#     reg = np.full(k + 1, 1.0 / c)
#     reg[-1] = 0.0
#     for _ in range(max_iter):
#         p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
#         grad = Xb.T @ (p - y) + reg * w
#         H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(k + 1)
#         step = np.linalg.solve(H, grad)
#         w -= step
#         if np.max(np.abs(step)) < tol:
#             break
#     return w[:-1], float(w[-1])

def standardiser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return mu, np.where(sd > 0, sd, 1.0)


def signed_distance(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return (X @ w + b) / np.linalg.norm(w)


def auroc(scores, labels) -> float | None:
    """Shared definition (common.metrics.auroc); None if only one class is present."""
    from common.metrics import auroc as shared_auroc

    return shared_auroc([float(x) for x in scores], [bool(y) for y in labels])[0]


# ---------------------------------------------------------------- features
def extract(runner, questions: list, system_prompt: str, layer: int = -1) -> np.ndarray:
    """Hidden state of the last prompt token at `layer` (-1 = final), one question at a time (no padding)."""
    import torch

    feats = []
    for i, q in enumerate(questions):
        prompt = runner._build_prompt(q["question"], system_prompt)
        inputs = runner.tokenizer(prompt, return_tensors="pt").to(runner.device)
        with torch.no_grad():
            out = runner.model(**inputs, output_hidden_states=True)
        feats.append(out.hidden_states[layer][0, -1].float().cpu().numpy())
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(questions)}", file=sys.stderr)
    return np.stack(feats)


def greedy_correct(responses_path) -> dict:
    """question_id -> 1 if the greedy answer was graded correct, 0 if incorrect or abstain; invalid rows are skipped."""
    out = {}
    with open(responses_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["is_greedy"] and r["outcome"] != "invalid":
                out[r["question_id"]] = int(r["outcome"] == "correct")
    return out


def cmd_extract(a):
    import yaml

    from common.model import ModelRunner

    model_cfg = yaml.safe_load((REPO_ROOT / "configs" / "model.yaml").read_text(encoding="utf-8"))
    dec_cfg = yaml.safe_load((REPO_ROOT / "configs" / "decoding.yaml").read_text(encoding="utf-8"))
    with open(a.questions, encoding="utf-8") as f:
        questions = [json.loads(l) for l in f if l.strip()][: a.limit]
    from p2_real_qa.generate_responses import pick_device

    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=a.adapter,
                         torch_dtype=model_cfg["torch_dtype"], device=pick_device(a.device))
    X = extract(runner, questions, dec_cfg["system_prompt"], a.layer)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(questions_file=str(a.questions), questions_sha256=hashlib.sha256(Path(a.questions).read_bytes()).hexdigest(),
                model_id=model_cfg["model_id"], model_revision=model_cfg["revision"], torch_dtype=model_cfg["torch_dtype"],
                adapter=a.adapter, layer=a.layer, pooling="last_prompt_token", prompt_version=dec_cfg["prompt_version"])
    np.savez_compressed(out, X=X.astype(np.float32), question_ids=np.array([q["question_id"] for q in questions]),
                        meta=json.dumps(meta))
    print(f"wrote {X.shape} features -> {out}")


def load_features(path) -> tuple[np.ndarray, list, dict]:
    z = np.load(path, allow_pickle=False)
    return z["X"].astype(np.float64), [str(q) for q in z["question_ids"]], json.loads(str(z["meta"]))


def cmd_fit(a):
    Xtr, ids_tr, meta_tr = load_features(a.train_features)
    ytr_map = greedy_correct(a.train_responses)
    keep = [i for i, q in enumerate(ids_tr) if q in ytr_map]
    Xtr, ytr = Xtr[keep], np.array([ytr_map[ids_tr[i]] for i in keep], float)
    mu, sd = standardiser(Xtr)
    w, b = fit_logistic((Xtr - mu) / sd, ytr, a.c, balanced=a.balanced)
    # w, b = fit_logistic((Xtr - mu) / sd, ytr, a.c)

    Xap, ids_ap, meta_ap = load_features(a.apply_features)
    if meta_ap.get("layer") != meta_tr.get("layer") or meta_ap.get("adapter") != meta_tr.get("adapter"):
        raise SystemExit("train and apply features come from different layers or models")
    d = signed_distance((Xap - mu) / sd, w, b)
    rows = [dict(question_id=q, distance=float(x)) for q, x in zip(ids_ap, d)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # report = dict(train_rows=len(ytr), train_positive_rate=float(ytr.mean()), c=a.c,
    #               train_auroc=auroc(signed_distance((Xtr - mu) / sd, w, b), ytr), layer=meta_tr.get("layer"))
    report = dict(train_rows=len(ytr), train_positive_rate=float(ytr.mean()), c=a.c, balanced=a.balanced,
                  train_auroc=auroc(signed_distance((Xtr - mu) / sd, w, b), ytr), layer=meta_tr.get("layer"))
    if a.apply_responses:                              # diagnostic only: GeoDe itself labels D1 by the probe's sign
        y_ap = greedy_correct(a.apply_responses)
        pairs = [(x, y_ap[q]) for q, x in zip(ids_ap, d) if q in y_ap]
        dd, yy = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        report["apply_auroc"] = auroc(dd, yy)
        report["apply_sign_agreement_with_correctness"] = float(((dd > 0).astype(int) == yy).mean())
        order = np.argsort(np.abs(dd))
        report["apply_auroc_by_distance_decile"] = [auroc(dd[c], yy[c]) for c in np.array_split(order, 10)]

        kept = np.argsort(-np.abs(dd))[: round(a.keep_fraction * len(dd))]       # the rows GeoDe would keep
        lab, true = (dd[kept] > 0).astype(int), yy[kept]
        known_recall = float(lab[true == 1].mean()) if (true == 1).any() else 0.0      # truly right -> labelled known
        unknown_recall = float(1 - lab[true == 0].mean()) if (true == 0).any() else 0.0  # truly wrong -> labelled IDK
        report["kept"] = dict(fraction=a.keep_fraction, rows=len(kept),
                              idk_share=float((lab == 0).mean()), true_wrong_share=float((true == 0).mean()),
                              known_recall=known_recall, unknown_recall=unknown_recall,
                              balanced_accuracy=(known_recall + unknown_recall) / 2)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--questions", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--adapter", help="LoRA adapter folder (default: base model)")
    e.add_argument("--layer", type=int, default=-1, help="hidden_states index; -1 = final layer (GeoDe TBG)")
    e.add_argument("--device", help="cuda | mps | cpu (default: auto)")
    e.add_argument("--limit", type=int)
    f = sp.add_parser("fit")
    f.add_argument("--train-features", required=True)
    f.add_argument("--train-responses", required=True, help="greedy-graded responses for the probe set (labels)")
    f.add_argument("--apply-features", required=True)
    f.add_argument("--apply-responses", help="optional: greedy-graded responses for the scored set (diagnostics only)")
    f.add_argument("--out", required=True)
    f.add_argument("--c", type=float, default=C)
    f.add_argument("--balanced", action="store_true", help="class-balanced weights (P3's probe setting)")
    f.add_argument("--keep-fraction", type=float, default=0.25, help="GeoDe's X, for the kept-set diagnostics only")
    a = ap.parse_args(argv)
    {"extract": cmd_extract, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
