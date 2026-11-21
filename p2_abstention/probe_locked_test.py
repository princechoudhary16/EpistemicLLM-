"""P3-style hidden-state probe vs the model's own log-probability, on the TriviaQA LOCKED TEST (run once).

Pre-specified before running (do not change after seeing locked-test numbers):
  * Model: untouched Qwen2.5-0.5B-Instruct, float32 (configs/model.yaml); shared prompt (ModelRunner._build_prompt).
  * Features: last prompt-token state at depths 6, 12, 18, 24 (= HF hidden_states[d], P3's convention).
  * Labels: was the base model's GREEDY answer graded correct (v5)? Abstain = not correct; invalid rows skipped.
  * Probe (P3's setting): standardise on train_probe only; class-balanced L2 logistic regression, C = 1.
    Fitted on train_probe (500); reported on validation (200, sanity check against P3) and locked_test (300).
  * Free signal: the greedy answer's average token log-probability (avg_logprob in the response rows).
  * Combinations (2-feature logistic regression, standardised, balanced, C = 1, fitted on train_probe):
      probe18 + log-prob    -- probe score for train rows is 5-fold out-of-fold, so the combiner never sees a probe
                               that was fitted on the same row
      log-prob + length     -- control: "does ANY second feature help?" (length = prompt tokens)
  * Headline: AUROC of each signal; paired question-bootstrap 95% intervals (2,000 draws, seed 0) for
      probe18 - log-prob, (probe18 + log-prob) - log-prob, (probe18 + log-prob) - (log-prob + length).

    python -m p2_abstention.probe_locked_test extract      # GPU, ~10 min: features for the three splits
    python -m p2_abstention.probe_locked_test analyse      # CPU, seconds: writes the JSON + markdown report
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2_abstention.probe import fit_logistic, standardiser  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train_probe", "validation", "locked_test")
DEPTHS = (6, 12, 18, 24)
FEAT = REPO_ROOT / "results/p3_locked/features"
OUT = REPO_ROOT / "results/final/p3_locked_test"
N_BOOT, SEED, FOLDS = 2000, 0, 5


def questions(split):
    with open(REPO_ROOT / f"data/real_qa/triviaqa_v0/{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_extract(a):
    import torch
    import yaml

    from common.model import ModelRunner

    model_cfg = yaml.safe_load((REPO_ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
    dec_cfg = yaml.safe_load((REPO_ROOT / "configs/decoding.yaml").read_text(encoding="utf-8"))
    if model_cfg["torch_dtype"] != "float32":
        raise SystemExit("configs/model.yaml must say float32")
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], torch_dtype="float32",
                         device="cuda" if torch.cuda.is_available() else "cpu")
    FEAT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        qs = questions(split)
        feats, lengths = {d: [] for d in DEPTHS}, []
        for q in qs:
            prompt = runner._build_prompt(q["question"], dec_cfg["system_prompt"])
            inputs = runner.tokenizer(prompt, return_tensors="pt").to(runner.device)
            with torch.no_grad():
                hs = runner.model(**inputs, output_hidden_states=True).hidden_states
            for d in DEPTHS:
                feats[d].append(hs[d][0, -1].float().cpu().numpy())
            lengths.append(int(inputs["input_ids"].shape[1]))
        np.savez_compressed(FEAT / f"{split}.npz", question_ids=np.array([q["question_id"] for q in qs]),
                            prompt_tokens=np.array(lengths), **{f"layer_{d}": np.stack(feats[d]) for d in DEPTHS})
        print(f"{split}: {len(qs)} questions -> {FEAT / f'{split}.npz'}", file=sys.stderr)


def greedy_rows(split):
    path = REPO_ROOT / f"results/raw/p2_triviaqa_v0_{split}_base.jsonl"
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["is_greedy"] and r["outcome"] != "invalid" and r.get("avg_logprob") is not None:
                out[r["question_id"]] = (int(r["outcome"] == "correct"), float(r["avg_logprob"]))
    return out


def load(split):
    z = np.load(FEAT / f"{split}.npz")
    rows = greedy_rows(split)
    ids = [str(q) for q in z["question_ids"]]
    keep = [i for i, q in enumerate(ids) if q in rows]
    y = np.array([rows[ids[i]][0] for i in keep], float)
    lp = np.array([rows[ids[i]][1] for i in keep])
    ln = z["prompt_tokens"][keep].astype(float)
    X = {d: z[f"layer_{d}"][keep].astype(np.float64) for d in DEPTHS}
    return dict(ids=[ids[i] for i in keep], y=y, logprob=lp, length=ln, X=X)


def auroc(s, y):
    """Mann-Whitney AUROC, ties count half (same definition as common.metrics.auroc)."""
    s, y = np.asarray(s, float), np.asarray(y, int)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


class Linear:
    """Standardise on the fitting rows, then balanced L2 logistic regression (C = 1)."""

    def fit(self, X, y):
        self.mu, self.sd = standardiser(X)
        self.w, self.b = fit_logistic((X - self.mu) / self.sd, y, 1.0, balanced=True)
        return self

    def score(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.b


def oof_scores(X, y, folds=FOLDS, seed=SEED):
    idx = np.random.default_rng(seed).permutation(len(y))
    s = np.empty(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        s[te] = Linear().fit(X[tr], y[tr]).score(X[te])
    return s


def cmd_analyse(a):
    tr, va, te = load("train_probe"), load("validation"), load("locked_test")
    signals = {}
    for name, d in (("validation", va), ("locked_test", te)):
        signals[name] = {"log-prob": d["logprob"], "length": d["length"]}
    for dep in DEPTHS:
        m = Linear().fit(tr["X"][dep], tr["y"])
        for name, d in (("validation", va), ("locked_test", te)):
            signals[name][f"probe L{dep}"] = m.score(d["X"][dep])
    # combinations, fitted on train_probe only
    tr_probe = oof_scores(tr["X"][18], tr["y"])
    comb = Linear().fit(np.c_[tr_probe, tr["logprob"]], tr["y"])
    ctrl = Linear().fit(np.c_[tr["logprob"], tr["length"]], tr["y"])
    for name, d in (("validation", va), ("locked_test", te)):
        signals[name]["probe L18 + log-prob"] = comb.score(np.c_[signals[name]["probe L18"], d["logprob"]])
        signals[name]["log-prob + length"] = ctrl.score(np.c_[d["logprob"], d["length"]])

    rng = np.random.default_rng(SEED)
    pairs = (("probe L18", "log-prob"), ("probe L18 + log-prob", "log-prob"),
             ("probe L18 + log-prob", "log-prob + length"))
    report = {"protocol": __doc__.split("    python")[0].strip(),
              "train_probe": dict(n=len(tr["y"]), correct=int(tr["y"].sum()))}
    for name, d in (("validation", va), ("locked_test", te)):
        y, sig = d["y"], signals[name]
        draws = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
        res = dict(n=len(y), correct=int(y.sum()), auroc={k: auroc(v, y) for k, v in sig.items()}, differences={})
        for p, q in pairs:
            diffs = sorted(auroc(sig[p][b], y[b]) - auroc(sig[q][b], y[b]) for b in draws)
            diffs = [x for x in diffs if not np.isnan(x)]
            res["differences"][f"{p} minus {q}"] = dict(
                estimate=res["auroc"][p] - res["auroc"][q],
                ci95=[diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]])
        report[name] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_vs_logprob.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Probe vs log-prob, TriviaQA (base model, float32)", "",
             "| Signal | Validation AUROC | Locked-test AUROC |", "|---|---:|---:|"]
    for k in signals["locked_test"]:
        lines.append(f"| {k} | {report['validation']['auroc'][k]:.3f} | {report['locked_test']['auroc'][k]:.3f} |")
    lines += ["", f"Validation: {va['y'].size} questions, {int(va['y'].sum())} correct. "
              f"Locked test: {te['y'].size} questions, {int(te['y'].sum())} correct.", "",
              "| Difference (AUROC) | Validation [95% CI] | Locked test [95% CI] |", "|---|---:|---:|"]
    for k in report["locked_test"]["differences"]:
        v, t = report["validation"]["differences"][k], report["locked_test"]["differences"][k]
        lines.append(f"| {k} | {v['estimate']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                     f"{t['estimate']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] |")
    (OUT / "probe_vs_logprob.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("extract")
    sp.add_parser("analyse")
    a = ap.parse_args(argv)
    {"extract": cmd_extract, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
