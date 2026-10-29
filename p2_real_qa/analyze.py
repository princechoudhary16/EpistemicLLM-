"""P2 analysis: K(q) distribution, hallucination-by-K (always with coverage), error taxonomy, metadata breakdown,
confident-hallucination count, figures.  (plan.md P2 bullets 4-7 and "Primary figures")

    python -m p2_real_qa.analyze --responses results/raw/p2_pilot_base.jsonl --questions data/real_qa/simpleqa_v0/pilot.jsonl --tag pilot

Everything numeric is computed by the SHARED code (common.metrics) so P2's numbers match P6's tables:
  K(q)                 -> metrics.knowledge_scores        (sampled responses only, invalid excluded)
  hallucination rate,  -> metrics.behavior_metrics        (W/N, coverage = (C+W)/N, selective accuracy = C/(C+W))
  coverage
  confident halluc.    -> metrics.confident_hallucinations
Confidence signals (avg_logprob, entropy) are already in every response row, written by common.model.ModelRunner.
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import metrics, schema  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSIST_TAU = 0.5      # mean pairwise Jaccard >= tau  => "consistent"          (TEAM: align with P4's consistency)
MIN_ATTEMPTS = 3       # need >= 3 non-abstaining samples to judge consistency
REFUSAL_MIN = 3        # >= 3 sampled runs abstain => refusal
BUCKETS = ["0/5", "1/5", "2/5", "3/5", "4/5", "5/5"]
_STARTERS = {"The", "A", "An", "As", "According", "In", "On", "At", "He", "She", "It", "They", "This", "That",
             "There", "His", "Her", "Their", "However", "Unfortunately", "I", "Based", "Without"}


# ---------------------------------------------------------------- consistency (heuristic until P4's lands)
def claim_signature(answer: str, question: str) -> frozenset:
    """Which entities/numbers did the model commit to? Capitalised words + numbers that are not in the question."""
    qtok = set(re.findall(r"[A-Za-z0-9]+", question.lower()))
    caps = [w for w in re.findall(r"\b[A-Z][A-Za-z\-']+\b", answer) if w not in _STARTERS]
    nums = re.findall(r"\b\d[\d,.]*\b", answer)
    return frozenset(t.lower() for t in caps + nums if t.lower().strip(",.") not in qtok)


def mean_pairwise_jaccard(sigs):
    sigs = [s for s in sigs if s]
    if len(sigs) < 2:
        return np.nan
    return float(np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(sigs, 2)]))


# ---------------------------------------------------------------- per-question table
def question_table(rows, qmeta):
    ks = {k[2]: v for k, v in metrics.knowledge_scores(rows).items()}          # key = (run_id, checkpoint_id, question_id)
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question_id"], []).append(r)
    out = []
    for qid, g in by_q.items():
        samp = [r for r in g if not r["is_greedy"] and r["outcome"] != "invalid"]
        att = [r for r in samp if r["outcome"] != "abstain"]
        greedy = next((r for r in g if r["is_greedy"]), None)
        lp = [r["avg_logprob"] for r in att if r["avg_logprob"] is not None]
        en = [r["entropy"] for r in att if r["entropy"] is not None]
        q = g[0]["question"]
        out.append(dict(
            question_id=qid, question=q, reference_answer=g[0]["reference_answer"],
            K=ks.get(qid, {}).get("knowledge_score", np.nan), n_samples=len(samp),
            n_correct=sum(r["outcome"] == "correct" for r in samp),
            n_abstain=sum(r["outcome"] == "abstain" for r in samp), n_attempt=len(att),
            greedy_outcome=greedy["outcome"] if greedy else None,
            truncated_frac=float(np.mean([bool(r["truncated"]) for r in samp])) if samp else np.nan,
            flag_partial=any(r["review_status"] in ("flagged:ref_partial_overlap", "flagged:numeric_ambiguous") for r in samp),
            mean_avg_logprob=float(np.mean(lp)) if lp else np.nan, mean_entropy=float(np.mean(en)) if en else np.nan,
            consistency=mean_pairwise_jaccard([claim_signature(r["model_answer"], q) for r in att]) if len(att) >= MIN_ATTEMPTS else np.nan,
        ))
    t = pd.DataFrame(out)
    t["k_bucket"] = pd.cut(t.K, [-0.01, 0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=BUCKETS).astype(str)
    if qmeta is not None:
        t = t.merge(qmeta[["question_id", "topic", "answer_type"]], on="question_id", how="left")
    return t


def assign_error_type(t: pd.DataFrame) -> pd.DataFrame:
    lp_hi = t.mean_avg_logprob.quantile(2 / 3)       # "confident" = top third by logprob AND bottom third by entropy
    en_lo = t.mean_entropy.quantile(1 / 3)           # (exploratory; P4 owns the frozen, validation-selected threshold)

    def f(r):
        if r.K >= 0.8:
            return "reliable (K>=0.8)"
        if r.K > 0:
            return "unstable_partial_knowledge (0<K<0.8)"
        if r.n_abstain >= REFUSAL_MIN:
            return "refusal"
        if r.flag_partial:
            return "partial_or_alias_candidate (manual check)"
        if np.isnan(r.consistency):
            return "unclassified (too few attempts)"
        if r.consistency < CONSIST_TAU:
            return "inconsistent_guessing"
        if r.mean_avg_logprob >= lp_hi and r.mean_entropy <= en_lo:
            return "unsupported_confident_answer"
        return "consistently_wrong"
    t["error_type"] = t.apply(f, axis=1)
    return t


# ---------------------------------------------------------------- tables
def by_k_bucket(rows, t):
    bucket = dict(zip(t.question_id, t.k_bucket))
    recs = []
    for b in BUCKETS:
        outs = [r["outcome"] for r in rows if not r["is_greedy"] and bucket.get(r["question_id"]) == b]
        if outs:
            recs.append({"k_bucket": b, "n_questions": int((t.k_bucket == b).sum()), **metrics.behavior_metrics(outs)})
    return pd.DataFrame(recs).set_index("k_bucket")


def confident_halluc(rows):
    samp = [r for r in rows if not r["is_greedy"]]
    lps = [r["avg_logprob"] for r in samp if r["avg_logprob"] is not None and r["outcome"] in ("correct", "incorrect")]
    if not lps:
        return None
    thr = float(np.quantile(lps, 2 / 3))
    conf = [r["avg_logprob"] if r["avg_logprob"] is not None else float("-inf") for r in samp]
    return metrics.confident_hallucinations([r["outcome"] for r in samp], conf, thr)


def figures(t, kb, out: Path, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping figures (pip install matplotlib)")
        return
    c = t.k_bucket.value_counts().reindex(BUCKETS).fillna(0)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(c.index, c.values); ax.set_yscale("symlog"); ax.set_xlabel("correct samples out of 5"); ax.set_ylabel("# questions (log)")
    for i, v in enumerate(c.values):
        ax.text(i, v, int(v), ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Distribution of K(q) ({tag}, n={len(t)})"); fig.tight_layout(); fig.savefig(out / f"p2_{tag}_k_distribution.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2)); x = np.arange(len(kb))
    ax.bar(x - 0.2, kb.hallucination_rate, 0.4, label="hallucination rate"); ax.bar(x + 0.2, kb.coverage, 0.4, label="answer coverage")
    ax.set_xticks(x, kb.index); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.set_xlabel("K(q) bucket")
    ax.set_title("Hallucination & coverage by K(q)"); fig.tight_layout(); fig.savefig(out / f"p2_{tag}_halluc_by_k.png", dpi=200); plt.close(fig)

    v = t.error_type.value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(v.index[::-1], v.values[::-1]); ax.set_xlabel("# questions"); ax.set_title("Error taxonomy")
    fig.tight_layout(); fig.savefig(out / f"p2_{tag}_taxonomy.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--questions", help="question JSONL (for topic / answer_type)")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--out", default="results/final")
    a = ap.parse_args()

    rows = schema.read_jsonl(a.responses)
    qmeta = pd.DataFrame(schema.read_jsonl(a.questions)) if a.questions else None
    out = REPO_ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t = assign_error_type(question_table(rows, qmeta))
    kb = by_k_bucket(rows, t)
    tax = t.error_type.value_counts().rename("n_questions").to_frame()
    tax["pct"] = (100 * tax.n_questions / len(t)).round(1)

    t.to_csv(out / f"p2_{a.tag}_question_table.csv", index=False)
    kb.to_csv(out / f"p2_{a.tag}_by_k_bucket.csv")
    tax.to_csv(out / f"p2_{a.tag}_taxonomy_table.csv")
    ex = ["question_id", "question", "reference_answer", "K", "n_abstain", "consistency", "mean_avg_logprob", "error_type"]
    pd.concat([t[t.error_type == "unsupported_confident_answer"].nlargest(5, "mean_avg_logprob")[ex],
               t[t.error_type == "consistently_wrong"].nlargest(5, "consistency")[ex],
               t[t.error_type == "inconsistent_guessing"].nsmallest(5, "consistency")[ex],
               t[t.error_type.str.startswith("unstable")][ex]]).to_csv(out / f"p2_{a.tag}_examples.csv", index=False)
    if qmeta is not None:
        for col in ("answer_type", "topic"):
            g = t.groupby(col).agg(n=("K", "size"), mean_K=("K", "mean"), mean_abstain=("n_abstain", "mean"))
            g[g.n >= 10].to_csv(out / f"p2_{a.tag}_by_{col}.csv")
    ch = confident_halluc(rows)
    (out / f"p2_{a.tag}_confident_hallucinations.json").write_text(json.dumps(ch, indent=2), encoding="utf-8")
    figures(t, kb, out, a.tag)

    print(tax, "\n", kb.round(3).to_string(), "\n", sep="\n")
    print("confident hallucinations (exploratory threshold = top third of avg_logprob):", ch)
    print(f"truncated-response share: {np.nanmean(t.truncated_frac):.1%}   -> wrote tables/figures to {out}")


if __name__ == "__main__":
    main()
