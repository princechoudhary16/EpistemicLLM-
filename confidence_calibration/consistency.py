"""Question-level summary: response consistency and knowledge score.

consistency  = share of the sampled responses whose normalised answer equals the greedy answer's.
               The greedy response is the target being judged; the samples are independent draws, so
               using consistency to predict greedy correctness is not circular.
majority_frac = size of the largest answer cluster among the samples / n_samples.
n_clusters    = number of distinct normalised answers among the samples.
knowledge_score = correct samples / n_samples (greedy excluded, plan.md §6). Descriptive only;
               it shares samples with consistency, so their relationship is partly by construction.

Abstentions share one cluster; empty outputs share another. Exact normalised match is strict
("Paris" != "Paris France"); semantic clustering is a possible later refinement.
"""

from collections import Counter, defaultdict

from .grading import is_abstention, normalize_answer

ABSTAIN_KEY = "<abstain>"
EMPTY_KEY = "<empty>"


def answer_key(text):
    norm = normalize_answer(text)
    if not norm:
        return EMPTY_KEY
    if is_abstention(text):
        return ABSTAIN_KEY
    return norm


def summarize(greedy_answer, sample_answers, sample_outcomes=None):
    n = len(sample_answers)
    if n == 0:
        return {"n_samples": 0, "consistency": None, "majority_frac": None, "n_clusters": 0,
                "knowledge_score": None}
    target = answer_key(greedy_answer)
    keys = [answer_key(a) for a in sample_answers]
    counts = Counter(keys)
    out = {
        "n_samples": n,
        "consistency": sum(k == target for k in keys) / n,
        "majority_frac": max(counts.values()) / n,
        "n_clusters": len(counts),
        "knowledge_score": None,
    }
    if sample_outcomes is not None and all(o is not None for o in sample_outcomes):
        out["knowledge_score"] = sum(o == "correct" for o in sample_outcomes) / n
    return out


GROUP_KEYS = ("run_id", "question_id", "checkpoint_id", "decoding_config_id")


def summarize_questions(rows):
    groups = defaultdict(lambda: {"greedy": None, "samples": []})
    for r in rows:
        key = tuple(r.get(k) for k in GROUP_KEYS)
        if r.get("is_greedy"):
            if groups[key]["greedy"] is not None:
                raise ValueError(f"two greedy responses for {key}")
            groups[key]["greedy"] = r
        else:
            groups[key]["samples"].append(r)
    out = []
    for key, g in groups.items():
        if g["greedy"] is None:
            raise ValueError(f"no greedy response for {key}")
        samples = sorted(g["samples"], key=lambda r: r["sample_index"])
        s = summarize(
            g["greedy"]["model_answer"],
            [r["model_answer"] for r in samples],
            [r.get("outcome") for r in samples],
        )
        out.append({**dict(zip(GROUP_KEYS, key)), **s})
    return out

