"""P1 exposure analysis: how does the number of times a fact was shown in training change what the
model does with it?

Reads response files in the shared row-level schema (one JSON object per generated response; needs
question_id, is_greedy, outcome; fact id, exposure, relation and split are joined from the dataset's
eval.jsonl) and writes tables, CSVs, a JSON of every number, and figures.

    python -m p1_exposure.analysis.exposure_analysis \
        --responses sft_matched=results/raw/p1_main_v0_sft_matched.jsonl base=results/raw/p1_main_v0_base.jsonl \
        --data-dir data/synthetic --version main_v0 --out-dir p1_exposure/analysis/results/main_v0

Uncertainty is a fact-level cluster bootstrap: whole facts are resampled with replacement within each
exposure level, because the two questions about a fact are not independent. Seeded, so reruns match.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BUCKETS = ("K=0", "inconsistent", "K>=0.8")

DEFINITIONS = """\
**Terms.** *Exposure*: how many times a fact appeared in the fine-tuning data (0 means never; the
fact is only asked about at test time). *Greedy response*: the single answer the model gives when it
always picks its most likely token. *Sampled responses*: 5 further answers per question drawn at
temperature 0.7. A response is *correct* if it contains the reference answer, *abstain* if it says it
does not know, otherwise *incorrect* (a hallucination: a substantive wrong answer).
*Accuracy* = correct / all; *hallucination* = incorrect / all; *abstention* = abstain / all;
*coverage* = (correct + incorrect) / all, the share of questions the model attempts;
*selective accuracy* = correct / (correct + incorrect), accuracy among attempted answers.
*K* of a question = fraction of its 5 sampled responses that are correct; *mean K* averages that over
questions. A fact's K pools all its sampled responses; facts are bucketed as *K=0* (never right),
*inconsistent* (0 < K < 0.8) or *K>=0.8* (reliably right).
*95% CI*: fact-level cluster bootstrap interval (see module docstring).
"""


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_eval(data_dir, version):
    rows = read_jsonl(Path(data_dir) / version / "eval.jsonl")
    return {r["question_id"]: r for r in rows}


def load_responses(path, eval_by_q, allow_extra=False):
    """allow_extra: skip responses whose question is outside eval_by_q (used when a split is selected)."""
    out = []
    for r in read_jsonl(path):
        q = eval_by_q.get(r["question_id"])
        if q is None and allow_extra:
            continue
        if q is None:
            raise ValueError(f"{path}: question_id {r['question_id']!r} is not in the eval set")
        conf = r.get("confidence_score")
        if conf is None:
            conf = r.get("avg_logprob")
        out.append({
            "question_id": r["question_id"], "fact_id": q["fact_id"], "relation": q["relation"],
            "split": q["split"], "exposure_level": int(q["exposure_level"]),
            "is_greedy": bool(r["is_greedy"]), "outcome": r["outcome"], "conf": conf,
            "grading_version": r.get("grading_version"), "decoding_config_id": r.get("decoding_config_id"),
            "method": r.get("method"),
        })
    return out


def check_coverage(label, rows, eval_by_q):
    greedy = Counter(r["question_id"] for r in rows if r["is_greedy"])
    missing = [q for q in eval_by_q if greedy[q] != 1]
    if missing:
        raise ValueError(f"{label}: {len(missing)} eval questions do not have exactly one greedy row "
                         f"(e.g. {missing[:3]})")


# ----------------------------------------------------------------------------------------
# Per-fact aggregation and bootstrap
# ----------------------------------------------------------------------------------------

def per_fact(rows):
    """One record per fact: greedy counts, sampled K sums, and bookkeeping."""
    facts = {}
    per_q_sampled = defaultdict(list)
    for r in rows:
        f = facts.setdefault(r["fact_id"], {
            "level": r["exposure_level"], "split": r["split"], "relation": r["relation"],
            "c": 0, "w": 0, "a": 0, "n": 0, "ks": 0.0, "kq": 0, "samp_c": 0, "samp_n": 0,
            "conf_correct": [], "conf_incorrect": []})
        if r["is_greedy"]:
            f["n"] += 1
            key = {"correct": "c", "incorrect": "w", "abstain": "a"}.get(r["outcome"])
            if key is not None:
                f[key] += 1
            if r["conf"] is not None and r["outcome"] in ("correct", "incorrect"):
                f["conf_correct" if r["outcome"] == "correct" else "conf_incorrect"].append(r["conf"])
        else:
            per_q_sampled[(r["fact_id"], r["question_id"])].append(r["outcome"] == "correct")
            f["samp_n"] += 1
            f["samp_c"] += r["outcome"] == "correct"
    for (fid, _), v in per_q_sampled.items():
        facts[fid]["ks"] += sum(v) / len(v)
        facts[fid]["kq"] += 1
    return facts


def arrays(fact_list):
    return {k: np.array([f[k] for f in fact_list], dtype=float) for k in ("c", "w", "a", "n", "ks", "kq")}


def stat_vectors(sums):
    """Statistics from summed arrays; each value may be a scalar or a vector over bootstrap draws."""
    c, w, a, n, ks, kq = (sums[k] for k in ("c", "w", "a", "n", "ks", "kq"))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "accuracy": c / n, "hallucination": w / n, "abstention": a / n,
            "coverage": (c + w) / n,
            "selective_accuracy": np.where((c + w) > 0, c / np.where((c + w) > 0, c + w, 1), np.nan),
            "mean_K": np.where(kq > 0, ks / np.where(kq > 0, kq, 1), np.nan),
        }


def interval(vec):
    vec = np.asarray(vec, dtype=float)
    if np.all(np.isnan(vec)):
        return None, None
    lo, hi = np.nanpercentile(vec, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_group(fact_list, rng, boot):
    """Point estimates plus bootstrap CIs for one group of facts."""
    if not fact_list:
        return None
    arr = arrays(fact_list)
    point = stat_vectors({k: v.sum() for k, v in arr.items()})
    idx = rng.integers(0, len(fact_list), size=(boot, len(fact_list)))
    draws = stat_vectors({k: v[idx].sum(axis=1) for k, v in arr.items()})
    out = {"n_facts": len(fact_list), "n_questions": int(arr["n"].sum()),
           "correct": int(arr["c"].sum()), "incorrect": int(arr["w"].sum()), "abstain": int(arr["a"].sum())}
    for name, val in point.items():
        lo, hi = interval(draws[name])
        out[name] = {"est": None if np.isnan(val) else float(val), "lo": lo, "hi": hi}
    return out, draws


def bucket(k):
    return "K=0" if k == 0 else "K>=0.8" if k >= 0.8 else "inconsistent"


def analyze_label(rows, levels, rng, boot):
    facts = per_fact(rows)
    result = {"exposure": {}, "relation": {}, "split": {}, "steps": {}, "confidence": {}, "chance_floor": {}}
    draws_by_level = {}
    for lvl in levels:
        fl = [f for f in facts.values() if f["level"] == lvl]
        summary, draws = summarize_group(fl, rng, boot)
        draws_by_level[lvl] = draws
        buckets = Counter(bucket(f["samp_c"] / f["samp_n"]) for f in fl if f["samp_n"] > 0)
        summary["k_buckets"] = {b: buckets.get(b, 0) for b in BUCKETS}
        result["exposure"][lvl] = summary

        cc = [x for f in fl for x in f["conf_correct"]]
        ci = [x for f in fl for x in f["conf_incorrect"]]
        if cc or ci:
            result["confidence"][lvl] = {
                "mean_conf_correct": float(np.mean(cc)) if cc else None, "n_correct": len(cc),
                "mean_conf_incorrect": float(np.mean(ci)) if ci else None, "n_incorrect": len(ci)}

        for rel in sorted({f["relation"] for f in fl}):
            sub = [f for f in fl if f["relation"] == rel]
            arr = arrays(sub)
            result["relation"].setdefault(rel, {})[lvl] = {
                "n_questions": int(arr["n"].sum()), "correct": int(arr["c"].sum()),
                "accuracy": float(arr["c"].sum() / arr["n"].sum())}
        for sp in sorted({f["split"] for f in fl}):
            sub = [f for f in fl if f["split"] == sp]
            s, _ = summarize_group(sub, rng, boot)
            result["split"].setdefault(sp, {})[lvl] = {"n_facts": s["n_facts"], "accuracy": s["accuracy"]}

    for lo_l, hi_l in zip(levels[:-1], levels[1:]):
        d = draws_by_level[hi_l]["accuracy"] - draws_by_level[lo_l]["accuracy"]
        est = result["exposure"][hi_l]["accuracy"]["est"] - result["exposure"][lo_l]["accuracy"]["est"]
        lo, hi = interval(d)
        result["steps"][f"{lo_l}->{hi_l}"] = {"est": est, "lo": lo, "hi": hi}

    zero = [r for r in rows if r["exposure_level"] == 0]
    for rel in sorted({r["relation"] for r in zero}):
        g = [r for r in zero if r["relation"] == rel and r["is_greedy"]]
        s = [r for r in zero if r["relation"] == rel and not r["is_greedy"]]
        result["chance_floor"][rel] = {
            "greedy": [sum(r["outcome"] == "correct" for r in g), len(g)],
            "sampled": [sum(r["outcome"] == "correct" for r in s), len(s)]}
    return result


# ----------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------

def pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def with_ci(m):
    if m is None or m["est"] is None:
        return "n/a"
    if m["lo"] is None:
        return pct(m["est"])
    return f"{pct(m['est'])} [{pct(m['lo'])}, {pct(m['hi'])}]"


def fnum(m):
    return "n/a" if m is None or m["est"] is None else f"{m['est']:.3f}"


def knum(m):
    if m is None or m["est"] is None:
        return "n/a"
    return f"{m['est']:.2f} [{m['lo']:.2f}, {m['hi']:.2f}]"


def render_markdown(results, levels, meta):
    L = ["# P1 exposure analysis", "", DEFINITIONS, "## Inputs", ""]
    for label, info in meta["inputs"].items():
        L.append(f"- `{label}`: `{info['path']}` ({info['rows']} rows, sha256 `{info['sha256'][:12]}`, "
                 f"grading {info['grading_versions']}, decoding {info['decoding_ids']})")
    split = meta.get("split", "all")
    scope = {"all": "all facts (dev and final)", "dev": "dev facts only (validation wording)",
             "final": "final facts only (test wording)"}[split]
    L += ["", f"Dataset `{meta['version']}`, {scope}; bootstrap draws {meta['boot']}, seed {meta['seed']}.", ""]
    for label, res in results.items():
        L += [f"## {label}: behaviour by exposure level", "",
              "| Exposure | Questions (facts) | Accuracy [95% CI] | Hallucination [95% CI] | Abstention | "
              "Coverage | Selective accuracy | Mean K [95% CI] | Facts K=0 / inconsistent / K>=0.8 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for lvl in levels:
            e = res["exposure"][lvl]
            kb = e["k_buckets"]
            L.append(f"| {lvl}x | {e['n_questions']} ({e['n_facts']}) | {with_ci(e['accuracy'])} | "
                     f"{with_ci(e['hallucination'])} | {pct(e['abstention']['est'])} | {pct(e['coverage']['est'])} | "
                     f"{pct(e['selective_accuracy']['est'])} | {knum(e['mean_K'])} | "
                     f"{kb['K=0']} / {kb['inconsistent']} / {kb['K>=0.8']} |")
        L += ["", "Change in accuracy between adjacent exposure levels (where the curve is steepest):", "",
              "| Step | Change [95% CI] |", "|---|---|"]
        for step, s in res["steps"].items():
            lo = "n/a" if s["lo"] is None else f"{100 * s['lo']:+.1f}"
            hi = "n/a" if s["hi"] is None else f"{100 * s['hi']:+.1f}"
            L.append(f"| {step.replace('->', 'x to ')}x | {100 * s['est']:+.1f} pts [{lo}, {hi}] |")
        rels = sorted(res["relation"])
        L += ["", f"### {label}: accuracy by relation (questions per cell in brackets)", "",
              "| Relation | " + " | ".join(f"{l}x" for l in levels) + " |", "|---|" + "---|" * len(levels)]
        for rel in rels:
            cells = [f"{pct(res['relation'][rel][l]['accuracy'])} ({res['relation'][rel][l]['n_questions']})"
                     for l in levels]
            L.append(f"| {rel} | " + " | ".join(cells) + " |")
        L += ["", f"### {label}: dev facts (validation wording) vs final facts (test wording)", "",
              "Dev and final differ in both the facts and the question wording, so this is a joint check; "
              "the held-out paraphrase run separates the two.", "",
              "| Split | " + " | ".join(f"{l}x" for l in levels) + " |", "|---|" + "---|" * len(levels)]
        for sp in sorted(res["split"]):
            L.append(f"| {sp} | " + " | ".join(with_ci(res["split"][sp][l]["accuracy"]) for l in levels) + " |")
        L += ["", f"### {label}: chance floor (correct answers at 0x, where the model has seen nothing)", "",
              "| Relation | Greedy | Sampled |", "|---|---|---|"]
        for rel, v in res["chance_floor"].items():
            L.append(f"| {rel} | {v['greedy'][0]}/{v['greedy'][1]} | {v['sampled'][0]}/{v['sampled'][1]} |")
        if res["confidence"]:
            L += ["", f"### {label}: mean confidence (average log-probability of the greedy answer)", "",
                  "| Exposure | Correct answers | Incorrect answers |", "|---|---|---|"]
            for lvl in levels:
                c = res["confidence"].get(lvl)
                if c:
                    mc = "n/a" if c["mean_conf_correct"] is None else f"{c['mean_conf_correct']:.3f} (n={c['n_correct']})"
                    mi = "n/a" if c["mean_conf_incorrect"] is None else f"{c['mean_conf_incorrect']:.3f} (n={c['n_incorrect']})"
                    L.append(f"| {lvl}x | {mc} | {mi} |")
        L.append("")
    return "\n".join(L) + "\n"


def write_csvs(results, levels, out_dir):
    with open(Path(out_dir) / "exposure_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "exposure", "n_questions", "n_facts", "accuracy", "accuracy_lo", "accuracy_hi",
                    "hallucination", "hallucination_lo", "hallucination_hi", "abstention", "coverage",
                    "selective_accuracy", "mean_K", "mean_K_lo", "mean_K_hi", "facts_K0", "facts_inconsistent",
                    "facts_K_ge_0.8"])
        for label, res in results.items():
            for lvl in levels:
                e = res["exposure"][lvl]
                kb = e["k_buckets"]
                w.writerow([label, lvl, e["n_questions"], e["n_facts"],
                            e["accuracy"]["est"], e["accuracy"]["lo"], e["accuracy"]["hi"],
                            e["hallucination"]["est"], e["hallucination"]["lo"], e["hallucination"]["hi"],
                            e["abstention"]["est"], e["coverage"]["est"], e["selective_accuracy"]["est"],
                            e["mean_K"]["est"], e["mean_K"]["lo"], e["mean_K"]["hi"],
                            kb["K=0"], kb["inconsistent"], kb["K>=0.8"]])
    with open(Path(out_dir) / "relation_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "relation", "exposure", "n_questions", "correct", "accuracy"])
        for label, res in results.items():
            for rel, by in res["relation"].items():
                for lvl in levels:
                    c = by[lvl]
                    w.writerow([label, rel, lvl, c["n_questions"], c["correct"], c["accuracy"]])


def make_figures(results, levels, primary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(levels))
    xt = [f"{l}x" for l in levels]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, (key, title) in zip(axes, [("accuracy", "Accuracy"), ("hallucination", "Hallucination rate"),
                                       ("coverage", "Coverage (share attempted)")]):
        for label, res in results.items():
            est = [res["exposure"][l][key]["est"] for l in levels]
            lo = [res["exposure"][l][key]["lo"] for l in levels]
            hi = [res["exposure"][l][key]["hi"] for l in levels]
            ax.plot(x, est, marker="o", label=label)
            ax.fill_between(x, lo, hi, alpha=0.2)
        ax.set_xticks(x, xt)
        ax.set_xlabel("Times the fact was shown in training (categorical axis)")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Greedy responses (shaded: 95% CI over facts)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_exposure_curves.png", dpi=150)
    plt.close(fig)

    res = results[primary]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rel, by in sorted(res["relation"].items()):
        ax.plot(x, [by[l]["accuracy"] for l in levels], marker="o", label=rel)
    ax.set_xticks(x, xt)
    ax.set_xlabel("Times the fact was shown in training (categorical axis)")
    ax.set_ylabel("Greedy accuracy")
    ax.set_title(f"Accuracy by relation ({primary})")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig_relation_breakdown.png", dpi=150)
    plt.close(fig)

    if res["confidence"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        lv = [l for l in levels if l in res["confidence"]]
        xs = np.arange(len(lv))
        for key, name in (("mean_conf_correct", "correct answers"), ("mean_conf_incorrect", "incorrect answers")):
            ax.plot(xs, [res["confidence"][l][key] if res["confidence"][l][key] is not None else np.nan for l in lv],
                    marker="o", label=name)
        ax.set_xticks(xs, [f"{l}x" for l in lv])
        ax.set_xlabel("Times the fact was shown in training (categorical axis)")
        ax.set_ylabel("Mean average log-probability of the answer")
        ax.set_title(f"Model confidence by exposure ({primary})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / "fig_confidence.png", dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def run(responses, data_dir, version, out_dir, boot=2000, seed=0, primary=None, notes="", figures=True,
        split="all"):
    """responses: {label: path}. split: "all", "dev" or "final" (only that split's facts are analysed).
    Returns (results, meta)."""
    eval_by_q = load_eval(data_dir, version)
    if split not in ("all", "dev", "final"):
        raise ValueError(f"split must be all, dev or final, got {split!r}")
    if split != "all":
        eval_by_q = {q: r for q, r in eval_by_q.items() if r["split"] == split}
    levels = sorted({int(q["exposure_level"]) for q in eval_by_q.values()})
    rng = np.random.default_rng(seed)
    results, inputs = {}, {}
    for label, path in responses.items():
        rows = load_responses(path, eval_by_q, allow_extra=split != "all")
        check_coverage(label, rows, eval_by_q)
        results[label] = analyze_label(rows, levels, rng, boot)
        inputs[label] = {
            "path": str(path) if not Path(path).is_absolute() else Path(path).name, "sha256": sha256(path), "rows": len(rows),
            "grading_versions": sorted({r["grading_version"] for r in rows if r["grading_version"]}) or ["not recorded"],
            "decoding_ids": sorted({r["decoding_config_id"] for r in rows if r["decoding_config_id"]}) or ["not recorded"],
            "methods": sorted({r["method"] for r in rows if r["method"]})}
    meta = {"version": version, "split": split, "boot": boot, "seed": seed, "levels": levels, "inputs": inputs,
            "notes": notes, "eval_sha256": sha256(Path(data_dir) / version / "eval.jsonl")}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    (out / "exposure_analysis.md").write_text(render_markdown(results, levels, meta), encoding="utf-8", newline="\n")
    write_csvs(results, levels, out)
    if figures:
        make_figures(results, levels, primary or next(iter(results)), out)
    return results, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--responses", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--version", default="main_v0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--primary", default=None, help="label used for the relation and confidence figures")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--split", choices=["all", "dev", "final"], default="all",
                    help="analyse only the dev or final facts (default: all)")
    a = ap.parse_args(argv)
    responses = {}
    for item in a.responses:
        if "=" not in item:
            sys.exit(f"--responses expects LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        responses[label] = path
    run(responses, a.data_dir, a.version, a.out_dir, a.boot, a.seed, a.primary, a.notes, not a.no_figures,
        a.split)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
