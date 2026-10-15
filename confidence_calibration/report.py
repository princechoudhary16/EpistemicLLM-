"""Behaviour metrics and signal discrimination for one scored run.

Target = greedy response. Discrimination population = greedy responses that attempted an answer
(correct/incorrect); label = correct. One row per question, so the bootstrap is question-level.
"""

import csv
import json
from pathlib import Path

from .consistency import GROUP_KEYS, summarize_questions
from .metrics import auprc, auroc, bootstrap_ci, outcome_metrics
from .signals import SIGNAL_DIRECTIONS, oriented


def _fmt(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _signal_table(attempted):
    labels = [int(r["outcome"] == "correct") for r in attempted]
    table = []
    for name, direction in SIGNAL_DIRECTIONS.items():
        values = [oriented(name, r.get(name)) for r in attempted]
        scores = [float("nan") if v is None else v for v in values]
        used = [(l, v) for l, v in zip(labels, values) if v is not None]
        ci_low, ci_high = bootstrap_ci(auroc, scores, labels)
        table.append({
            "signal": name,
            "confident_when": "higher" if direction > 0 else "lower",
            "n_used": len(used),
            "n_missing": len(values) - len(used),
            "n_correct": sum(l for l, _ in used),
            "n_incorrect": sum(1 - l for l, _ in used),
            "auroc": auroc(scores, labels),
            "auroc_ci_low": ci_low,
            "auroc_ci_high": ci_high,
            "auprc": auprc(scores, labels),
        })
    return table


def build_report(responses):
    """Return (report dict, question-level summaries). Responses must be graded and scored."""
    questions = summarize_questions(responses)
    summary_by_key = {tuple(q[k] for k in GROUP_KEYS): q for q in questions}
    greedy = [r for r in responses if r.get("is_greedy")]
    samples = [r for r in responses if not r.get("is_greedy")]
    for r in greedy:
        summary = summary_by_key[tuple(r.get(k) for k in GROUP_KEYS)]
        r.update({k: v for k, v in summary.items() if k not in GROUP_KEYS})

    attempted = [r for r in greedy if r["outcome"] in ("correct", "incorrect")]
    checked = [r for r in greedy if "argmax_match" in r]
    generation_gaps = [abs(r["generation_avg_logprob"] - r["avg_logprob"]) for r in greedy
                       if r.get("generation_avg_logprob") is not None and r.get("avg_logprob") is not None]
    knowledge = [r["knowledge_score"] for r in greedy if r.get("knowledge_score") is not None]
    report = {
        "greedy_behaviour": outcome_metrics(r["outcome"] for r in greedy),
        "sample_behaviour": outcome_metrics(r["outcome"] for r in samples),
        "signals": _signal_table(attempted),
        "diagnostics": {
            "n_questions": len(greedy),
            "n_sample_responses": len(samples),
            "greedy_truncated": sum(bool(r.get("truncated")) for r in greedy),
            "sample_truncated": sum(bool(r.get("truncated")) for r in samples),
            "retokenized": sum(bool(r.get("retokenized")) for r in responses),
            "greedy_argmax_match_rate": sum(r["argmax_match"] for r in checked) / len(checked) if checked else None,
            "generation_logprob_max_abs_gap": max(generation_gaps) if generation_gaps else None,
            "mean_knowledge_score": sum(knowledge) / len(knowledge) if knowledge else None,
        },
    }
    return report, questions


def to_markdown(report, title):
    g, diag = report["greedy_behaviour"], report["diagnostics"]
    lines = [
        f"## {title}", "",
        "**Greedy behaviour** (N excludes invalid outputs):", "",
        "| N | correct | incorrect | abstain | invalid | accuracy | hallucination | coverage | selective acc. |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {g['N']} | {g['C']} | {g['W']} | {g['A']} | {g['invalid']} | {_fmt(g['accuracy'])} | "
        f"{_fmt(g['hallucination_rate'])} | {_fmt(g['coverage'])} | {_fmt(g['selective_accuracy'])} |",
        "",
        "**Signal discrimination** (attempted greedy answers; label = correct; 95% question-bootstrap CI):", "",
        "| signal | confident when | n (correct/incorrect) | AUROC [95% CI] | AUPRC |",
        "|---|---|---|---|---|",
    ]
    for s in report["signals"]:
        lines.append(
            f"| {s['signal']} | {s['confident_when']} | {s['n_used']} ({s['n_correct']}/{s['n_incorrect']}) | "
            f"{_fmt(s['auroc'])} [{_fmt(s['auroc_ci_low'])}, {_fmt(s['auroc_ci_high'])}] | {_fmt(s['auprc'])} |"
        )
    lines += ["", "**Diagnostics:**", ""]
    lines += [f"- {k}: {_fmt(v, 4) if isinstance(v, float) else v}" for k, v in diag.items()]
    return "\n".join(lines) + "\n"


def write_report(report, output_dir, title):
    """Write report.md, report.json and signals.csv; return the markdown."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "signals.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["signals"][0]))
        writer.writeheader()
        writer.writerows(report["signals"])
    (out / "report.json").write_text(json.dumps(report, indent=2))
    markdown = to_markdown(report, title)
    (out / "report.md").write_text(markdown)
    return markdown
