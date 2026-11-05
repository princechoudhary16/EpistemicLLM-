"""Compare methods on the same held-out questions under the same confidence policy.

Each run was fitted and frozen on its own development rows, so this applies one procedure per method
rather than one threshold. Matched coverage reads no labels: it sets each method's threshold so that it
answers the same share of questions, which is what makes selective risk comparable (plan.md section 6).
"""

import csv
import html
import json
from pathlib import Path

from common.metrics import (CORRECT, INCORRECT, area_under_risk_coverage, auroc, bootstrap_ci, paired_bootstrap_diff,
                            risk_coverage_curve)

from .calibration import MISSING_CONFIDENCE, select_threshold
from .calibration_report import MUTED_INK, SERIES_COLORS, fmt, pct, plot_frame, svg_text

N_BOOTSTRAP = 1000
ATTEMPTED = (CORRECT, INCORRECT)


def _confidence(row):
    return MISSING_CONFIDENCE if row["probability_correct"] is None else row["probability_correct"]


def matched_risk(rows, coverage):
    """Selective risk once the threshold keeps `coverage` of the questions. Returns (risk, achieved coverage)."""
    outcomes, confidence = [r["outcome"] for r in rows], [_confidence(r) for r in rows]
    selected = select_threshold(outcomes, confidence, coverage)
    if selected is None:
        return None, None
    kept = [o for o, c in zip(outcomes, confidence) if o in ATTEMPTED and c >= selected["probability_correct"]]
    return sum(o == INCORRECT for o in kept) / len(kept), selected["selection_coverage"]


def _aurc(rows):
    return area_under_risk_coverage(risk_coverage_curve([r["outcome"] for r in rows], [_confidence(r) for r in rows]))


def load_run(label, directory, signal):
    directory = Path(directory)
    summary = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    if signal not in summary["signals"]:
        raise ValueError(f"{label}: {signal!r} was not evaluated in {directory}; found {sorted(summary['signals'])}")
    with (directory / "calibrated_predictions.jsonl").open(encoding="utf-8") as handle:
        predictions = [json.loads(line) for line in handle]
    return {"label": label, "signal": signal, "summary": summary, "metrics": summary["signals"][signal]["metrics"],
            "rows": [row for row in predictions if row["signal"] == signal]}


def _cluster_field(rows):
    return "fact_id" if all(row.get("fact_id") for row in rows) else "question_id"


def _check_same_questions(runs):
    question_sets = {run["label"]: {row["question_id"] for row in run["rows"]} for run in runs}
    first = next(iter(question_sets.values()))
    different = [label for label, questions in question_sets.items() if questions != first]
    if different:
        raise ValueError(f"methods must be compared on identical questions; {different} differ")
    versions = {run["summary"]["provenance"]["dataset_version"] for run in runs}
    if len(versions) > 1:
        raise ValueError(f"runs come from different datasets: {sorted(versions)}")


def compare_runs(runs, reference, coverages, seed=0):
    """Returns a JSON-serializable comparison. `reference` is the label differences are taken against."""
    _check_same_questions(runs)
    by_label = {run["label"]: run for run in runs}
    if reference not in by_label:
        raise ValueError(f"reference {reference!r} is not one of {sorted(by_label)}")
    base = by_label[reference]["rows"]
    cluster = _cluster_field(base)
    results = {}
    for run in runs:
        rows, metrics = run["rows"], run["metrics"]
        entry = {
            "signal": run["signal"], "checkpoint_id": run["summary"]["provenance"]["checkpoint_id"],
            "training_seed": run["summary"]["provenance"]["training_seed"],
            "answer_everything": metrics["policy"]["answer_everything"], "auroc": metrics["auroc"],
            "auroc_ci_95": metrics.get("auroc_ci_95"), "brier": metrics["brier"], "ece_10_bins": metrics["ece_10_bins"],
            "aurc": _aurc(rows), "operating_points": {},
        }
        if run["label"] != reference:
            entry["aurc_difference"] = _difference(rows, base, _aurc, cluster, seed)
        for coverage in coverages:
            key = f"{coverage:.2f}"
            risk, achieved = matched_risk(rows, coverage)
            point = {"matched_selective_risk": risk, "matched_coverage": achieved,
                     "frozen": metrics["policy"]["operating_points"].get(key)}
            if run["label"] != reference:
                point["matched_risk_difference"] = _difference(
                    rows, base, lambda sample: matched_risk(sample, coverage)[0], cluster, seed)
            entry["operating_points"][key] = point
        results[run["label"]] = entry
    return {"reference": reference, "coverages": [f"{c:.2f}" for c in coverages], "n_questions": len(base),
            "cluster": cluster, "bootstrap_replicates": N_BOOTSTRAP, "seed": seed,
            "dataset_version": runs[0]["summary"]["provenance"]["dataset_version"], "methods": results}


def _difference(rows_a, rows_b, metric, cluster, seed):
    result = paired_bootstrap_diff(rows_a, rows_b, metric, cluster, n_boot=N_BOOTSTRAP, seed=seed)
    return {"estimate": result["estimate"], "low": result["ci_low"], "high": result["ci_high"],
            "n_clusters": result["n_clusters"]}


def compare_probe(run, predictions_csv, probe, seed=0):
    """A hidden-state probe's AUROC against the output signal, on the attempted questions both scored."""
    with Path(predictions_csv).open(newline="", encoding="utf-8") as handle:
        scores = {row["question_id"]: float(row["probability"]) for row in csv.DictReader(handle)
                  if row["probe"] == probe}
    if not scores:
        raise ValueError(f"{predictions_csv} has no predictions for probe {probe!r}")
    shared = [{**row, "probe_score": scores[row["question_id"]]} for row in run["rows"]
              if row["question_id"] in scores and row["outcome"] in ATTEMPTED and row["raw_score"] is not None]
    if not shared:
        return {"probe": probe, "signal": run["signal"], "n": 0, "status": "no questions scored by both"}

    def score_auroc(rows, field):
        return auroc([row[field] for row in rows], [row["outcome"] == CORRECT for row in rows])[0]

    def difference(rows):
        first, second = score_auroc(rows, "probe_score"), score_auroc(rows, "probability_correct")
        return None if first is None or second is None else first - second

    cluster = _cluster_field(shared)
    gap = bootstrap_ci(shared, difference, cluster, n_boot=N_BOOTSTRAP, seed=seed)
    labels = [row["outcome"] == CORRECT for row in shared]
    return {"probe": probe, "signal": run["signal"], "n": len(shared), "n_correct": sum(labels),
            "n_incorrect": len(labels) - sum(labels), "n_clusters": gap["n_clusters"],
            "probe_auroc": score_auroc(shared, "probe_score"), "signal_auroc": score_auroc(shared, "probability_correct"),
            "difference": {"estimate": gap["estimate"], "low": gap["ci_low"], "high": gap["ci_high"]}}


def _signed(difference):
    if not difference or difference["estimate"] is None:
        return "reference"
    return f"{difference['estimate']:+.3f} [{difference['low']:+.3f}, {difference['high']:+.3f}]"


def render_markdown(comparison, probe, title):
    reference = comparison["reference"]
    lines = [f"# {title}", "",
             f"{comparison['n_questions']} held-out questions from `{comparison['dataset_version']}`; every method answers "
             f"the same questions. Differences are against `{reference}`, resampled jointly by `{comparison['cluster']}` "
             f"({comparison['bootstrap_replicates']} replicates). Each method's policy was fitted and frozen on its own "
             "development rows.", "", "## Ranking and calibration", "",
             "| method | signal | training seed | accuracy, answering everything | AUROC [95% CI] | Brier | ECE | "
             "area under risk-coverage | difference [95% CI] |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, entry in comparison["methods"].items():
        interval = entry["auroc_ci_95"] or {}
        lines.append(f"| {label} | `{entry['signal']}` | {entry['training_seed']} | "
                     f"{pct(entry['answer_everything']['accuracy'])} | {fmt(entry['auroc'])} "
                     f"[{fmt(interval.get('low'))}, {fmt(interval.get('high'))}] | {fmt(entry['brier'])} | "
                     f"{fmt(entry['ece_10_bins'])} | {fmt(entry['aurc'])} | {_signed(entry.get('aurc_difference'))} |")
    lines += ["", "Lower is better for Brier, ECE and area under risk-coverage. A difference whose interval includes "
              "zero is not evidence that either method is better.", "", "## Frozen and matched coverage", "",
              "*Frozen* applies the threshold chosen on validation, so achieved coverage can drift from its target. "
              "*Matched* sets each method's threshold, without labels, to answer the same share of these questions, "
              "and is the fair comparison of selective risk.", "",
              "| target | method | frozen coverage | frozen selective accuracy | frozen hallucination | confident errors | "
              "matched coverage | matched selective risk | difference [95% CI] |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in comparison["coverages"]:
        for label, entry in comparison["methods"].items():
            point = entry["operating_points"][key]
            frozen = point["frozen"] or {}
            behavior = frozen.get("behavior") or {}
            errors = (frozen.get("confident_hallucinations") or {}).get("count", "N/A")
            lines.append(f"| {float(key):.0%} | {label} | {pct(behavior.get('coverage'))} | "
                         f"{pct(behavior.get('selective_accuracy'))} | {pct(behavior.get('hallucination_rate'))} | {errors} | "
                         f"{pct(point['matched_coverage'])} | {pct(point['matched_selective_risk'])} | "
                         f"{_signed(point.get('matched_risk_difference'))} |")
    if probe:
        lines += ["", "## Hidden-state probe against the output signal", ""]
        if probe["n"]:
            lines += [f"`{probe['probe']}` on the {probe['n']} attempted questions both scored "
                      f"({probe['n_correct']} correct, {probe['n_incorrect']} wrong, {probe['n_clusters']} facts): probe AUROC "
                      f"{fmt(probe['probe_auroc'])}, `{probe['signal']}` AUROC {fmt(probe['signal_auroc'])}, difference "
                      f"{_signed(probe['difference'])}. This compares ranking only, and the probe's layer was chosen on "
                      "these same questions, which favours the probe.", ""]
        else:
            lines += [f"`{probe['probe']}`: {probe['status']}.", ""]
    lines += ["![Risk-coverage by method](risk_coverage_by_method.svg)", ""]
    return "\n".join(lines)


def risk_coverage_svg(runs):
    n_rows = (len(runs) + 2) // 3
    parts, (left, top, plot_w, plot_h, legend_y) = plot_frame(
        "Selective risk against coverage, by method", "Coverage (share of all valid questions answered)",
        "Selective risk (wrong share of answers kept)", n_rows)
    for index, run in enumerate(runs):
        curve = risk_coverage_curve([r["outcome"] for r in run["rows"]], [_confidence(r) for r in run["rows"]])
        coords = " ".join(f"{left + p['coverage'] * plot_w:.1f},{top + (1 - p['selective_risk']) * plot_h:.1f}"
                          for p in curve if p["threshold"] >= 0)
        label = f"{run['label']}: {run['signal']}"
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{SERIES_COLORS[index]}" stroke-width="2">'
                     f"<title>{html.escape(label)}</title></polyline>")
        x, y = left + (index % 3) * 240, legend_y + (index // 3) * 22
        parts.append(f'<path d="M{x},{y} h18" stroke="{SERIES_COLORS[index]}" stroke-width="3"/>')
        parts.append(svg_text(x + 24, y + 4, label, 11, "start", MUTED_INK))
    parts.append("</svg>")
    return "\n".join(parts)


def write_comparison(comparison, probe, runs, output_dir, title):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {**comparison, "probe_comparison": probe}
    (out / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "risk_coverage_by_method.svg").write_text(risk_coverage_svg(runs), encoding="utf-8")
    markdown = render_markdown(comparison, probe, title)
    (out / "comparison_report.md").write_text(markdown, encoding="utf-8")
    return markdown
