"""Markdown tables and dependency-free SVG figures for one calibration stage."""

import html
import json
from pathlib import Path

from .error_analysis import write_error_table
from .signals import SIGNAL_DIRECTIONS

SURFACE, INK, MUTED_INK, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Colour follows the signal, in every figure of every run.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SIGNAL_COLORS = dict(zip(SIGNAL_DIRECTIONS, SERIES_COLORS))
REFERENCE_SIGNAL = "avg_logprob"
N_ERROR_EXAMPLES = 15


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def pct(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def _with_ci(value, interval):
    interval = interval or {}
    return f"{fmt(value)} [{fmt(interval.get('low'))}, {fmt(interval.get('high'))}]"


def svg_text(x, y, value, size=12, anchor="middle", color=INK):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{color}">{html.escape(str(value))}</text>')


def plot_frame(title, x_label, y_label, n_legend_rows, width=820, height=540):
    left, top, right, bottom = 78, 58, 30, 66 + 22 * n_legend_rows
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
             svg_text(width / 2, 28, title, 16)]
    for tick in range(6):
        value = tick / 5
        x, y = left + value * plot_w, top + plot_h - value * plot_h
        parts.append(f'<path d="M{x:.1f},{top} V{top + plot_h}" stroke="{GRID}"/>')
        parts.append(f'<path d="M{left},{y:.1f} H{left + plot_w}" stroke="{GRID}"/>')
        parts.append(svg_text(x, top + plot_h + 19, f"{value:.1f}", 11, color=MUTED_INK))
        parts.append(svg_text(left - 10, y + 4, f"{value:.1f}", 11, "end", MUTED_INK))
    parts.append(svg_text(left + plot_w / 2, top + plot_h + 42, x_label, 12, color=MUTED_INK))
    parts.append(f'<text transform="translate(22 {top + plot_h / 2}) rotate(-90)" font-family="Arial, sans-serif" '
                 f'font-size="12" text-anchor="middle" fill="{MUTED_INK}">{html.escape(y_label)}</text>')
    return parts, (left, top, plot_w, plot_h, top + plot_h + 58)


def _legend(signals, selected, x0, y0):
    parts = []
    for index, signal in enumerate(signals):
        x, y = x0 + (index % 4) * 180, y0 + (index // 4) * 22
        parts.append(f'<path d="M{x},{y} h18" stroke="{SIGNAL_COLORS[signal]}" stroke-width="3"/>')
        label = f"{signal} (selected)" if signal == selected else signal
        parts.append(svg_text(x + 24, y + 4, label, 11, "start", MUTED_INK))
    return parts


def _legend_rows(count):
    return (count + 3) // 4


def _headline_signals(summary):
    """The selected signal and the shared reference; the two every table and figure leads with."""
    return [s for s in dict.fromkeys((summary["selected_signal"], REFERENCE_SIGNAL)) if s in summary["signals"]]


def _reliability_svg(summary):
    selected = summary["selected_signal"]
    signals = _headline_signals(summary)
    parts, (left, top, plot_w, plot_h, legend_y) = plot_frame(
        f"Reliability on {summary['evaluated_split']}", "Mean predicted P(correct)", "Observed accuracy",
        _legend_rows(len(signals)))
    parts.append(f'<path d="M{left},{top + plot_h} L{left + plot_w},{top}" stroke="{MUTED_INK}" '
                 'stroke-dasharray="6 5" stroke-width="1.5"/>')
    for signal in signals:
        color = SIGNAL_COLORS[signal]
        points = []
        for item in summary["signals"][signal]["metrics"]["reliability_bins"]:
            if item["n"]:
                points.append((left + item["mean_confidence"] * plot_w, top + (1 - item["accuracy"]) * plot_h, item))
        if points:
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y, item in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="{SURFACE}" stroke-width="2">'
                         f'<title>{html.escape(signal)}: predicted {item["mean_confidence"]:.3f}, '
                         f'observed {item["accuracy"]:.3f}, n={item["n"]}</title></circle>')
    parts += _legend(signals, selected, left, legend_y)
    parts.append("</svg>")
    return "\n".join(parts)


def _risk_coverage_svg(summary):
    selected = summary["selected_signal"]
    signals = list(summary["signals"])
    parts, (left, top, plot_w, plot_h, legend_y) = plot_frame(
        f"Selective risk against coverage on {summary['evaluated_split']}", "Coverage (share of all valid questions answered)",
        "Selective risk (wrong share of answers kept)", _legend_rows(len(signals)))
    for signal in signals:
        curve = summary["signals"][signal]["metrics"]["policy"]["risk_coverage_curve"]
        coords = " ".join(f"{left + p['coverage'] * plot_w:.1f},{top + (1 - p['selective_risk']) * plot_h:.1f}"
                          for p in curve if p["threshold"] >= 0)
        if coords:
            width = 2.5 if signal == selected else 1.5
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{SIGNAL_COLORS[signal]}" '
                         f'stroke-width="{width}"><title>{html.escape(signal)}</title></polyline>')
    parts += _legend(signals, selected, left, legend_y)
    parts.append("</svg>")
    return "\n".join(parts)


def _inputs_table(summary):
    lines = ["## Inputs", "", "| file | rows | sha256 | arrived graded with | outcomes changed by regrade |",
             "|---|---:|---|---|---:|"]
    for item in summary["inputs"]:
        versions = ", ".join(sorted(item["arrived_grading_versions"]))
        lines.append(f"| `{item['path']}` | {item['rows']} | `{item['sha256'][:12]}` | {versions} | "
                     f"{item['outcomes_changed_by_regrade']} |")
    return lines + [""]


def _split_table(summary):
    lines = ["## Split counts", "", "Greedy responses, the target being judged.", "",
             "| split | questions | correct | incorrect | abstain | invalid |", "|---|---:|---:|---:|---:|---:|"]
    for split, counts in summary["split_counts"].items():
        outcomes = counts["outcomes"]
        lines.append(f"| {split} | {counts['n_questions']} | {outcomes['correct']} | {outcomes['incorrect']} | "
                     f"{outcomes['abstain']} | {outcomes['invalid']} |")
    return lines + [""]


def _signal_table(summary):
    differences = (summary.get("reference_comparison") or {}).get("differences", {})
    lines = [f"## Signal comparison on {summary['evaluated_split']}", "",
             "Attempted greedy answers with a usable signal. Intervals are 95% cluster-bootstrap percentile intervals.",
             f"The difference column is AUROC minus `{REFERENCE_SIGNAL}` on the same questions, resampled jointly.", "",
             "| signal | N (correct/wrong) | AUROC [95% CI] | difference [95% CI] | AUPRC | Brier [95% CI] | "
             "base-rate Brier | ECE [95% CI] | ECE read raw |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for signal, result in summary["signals"].items():
        m = result["metrics"]
        difference = differences.get(signal)
        diff_text = "reference" if signal == REFERENCE_SIGNAL else "N/A"
        if difference and difference["auroc_difference"] is not None:
            diff_text = (f"{difference['auroc_difference']:+.3f} [{difference['low']:+.3f}, "
                         f"{difference['high']:+.3f}]")
        raw = (m.get("uncalibrated") or {}).get("ece_10_bins")
        name = f"**{signal}**" if signal == summary["selected_signal"] else signal
        lines.append(f"| {name} | {m['n']} ({m['n_correct']}/{m['n_incorrect']}) | "
                     f"{_with_ci(m['auroc'], m.get('auroc_ci_95'))} | {diff_text} | {fmt(m['auprc'])} | "
                     f"{_with_ci(m['brier'], m.get('brier_ci_95'))} | {fmt(m['brier_base_rate'])} | "
                     f"{_with_ci(m['ece_10_bins'], m.get('ece_10_bins_ci_95'))} | {fmt(raw)} |")
    lines += ["", f"The selected signal (validation Brier) is in bold. Base-rate Brier predicts the training accuracy "
              f"({pct(summary['train_base_rate'])}) for every answer, so a useful signal must beat it. "
              "ECE read raw treats the signal itself as a probability (`exp` of a log-probability, or the agreement "
              "share); N/A means it has no such reading.", ""]
    return lines


def _policy_table(summary):
    signals = _headline_signals(summary)
    baseline = summary["signals"][signals[0]]["metrics"]["policy"]["answer_everything"]
    lines = [f"## Abstention policies on {summary['evaluated_split']}", "",
             f"Attempted answers below the threshold become abstentions. Rates use all {baseline['N']} valid questions, "
             "abstentions included; selective accuracy uses the answers kept. Thresholds come from validation.", "",
             f"Answering everything: accuracy {pct(baseline['accuracy'])}, hallucination "
             f"{pct(baseline['hallucination_rate'])}, abstention {pct(baseline['abstention_rate'])}, "
             f"coverage {pct(baseline['coverage'])}, selective accuracy {pct(baseline['selective_accuracy'])}.", "",
             "| signal | target coverage | threshold P(correct) | coverage | accuracy | selective accuracy | "
             "hallucination | confident errors |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for signal in signals:
        points = summary["signals"][signal]["metrics"]["policy"]["operating_points"]
        for key in sorted(points, reverse=True):
            point = points[key]
            b = point["behavior"]
            lines.append(f"| {signal} | {float(key):.0%} | {fmt(point['threshold'])} | {pct(b['coverage'])} | "
                         f"{pct(b['accuracy'])} | {pct(b['selective_accuracy'])} | {pct(b['hallucination_rate'])} | "
                         f"{point['confident_hallucinations']['count']} |")
    lines += ["", "Tied scores can make a threshold keep more answers than its target; the coverage column shows what "
              "was actually kept.", ""]
    return lines


def _threshold_stability_table(summary):
    rows = [(signal, result["metrics"]["threshold_stability_ci_95"])
            for signal, result in summary["signals"].items() if "threshold_stability_ci_95" in result["metrics"]]
    if not rows:
        return []
    key = summary["primary_coverage"]
    lines = ["## Threshold stability", "",
             f"How much the {float(key):.0%}-coverage threshold moves when validation is resampled.", "",
             "| signal | threshold | 95% interval |", "|---|---:|---:|"]
    for signal, interval in rows:
        threshold = summary["signals"][signal]["metrics"]["policy"]["operating_points"][key]["threshold"]
        lines.append(f"| {signal} | {fmt(threshold)} | [{fmt(interval['low'])}, {fmt(interval['high'])}] |")
    return lines + [""]


def _stability_table(summary):
    signals = _headline_signals(summary)
    lines = ["## Signal stability across groups", "",
             "Descriptive. AUROC is N/A where a group has only one outcome. Sampled K does not include the greedy answer "
             "being judged, but it shares its samples with the consistency signals.", "",
             "| signal | grouping | group | questions | correct/wrong | AUROC [95% CI] | Brier | ECE |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    found = False
    for signal in signals:
        for dimension, groups in summary["signals"][signal]["metrics"]["stability"].items():
            for label, m in groups.items():
                found = True
                lines.append(f"| {signal} | {dimension.replace('_', ' ')} | {label} | {m['n_questions']} | "
                             f"{m['n_correct']}/{m['n_incorrect']} | {_with_ci(m['auroc'], m.get('auroc_ci_95'))} | "
                             f"{fmt(m['brier'])} | {fmt(m['ece_10_bins'])} |")
    return lines + [""] if found else []


def _error_section(summary):
    analysis = summary.get("confident_errors")
    if not analysis:
        return []
    patterns = analysis["by_pattern"]
    lines = [f"## High-confidence errors on {summary['evaluated_split']}", "",
             f"`{analysis['signal']}` at the {float(summary['primary_coverage']):.0%}-coverage threshold "
             f"(P(correct) >= {fmt(analysis['threshold'])}): {analysis['n_errors']} wrong out of "
             f"{analysis['n_answers_kept']} answers kept.", "",
             "| pattern | meaning | count |", "|---|---|---:|",
             f"| stable_wrong | K=0; the samples repeat the wrong answer | {patterns['stable_wrong']} |",
             f"| wavering_wrong | K=0; the samples disagree with it | {patterns['wavering_wrong']} |",
             f"| knows_sometimes | K>0; some samples are right | {patterns['knows_sometimes']} |", "",
             f"Flagged by the shared grader for review: {analysis['n_flagged_by_grader']}. "
             f"Every error is listed in `confident_errors.csv`."]
    if analysis["by_exposure"]:
        by_level = ", ".join(f"{level}x: {count}" for level, count in analysis["by_exposure"].items())
        lines += ["", f"By exposure level: {by_level}."]
    lines += ["", "Most confident errors:", "", "| P(correct) | pattern | question | answer | reference |",
              "|---:|---|---|---|---|"]
    for example in analysis["examples"][:N_ERROR_EXAMPLES]:
        cells = [_table_text(example[field]) for field in ("question", "model_answer", "reference_answer")]
        lines.append(f"| {fmt(example['probability_correct'])} | {example['pattern']} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return lines + [""]


def _table_text(value, limit=110):
    text = " ".join(str(value).split()).replace("|", "/")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def render_markdown(summary, title):
    unavailable = ", ".join(f"`{name}`" for name in summary["unavailable_signals"]) or "none"
    provenance = summary["provenance"]
    lines = [f"# {title}", "",
             f"Model `{provenance['model_id']}` at `{provenance['model_revision'][:8]}`, checkpoint "
             f"`{provenance['checkpoint_id']}`; prompt `{provenance['prompt_version']}`, decoding "
             f"`{provenance['decoding_config_id']}`, grader `{provenance['grading_version']}`. Calibrators are "
             f"fitted on train only and every choice is made on validation. Signals not available: {unavailable}.", ""]
    if summary["stage"] == "evaluation":
        fitted_on = summary["policy_provenance"]
        lines += [f"Frozen policy `{summary['policy_sha256'][:12]}`, fitted on `{fitted_on['checkpoint_id']}` / "
                  f"`{fitted_on['dataset_version']}`, applied unchanged"
                  + (" as a transfer test: this model or dataset differs from the one it was fitted on." if summary["transfer"]
                     else ". Nothing was refitted or reselected."), ""]
    for section in (_inputs_table, _split_table, _signal_table, _policy_table, _threshold_stability_table,
                    _stability_table, _error_section):
        lines += section(summary)
    lines += ["![Reliability diagram](reliability_diagram.svg)", "", "![Risk-coverage curves](risk_coverage.svg)", ""]
    return "\n".join(lines)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_report(summary, predictions, output_dir, title, report_name):
    """Write machine-readable results, per-question predictions, Markdown and figures. Returns the Markdown."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / f"{report_name}.json", summary)
    with (out / "calibrated_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (out / "reliability_diagram.svg").write_text(_reliability_svg(summary), encoding="utf-8")
    (out / "risk_coverage.svg").write_text(_risk_coverage_svg(summary), encoding="utf-8")
    if summary.get("confident_errors"):
        write_error_table(summary["confident_errors"], out / "confident_errors.csv")
    markdown = render_markdown(summary, title)
    (out / f"{report_name}_report.md").write_text(markdown, encoding="utf-8")
    return markdown


def write_policy(policy, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "frozen_policy.json", policy)
