"""Compact, dependency-free artifacts for the P4 probability-calibration analysis."""

import html
import json
from pathlib import Path


COLORS = ("#1769aa", "#d1495b", "#2a9d8f", "#8b5cf6", "#e07a16", "#59636e", "#b23aee")


def _fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def _svg_text(x, y, value, size=12, anchor="middle", color="#263238"):
    return (f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{color}">{html.escape(str(value))}</text>')


def _plot_frame(title, x_label, y_label, width=820, height=540):
    left, top, right, bottom = 78, 58, 30, 88
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             _svg_text(width / 2, 28, title, 17),
             f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="#334155"/>']
    for tick in range(6):
        value = tick / 5
        x = left + value * plot_w
        y = top + plot_h - value * plot_h
        parts.append(f'<path d="M{x:.1f},{top} V{top + plot_h}" stroke="#e5e7eb"/>')
        parts.append(f'<path d="M{left},{y:.1f} H{left + plot_w}" stroke="#e5e7eb"/>')
        parts.append(_svg_text(x, top + plot_h + 19, f"{value:.1f}", 10))
        parts.append(_svg_text(left - 12, y + 4, f"{value:.1f}", 10, "end"))
    parts.append(_svg_text(width / 2, height - 25, x_label, 12))
    parts.append(f'<text transform="translate(20 {top + plot_h / 2}) rotate(-90)" '
                 'font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#263238">'
                 f'{html.escape(y_label)}</text>')
    return parts, (left, top, plot_w, plot_h, width, height)


def _reliability_svg(summary):
    parts, frame = _plot_frame("Reliability diagram — locked test", "Mean predicted P(correct)",
                               "Observed accuracy")
    left, top, plot_w, plot_h, width, _ = frame
    parts.append(f'<path d="M{left},{top + plot_h} L{left + plot_w},{top}" '
                 'stroke="#475569" stroke-dasharray="6 5" stroke-width="1.5"/>')
    selected = summary.get("selected_signal_on_validation")
    for index, (signal, result) in enumerate(summary.get("signals", {}).items()):
        test = result.get("test") or {}
        bins = test.get("reliability_bins", [])
        color = COLORS[index % len(COLORS)]
        points = []
        for item in bins:
            confidence, accuracy = item.get("mean_confidence"), item.get("accuracy")
            if item.get("n", 0) and confidence is not None and accuracy is not None:
                x = left + confidence * plot_w
                y = top + (1 - accuracy) * plot_h
                points.append((x, y))
        if points:
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_x = 90 + (index % 4) * 175
        legend_y = 490 + (index // 4) * 20
        parts.append(f'<path d="M{legend_x},{legend_y - 4} h18" stroke="{color}" stroke-width="3"/>')
        label = signal + (" *" if signal == selected else "")
        parts.append(_svg_text(legend_x + 24, legend_y, label, 10, "start"))
    parts.append('</svg>')
    return "\n".join(parts)


def _risk_coverage_svg(summary):
    parts, frame = _plot_frame("Selective risk vs coverage — locked test", "Coverage (all valid responses)",
                               "Selective risk (incorrect / retained answers)")
    left, top, plot_w, plot_h, _, _ = frame
    selected = summary.get("selected_signal_on_validation")
    for index, (signal, result) in enumerate(summary.get("signals", {}).items()):
        curve = ((result.get("test") or {}).get("policy") or {}).get("risk_coverage_curve", [])
        color = COLORS[index % len(COLORS)]
        points = [(left + p["coverage"] * plot_w, top + (1 - p["selective_risk"]) * plot_h)
                  for p in curve if p.get("coverage") is not None and p.get("selective_risk") is not None]
        if points:
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = 90 + (index % 4) * 175
        legend_y = 490 + (index // 4) * 20
        parts.append(f'<path d="M{legend_x},{legend_y - 4} h18" stroke="{color}" stroke-width="3"/>')
        label = signal + (" *" if signal == selected else "")
        parts.append(_svg_text(legend_x + 24, legend_y, label, 10, "start"))
    parts.append('</svg>')
    return "\n".join(parts)


def _markdown(summary, title):
    lines = [f"# {title}", "", "## Protocol", "",
             f"- Probability mapping: {summary['protocol']['probability_method']}",
             f"- Fit / select / evaluate: {summary['protocol']['fit_split']} / "
             f"{summary['protocol']['selection_split']} / {summary['protocol']['evaluation_split']}",
             f"- Signal selected on validation by Brier score: **{summary.get('selected_signal_on_validation') or 'N/A'}**",
             f"- Threshold operating point: validation-selected for requested coverage "
             f"{summary['protocol']['threshold_target_coverage']:.0%}",
             "- Test labels are not used for fitting, signal selection, or threshold selection.",
             "- Brier/ECE use attempted greedy answers with usable signals; abstentions and invalid outputs "
             "remain in the behavior-policy denominator, not the probability denominator.", "",
             "## Split counts", "", "| split | questions | correct | incorrect | abstain | invalid |",
             "|---|---:|---:|---:|---:|---:|"]
    for split, counts in summary["split_counts"].items():
        outcomes = counts["outcomes"]
        lines.append(f"| {split} | {counts['n_questions']} | {outcomes['correct']} | {outcomes['incorrect']} | "
                     f"{outcomes['abstain']} | {outcomes['invalid']} |")
    lines += ["", "## Calibration and confidence-policy results", "",
              "| signal | train N | validation Brier | test N (C/W) | AUROC [95% CI] | AUPRC | "
              "Brier [95% CI] | ECE [95% CI] | selected threshold | test coverage | confident errors |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for signal, result in summary["signals"].items():
        train, val, test = result["training"], result["validation"] or {}, result["test"] or {}
        policy = test.get("policy") or {}
        discrimination = test.get("discrimination") or {}
        auroc_ci = test.get("auroc_ci_95") or {}
        brier_ci = test.get("brier_ci_95") or {}
        ece_ci = test.get("ece_10_bins_ci_95") or {}
        halluc = policy.get("confident_hallucinations") or {}
        behavior = policy.get("behavior") or {}
        lines.append("| {} | {} | {} | {} ({}/{}) | {} [{}, {}] | {} | {} [{}, {}] | {} [{}, {}] | {} | {} | {} |".format(
            signal, train.get("n", 0), _fmt(val.get("brier") if val else None), test.get("n", 0),
            discrimination.get("n_correct", 0), discrimination.get("n_incorrect", 0),
            _fmt(discrimination.get("auroc")), _fmt(auroc_ci.get("low")), _fmt(auroc_ci.get("high")),
            _fmt(discrimination.get("auprc")), _fmt(test.get("brier")), _fmt(brier_ci.get("low")),
            _fmt(brier_ci.get("high")), _fmt(test.get("ece_10_bins")), _fmt(ece_ci.get("low")),
            _fmt(ece_ci.get("high")), _fmt(result.get("threshold", {}).get("probability_correct")),
            _fmt(behavior.get("coverage")), halluc.get("count", "N/A")))
    selected = summary.get("selected_signal_on_validation")
    selected_test = (summary.get("signals", {}).get(selected, {}).get("test", {}) if selected else {})
    strata = selected_test.get("stratified", {})
    lines += ["", "## Selected-signal test slices", "",
              "Descriptive only; thresholds and signal choice remain frozen from validation. Sample-K slices "
              "use sampled generations, not the greedy response being scored.", "",
              "| dimension | group | questions | correct | incorrect | coverage | Brier | ECE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dimension, groups in strata.items():
        for label, metrics in groups.items():
            behavior = metrics["behavior"]
            lines.append(f"| {dimension} | {label} | {metrics['n_questions']} | {behavior['C']} | {behavior['W']} | "
                         f"{_fmt(behavior.get('coverage'))} | {_fmt(metrics.get('brier'))} | "
                         f"{_fmt(metrics.get('ece_10_bins'))} |")
    if not any(strata.values()):
        lines.append("| N/A | No exposure or sample-K grouping metadata is available | 0 | N/A | N/A | N/A | N/A | N/A |")
    lines += ["", "![Reliability diagram](reliability_diagram.svg)", "",
              "![Risk-coverage curves](risk_coverage.svg)", "",
              "The selected signal and operating thresholds are development choices. Interpret test estimates "
              "with their intervals and class counts; small locked-test sets can produce wide uncertainty.", ""]
    return "\n".join(lines)


def write_calibration_report(summary, predictions, output_dir, title="P4 confidence calibration"):
    """Write machine-readable results, downstream predictions, Markdown and vector figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    with (out / "calibrated_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (out / "reliability_diagram.svg").write_text(_reliability_svg(summary), encoding="utf-8")
    (out / "risk_coverage.svg").write_text(_risk_coverage_svg(summary), encoding="utf-8")
    markdown = _markdown(summary, title)
    (out / "calibration_report.md").write_text(markdown, encoding="utf-8")
    return markdown
