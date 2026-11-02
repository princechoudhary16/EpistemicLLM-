"""Compare paired base/SFT prompt representations with a train-fit joint PCA."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


EXPOSURE_COLORS = {
    0: "#636363",
    1: "#1f78b4",
    2: "#33a02c",
    5: "#ff7f00",
    10: "#e31a1c",
    20: "#6a3d9a",
}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_run(run_dir):
    run_dir = Path(run_dir)
    data = np.load(run_dir / "features.npz", allow_pickle=False)
    ids = data["question_ids"].astype(str).tolist()
    predictions = {row["question_id"]: row for row in read_jsonl(run_dir / "predictions.jsonl")}
    if set(predictions) != set(ids):
        raise ValueError(f"Prediction/feature ID mismatch in {run_dir}")
    exposures = {qid: predictions[qid]["exposure_level"] for qid in ids}
    return data, ids, predictions, exposures


def svg_text(x, y, value, size=13, anchor="start", weight="normal", fill="#222"):
    value = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{fill}">{value}</text>')


def make_svg(path, projected, variances, exposures, ids, split_seed):
    width, height = 1360, 1120
    panel_w, panel_h = 620, 450
    panel_origins = [(48, 88), (692, 88), (48, 610), (692, 610)]
    layers = [6, 12, 18, 24]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, "Base vs matched SFT prompt representations", 22, "middle", "bold"),
        svg_text(width / 2, 58, f"Joint PCA fit on training questions only; held-out facts from split seed {split_seed}", 13, "middle", fill="#555"),
    ]
    # Shared legend: color is exposure, marker shape is checkpoint.
    lx, ly = 100, 78
    for i, exp in enumerate(sorted(EXPOSURE_COLORS)):
        x = lx + i * 130
        parts.append(f'<circle cx="{x}" cy="{ly}" r="6" fill="{EXPOSURE_COLORS[exp]}"/>')
        parts.append(svg_text(x + 12, ly + 4, f"{exp}x exposure", 12))
    parts += [
        '<circle cx="1000" cy="78" r="6" fill="#555"/>',
        svg_text(1013, 82, "Base", 12),
        '<polygon points="1080,71 1087,84 1073,84" fill="#555"/>',
        svg_text(1094, 82, "SFT", 12),
        '<line x1="1150" y1="78" x2="1182" y2="78" stroke="#aaa" stroke-width="1"/>',
        svg_text(1190, 82, "same question", 12),
    ]

    for panel, layer in enumerate(layers):
        ox, oy = panel_origins[panel]
        x0, y0, plot_w, plot_h = ox + 66, oy + 42, panel_w - 100, panel_h - 95
        base_xy, sft_xy = projected[layer]
        all_xy = np.vstack([base_xy, sft_xy])
        mins = all_xy.min(axis=0)
        maxs = all_xy.max(axis=0)
        margin = (maxs - mins) * 0.08
        mins -= margin
        maxs += margin
        spans = np.maximum(maxs - mins, 1e-8)

        def coords(point):
            px = x0 + (point[0] - mins[0]) / spans[0] * plot_w
            py = y0 + plot_h - (point[1] - mins[1]) / spans[1] * plot_h
            return px, py

        parts.append(svg_text(ox + panel_w / 2, oy + 20,
                              f"Layer {layer} · PC1+PC2 {variances[layer] * 100:.1f}%", 16, "middle", "bold"))
        parts.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="#555" stroke-width="1"/>')
        # Light grid and tick labels.
        for tick in range(5):
            frac = tick / 4
            xx = x0 + frac * plot_w
            yy = y0 + (1 - frac) * plot_h
            vx = mins[0] + frac * spans[0]
            vy = mins[1] + frac * spans[1]
            parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + plot_h}" stroke="#e5e5e5"/>')
            parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + plot_w}" y2="{yy:.1f}" stroke="#e5e5e5"/>')
            parts.append(svg_text(xx, y0 + plot_h + 18, f"{vx:.1f}", 10, "middle", fill="#555"))
            parts.append(svg_text(x0 - 8, yy + 3, f"{vy:.1f}", 10, "end", fill="#555"))
        parts.append(svg_text(x0 + plot_w / 2, y0 + plot_h + 39, "PC1", 12, "middle"))
        parts.append(f'<text x="{ox + 15}" y="{y0 + plot_h / 2}" font-family="Arial" font-size="12" text-anchor="middle" transform="rotate(-90 {ox + 15} {y0 + plot_h / 2})">PC2</text>')

        # Connect a deterministic subset of paired prompt vectors to show their shift.
        for idx in range(0, len(ids), 3):
            bx, by = coords(base_xy[idx])
            sx, sy = coords(sft_xy[idx])
            parts.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{sx:.1f}" y2="{sy:.1f}" stroke="#888" stroke-opacity="0.24" stroke-width="0.8"/>')
        for idx, qid in enumerate(ids):
            color = EXPOSURE_COLORS[int(exposures[qid])]
            bx, by = coords(base_xy[idx])
            sx, sy = coords(sft_xy[idx])
            parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.2" fill="{color}" fill-opacity="0.78" stroke="white" stroke-width="0.5"/>')
            parts.append(f'<polygon points="{sx:.1f},{sy - 5:.1f} {sx + 4.5:.1f},{sy + 3.5:.1f} {sx - 4.5:.1f},{sy + 3.5:.1f}" fill="{color}" fill-opacity="0.84" stroke="white" stroke-width="0.5"/>')

    parts.append(svg_text(width / 2, height - 15,
                          "Each point is a held-out question prompt; colors show training exposure. Axes are joint within each layer, not comparable across layers.",
                          12, "middle", fill="#555"))
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, default=Path("results/p3/synthetic_main_v0_base_prompt_v0_bf16_comparison"))
    parser.add_argument("--sft-run", type=Path, default=Path("results/p3/synthetic_main_v0_sft_matched_seed0"))
    parser.add_argument("--split", type=Path, default=Path("results/p3/synthetic_main_v0_base_prompt_v0/probe/split.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/p3_representation_comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base, base_ids, base_preds, base_exp = load_run(args.base_run)
    sft, sft_ids, sft_preds, sft_exp = load_run(args.sft_run)
    if base_ids != sft_ids:
        raise ValueError("Base and SFT feature IDs/order differ")
    if base_exp != sft_exp:
        raise ValueError("Exposure assignments differ across runs")
    for layer in (6, 12, 18, 24):
        if base[f"layer_{layer}"].shape != sft[f"layer_{layer}"].shape:
            raise ValueError(f"Layer {layer} shape mismatch")

    split = json.loads(args.split.read_text())
    id_index = {qid: i for i, qid in enumerate(base_ids)}
    train_ids, val_ids = split["training_ids"], split["validation_ids"]
    if set(train_ids) & set(val_ids):
        raise ValueError("Train/validation question IDs overlap")
    train_groups = {base["group_ids"][id_index[qid]].item() for qid in train_ids}
    val_groups = {base["group_ids"][id_index[qid]].item() for qid in val_ids}
    if train_groups & val_groups:
        raise ValueError("Train/validation fact groups overlap")
    train_idx = np.asarray([id_index[qid] for qid in train_ids])
    val_idx = np.asarray([id_index[qid] for qid in val_ids])
    val_order = [base_ids[i] for i in val_idx]
    exposures = {qid: base_exp[qid] for qid in val_order}

    projected = {}
    variances = {}
    rows = []
    for layer in (6, 12, 18, 24):
        xb = base[f"layer_{layer}"]
        xs = sft[f"layer_{layer}"]
        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(np.concatenate([xb[train_idx], xs[train_idx]], axis=0))
        projected[layer] = (pca.transform(xb[val_idx]), pca.transform(xs[val_idx]))
        variances[layer] = float(pca.explained_variance_ratio_.sum())
        for exp in sorted(EXPOSURE_COLORS):
            mask = np.asarray([exposures[qid] == exp for qid in val_order])
            b, s = xb[val_idx][mask], xs[val_idx][mask]
            cos = np.sum(b * s, axis=1) / (np.linalg.norm(b, axis=1) * np.linalg.norm(s, axis=1) + 1e-12)
            shift = np.linalg.norm(s - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-12)
            rows.append({
                "layer": layer,
                "exposure_level": exp,
                "validation_questions": int(mask.sum()),
                "mean_cosine_similarity": float(cos.mean()),
                "median_cosine_similarity": float(np.median(cos)),
                "mean_relative_l2_shift": float(shift.mean()),
                "median_relative_l2_shift": float(np.median(shift)),
                "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
                "pca_pc2_variance": float(pca.explained_variance_ratio_[1]),
            })

    svg_path = args.output_dir / "base_vs_sft_joint_pca.svg"
    make_svg(svg_path, projected, variances, exposures, val_order, split["seed"])
    csv_path = args.output_dir / "paired_representation_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Base vs SFT representation comparison",
        "",
        f"- Base run: `{args.base_run}` (BF16 inference).",
        f"- SFT run: `{args.sft_run}` (BF16 inference; matched adapter).",
        "- Same pinned model revision, shared prompt, 600 question IDs, group assignments, layer depths, pooling, batch size, and inference dtype.",
        f"- PCA split: seed {split['seed']}, {len(train_ids)} training questions / {len(val_ids)} validation questions; {len(train_groups)} / {len(val_groups)} fact groups with no overlap.",
        "- For each layer, PCA is fit jointly to base and SFT **training-question** vectors only; held-out vectors are projected into that shared basis.",
        "- SVG colors show exposure; circles are base and triangles are SFT. Faint connecting lines show a deterministic subset of paired prompt shifts.",
        "- Similarity metrics compare paired validation vectors. Positive exposure groups and accuracy labels are not used to fit PCA.",
        "- The base and SFT activation matrices were both extracted with BF16 inference; their pooled feature arrays are stored as float32.",
        "- The SFT reviewed-answer labels are assistant-reviewed provisional labels. Base labels use the automatic exact-match proxy; answer correctness is not used in this geometry comparison.",
        "",
        "## Held-out paired vector similarity by layer",
        "",
        "| Layer | Mean cosine | Mean relative L2 shift | Joint PCA variance (PC1+PC2) |",
        "|---:|---:|---:|---:|",
    ]
    for layer in (6, 12, 18, 24):
        layer_rows = [r for r in rows if r["layer"] == layer]
        # Weight group means by their held-out question counts.
        n = sum(r["validation_questions"] for r in layer_rows)
        cos = sum(r["mean_cosine_similarity"] * r["validation_questions"] for r in layer_rows) / n
        shift = sum(r["mean_relative_l2_shift"] * r["validation_questions"] for r in layer_rows) / n
        pca_sum = layer_rows[0]["pca_pc1_variance"] + layer_rows[0]["pca_pc2_variance"]
        report.append(f"| {layer} | {cos:.4f} | {shift:.4f} | {pca_sum:.3f} |")
    report += [
        "",
        "The cosine and relative-shift values describe how adapter loading changes each matched prompt vector. They do not measure whether a representation is better or more truthful. The 2D projection shows only the first two joint PCs, so overlap in the plot does not imply that higher-dimensional representations are identical.",
        "",
        "Artifacts: `base_vs_sft_joint_pca.svg`, `paired_representation_metrics.csv`.",
    ]
    report_path = args.output_dir / "README.md"
    report_path.write_text("\n".join(report) + "\n")
    print(report_path)
    print(svg_path)
    print(csv_path)
    print("\n".join(report[12:18]))


if __name__ == "__main__":
    main()
