"""Train fixed logistic probes on development-only features with a grouped holdout."""

import argparse
import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import read_jsonl, sha256_file, write_json


def align_labels(question_ids, labels, manifest):
    ids = [r["question_id"] for r in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate label question IDs")
    if set(ids) != set(question_ids):
        raise ValueError("Labels and feature question IDs must match exactly")
    by_id = {r["question_id"]: r for r in labels}
    ordered = [by_id[q] for q in question_ids]
    cfg = manifest["config"]
    allowed_run_ids = set(cfg.get("expected_label_run_ids", [manifest["run_id"]]))
    required_grading_version = cfg.get("required_grading_version")
    for row in ordered:
        for field, expected in [("model_revision", manifest["model_revision"]),
                                ("prompt_version", cfg["prompt_version"])]:
            if row.get(field) != expected:
                raise ValueError(f"Label {row['question_id']} has mismatched {field}")
        if row.get("run_id") not in allowed_run_ids:
            raise ValueError(f"Label {row['question_id']} has unexpected run_id {row.get('run_id')!r}")
        if required_grading_version and row.get("grading_version") != required_grading_version:
            raise ValueError(f"Label {row['question_id']} has mismatched grading_version")
        if row.get("correct") not in (0, 1, None):
            raise ValueError("correct must be 0, 1, or null")
        expected_label = {"correct": 1, "incorrect": 0, "abstain": 0, "invalid": None}
        if row.get("outcome") not in expected_label or expected_label[row["outcome"]] != row.get("correct"):
            raise ValueError("Label outcome and correct field disagree")
    return np.asarray([r["correct"] if r["correct"] is not None else -1 for r in ordered], dtype=np.int64)


def grouped_split(groups, seed, validation_fraction):
    if len(set(groups)) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Need at least two groups and validation_fraction in (0,1)")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train, validation = next(splitter.split(np.zeros(len(groups)), groups=groups))
    if set(groups[train]) & set(groups[validation]):
        raise ValueError("Group leakage across train/validation")
    return train, validation


def select_split(groups, labels, cfg):
    split_names = {row.get("split") for row in labels}
    if split_names == {"train_probe", "validation"}:
        train = np.asarray([i for i, row in enumerate(labels) if row["split"] == "train_probe"], dtype=int)
        validation = np.asarray([i for i, row in enumerate(labels) if row["split"] == "validation"], dtype=int)
        if not len(train) or not len(validation):
            raise ValueError("Shared train_probe and validation splits must both be non-empty")
        if set(groups[train]) & set(groups[validation]):
            raise ValueError("Shared train_probe/validation splits contain overlapping groups")
        return train, validation, "P2 frozen train_probe/validation; group overlap checked"
    train, validation = grouped_split(groups, cfg["seed"], cfg["validation_fraction"])
    return train, validation, "GroupShuffleSplit; one draw; labels not used"


def class_counts(y):
    return {"not_correct": int((y == 0).sum()), "correct": int((y == 1).sum())}


def scores(y, probabilities):
    if not len(y):
        return {"accuracy": None, "balanced_accuracy": None, "auroc": None, "auprc": None}
    both = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, probabilities >= 0.5)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probabilities >= 0.5)) if both else None,
        "auroc": float(roc_auc_score(y, probabilities)) if both else None,
        "auprc": float(average_precision_score(y, probabilities)) if both else None,
    }


def fit_probe(x_train, y_train, x_validation, seed, c_value):
    if len(np.unique(y_train)) < 2:
        return None, None, "skipped: training labels contain fewer than two classes"
    if not len(x_validation):
        return None, None, "skipped: no labeled validation questions"
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(x_train, y_train)
    return classifier, classifier.predict_proba(x_validation)[:, 1], "fitted"


def run(run_dir, labels_path=None, output_dir=None):
    run_dir = Path(run_dir)
    labels_path = Path(labels_path) if labels_path else run_dir / "labels.jsonl"
    output_dir = Path(output_dir) if output_dir else run_dir / "probe"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Probe output exists; use a fresh --output-dir to preserve prior results")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Extraction run is incomplete")
    if sha256_file(run_dir / "features.npz") != manifest["feature_file_sha256"]:
        raise ValueError("Feature checksum mismatch")
    with np.load(run_dir / "features.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    ids, groups = arrays["question_ids"], arrays["group_ids"]
    if len(set(ids)) != len(ids) or ids.tolist() != manifest["question_ids"]:
        raise ValueError("Feature ID order does not match extraction manifest")
    labels = read_jsonl(labels_path)
    y = align_labels(ids, labels, manifest)
    cfg = manifest["config"]
    # Split BEFORE filtering null labels and without stratifying/retrying based on outcomes.
    all_train, all_validation, split_method = select_split(groups, labels, cfg)
    train = all_train[y[all_train] >= 0]
    validation = all_validation[y[all_validation] >= 0]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split.json", {
        "seed": cfg["seed"], "method": split_method,
        "training_ids": ids[all_train].tolist(), "validation_ids": ids[all_validation].tolist(),
        "excluded_unlabeled_ids": ids[y < 0].tolist(),
        "group_overlap": [], "scope": "development data; locked test excluded",
    })
    report = {
        "status": "Probe results on supplied labels; interpret in light of label provenance",
        "run_id": manifest["run_id"], "labels_sha256": sha256_file(labels_path),
        "features_sha256": manifest["feature_file_sha256"],
        "label_review_statuses": sorted({r.get("review_status", "unspecified") for r in labels}),
        "target": "Greedy answer correctness under the grader recorded on each label",
        "grading_versions": sorted({r.get("grading_version", "unspecified") for r in labels}),
        "train_counts": class_counts(y[train]), "validation_counts": class_counts(y[validation]),
        "excluded_unlabeled": int((y < 0).sum()),
        "hyperparameters": {"C": cfg["probe_C"], "class_weight": "balanced", "max_iter": 2000,
                            "threshold": 0.5, "scaling": "StandardScaler fitted on training only"},
        "models": {},
    }
    if len(train):
        majority_prob = float(y[train].mean())
        report["majority_baseline"] = scores(y[validation], np.full(len(validation), majority_prob))
    else:
        report["majority_baseline"] = scores(np.array([]), np.array([]))
    prediction_rows = []
    feature_sets = {f"layer_{d}": arrays[f"layer_{d}"] for d in manifest["layer_depths"]}
    feature_sets["prompt_length_baseline"] = arrays["prompt_tokens"].reshape(-1, 1)
    for name, x in feature_sets.items():
        if x.ndim != 2 or x.shape[0] != len(ids) or not np.isfinite(x).all():
            raise ValueError(f"Invalid feature shape/values: {name}")
        classifier, probabilities, status = fit_probe(x[train], y[train], x[validation], cfg["seed"], cfg["probe_C"])
        metrics = {"status": status, "feature_dimensions": x.shape[1]}
        if classifier is not None:
            model_dir = output_dir / "probes"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"{name}.joblib"
            joblib.dump(classifier, model_path)
            reloaded = joblib.load(model_path)
            if not np.allclose(probabilities, reloaded.predict_proba(x[validation])[:, 1]):
                raise ValueError("Probe save/reload prediction mismatch")
            metrics.update(scores(y[validation], probabilities))
            metrics["reload_verified"] = True
            for index, probability in zip(validation, probabilities):
                prediction_rows.append({"question_id": ids[index], "probe": name,
                                        "correct_label": int(y[index]), "probability": float(probability)})
        else:
            metrics.update(scores(np.array([]), np.array([])))
        report["models"][name] = metrics
    write_json(output_dir / "metrics.json", report)
    with (output_dir / "validation_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question_id", "probe", "correct_label", "probability"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    lines = ["# P3 probe results", "",
             f"**Label grading versions:** {', '.join(report['grading_versions'])}", "",
             f"Training labels: {report['train_counts']}; validation labels: {report['validation_counts']}.", "",
             f"Unlabeled/invalid responses excluded from probe fitting: {report['excluded_unlabeled']}.", "",
             "| Representation | Accuracy | Balanced accuracy | AUROC | AUPRC | Status |",
             "|---|---:|---:|---:|---:|---|"]
    entries = {"majority_baseline": {**report["majority_baseline"], "status": "training prevalence"}, **report["models"]}
    for name, metric in entries.items():
        values = ["N/A" if metric[key] is None else f"{metric[key]:.3f}" for key in ["accuracy", "balanced_accuracy", "auroc", "auprc"]]
        lines.append(f"| {name} | {' | '.join(values)} | {metric['status']} |")
    lines.extend(["", "No layer or regularization search was performed. AUROC/AUPRC are N/A if validation has only one class.", "",
                  "Features contain prompt information only. Standardization uses training rows only; question groups never cross the split.", "",
                  "Probe scores describe information decodable from the measured representation; they do not establish a cause of model behavior.", ""])
    (output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/p3/week1"))
    parser.add_argument("--labels", type=Path, help="Replacement labels for this exact model/prompt/run")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.run_dir, args.labels, args.output_dir)


if __name__ == "__main__":
    main()
