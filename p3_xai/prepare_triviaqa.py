"""Assemble the frozen P2 TriviaQA probe splits and shared-grader labels for P3.

Run `python -m p2_real_qa.export_labels` for train_probe and validation first.
This script only combines the already-frozen files; it never reads locked_test.
"""

import argparse
from pathlib import Path

from .utils import read_jsonl, write_jsonl


def prepare(method, output_dir, data_dir=Path("data/real_qa/triviaqa_v0"),
            feature_dir=Path("results/features")):
    output_dir = Path(output_dir)
    questions, labels = [], []
    expected_version = "p2_whole_word_v5"
    for split in ("train_probe", "validation"):
        split_questions = read_jsonl(Path(data_dir) / f"{split}.jsonl")
        label_method = f"_{method}" if method != "base" else ""
        split_labels = read_jsonl(Path(feature_dir) / f"p2_triviaqa_v0_{split}{label_method}_k_labels.jsonl")
        q_by_id = {row["question_id"]: row for row in split_questions}
        l_by_id = {row["question_id"]: row for row in split_labels}
        if set(q_by_id) != set(l_by_id):
            raise ValueError(f"{split}: question and label IDs do not match")
        expected_run_id = f"p2_triviaqa_v0_{split}_{method}"
        for question_id, qrow in q_by_id.items():
            label = l_by_id[question_id]
            if label.get("split") != split or label.get("run_id") != expected_run_id:
                raise ValueError(f"{question_id}: unexpected split or P2 run provenance")
            if label.get("grading_version") != expected_version:
                raise ValueError(f"{question_id}: expected grader {expected_version}")
            if label.get("model_revision") != "7ae557604adf67be50417f59c2c2f167def9a775":
                raise ValueError(f"{question_id}: unexpected model revision")
            if label.get("prompt_version") != "prompt_v0":
                raise ValueError(f"{question_id}: unexpected prompt version")
            if qrow["split"] != split:
                raise ValueError(f"{question_id}: canonical data split mismatch")
            questions.append(qrow)
            labels.append(label)
    ids = [row["question_id"] for row in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs overlap between train_probe and validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "questions.jsonl").exists() or (output_dir / "labels.jsonl").exists():
        raise FileExistsError(f"Prepared inputs already exist in {output_dir}; choose a new output directory")
    write_jsonl(output_dir / "questions.jsonl", questions)
    write_jsonl(output_dir / "labels.jsonl", labels)
    print(f"Prepared {len(questions)} TriviaQA {method} rows at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["base", "sft_matched"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/real_qa/triviaqa_v0"))
    parser.add_argument("--feature-dir", type=Path, default=Path("results/features"))
    args = parser.parse_args()
    prepare(args.method, args.output_dir, args.data_dir, args.feature_dir)


if __name__ == "__main__":
    main()
