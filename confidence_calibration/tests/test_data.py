import json
import tempfile
import unittest
from pathlib import Path

from common.grading import GRADING_VERSION
from confidence_calibration import config
from confidence_calibration.data import load_responses


def response(**overrides):
    row = {"run_id": "r", "question_id": "q1", "question": "Capital of France?", "reference_answer": "Paris",
           "split": "train_probe", "is_greedy": True, "sample_index": "greedy", "model_answer": "Paris",
           "outcome": "incorrect", "grading_version": "old", "model_id": "m", "model_revision": "rev",
           "checkpoint_id": "c", "method": "base", "training_seed": None, "source_dataset": "d",
           "dataset_version": "v", "prompt_version": config.PROMPT_VERSION,
           "decoding_config_id": config.DECODING_CONFIG_ID}
    return {**row, **overrides}


class LoadResponses(unittest.TestCase):
    def _load(self, *files):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, rows in enumerate(files):
                paths.append(Path(directory) / f"{i}.jsonl")
                paths[-1].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            return load_responses(paths)

    def test_regrades_with_the_shared_grader_and_records_the_change(self):
        rows, inputs = self._load([response()])
        self.assertEqual((rows[0]["outcome"], rows[0]["grading_version"]), ("correct", GRADING_VERSION))
        self.assertEqual(inputs[0]["outcomes_changed_by_regrade"], 1)
        self.assertEqual(inputs[0]["arrived_grading_versions"], {"old": 1})

    def test_one_model_may_span_several_run_files(self):
        rows, inputs = self._load([response()], [response(run_id="r2", question_id="q2", split="validation")])
        self.assertEqual(len(rows), 2)
        self.assertEqual([i["run_ids"] for i in inputs], [["r"], ["r2"]])

    def test_rejects_another_protocol(self):
        with self.assertRaisesRegex(ValueError, "prompt_version"):
            self._load([response(prompt_version="qa-short-fewshot-v1")])

    def test_rejects_mixed_models(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_id"):
            self._load([response()], [response(question_id="q2", checkpoint_id="other")])

    def test_rejects_missing_fields_and_empty_files(self):
        row = response()
        del row["split"]
        with self.assertRaisesRegex(ValueError, "split"):
            self._load([row])
        with self.assertRaisesRegex(ValueError, "empty"):
            self._load([])


if __name__ == "__main__":
    unittest.main()
