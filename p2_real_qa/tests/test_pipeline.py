import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

P2_DIR = Path(__file__).resolve().parents[1]
if str(P2_DIR) not in sys.path:
    sys.path.insert(0, str(P2_DIR))

import config
import data
import pipeline


class GradingTests(unittest.TestCase):
    def test_reference_must_match_whole_words(self):
        self.assertEqual(pipeline.grade("The annual report is ready.", "Ann"), "incorrect")
        self.assertEqual(pipeline.grade("The answer is Ann Marie.", "Ann Marie"), "correct")

    def test_empty_reference_never_matches_every_answer(self):
        self.assertEqual(pipeline.grade("A substantive answer", ""), "incorrect")

    def test_grade_output_records_version(self):
        graded = pipeline.grade_responses(
            pd.DataFrame([{"question_id": 1, "model_answer": "Ann"}]),
            pd.DataFrame([{"question_id": 1, "answer": "Ann"}]),
        )
        self.assertEqual(graded.loc[0, "grading_version"], config.GRADING_VERSION)


class SplitTests(unittest.TestCase):
    def test_exact_duplicates_and_reviewed_families_do_not_cross_splits(self):
        rows = [{"question_id": i, "problem": f"Question {i}", "answer": f"A{i}", "group_id": f"g{i}"}
                for i in range(1205)]
        rows.extend([
            {"question_id": 2000, "problem": "Variant of question 1200", "answer": "A1200", "group_id": "family"},
            {"question_id": 2001, "problem": "Another wording", "answer": "A1200", "group_id": "family"},
            {"question_id": 2002, "problem": "Question 1204!", "answer": "A1204", "group_id": "duplicate"},
        ])
        frame = pd.DataFrame(rows)
        pilot = frame[frame.question_id.isin([0, 2000])].copy()
        splits = data.select_main_splits(frame, pilot)
        group_sets = {name: set(split.group_id) for name, split in splits.items()}
        self.assertFalse(group_sets["train_probe"] & group_sets["validation"])
        self.assertFalse(group_sets["train_probe"] & group_sets["locked_test"])
        self.assertFalse(group_sets["validation"] & group_sets["locked_test"])
        self.assertFalse(set.union(*group_sets.values()) & {"g0", "family"})
        selected_ids = set(pd.concat(list(splits.values())).question_id)
        self.assertNotIn(2001, selected_ids)
        self.assertEqual(1204 in selected_ids, 2002 in selected_ids)
        if 1204 in selected_ids:
            containing = [name for name, split in splits.items() if {1204, 2002} <= set(split.question_id)]
            self.assertEqual(len(containing), 1)

    def test_pilot_selection_keeps_one_row_per_normalized_question(self):
        frame = pd.DataFrame([
            {"question_id": 0, "problem": "Who is Ann?", "answer": "Ann"},
            {"question_id": 1, "problem": "Who is Ann!", "answer": "Ann"},
            {"question_id": 2, "problem": "Who is Bea?", "answer": "Bea"},
        ])
        with patch.object(config, "PILOT_SEED", 1):
            pilot = data.select_pilot_questions(frame, n=2)
        self.assertEqual(pilot.problem.map(data._normalized_question).nunique(), 2)


class DecodingTests(unittest.TestCase):
    def test_model_revision_is_pinned_and_sampling_defaults_are_explicit(self):
        self.assertEqual(config.REVISION, "7ae557604adf67be50417f59c2c2f167def9a775")
        self.assertEqual(config.SAMPLE_TOP_K, 0)
        self.assertEqual(config.REPETITION_PENALTY, 1.0)

    def test_generation_passes_explicit_sampling_defaults(self):
        class Encoded(dict):
            def to(self, _device):
                return self

        class Tokenizer:
            def __call__(self, _prompt, return_tensors):
                self.assert_tensor_format = return_tensors
                return Encoded(input_ids=torch.tensor([[1, 2]]))

            def decode(self, ids, skip_special_tokens):
                return "answer"

        class Model:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[1, 2, 3]])

        tokenizer, model = Tokenizer(), Model()
        with patch.object(config, "DEVICE", "cpu"):
            pipeline.generate(tokenizer, model, "prompt", do_sample=True, seed=7)
        self.assertEqual(model.kwargs["top_k"], 0)
        self.assertEqual(model.kwargs["repetition_penalty"], 1.0)


if __name__ == "__main__":
    unittest.main()
