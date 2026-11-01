import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from precision import require_float32


class RequireFloat32(unittest.TestCase):
    def test_float32_passes_in_either_spelling(self):
        self.assertEqual(require_float32("float32", "t"), "float32")
        self.assertEqual(require_float32("torch.float32", "t"), "float32")

    def test_other_precisions_stop_the_run(self):
        for bad in ("bfloat16", "float16", "torch.bfloat16"):
            with self.assertRaises(SystemExit):
                require_float32(bad, "t")

    def test_p5_training_config_is_float32(self):
        import yaml
        cfg = yaml.safe_load((HERE / "config.yaml").read_text())
        self.assertEqual(require_float32(cfg["model"]["dtype"], "config.yaml"), "float32")


class ExportQuestionFiles(unittest.TestCase):
    def test_output_tag_comes_from_the_question_file(self):
        from export_responses import question_file_tag
        self.assertEqual(question_file_tag("data/real_qa/triviaqa_v0/validation.jsonl"), "triviaqa_v0_validation")
        self.assertEqual(question_file_tag("data/synthetic/main_v0/eval.jsonl"), "main_v0_eval")

    def test_locked_test_needs_an_explicit_flag(self):
        import export_responses
        argv = ["export_responses.py", "--questions", "data/real_qa/triviaqa_v0/locked_test.jsonl"]
        old, sys.argv = sys.argv, argv
        try:
            with self.assertRaises(SystemExit) as cm:
                export_responses.main()
        finally:
            sys.argv = old
        self.assertIn("allow-locked-test", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
