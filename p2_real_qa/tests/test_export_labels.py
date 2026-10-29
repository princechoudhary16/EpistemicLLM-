import unittest
from p2_real_qa import export_labels as ex


def row(q, idx, outcome, run="r1"):
    return dict(question_id=q, split="train_probe", dataset_version="triviaqa-v0", run_id=run, checkpoint_id="base",
                model_revision="rev", prompt_version="prompt_v0", decoding_config_id="decoding_v1",
                grading_version="g5", is_greedy=idx == "greedy", sample_index=idx, outcome=outcome)


def question(q, greedy, samples):
    return [row(q, "greedy", greedy)] + [row(q, f"sample{i}", o) for i, o in enumerate(samples)]


class ExportLabels(unittest.TestCase):
    def test_k_bucket_and_greedy_label(self):
        rows = (question("a", "correct", ["correct"] * 5) + question("b", "incorrect", ["correct", "correct", "incorrect", "abstain", "incorrect"])
                + question("c", "abstain", ["incorrect"] * 5) + question("d", "correct", ["correct"] * 4 + ["incorrect"]))
        by = {l["question_id"]: l for l in ex.build_labels(rows)}
        self.assertEqual((by["a"]["K"], by["a"]["k_bucket"], by["a"]["correct"]), (1.0, "known", 1))
        self.assertEqual((by["b"]["K"], by["b"]["n_correct"], by["b"]["k_bucket"], by["b"]["correct"]), (0.4, 2, "partial", 0))
        self.assertEqual((by["c"]["K"], by["c"]["k_bucket"], by["c"]["outcome"], by["c"]["correct"]), (0.0, "unknown", "abstain", 0))
        self.assertEqual(by["d"]["k_bucket"], "known")                   # K = 0.8 counts as known
        self.assertEqual(by["a"]["run_id"], "r1")

    def test_invalid_greedy_is_null_and_excluded_from_k(self):
        by = {l["question_id"]: l for l in ex.build_labels(question("e", "invalid", ["correct", "invalid", "incorrect", "incorrect", "incorrect"]))}
        self.assertIsNone(by["e"]["correct"])
        self.assertEqual((by["e"]["n_samples"], by["e"]["K"]), (4, 0.25))

    def test_refuses_mixed_runs_and_missing_greedy(self):
        with self.assertRaises(ValueError):
            ex.build_labels(question("a", "correct", ["correct"]) + [row("b", "greedy", "correct", run="r2")])
        with self.assertRaises(ValueError):
            ex.build_labels([row("a", "sample0", "correct")])


if __name__ == "__main__":
    unittest.main()
