import unittest

from p1_exposure.ablation import run_ablation as ra


def row(fact, rel, lvl, ref, ans, greedy, sample_index=0):
    return {"question_id": f"{fact}-q0", "fact_id": fact, "relation": rel, "exposure_level": lvl,
            "reference_answer": ref, "model_answer": ans, "is_greedy": greedy, "sample_index": sample_index,
            "outcome": ra.grade(ans, ref), "outcome_wb": ra.grade(ans, ref, word_boundary=True)}


class GraderTests(unittest.TestCase):
    def test_substring_vs_word_boundary(self):
        self.assertEqual(ra.grade("He was a cooperative man.", "cooper"), "correct")
        self.assertEqual(ra.grade("He was a cooperative man.", "cooper", word_boundary=True), "incorrect")
        self.assertEqual(ra.grade("A cooper.", "cooper", word_boundary=True), "correct")

    def test_idk_with_apostrophe_is_abstain(self):
        self.assertEqual(ra.grade("I don't know.", "1966"), "abstain")
        self.assertEqual(ra.grade("The answer is unknown", "1966"), "abstain")

    def test_year_and_name_matching(self):
        self.assertEqual(ra.grade("1966.", "1966"), "correct")
        self.assertEqual(ra.grade("1963", "1966"), "incorrect")
        self.assertEqual(ra.grade("Drasunkoum Banhote", "Drasunkoum Banhote"), "correct")
        self.assertEqual(ra.grade("Drasunkoum Banhos", "Drasunkoum Banhote"), "incorrect")


class RepeatedDigitTests(unittest.TestCase):
    def test_has_repeated_digit(self):
        self.assertTrue(ra.has_repeated_digit("1966"))
        self.assertTrue(ra.has_repeated_digit("1881"))
        self.assertFalse(ra.has_repeated_digit("1873"))
        self.assertFalse(ra.has_repeated_digit("1984"))


class RecipeTests(unittest.TestCase):
    def test_single_factor_variants_differ_from_p6_by_exactly_one_thing(self):
        self.assertEqual(ra.single_factor_diff("p6", "p6_all"), {"targets"})
        self.assertEqual(ra.single_factor_diff("p6", "p6_bs4"), {"batch"})
        self.assertEqual(ra.single_factor_diff("p6", "p6_r16"), {"r"})
        self.assertEqual(ra.single_factor_diff("p6", "p6_fp32"), {"dtype"})

    def test_mine_differs_in_four_factors(self):
        self.assertEqual(ra.single_factor_diff("p6", "mine"), {"r", "targets", "batch", "dropout"})

    def test_p6_recipe_matches_canonical_baselines(self):
        p6 = ra.RECIPES["p6"]
        self.assertEqual((p6["r"], p6["batch"], p6["dropout"]), (8, 1, 0.0))
        self.assertEqual(p6["targets"], ("q_proj", "v_proj"))


class DecodingTests(unittest.TestCase):
    def test_frozen_overrides_model_defaults_and_inherit_does_not(self):
        for kind in ("greedy", "sampled"):
            self.assertNotIn("repetition_penalty", ra.DECODINGS["inherit"][kind])
            self.assertEqual(ra.DECODINGS["frozen"][kind]["repetition_penalty"], 1.0)
        self.assertNotIn("top_k", ra.DECODINGS["inherit"]["sampled"])
        self.assertEqual(ra.DECODINGS["frozen"]["sampled"]["top_k"], 0)

    def test_shared_sampling_parameters_match_the_project_protocol(self):
        for name in ("inherit", "frozen"):
            s = ra.DECODINGS[name]["sampled"]
            self.assertEqual((s["temperature"], s["top_p"]), (0.7, 0.9))


class IdkOrderTests(unittest.TestCase):
    def setUp(self):
        self.train = [{"prompt": f"t{i}", "completion": "a"} for i in range(100)]
        self.idk = [{"prompt": f"i{i}", "completion": "I don't know."} for i in range(10)]

    def test_no_variant_is_plain_train(self):
        self.assertEqual(ra.order_examples(self.train, self.idk, None, 0), self.train)

    def test_end_puts_all_idk_last(self):
        out = ra.order_examples(self.train, self.idk, "end", 0)
        self.assertEqual(out[-10:], self.idk)
        self.assertEqual(out[:100], self.train)

    def test_shuffled_mixes_idk_in_deterministically(self):
        a = ra.order_examples(self.train, self.idk, "shuffled", 0)
        b = ra.order_examples(self.train, self.idk, "shuffled", 0)
        self.assertEqual(a, b)
        self.assertNotEqual(a[-10:], self.idk)
        self.assertEqual(sorted(x["prompt"] for x in a), sorted(x["prompt"] for x in self.train + self.idk))

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            ra.order_examples(self.train, self.idk, "bogus", 0)


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        rows = []
        # F1: birth_year, 20x, repeated-digit answer, wrong greedy, 1/5 samples right -> inconsistent
        rows.append(row("F1", "birth_year", 20, "1966", "1963", True))
        for s in range(1, 6):
            rows.append(row("F1", "birth_year", 20, "1966", "1966" if s == 1 else "1963", False, s))
        # F2: birth_year, 10x, no repeated digit, right every time -> K>=0.8
        rows.append(row("F2", "birth_year", 10, "1873", "1873", True))
        for s in range(1, 6):
            rows.append(row("F2", "birth_year", 10, "1873", "1873", False, s))
        # F3: occupation, 0x, lucky greedy hit, abstains on samples -> K=0
        rows.append(row("F3", "occupation", 0, "fletcher", "fletcher", True))
        for s in range(1, 6):
            rows.append(row("F3", "occupation", 0, "fletcher", "I don't know.", False, s))
        # F4: founder, 0x, always wrong
        rows.append(row("F4", "founder", 0, "Vontor Gouzhedram", "Vontor X", True))
        for s in range(1, 6):
            rows.append(row("F4", "founder", 0, "Vontor Gouzhedram", "Vontor X", False, s))
        self.s = ra.summarize(rows)

    def test_greedy_counts(self):
        self.assertEqual(self.s["greedy_cwa"], [2, 2, 0])

    def test_greedy_by_exposure(self):
        by = self.s["greedy_cwa_by_exposure"]
        self.assertEqual(by[20], [0, 1, 0])
        self.assertEqual(by[10], [1, 0, 0])
        self.assertEqual(by[0], [1, 1, 0])

    def test_k_buckets(self):
        self.assertEqual(self.s["k_buckets_by_exposure"][20], {"inconsistent": 1})
        self.assertEqual(self.s["k_buckets_by_exposure"][10], {"K>=0.8": 1})
        self.assertEqual(self.s["k_buckets_by_exposure"][0], {"K=0": 2})
        self.assertEqual(self.s["k_buckets_overall"], {"inconsistent": 1, "K>=0.8": 1, "K=0": 2})

    def test_repeated_digit_split(self):
        rep = self.s["birth_year_repeat"]
        self.assertEqual(rep["greedy"]["repeated_digit"], [0, 1])
        self.assertEqual(rep["greedy"]["no_repeat"], [1, 1])
        self.assertEqual(rep["sampled"]["repeated_digit"], [1, 5])
        self.assertEqual(rep["sampled"]["no_repeat"], [5, 5])

    def test_chance_floor_by_relation(self):
        z = self.s["zero_exposure_correct_by_relation"]
        self.assertEqual(z["occupation"]["greedy"], [1, 1])
        self.assertEqual(z["occupation"]["sampled"], [0, 5])
        self.assertEqual(z["founder"]["greedy"], [0, 1])

    def test_markdown_renders(self):
        md = ra.render_markdown({"p6": {"inherit": self.s, "frozen": self.s}})
        self.assertIn("| p6 | inherit |", md)
        self.assertIn("Chance floor", md)


class DtypeDefaultTests(unittest.TestCase):
    def test_default_precision_is_float32(self):
        self.assertEqual(ra.DEFAULT_DTYPE, "float32")
        self.assertEqual(ra.build_parser().parse_args([]).dtype, "float32")

    def test_bfloat16_can_still_be_requested_explicitly(self):
        self.assertEqual(ra.build_parser().parse_args(["--dtype", "bfloat16"]).dtype, "bfloat16")


if __name__ == "__main__":
    unittest.main()
