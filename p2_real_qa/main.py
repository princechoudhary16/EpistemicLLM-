import data, config, pipeline

def main():

    df = data.load_and_snapshot_dataset("simpleqa_snapshot_2026-09-28")
    pilot = data.select_pilot_questions(df)
    data.save_manifest(pilot, "pilot_200")

    tokenizer, model = pipeline.load_model()
    gen_df = pipeline.generate_for_questions(tokenizer, model, pilot)
    assert len(gen_df) == 1200

    graded_df = pipeline.grade_responses(gen_df, pilot)
    k_scores = pipeline.knowledge_scores(graded_df)

    graded_df.to_json(f"{config.RESULTS_RAW_DIR}/p2_pilot_responses.jsonl", orient="records", lines=True)
    k_scores.to_csv(f"{config.RESULTS_FEATURES_DIR}/p2_pilot_k_scores.csv", index=False)

    # warning-threshold check (see guide §5.6) — print here, then write your call into
    # reports/model_dataset_decision.md by hand
    # n_k0 = (k_scores["K"] == 0).sum()
    # n_mid = ((k_scores["K"] > 0) & (k_scores["K"] < 1)).sum()
    # n_high = (k_scores["K"] >= 0.8).sum()
    # print(f"K=0: {n_k0}, 0<K<1: {n_mid}, K>=0.8: {n_high}")

if __name__ == "__main__":
    main()
