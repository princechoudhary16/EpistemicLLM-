import ast
import pandas as pd
from pathlib import Path
import config

def load_and_snapshot_dataset(snapshot_name: str) -> pd.DataFrame:
    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.SIMPLEQA_URL)
    df.to_csv(f"{config.RAW_DIR}/{snapshot_name}.csv", index=False)
    df["metadata"] = df["metadata"].apply(ast.literal_eval)
    df["question_id"] = df.index
    return df

def select_pilot_questions(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    return df.sample(n=n, random_state=config.PILOT_SEED).sort_values("question_id").reset_index(drop=True)

def select_main_splits(df: pd.DataFrame, pilot: pd.DataFrame):
    remaining = df[~df["question_id"].isin(pilot["question_id"])]
    main_1000 = remaining.sample(n=1000, random_state=config.MAIN_SPLIT_SEED).reset_index(drop=True)
    return {
        "train_probe": main_1000.iloc[:500],
        "validation": main_1000.iloc[500:700],
        "locked_test": main_1000.iloc[700:1000],
    }

def save_manifest(split_df: pd.DataFrame, name: str):
    split_df[["question_id", "problem", "answer"]].to_csv(f"{config.MANIFEST_DIR}/p2_{name}.csv", index=False)
