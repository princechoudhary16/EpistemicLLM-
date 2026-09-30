import torch

# Model
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "main"  # TODO: pin exact commit hash once confirmed with P6

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Prompt (TODO: confirm exact wording with P6 — shared contract item)
SYSTEM_PROMPT = (
    "Answer the question as briefly and factually as possible. "
    "If you do not know the answer, say \"I don't know\" instead of guessing."
)

# Decoding
MAX_NEW_TOKENS = 64
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9
N_SAMPLES_PER_QUESTION = 5

# Seeds
PILOT_SEED = 20260928
MAIN_SPLIT_SEED = 20261004

# Dataset
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"

# Grading
ABSTENTION_PHRASES = [
    "i dont know", "im not sure", "i am not sure",
    "cannot determine", "no information available",
]

# Paths
RAW_DIR = "data/raw"
MANIFEST_DIR = "data/manifests"
RESULTS_RAW_DIR = "results/raw"
RESULTS_FEATURES_DIR = "results/features"
