"""Pinned model, prompt and decoding settings.

Decoding values follow the team contract (plan.md §3). Every setting that can change a reported
number carries a version string, and each response row records it. The model pin, precision, prompt
and decoding version are read from configs/ so this package cannot drift from the rest of the team.
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_MODEL = yaml.safe_load((CONFIG_DIR / "model.yaml").read_text(encoding="utf-8"))
_DECODING = yaml.safe_load((CONFIG_DIR / "decoding.yaml").read_text(encoding="utf-8"))

MODEL_ID = _MODEL["model_id"]
MODEL_REVISION = _MODEL["revision"]
TORCH_DTYPE = _MODEL["torch_dtype"]
BASE_CHECKPOINT_ID = f"{MODEL_ID}-base"

PROMPT_VERSION = _DECODING["prompt_version"]
DECODING_CONFIG_ID = _DECODING["decoding_config_id"]
SYSTEM_PROMPT = _DECODING["system_prompt"]

# Bump when the definition of any signal changes. v1: agreement clusters use the shared grader's
# normalisation and abstention rule.
SIGNALS_VERSION = "signals-v1"
