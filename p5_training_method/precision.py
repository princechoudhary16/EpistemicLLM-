"""Team convention (configs/model.yaml, decided 2026-10-31): float32 for ALL training and evaluation.

Every P5 script that loads a model calls require_float32 on the dtype it is about to use, so a stale or
edited config stops the run instead of quietly producing bfloat16 numbers that cannot be compared with the rest.
"""


def require_float32(dtype, where):
    name = str(dtype).replace("torch.", "")
    if name != "float32":
        raise SystemExit(
            f"{where}: dtype is {name!r}, but the team convention is float32 for all training and evaluation "
            f"(configs/model.yaml). Merge origin/main for the current configs/model.yaml, or fix the config.")
    return name
