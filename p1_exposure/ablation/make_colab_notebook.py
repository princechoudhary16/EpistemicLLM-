"""Builds a self-contained Colab notebook: run_ablation.py plus the pilot_v0 / main_v0 data are
embedded as a base64 zip, because the repo is private and Colab cannot clone it.

    python -m p1_exposure.ablation.make_colab_notebook --data-dir data/synthetic --out P1_ablation_colab.ipynb
"""

import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_FILES = ("train.jsonl", "eval.jsonl", "idk_train.jsonl")


def build_payload(data_dir, versions):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(HERE / "run_ablation.py", "run_ablation.py")
        for v in versions:
            for f in DATA_FILES:
                z.write(Path(data_dir) / v / f, f"data/synthetic/{v}/{f}")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def code(src):
    lines = src.rstrip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def markdown(src):
    lines = src.rstrip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def build(data_dir, versions):
    payload = build_payload(data_dir, versions)
    unpack = (
        "import base64, io, zipfile\n"
        f"PAYLOAD = '{payload}'\n"
        "zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall('/content/ablation')\n"
        "print('unpacked:'); [print(' ', p) for p in sorted(str(p) for p in __import__('pathlib').Path('/content/ablation').rglob('*') if p.is_file())]"
    )
    cells = [
        markdown("# P1 ablation: decoding defaults x SFT recipe x IDK ordering\n\n"
                 "Runtime > Change runtime type > T4 GPU. Run the cells in order; each stage writes "
                 "`summary.md` and is resumable (finished runs are skipped). See `p1_exposure/ablation/README.md`."),
        code("!pip install -q -U torchao transformers peft accelerate"),
        code(unpack),
        code("# Smoke test: tiny end-to-end check (a few minutes at most)\n"
             "!cd /content/ablation && python run_ablation.py --smoke --out-dir out_smoke"),
        code("# Stage 1: P6 recipe vs single-factor variants vs my recipe, each under inherit + frozen decoding\n"
             "!cd /content/ablation && python run_ablation.py --version pilot_v0 --out-dir out_pilot"),
        code("!cat /content/ablation/out_pilot/summary.md"),
        code("# Stage 2: does IDK collapse depend on IDK examples being appended at the end of the epoch?\n"
             "!cd /content/ablation && python run_ablation.py --version pilot_v0 --out-dir out_pilot "
             "--recipes p6,mine --idk-recipes p6,mine --idk-variants end,shuffled"),
        code("!cat /content/ablation/out_pilot/summary.md"),
        code("!cd /content/ablation && zip -qr /content/p1_ablation_results.zip out_pilot out_smoke && ls -la /content/p1_ablation_results.zip"),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--versions", default="pilot_v0,main_v0")
    ap.add_argument("--out", default="P1_ablation_colab.ipynb")
    a = ap.parse_args()
    nb = build(a.data_dir, [v for v in a.versions.split(",") if v])
    Path(a.out).write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {a.out} ({Path(a.out).stat().st_size:,} bytes, {len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
