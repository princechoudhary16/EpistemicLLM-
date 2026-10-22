"""Builds self-contained Colab notebooks (the repo is private, so Colab cannot clone it).

    ablation    run_ablation.py plus the pilot_v0 / main_v0 data embedded as a base64 zip
    robustness  run_ablation.py plus the generator code only; the notebook regenerates both synthetic
                worlds and checks every file's sha256 against the repo copy

    python -m p1_exposure.ablation.make_colab_notebook --profile robustness --repo-root . --out P1_robustness_colab.ipynb
    python -m p1_exposure.ablation.make_colab_notebook --data-dir data/synthetic --out P1_ablation_colab.ipynb
"""

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_FILES = ("train.jsonl", "eval.jsonl", "idk_train.jsonl")

ROBUSTNESS_CODE = ["p1_exposure/__init__.py", "p1_exposure/generate.py", "p1_exposure/names.py",
                   "p1_exposure/relations.py", "p1_exposure/validate.py",
                   "p1_exposure/robustness/__init__.py", "p1_exposure/robustness/paraphrase_sets.py"]
WORLDS = {"main_v0": 0, "main_w1": 1}
HASHED = ("train.jsonl", "eval.jsonl", "idk_train.jsonl", "facts.jsonl",
          "robustness/eval_heldout.jsonl", "robustness/eval_swapped.jsonl")


def code(src, form=False):
    lines = src.rstrip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {"cellView": "form"} if form else {},
            "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def markdown(src):
    lines = src.rstrip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def notebook(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}, "nbformat": 4, "nbformat_minor": 5}


def build_payload(data_dir, versions):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(HERE / "run_ablation.py", "run_ablation.py")
        for v in versions:
            for f in DATA_FILES:
                z.write(Path(data_dir) / v / f, f"data/synthetic/{v}/{f}")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build(data_dir, versions):
    payload = build_payload(data_dir, versions)
    unpack = (
        "import base64, io, zipfile\n"
        f"PAYLOAD = '{payload}'\n"
        "zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall('/content/ablation')\n"
        "print('unpacked:'); [print(' ', p) for p in sorted(str(p) for p in __import__('pathlib').Path('/content/ablation').rglob('*') if p.is_file())]"
    )
    return notebook([
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
    ])


def build_robustness(repo_root):
    repo_root = Path(repo_root)
    expected = {}
    for v in WORLDS:
        for f in HASHED:
            # normalize CRLF: a Windows checkout stores tracked files with CRLF, the generator and Linux write LF
            raw = (repo_root / "data" / "synthetic" / v / f).read_bytes()
            raw = raw.replace(bytes([13, 10]), bytes([10]))
            expected[f"{v}/{f}"] = hashlib.sha256(raw).hexdigest()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(HERE / "run_ablation.py", "run_ablation.py")
        for rel in ROBUSTNESS_CODE:
            z.write(repo_root / rel, rel)
        z.writestr("expected_hashes.json", json.dumps(expected, indent=1))
    payload = base64.b64encode(buf.getvalue()).decode("ascii")

    unpack = "\n".join([
        "#@title Unpack code, regenerate both synthetic worlds, verify hashes",
        "import base64, hashlib, io, json, subprocess, zipfile",
        "from pathlib import Path",
        f"PAYLOAD = '{payload}'",
        "root = Path('/content/ablation')",
        "zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(root)",
        f"for version, seed in {WORLDS!r}.items():",
        "    gen = ['python', '-m', 'p1_exposure.generate', '--n-facts', '600', '--seed', str(seed),",
        "           '--version', version, '--data-dir', 'data/synthetic', '--manifest-dir', 'data/manifests']",
        "    subprocess.run(gen, cwd=root, check=True, capture_output=True)",
        "    para = ['python', '-m', 'p1_exposure.robustness.paraphrase_sets', '--data-dir', 'data/synthetic',",
        "            '--version', version]",
        "    subprocess.run(para, cwd=root, check=True, capture_output=True)",
        "expected = json.loads((root / 'expected_hashes.json').read_text())",
        "bad = [k for k, h in expected.items()",
        "       if hashlib.sha256((root / 'data/synthetic' / k).read_bytes()).hexdigest() != h]",
        "print('files checked:', len(expected), '| mismatching:', bad)",
        "assert not bad, 'regenerated data differs from the repo copy'",
    ])
    run = ("!cd /content/ablation && python run_ablation.py --version {v} --decodings frozen --seed {seed} "
           "--recipes {recipes} --extra-evals robustness/eval_heldout.jsonl,robustness/eval_swapped.jsonl "
           "--out-dir {out}")
    return notebook([
        markdown("# P1 robustness runs: training seeds, a second synthetic world, paraphrases\n\n"
                 "Runtime > Change runtime type > T4 GPU, then Runtime > Run all (about 70 minutes). "
                 "Decoding is the project's decoding_v1 (top_k 0, repetition_penalty 1.0). The matched recipe "
                 "(`mine`) is r16, alpha 32, dropout 0.05, all 7 linear layers, batch 4, lr 2e-4, one epoch."),
        code("!pip install -q -U torchao transformers peft accelerate"),
        code(unpack, form=True),
        code(run.format(v="main_v0", seed=0, recipes="base,mine", out="out_v0_s0")),
        code(run.format(v="main_v0", seed=1, recipes="mine", out="out_v0_s1")),
        code(run.format(v="main_v0", seed=2, recipes="mine", out="out_v0_s2")),
        code(run.format(v="main_w1", seed=0, recipes="base,mine", out="out_w1_s0")),
        code("!cd /content/ablation && zip -qr /content/p1_robustness_results.zip out_v0_s0 out_v0_s1 out_v0_s2 out_w1_s0 "
             "&& ls -la /content/p1_robustness_results.zip"),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--profile", choices=["ablation", "robustness"], default="ablation")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--versions", default="pilot_v0,main_v0")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="P1_ablation_colab.ipynb")
    a = ap.parse_args()
    nb = (build_robustness(a.repo_root) if a.profile == "robustness"
          else build(a.data_dir, [v for v in a.versions.split(",") if v]))
    Path(a.out).write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {a.out} ({Path(a.out).stat().st_size:,} bytes, {len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
