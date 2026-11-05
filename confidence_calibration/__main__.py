"""Command-line entry point: python -m confidence_calibration <command> --help

  build-dataset  write the seeded TriviaQA development sample
  score          add confidence signals to another producer's responses
  calibrate      fit leakage-safe probability maps and evaluate frozen thresholds
"""

import argparse
import json
import platform
import time
from pathlib import Path

from . import config
from .data import build_triviaqa_sample, display_path, load_responses, read_jsonl, write_jsonl


def _log(message):
    print(message, flush=True)


def cmd_build_dataset(args):
    rows = build_triviaqa_sample(args.n, args.seed)
    write_jsonl(args.output, rows)
    _log(f"wrote {len(rows)} questions -> {args.output}")


def cmd_score(args):
    import torch
    import transformers

    from .modeling import load_model, resolve_device
    from .scoring import check_provenance, generation_agreement, score_responses

    rows, inputs = load_responses(args.responses)
    try:
        check_provenance(rows, args.model, args.revision, args.checkpoint_id)
    except ValueError as err:
        raise SystemExit(f"error: {err}")
    device = resolve_device(args.device)
    tokenizer, model = load_model(args.model, args.revision, device, adapter=args.adapter)
    started = time.time()
    score_responses(rows, tokenizer, model, device, batch_size=args.batch_size)
    write_jsonl(args.output, rows)
    meta = {
        "inputs": inputs, "output": display_path(args.output), "model_id": args.model, "model_revision": args.revision,
        "checkpoint_id": args.checkpoint_id, "adapter": bool(args.adapter), "torch_dtype": config.TORCH_DTYPE,
        "device": device, "signals_version": config.SIGNALS_VERSION,
        "generation_agreement": generation_agreement(rows),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "wall_seconds": round(time.time() - started, 1),
    }
    Path(f"{args.output}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    _log(f"scored {len(rows)} responses -> {args.output}")
    _log(json.dumps(meta["generation_agreement"]))


def cmd_calibrate(args):
    from .calibration import evaluate_calibration
    from .calibration_report import write_calibration_report

    summary, predictions = evaluate_calibration(
        read_jsonl(args.responses), seed=args.seed, target_coverage=args.target_coverage,
    )
    output_dir = Path(args.output_dir)
    _log(write_calibration_report(summary, predictions, output_dir, args.title or output_dir.name))
    _log(f"calibration artifacts -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(prog="python -m confidence_calibration", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("build-dataset", help="write the seeded TriviaQA development sample")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="data/triviaqa/validation_sample_n300_seed0.jsonl")
    p.set_defaults(func=cmd_build_dataset)

    p = commands.add_parser("score", help="add confidence signals to another producer's responses")
    p.add_argument("--responses", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=config.MODEL_ID)
    p.add_argument("--revision", default=config.MODEL_REVISION)
    p.add_argument("--adapter", help="PEFT adapter directory for fine-tuned checkpoints")
    p.add_argument("--checkpoint-id", default=config.BASE_CHECKPOINT_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.set_defaults(func=cmd_score)

    p = commands.add_parser("calibrate", help="fit and evaluate confidence calibration on explicit data splits")
    p.add_argument("--responses", required=True, help="graded and scored response JSONL for one run")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--title", help="report title (defaults to the output directory name)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed used only when deriving train/validation groups from dev/final data")
    p.add_argument("--target-coverage", type=float, default=0.8,
                   help="validation coverage operating point for selecting confidence thresholds")
    p.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
