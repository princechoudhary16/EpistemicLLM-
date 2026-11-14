"""Command-line entry point: python -m confidence_calibration <command> --help

  build-dataset  write the seeded TriviaQA development sample
  score          add confidence signals to another producer's responses
  calibrate      fit on train, select on validation, write a frozen policy
  evaluate       apply a frozen policy to held-out rows, once
  compare        compare methods on the same held-out questions at matched coverage
"""

import argparse
import json
import platform
import time
from pathlib import Path

from . import config
from .data import build_triviaqa_sample, display_path, load_responses, write_jsonl


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
    from .calibration_report import write_policy, write_report
    from .policy import fit_policy

    rows, inputs = load_responses(args.responses)
    policy, summary, predictions = fit_policy(rows, inputs, seed=args.seed, target_coverage=args.target_coverage,
                                              coverage_grid=args.coverage_grid)
    output_dir = Path(args.output_dir)
    write_policy(policy, output_dir)
    write_report(summary, predictions, output_dir, args.title or output_dir.name, "calibration")
    _log(f"selected {policy['selected_signal']} on validation; frozen policy -> {output_dir / 'frozen_policy.json'}")


def cmd_evaluate(args):
    from .calibration_report import write_report
    from .data import sha256_file
    from .policy import evaluate_policy, load_policy

    output_dir = Path(args.output_dir)
    if (output_dir / "evaluation.json").exists():
        raise SystemExit(f"error: {output_dir} already holds an evaluation. A held-out split is evaluated once; "
                         "keep that result and report any further run as a separate, disclosed analysis.")
    policy = load_policy(args.policy)
    rows, inputs = load_responses(args.responses)
    summary, predictions = evaluate_policy(policy, sha256_file(args.policy), rows, inputs, seed=args.seed,
                                           allow_transfer=args.allow_transfer)
    write_report(summary, predictions, output_dir, args.title or output_dir.name, "evaluation")
    _log(f"evaluated {len(summary['signals'])} frozen signals -> {output_dir}")


def cmd_compare(args):
    from .method_comparison import compare_probe, compare_runs, load_run, write_comparison

    runs = []
    for item in args.run:
        label, separator, rest = item.partition("=")
        directory, at, signal = rest.rpartition("@")
        if not (separator and at):
            raise SystemExit(f"error: --run expects LABEL=DIRECTORY@SIGNAL, got {item!r}")
        runs.append(load_run(label, directory, signal))
    comparison = compare_runs(runs, args.reference or runs[0]["label"], args.coverage_grid, args.seed)
    probe = None
    if args.probe_predictions:
        if not args.probe:
            raise SystemExit("error: --probe-predictions needs --probe")
        reference = next(run for run in runs if run["label"] == comparison["reference"])
        probe = compare_probe(reference, args.probe_predictions, args.probe, args.seed)
    write_comparison(comparison, probe, runs, args.output_dir, args.title or Path(args.output_dir).name)
    _log(f"compared {len(runs)} methods -> {args.output_dir}")


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

    p = commands.add_parser("calibrate", help="fit on train, select on validation, write a frozen policy")
    p.add_argument("--responses", nargs="+", required=True,
                   help="response JSONL files for one model; rows from held-out splits are ignored unread")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--title", help="report title (defaults to the output directory name)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the bootstrap and for dividing dev facts into train/validation")
    p.add_argument("--target-coverage", type=float, default=config.PRIMARY_COVERAGE, help="primary validation coverage operating point")
    p.add_argument("--coverage-grid", type=float, nargs="+", default=list(config.COVERAGE_GRID),
                   help="further coverage targets that each get a validation-selected threshold")
    p.set_defaults(func=cmd_calibrate)

    p = commands.add_parser("evaluate", help="apply a frozen policy to held-out rows, once")
    p.add_argument("--policy", required=True, help="frozen_policy.json written by calibrate")
    p.add_argument("--responses", nargs="+", required=True, help="response JSONL files holding the held-out rows")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--title", help="report title (defaults to the output directory name)")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--allow-transfer", action="store_true",
                   help="apply the policy to another model or dataset under the same protocol")
    p.set_defaults(func=cmd_evaluate)

    p = commands.add_parser("compare", help="compare methods on the same held-out questions at matched coverage")
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIRECTORY@SIGNAL",
                   help="an evaluate output directory and the signal to compare; repeat per method")
    p.add_argument("--reference", help="label differences are taken against (default: the first --run)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--title", help="report title (defaults to the output directory name)")
    p.add_argument("--coverage-grid", type=float, nargs="+", default=list(config.COVERAGE_GRID))
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--probe-predictions", help="CSV of question_id,probe,probability from a hidden-state probe")
    p.add_argument("--probe", help="which probe in that CSV, e.g. layer_18")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
