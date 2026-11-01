"""Command-line entry point: python -m confidence_calibration <command> --help

  build-dataset  write the seeded TriviaQA development sample
  run            generate -> grade -> score -> report for a question file (one model load)
  score          add confidence signals to another producer's graded responses
  report         rebuild the report for a scored response file
  calibrate      fit leakage-safe probability maps and evaluate frozen thresholds
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .data import build_triviaqa_sample, load_questions, read_jsonl, write_jsonl

DEFAULT_RESULTS_DIR = Path("results/raw/confidence_calibration")


def _log(message):
    print(message, flush=True)


def cmd_build_dataset(args):
    rows = build_triviaqa_sample(args.n, args.seed)
    write_jsonl(args.output, rows)
    _log(f"wrote {len(rows)} questions -> {args.output}")


def cmd_run(args):
    import torch
    import transformers

    from .generation import decoding_settings, generate_responses
    from .grading import grade_responses
    from .modeling import end_token_ids, load_model, resolve_device
    from .report import build_report, write_report
    from .scoring import score_responses

    run_id = args.run_id or f"{Path(args.questions).stem}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    output_dir = Path(args.output_dir or DEFAULT_RESULTS_DIR / run_id)
    device = resolve_device(args.device)
    questions = load_questions(args.questions, args.limit)
    run_meta = {"run_id": run_id, "model_id": config.MODEL_ID, "model_revision": config.MODEL_REVISION,
                "checkpoint_id": config.BASE_CHECKPOINT_ID, "method": "base", "training_seed": None}
    started = time.time()

    tokenizer, model = load_model(config.MODEL_ID, config.MODEL_REVISION, device)
    responses = generate_responses(
        questions, tokenizer, model, device, run_meta=run_meta, prompt_version=args.prompt_version,
        n_samples=args.n_samples, batch_size=args.batch_size, seed=args.seed,
        record_generation_logprobs=args.verify_scoring, log=_log,
    )
    grade_responses(responses)
    score_responses(responses, tokenizer, model, device, batch_size=args.score_batch_size)
    report, question_summaries = build_report(responses)

    write_jsonl(output_dir / "responses.jsonl", responses)
    write_jsonl(output_dir / "questions.jsonl", question_summaries)
    run_config = {
        **run_meta, "questions": args.questions, "n_questions": len(questions), "device": device,
        "prompt_version": args.prompt_version, "grading_version": config.GRADING_VERSION,
        "signals_version": config.SIGNALS_VERSION, "decoding_config_id": config.DECODING_CONFIG_ID,
        "decoding": decoding_settings(args.n_samples, end_token_ids(tokenizer)),
        "seed": args.seed, "batch_size": args.batch_size,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "wall_seconds": round(time.time() - started, 1),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    _log(write_report(report, output_dir, run_id))
    _log(f"outputs -> {output_dir}")


def cmd_score(args):
    from .modeling import load_model, resolve_device
    from .scoring import check_provenance, score_responses

    rows = read_jsonl(args.responses)
    if not args.skip_provenance_check:
        try:
            check_provenance(rows, args.model, args.checkpoint_id)
        except ValueError as err:
            raise SystemExit(f"error: {err} (pass --skip-provenance-check to override)")
    device = resolve_device(args.device)
    tokenizer, model = load_model(args.model, args.revision, device, adapter=args.adapter)
    score_responses(rows, tokenizer, model, device, batch_size=args.batch_size)
    write_jsonl(args.output, rows)
    _log(f"scored {len(rows)} responses -> {args.output}")


def cmd_report(args):
    from .report import build_report, write_report

    report, question_summaries = build_report(read_jsonl(args.responses))
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "questions.jsonl", question_summaries)
    _log(write_report(report, output_dir, args.title or output_dir.name))


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

    p = commands.add_parser("run", help="generate, grade, score and report a question file")
    p.add_argument("--questions", required=True)
    p.add_argument("--output-dir", help=f"default: {DEFAULT_RESULTS_DIR}/<run_id>")
    p.add_argument("--run-id")
    p.add_argument("--limit", type=int, help="first N questions only (smoke tests)")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--score-batch-size", type=int, default=16)
    p.add_argument("--n-samples", type=int, default=config.N_SAMPLES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-version", default=config.PROMPT_VERSION, choices=sorted(config.PROMPTS))
    p.add_argument("--verify-scoring", action="store_true",
                   help="record generate()'s greedy log-probs to confirm scoring reproduces them")
    p.set_defaults(func=cmd_run)

    p = commands.add_parser("score", help="add confidence signals to graded responses from any producer")
    p.add_argument("--responses", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=config.MODEL_ID)
    p.add_argument("--revision", default=config.MODEL_REVISION)
    p.add_argument("--adapter", help="PEFT adapter directory for fine-tuned checkpoints")
    p.add_argument("--checkpoint-id", default=config.BASE_CHECKPOINT_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-provenance-check", action="store_true")
    p.set_defaults(func=cmd_score)

    p = commands.add_parser("report", help="rebuild the report for a scored response file")
    p.add_argument("--responses", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--title")
    p.set_defaults(func=cmd_report)

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
