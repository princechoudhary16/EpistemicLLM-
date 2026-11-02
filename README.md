# EpistemicLLM-
Studying how factual exposure shapes what an LLM knows, how confidently it behaves, and how that knowledge is represented internally plus an evidence-aware fine-tuning method to reduce confident hallucinations. Behavioral, calibration, and XAI analysis on synthetic + real-world QA.

## P3 Week 1

The hidden-state extraction and probe pipeline has been run on 100 SimpleQA development questions with Qwen2.5-0.5B-Instruct.

- [Setup and reproduction commands](p3_xai/README.md)
- [Week 1 completion report and limitations](reports/p3_week1.md)
- [Detailed team plan](plan.md)

These are provisional pipeline results. The validation labels contain no correct answers, so probe AUROC is unavailable; high raw accuracy does not demonstrate knowledge detection.

The follow-up P1 synthetic dev extraction completed on 600 questions. Its provisional greedy labels also contained no correct answers, so probe fitting was skipped. See the [run report](reports/p3_synthetic_dev_probe.md) for counts and next steps.

An exploratory fact-cluster bootstrap for the fixed layer 18 matched-SFT probe is saved beside its reviewed-label metrics. Its intervals reflect only the current 75-fact development holdout and provisional labels.
