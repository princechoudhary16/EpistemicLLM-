# P6 Week 1 — how the evaluation actually worked (for the team)

Summary for the group: what data was used, how "evaluation" actually happened
end-to-end, what the numbers mean, and — most importantly — **what's still a
placeholder and must not be quoted as a real result yet.**

## The two things that got tested, and why neither is "the" project data

Per `plan.md` §2, P6's job is the shared infrastructure everyone else's results will
run through — the model runner, the metrics, the eventual baselines. Before trusting
that infrastructure with anyone's real data, it had to be proven against two small,
throwaway datasets built specifically for this:

1. **A 5-question smoke set** (general knowledge questions like "capital of France,"
   plus one deliberately unanswerable fictional question) — to test the
   *generation + grading + metrics* pipeline.
2. **A 3-fact toy training set** (fictional facts like "the call sign of the research
   vessel Kestrahl is Umbraline-9") — to test the *fine-tuning* pipeline.

**Neither of these is P1's synthetic data or P2's real QA data.** They're deliberately
independent, minimal datasets whose only job is to prove the plumbing works before
anyone's actual experiment runs through it. `plan.md`'s dependency table explicitly
allows this — P6 only needs "dummy prediction files + base model" to start, precisely so
P6 isn't blocked waiting on P1/P2.

## Part 1: evaluating the base model (the smoke test)

**Setup frozen first** (`plan.md` §3): exact model pinned by commit revision
(`Qwen2.5-0.5B-Instruct`), exact prompt template, exact decoding settings (1 greedy
answer + 5 sampled answers per question, temperature 0.7, top-p 0.9).

**What actually happened, step by step:**

1. For each of the 5 questions, the model generated **6 responses**: 1 "greedy" (its
   single most confident answer) + 5 "sampled" (randomized re-answers of the same
   question) → **30 total responses**.
2. Each response was **graded** — correct / incorrect / abstained ("I don't know") —
   against the known right answer.
3. Every response, with its grade, was written into one shared table format that
   every team member's future results will also use (so results can be merged later
   without custom glue code per pair of people).
4. From that table, standard metrics got computed: **accuracy** (how often it's
   right), **coverage** (how often it even tries to answer), **selective accuracy**
   (accuracy only among attempted answers), and — because each question was asked 5
   times — a **knowledge score per question** (did it get the same question right
   consistently across all 5 tries, or was it just guessing sometimes?).

**Actual result:** 4/5 correct on the greedy answer. The one deliberately-unanswerable
question (about a fictional explorer) scored knowledge = 0.0 — meaning across all 5
tries, the model never happened to guess something matching a "correct" answer, which is
exactly the behavior that question was designed to check.

**⚠️ Important for the team:** the grading used here is a **temporary placeholder** —
a simple text-matching check, not the real grader. `plan.md` §2 explicitly assigns real
grading to P2. **Nobody should cite this 4/5 number, or any number from this smoke test,
as a project finding.** It only proves the pipeline runs correctly end-to-end.

## Part 2: proving the fine-tuning pipeline (the SFT sanity check)

`plan.md`'s Week 1 checklist for P6 specifically requires proving that a **standard
fine-tuning run** works — not just generation. This is a separate, smaller test:

1. **Before training:** asked the untouched base model 3 made-up factual questions it
   has no way of already knowing (e.g. "who is the warden of the Thornmere Archive?").
   It got all 3 wrong — and notably, it didn't say "I don't know," it *confidently made
   up a plausible-sounding wrong answer* for each one. That's a live example of exactly
   the "confident hallucination" failure mode this whole project is about.
2. **Training:** ran a small LoRA fine-tune (a lightweight, fast way to adjust a
   model without retraining the whole thing) on those same 3 facts, repeated 10 times
   each. Training loss dropped from **7.26 → 0.007** over 90 steps — a clear sign the
   model actually learned something, not just that the script ran.
3. **Saved the trained adapter to a file, then threw away everything in memory and
   loaded it back fresh from that file** — deliberately the strictest possible test,
   since this proves the *saved checkpoint* actually works, not just the object that
   was sitting in memory right after training.
4. **After training,** asked the same 3 questions again. Got **3/3 correct.**

This is the proof that P6's training infrastructure — not just its generation/grading
infrastructure — is solid enough for P5 (who owns the actual proposed method) and for
the standard-SFT and IDK baselines still to come.

## What this does and doesn't mean for the team right now

**Solid and reusable by everyone:**
- The pinned model/config, so nobody's results silently drift apart from someone else's
- The shared output format everyone's real results will also use
- The full metrics toolkit (accuracy, coverage, calibration, confidence-threshold
  analysis, statistical confidence intervals) — implemented and tested against
  hand-worked-out example numbers, so it's not just "looks about right"
- A working, verified generation pipeline *and* a working, verified fine-tuning pipeline

**Not yet real results, and not to be presented as such:**
- Any accuracy/correctness number from these two test runs — both used throwaway
  toy data, and the smoke test used placeholder grading
- The standard-SFT and IDK baselines that will actually go in the final comparison
  table — those come next, once real data is ready
- Nothing here uses P1's synthetic facts or P2's real QA data yet

## Where to point people for more detail

- `p6_baselines/README.md` — quick reference, how to actually run any of this
- `reports/p6_week1_walkthrough.md` — the reasoning behind each design decision
- `reports/p6_code_explained_line_by_line.md` / `metrics_technical_reference.md` — the
  actual code, explained at different levels of depth
