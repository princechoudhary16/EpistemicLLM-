# P1 — Controlled synthetic exposure

Generates a fictional factual world where every fact's number of training presentations
(0, 1, 2, 5, 10, 20) is fixed by construction. Standard library only; no GPU or model needed.

## Run

```bash
# smoke run (pilot: 120 facts, seed 0) -> data/synthetic/pilot_v0, data/manifests/synthetic_pilot_v0
python -m p1_exposure.generate

# main dataset (100 facts per exposure level)
python -m p1_exposure.generate --n-facts 600 --version main_v0

# re-check any generated directory
python -m p1_exposure.validate data/synthetic/pilot_v0

# tests (includes corruption tests proving each check can fail)
python -m unittest discover -s p1_exposure/tests -t .
```

`--n-facts` must be divisible by `relations x levels` (5 x 6 = 30). Same seed and config produce
byte-identical files (sha256s are in `audit.json`).

## Outputs

| File | Contents |
|---|---|
| `data/synthetic/<v>/train.jsonl` | One row per presentation: `prompt`, `completion` (answer only). Facts at exposure *e* appear exactly *e* times; corpus is pre-expanded and shuffled, so train **one pass, one epoch**. 0x facts are absent. |
| `data/synthetic/<v>/eval.jsonl` | Two questions per fact, using wording never seen in training. Fields follow the shared schema (`question_id`, `fact_id`, `exposure_level`, `question`, `reference_answer`, `split`, ...). |
| `data/synthetic/<v>/facts.jsonl` | Full fact table including answers. Evaluation-side only; never feed to a trainer. |
| `data/synthetic/<v>/idk_train.jsonl` | Separate unknown entities with completion "I don't know." for P6's IDK baseline. Disjoint from every evaluated fact. |
| `data/manifests/synthetic_<v>/facts_manifest.jsonl` | IDs, relation, exposure, split only (no names/answers). |
| `data/manifests/synthetic_<v>/audit.json` | Counts per level/relation/split, check results, file hashes. |

## Design

- **Relations (5):** birth_year, founder, capital, headquarters, occupation. Answer types: year, person, place, closed-set category.
- **Balance:** each (relation, level) cell has the same number of facts; exposure levels are dealt over an answer-sorted order so each level gets the same mix of birth decades / occupations.
- **Split:** within every cell, half the facts are `dev` (evaluated with *val* wording) and half `final` (evaluated with *test* wording). Both halves are trained on; only the questions differ.
- **Names:** one shared syllable model for all entity types, so a name's shape never hints at its role or answer. No word contains another, and all names are unique across the world.
- **Templates:** 4 train / 2 val / 2 test question wordings per relation, all distinct strings.

## Decisions to confirm with the team

1. **Training format is Q -> A, not declarative statements**, so it drops straight into chat SFT with answer-only loss and matches the eval prompt style. Statement-style training is a possible Week 2 change if high-exposure facts are not learned.
2. **Wording rotates across presentations** (fact at 20x uses each of the 4 train wordings 5 times; at 1x uses one). So low exposures also see fewer distinct wordings. That is inherent to counting presentations, but worth stating in the report.
3. **Dev and final facts use different eval wordings** (val vs test), so a dev-to-final gap can partly reflect wording.
4. **5 relations, not 6:** 120 pilot facts split evenly only for 5 (or 4, 10...). Add a relation in `relations.py` if you want more, keeping `n-facts` divisible.

## Where the rest of P1 lives

| Folder | What it is |
|---|---|
| `colab/` | First pilot fine-tune and evaluation (self-contained Colab notebook) |
| `ablation/` | Decoding defaults x SFT recipe x IDK ordering; `run_ablation.py` is the training and evaluation harness, with a SLURM script and a Colab notebook builder |
| `analysis/` | Exposure tables and figures with fact-level intervals, and the Week 4 robustness report |
| `robustness/` | Held-out and swapped question wordings (`paraphrase_sets.py`) |
| `aux_prompts.py` | Answer-masked training-wording prompts for the 0x facts, for a head that needs 0x examples (written to `data/synthetic/<v>/zero_prompts/`) |

Second world for the robustness check: `python -m p1_exposure.generate --n-facts 600 --seed 1 --version main_w1`
(different invented names and a different fact-to-exposure assignment; nothing shared with `main_v0`).
Extra wordings for any version: `python -m p1_exposure.robustness.paraphrase_sets --version main_v0`,
written to `data/synthetic/<v>/robustness/` and never mixed into `eval.jsonl`.

Auxiliary 0x prompts for any version: `python -m p1_exposure.aux_prompts --version main_v0`. One row per 0x fact and
training wording (4 per fact), with no completion and no answer in the row; checks confirm no answer text appears in a
prompt and no 0x name appears in `train.jsonl` or `idk_train.jsonl`. Use them with no language-model loss. They name the
0x entities, so a head trained on them has seen those names (never their answers).
