"""P1 robustness report (plan.md Week 4): does the exposure curve survive a different training seed,
a different synthetic world, and different question wording?

Reads the output folders written by p1_exposure/ablation/run_ablation.py (see make_colab_notebook.py,
profile "robustness"): <dir>/responses__<run>__<decoding>.jsonl for the standard eval questions and
<dir>/extra/eval_heldout|eval_swapped/responses__<run>__<decoding>.jsonl for the paraphrase sets.

    python -m p1_exposure.analysis.robustness_report --data-dir data/synthetic \
        --world main_v0=out_v0_s0,out_v0_s1,out_v0_s2 --world main_w1=out_w1_s0 \
        --run mine --decoding frozen --out-dir p1_exposure/analysis/results/robustness

All intervals are fact-level bootstrap percentile intervals (whole facts are resampled, because the
questions about one fact are not independent). Comparisons between wordings use the same facts, so
they are paired: the interval is on the per-fact difference.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from p1_exposure.analysis.exposure_analysis import (interval, load_eval, load_responses, per_fact, pct, sha256)

PARAPHRASE_SETS = (("heldout", "eval_heldout", "robustness/eval_heldout.jsonl"),
                   ("swapped", "eval_swapped", "robustness/eval_swapped.jsonl"))

DEFINITIONS = """\
**Terms.** *Exposure*: how many times a fact appeared in the fine-tuning data (0 means never).
*Accuracy*: share of greedy answers that contain the reference answer. *Training seed*: the random
seed for the adapter's initial weights and dropout; the training data and its order are unchanged.
*World*: one complete synthetic dataset (invented names, answers, and which facts get which exposure);
world 1 is generated with a different generator seed, so it has different facts and a different
fact-to-exposure assignment. *Held-out wording*: two new question wordings per relation that appear in no
training, validation or test template. *Swapped wording*: each fact asked with the other split's
wording (dev facts get the test wordings, final facts get the validation wordings). *Paired*: the same
facts are compared under two wordings, and the interval is on the per-fact difference.
*95% CI*: fact-level bootstrap interval.
"""


def run_paths(run_dir, run, decoding):
    d = Path(run_dir)
    main = d / f"responses__{run}__{decoding}.jsonl"
    extras = {name: d / "extra" / folder / f"responses__{run}__{decoding}.jsonl"
              for name, folder, _ in PARAPHRASE_SETS}
    return main, extras


def read_eval_file(path):
    with open(path, encoding="utf-8") as fh:
        return {r["question_id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def fact_records(responses_path, eval_by_q):
    rows = load_responses(responses_path, eval_by_q)
    return per_fact(rows)


def level_acc(facts, level):
    fl = [f for f in facts.values() if f["level"] == level]
    return sum(f["c"] for f in fl) / sum(f["n"] for f in fl) if fl else None


def fact_vector(facts_list, fact_ids):
    """Per-fact greedy accuracy averaged over the given runs (seeds)."""
    out = []
    for fid in fact_ids:
        accs = [fl[fid]["c"] / fl[fid]["n"] for fl in facts_list]
        out.append(sum(accs) / len(accs))
    return np.array(out)


def boot_mean(vec, rng, boot):
    idx = rng.integers(0, len(vec), size=(boot, len(vec)))
    draws = vec[idx].mean(axis=1)
    lo, hi = interval(draws)
    return float(vec.mean()), lo, hi, draws


def boot_diff_paired(a, b, rng, boot):
    d = b - a
    idx = rng.integers(0, len(d), size=(boot, len(d)))
    lo, hi = interval(d[idx].mean(axis=1))
    return float(d.mean()), lo, hi


def boot_diff_independent(a, b, rng, boot):
    ia = rng.integers(0, len(a), size=(boot, len(a)))
    ib = rng.integers(0, len(b), size=(boot, len(b)))
    lo, hi = interval(b[ib].mean(axis=1) - a[ia].mean(axis=1))
    return float(b.mean() - a.mean()), lo, hi


def analyze(data_dir, worlds, run, decoding, boot, seed):
    rng = np.random.default_rng(seed)
    out = {"worlds": {}, "levels": None}
    for version, dirs in worlds.items():
        eval_std = load_eval(data_dir, version)
        levels = sorted({int(q["exposure_level"]) for q in eval_std.values()})
        out["levels"] = levels
        std_facts = []
        out.setdefault("inputs", {})[version] = []
        for d in dirs:
            main, _ = run_paths(d, run, decoding)
            out["inputs"][version].append({"dir": Path(d).name, "file": main.name, "sha256": sha256(main)})
            std_facts.append(fact_records(main, eval_std))
        fact_ids = sorted(std_facts[0])
        fact_level = {fid: std_facts[0][fid]["level"] for fid in fact_ids}
        w = {"seeds": len(dirs), "per_seed": {}, "seed_mean": {}, "paraphrase": {}}

        for lvl in levels:
            accs = [level_acc(fl, lvl) for fl in std_facts]
            ids = [fid for fid in fact_ids if fact_level[fid] == lvl]
            vec = fact_vector(std_facts, ids)
            est, lo, hi, _ = boot_mean(vec, rng, boot)
            w["per_seed"][lvl] = accs
            w["seed_mean"][lvl] = {"est": est, "lo": lo, "hi": hi, "range": max(accs) - min(accs), "n_facts": len(ids)}

        for name, folder, rel_path in PARAPHRASE_SETS:
            extra_rows = Path(data_dir) / version / rel_path
            if not extra_rows.exists():
                continue
            eval_extra = read_eval_file(extra_rows)
            extra_facts = []
            for d in dirs:
                _, extras = run_paths(d, run, decoding)
                if not extras[name].exists():
                    extra_facts = None
                    break
                extra_facts.append(fact_records(extras[name], eval_extra))
            if extra_facts is None:
                continue
            per_level = {}
            for lvl in levels:
                ids = [fid for fid in fact_ids if fact_level[fid] == lvl]
                a, b = fact_vector(std_facts, ids), fact_vector(extra_facts, ids)
                est, lo, hi = boot_diff_paired(a, b, rng, boot)
                per_level[lvl] = {"standard": float(a.mean()), "paraphrase": float(b.mean()),
                                  "diff": est, "lo": lo, "hi": hi}
            learn = [fid for fid in fact_ids if fact_level[fid] >= 2]
            a, b = fact_vector(std_facts, learn), fact_vector(extra_facts, learn)
            est, lo, hi = boot_diff_paired(a, b, rng, boot)
            by_split = {}
            for sp in ("dev", "final"):
                ids = [fid for fid in learn if std_facts[0][fid]["split"] == sp]
                if ids:
                    sa, sb = fact_vector(std_facts, ids), fact_vector(extra_facts, ids)
                    e, l, h = boot_diff_paired(sa, sb, rng, boot)
                    by_split[sp] = {"standard": float(sa.mean()), "paraphrase": float(sb.mean()), "diff": e,
                                    "lo": l, "hi": h, "n_facts": len(ids)}
            w["paraphrase"][name] = {"per_level": per_level, "overall_ge2": {"diff": est, "lo": lo, "hi": hi,
                                     "standard": float(a.mean()), "paraphrase": float(b.mean())},
                                     "by_split_ge2": by_split}
        out["worlds"][version] = w
        w["_vectors"] = {"fact_ids": fact_ids, "fact_level": fact_level, "std_facts": std_facts}

    versions = list(worlds)
    if len(versions) >= 2:
        base, other = versions[0], versions[1]
        cmp = {}
        for lvl in out["levels"]:
            va = out["worlds"][base]["_vectors"]
            vb = out["worlds"][other]["_vectors"]
            ia = [f for f in va["fact_ids"] if va["fact_level"][f] == lvl]
            ib = [f for f in vb["fact_ids"] if vb["fact_level"][f] == lvl]
            a = fact_vector(va["std_facts"], ia)
            b = fact_vector(vb["std_facts"], ib)
            est, lo, hi = boot_diff_independent(a, b, rng, boot)
            cmp[lvl] = {"world_a": float(a.mean()), "world_b": float(b.mean()), "diff": est, "lo": lo, "hi": hi}
        out["world_comparison"] = {"a": base, "b": other, "per_level": cmp}
    for w in out["worlds"].values():
        w.pop("_vectors", None)
    return out


def render(out, run, decoding, boot, seed):
    levels = out["levels"]
    L = ["# P1 robustness report", "", DEFINITIONS,
         f"Run `{run}` (matched recipe unless stated), decoding `{decoding}`, {boot} bootstrap draws, seed {seed}.", ""]
    for version, items in out.get("inputs", {}).items():
        L.append(f"- `{version}` inputs: " + ", ".join(f"`{i['dir']}/{i['file']}` (sha256 `{i['sha256'][:12]}`)" for i in items))
    L.append("")
    for version, w in out["worlds"].items():
        L += [f"## World `{version}`: training seeds ({w['seeds']} seeds)", "",
              "| Exposure | " + " | ".join(f"seed {i}" for i in range(w["seeds"])) +
              " | Mean [95% CI over facts] | Spread across seeds (max minus min) |",
              "|---|" + "---|" * (w["seeds"] + 2)]
        for lvl in levels:
            sm = w["seed_mean"][lvl]
            L.append(f"| {lvl}x | " + " | ".join(pct(a) for a in w["per_seed"][lvl]) +
                     f" | {pct(sm['est'])} [{pct(sm['lo'])}, {pct(sm['hi'])}] | {100 * sm['range']:.1f} pts |")
        for name, p in w["paraphrase"].items():
            L += ["", f"### `{version}`: {name} wording vs standard wording (paired by fact, averaged over seeds)", "",
                  "| Exposure | Standard | " + name.capitalize() + " | Difference [95% CI] |", "|---|---|---|---|"]
            for lvl in levels:
                c = p["per_level"][lvl]
                L.append(f"| {lvl}x | {pct(c['standard'])} | {pct(c['paraphrase'])} | "
                         f"{100 * c['diff']:+.1f} pts [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] |")
            o = p["overall_ge2"]
            L.append(f"| all facts at 2x and above | {pct(o['standard'])} | {pct(o['paraphrase'])} | "
                     f"{100 * o['diff']:+.1f} pts [{100 * o['lo']:+.1f}, {100 * o['hi']:+.1f}] |")
            for sp, s in p["by_split_ge2"].items():
                L.append(f"| {sp} facts at 2x and above (n={s['n_facts']}) | {pct(s['standard'])} | {pct(s['paraphrase'])} | "
                         f"{100 * s['diff']:+.1f} pts [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}] |")
        L.append("")
    if "world_comparison" in out:
        wc = out["world_comparison"]
        L += [f"## Second world: `{wc['b']}` minus `{wc['a']}` (seed-averaged accuracy, different facts)", "",
              "| Exposure | " + wc["a"] + " | " + wc["b"] + " | Difference [95% CI] |", "|---|---|---|---|"]
        for lvl in levels:
            c = wc["per_level"][lvl]
            L.append(f"| {lvl}x | {pct(c['world_a'])} | {pct(c['world_b'])} | "
                     f"{100 * c['diff']:+.1f} pts [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] |")
        L.append("")
    return "\n".join(L) + "\n"


def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = out["levels"]
    x = np.arange(len(levels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for version, w in out["worlds"].items():
        for i in range(w["seeds"]):
            ax.plot(x, [w["per_seed"][l][i] for l in levels], marker="o", alpha=0.45,
                    linewidth=1, label=f"{version}, seed {i}")
    ax.set_xticks(x, [f"{l}x" for l in levels])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Times the fact was shown in training (categorical axis)")
    ax.set_ylabel("Greedy accuracy")
    ax.set_title("Training seeds and a second synthetic world")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    n_groups = max(1, sum(len(w["paraphrase"]) for w in out["worlds"].values()))
    width = 0.8 / n_groups
    k = 0
    for version, w in out["worlds"].items():
        for name, p in w["paraphrase"].items():
            d = [100 * p["per_level"][l]["diff"] for l in levels]
            lo = [100 * (p["per_level"][l]["diff"] - p["per_level"][l]["lo"]) for l in levels]
            hi = [100 * (p["per_level"][l]["hi"] - p["per_level"][l]["diff"]) for l in levels]
            ax.bar(x + (k - (n_groups - 1) / 2) * width, d, width, yerr=[lo, hi], capsize=2,
                   label=f"{version}: {name}")
            k += 1
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{l}x" for l in levels])
    ax.set_xlabel("Times the fact was shown in training (categorical axis)")
    ax.set_ylabel("Accuracy change vs standard wording (points)")
    ax.set_title("Different question wording (paired by fact, 95% CI)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--world", action="append", required=True, metavar="VERSION=DIR1,DIR2",
                    help="dataset version and the run folders (one per training seed); repeat per world")
    ap.add_argument("--run", default="mine")
    ap.add_argument("--decoding", default="frozen")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    worlds = {}
    for item in a.world:
        if "=" not in item:
            sys.exit(f"--world expects VERSION=DIR1,DIR2, got {item!r}")
        version, dirs = item.split("=", 1)
        worlds[version] = [d for d in dirs.split(",") if d]
    out = analyze(a.data_dir, worlds, a.run, a.decoding, a.boot, a.seed)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "robustness.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "robustness_report.md").write_text(render(out, a.run, a.decoding, a.boot, a.seed),
                                                   encoding="utf-8", newline="\n")
    if not a.no_figure:
        make_figure(out, out_dir / "fig_robustness.png")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
