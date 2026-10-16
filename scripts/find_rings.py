"""Find scam rings in the extracted complaints and measure how well it worked.

  python scripts/find_rings.py                    # uses Nemotron Nano extraction
  python scripts/find_rings.py --extracted data/extracted/rules.jsonl

Three stages are compared on the validation and test splits:
  1. Plain matching: complaints sharing any identifier are one group.
  2. + Shared-mule splitting.
  3. + Link predictor (CaseLink): attaches complaints that share no identifier.

The attach threshold is chosen on the validation split by a rule fixed in
advance (at most 5% of one-off scams wrongly attached, then as many
story-only complaints linked as possible). The test split is never used
for any choice.

No model calls: this step is free.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from caselink.evaluation import load_dataset  # noqa: E402
from caselink.linker import LinkPredictor, attach  # noqa: E402
from caselink.rings import StoryModel, evaluate, find_rings, load_cases, summarize_rings  # noqa: E402

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90
MAX_NOISE_ATTACHED = 0.05


def pick_threshold(labels, scored, truth, val_ids) -> tuple[float, list[dict]]:
    rows = []
    for t in THRESHOLDS:
        m = evaluate(attach(labels, scored, t), truth, val_ids)
        rows.append({"threshold": t, **m})
    ok = [r for r in rows if r["noise_wrongly_attached"] / max(1, r["noise"]) <= MAX_NOISE_ATTACHED]
    best = max(ok or rows, key=lambda r: (r["story_only_linked"], r["ari"], -r["threshold"]))
    return best["threshold"], rows


def fmt(m: dict) -> str:
    return (f"{m['ring_complaints_correct'] / m['ring_complaints']:>7.1%} "
            f"{m['story_only_linked']:>5}/{m['story_only']:<4}"
            f"{m['noise_wrongly_attached']:>5}/{m['noise']:<4}"
            f"{len(m['rings_wrongly_merged']):>7}   {m['ari']:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extracted", default="data/extracted/cascade_nano.jsonl")
    p.add_argument("--data", default="data/synthetic")
    args = p.parse_args()

    start = time.perf_counter()
    _, truth = load_dataset(args.data)
    cases = load_cases(args.extracted, args.data)
    story = StoryModel(cases)
    split_ids = {s: [c for c in cases if cases[c].split == s] for s in ("val", "test")}

    plain, _, _ = find_rings(cases, story, split_mules=False)
    labels, mules, _ = find_rings(cases, story, split_mules=True)
    print(f"Linked {len(cases)} complaints; shared mule accounts found: {len(mules)}")
    for m in mules:
        print(f"  {m['identifier']} bridges two groups of {m['sides'][0]} and {m['sides'][1]} complaints "
              f"(story similarity {m['story_similarity']}; scam types {m['scam_types']})")

    print("Training the link predictor on the train split...")
    lp = LinkPredictor()
    X, y = lp.training_rows(cases, labels, truth, story)
    lp.fit(X, y)
    scored = lp.score_orphans(cases, labels, story)
    threshold, sweep = pick_threshold(labels, scored, truth, split_ids["val"])
    final = attach(labels, scored, threshold)
    linked = {s["complaint"]: s["probability"] for s in scored if s["probability"] >= threshold}
    print(f"  {len(y):,} training pairs; attach threshold chosen on validation: {threshold}")

    stages = {"Plain matching": plain, "+ shared-mule splitting": labels, "+ link predictor (CaseLink)": final}
    report = {"extracted": args.extracted, "threshold": threshold, "mules": mules, "sweep_val": sweep, "stages": {}}
    for split in ("val", "test"):
        print(f"\n{split.upper()} split")
        print(f"  {'stage':<30}{'correct':>8} {'story-only':>10} {'one-offs':>9} {'merged':>7}   ARI")
        for name, lab in stages.items():
            m = evaluate(lab, truth, split_ids[split])
            report["stages"].setdefault(name, {})[split] = m
            print(f"  {name:<30}{fmt(m)}")
    print("\n  correct = ring complaints in the right ring; story-only = linked with no shared identifier;")
    print("  one-offs = unrelated single scams wrongly attached to a ring; merged = two rings wrongly joined")

    rings, alone = summarize_rings(final, cases, mules, linked)
    out = Path("data/rings")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rings.json").write_text(json.dumps({"rings": rings, "unlinked_complaints": alone}, indent=2,
                                               ensure_ascii=False), encoding="utf-8")
    Path("results").mkdir(exist_ok=True)
    Path("results/rings_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{len(rings)} rings found, {len(alone)} complaints left on their own. Largest by losses:")
    for r in rings[:5]:
        print(f"  {r['ring']}: {r['victims']} victims ({r['complaints']} complaints), "
              f"NPR {r['total_loss_npr']:,}, {r['scam_type']}, {r['active_from']} to {r['active_to']}, "
              f"{len(r['identifiers'])} leads")
    print(f"\nSaved data/rings/rings.json and results/rings_eval.json ({time.perf_counter() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())
