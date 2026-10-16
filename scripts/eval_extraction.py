"""Score identifier extraction against the ground truth.

  python scripts/eval_extraction.py                  # rules (Tier 1), all splits
  python scripts/eval_extraction.py --split test     # held-out test split only
  python scripts/eval_extraction.py --errors 10      # also print 10 mistakes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from caselink.evaluation import ID_TYPES, load_dataset, score_extraction, truth_keys  # noqa: E402
from caselink.extract_rules import extract, needs_llm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--errors", type=int, default=0, help="print N complaints with mistakes")
    p.add_argument("--out", default=None, help="save scores as JSON")
    args = p.parse_args()

    complaints, truth = load_dataset(args.data)
    ids = [cid for cid, c in complaints.items() if args.split == "all" or c["split"] == args.split]

    pred, escalate = {}, 0
    for cid in ids:
        hits = extract(complaints[cid]["text"])
        pred[cid] = {(h.type, h.value) for h in hits}
        escalate += needs_llm(hits, complaints[cid]["text"])

    s = score_extraction(pred, truth, ids)
    print(f"Rule-based extraction (Tier 1) on {len(ids)} complaints ({args.split})\n")
    print(f"  {'type':<16}{'precision':>10}{'recall':>9}{'F1':>8}{'count':>8}")
    for t in ID_TYPES:
        r = s["by_type"][t]
        print(f"  {t:<16}{r['precision']:>10.1%}{r['recall']:>9.1%}{r['f1']:>8.1%}{r['support']:>8}")
    o, n = s["overall"], s["numbers_any_label"]
    print(f"  {'ALL':<16}{o['precision']:>10.1%}{o['recall']:>9.1%}{o['f1']:>8.1%}{o['support']:>8}")
    print(f"\n  Phone/wallet numbers, ignoring which label: precision {n['precision']:.1%}, recall {n['recall']:.1%}")
    print(f"  Complaints the cascade would send to Nemotron: {escalate} of {len(ids)} ({escalate / len(ids):.0%})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({**s, "escalated": escalate, "n": len(ids)}, indent=2))

    shown = 0
    for cid in ids:
        if shown >= args.errors:
            break
        t = truth_keys(truth[cid])
        missed, extra = t - pred[cid], pred[cid] - t
        if missed or extra:
            shown += 1
            print("\n" + "-" * 72 + f"\n{cid}\n" + complaints[cid]["text"])
            print(f"  missed: {sorted(missed)}\n  extra:  {sorted(extra)}")
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())
