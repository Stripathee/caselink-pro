"""Run extraction over the dataset and score it against the ground truth.

  python scripts/run_extraction.py --mode rules                      # free, instant
  python scripts/run_extraction.py --mode cascade --limit 30         # rules + Nemotron Nano, small test
  python scripts/run_extraction.py --mode cascade                    # all 1,500 complaints
  python scripts/run_extraction.py --mode llm --model super --limit 30   # model only, for comparison

Results are saved to data/extracted/ and every model call is cached, so
re-running costs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import APIStatusError  # noqa: E402

from caselink.config import MODELS  # noqa: E402
from caselink.evaluation import ID_TYPES, load_dataset, score_extraction, score_names  # noqa: E402
from caselink.extract_llm import extract_complaint  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["rules", "cascade", "llm"], default="cascade")
    p.add_argument("--model", default="nano", help="model tier (nano/super/ultra) or a full model ID")
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--limit", type=int, default=0, help="only the first N complaints (0 = all)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    model = MODELS.get(args.model, args.model)
    complaints, truth = load_dataset(args.data)
    ids = [cid for cid, c in complaints.items() if args.split == "all" or c["split"] == args.split]
    if args.limit:
        ids = ids[:args.limit]

    who = "rules only (no model calls)" if args.mode == "rules" else f"{args.mode} with {model}"
    print(f"Extracting from {len(ids)} complaints: {who}")
    results: dict[str, dict] = {}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_complaint, complaints[cid]["text"], model, args.mode,
                               complaints[cid]["filed_date"]): cid for cid in ids}
        try:
            for i, fut in enumerate(as_completed(futures), start=1):
                results[futures[fut]] = fut.result()
                if i % 100 == 0 or i == len(ids):
                    print(f"  done {i}/{len(ids)}  ({time.perf_counter() - start:.0f}s)")
        except APIStatusError as err:
            for f in futures:
                f.cancel()
            print(f"\nStopped: the API returned {err.status_code}: {err.message}")
            print("Finished complaints are cached. Fix the problem and run the same command again.")
            return 1

    tag = args.mode if args.mode == "rules" else f"{args.mode}_{args.model}"
    out_dir = Path("data/extracted")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{tag}.jsonl").open("w", encoding="utf-8") as f:
        for cid in ids:
            f.write(json.dumps({"complaint_id": cid, **results[cid]}, ensure_ascii=False) + "\n")

    pred = {cid: {(i["type"], i["value"]) for i in r["identifiers"]} for cid, r in results.items()}
    s = score_extraction(pred, truth, ids)
    print(f"\nIdentifiers ({len(ids)} complaints)")
    print(f"  {'type':<16}{'precision':>10}{'recall':>9}{'F1':>8}{'count':>8}")
    for t in ID_TYPES:
        r = s["by_type"][t]
        print(f"  {t:<16}{r['precision']:>10.1%}{r['recall']:>9.1%}{r['f1']:>8.1%}{r['support']:>8}")
    o = s["overall"]
    print(f"  {'ALL':<16}{o['precision']:>10.1%}{o['recall']:>9.1%}{o['f1']:>8.1%}{o['support']:>8}")

    report = {"mode": args.mode, "model": None if args.mode == "rules" else model, "n": len(ids),
              "identifiers": s}
    if args.mode != "rules":
        names = score_names({cid: r["people"] + r["organisations"] for cid, r in results.items()}, truth, ids,
                            {cid: complaints[cid]["text"] for cid in ids})
        typed = [cid for cid in ids if results[cid]["scam_type"]]
        type_acc = sum(results[c]["scam_type"] == truth[c]["scam_type"] for c in ids) / len(ids)
        amt_ok = sum(results[c]["amount_lost_npr"] == truth[c]["amount_npr"] for c in ids) / len(ids)
        dated = [c for c in ids if results[c]["incident_date"]]
        date_ok = sum(results[c]["incident_date"] == truth[c]["incident_date"] for c in dated) / max(1, len(dated))
        tokens_in = sum(r["stats"]["prompt_tokens"] for r in results.values() if not r["stats"]["cached"])
        tokens_out = sum(r["stats"]["completion_tokens"] for r in results.values() if not r["stats"]["cached"])
        failed = sum(r.get("parse_failed", False) for r in results.values())
        dropped = sum(r["stats"]["dropped"] for r in results.values())
        print(f"\nNames (aliases and organisations): precision {names['precision']:.1%}, "
              f"recall {names['recall']:.1%}, F1 {names['f1']:.1%}")
        print(f"Scam type correct: {type_acc:.1%}  ({len(typed)} labelled)")
        print(f"Amount lost exactly right: {amt_ok:.1%}")
        print(f"Incident date: {len(dated)} of {len(ids)} dated, {date_ok:.1%} of those exactly right")
        print(f"Unreadable answers: {failed} | items removed by validation: {dropped}")
        print(f"New tokens this run: {tokens_in:,} in, {tokens_out:,} out")
        report.update(names=names, scam_type_accuracy=type_acc, amount_accuracy=amt_ok,
                      date_coverage=len(dated) / len(ids), date_accuracy=date_ok,
                      parse_failed=failed, dropped=dropped, tokens_in=tokens_in, tokens_out=tokens_out)

    Path("results").mkdir(exist_ok=True)
    (Path("results") / f"extraction_{tag}_{args.split}.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved: data/extracted/{tag}.jsonl and results/extraction_{tag}_{args.split}.json")
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())
