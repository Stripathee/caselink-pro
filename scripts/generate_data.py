"""Generate the synthetic complaint dataset.

Examples (run from the project root):
  python scripts/generate_data.py --sample 12          # 12 varied complaints, a few cents
  python scripts/generate_data.py                      # full dataset, about 1,500 complaints
  python scripts/generate_data.py --writer template    # no model calls, free (for testing)

Everything is cached, so if a run stops halfway, run the same command again:
finished complaints come from the cache and cost nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import APIStatusError  # noqa: E402

from caselink.config import MODELS  # noqa: E402
from caselink.synth import SCAM_TYPES, build_world, ring_to_dict  # noqa: E402
from caselink.writer import write_with_llm, write_with_template  # noqa: E402


def pick_sample(complaints: list[dict], n: int) -> list[dict]:
    """A small sample that still covers every scam type and every hard case."""
    chosen: list[dict] = []

    def take(pred) -> None:
        for c in complaints:
            if pred(c) and c not in chosen:
                chosen.append(c)
                return

    for t in SCAM_TYPES:
        take(lambda c, t=t: c["scam_type"] == t and not c["flags"]["noise"])
    take(lambda c: c["flags"]["noise"])
    take(lambda c: c["flags"]["script_only"])
    take(lambda c: c["flags"]["duplicate_of"] is not None)
    take(lambda c: c["flags"]["has_typo"])
    take(lambda c: c["flags"]["shared_mule"])
    take(lambda c: c["style"]["nepali_words"])
    for c in complaints:
        if len(chosen) >= n:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen[:n]


def write_all(complaints: list[dict], writer: str, model: str, workers: int) -> dict[str, dict]:
    results: dict[str, dict] = {}
    start = time.perf_counter()

    def job(c: dict) -> tuple[str, dict]:
        out = write_with_template(c) if writer == "template" else write_with_llm(c, model)
        return c["complaint_id"], out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, c) for c in complaints]
        failed = empty = 0
        warned = False
        try:
            for i, fut in enumerate(as_completed(futures), start=1):
                cid, out = fut.result()
                results[cid] = out
                failed += out["stats"]["failed"]
                empty += out["stats"]["empty"]
                if i % 50 == 0 or i == len(futures):
                    print(f"  written {i}/{len(futures)}  ({time.perf_counter() - start:.0f}s)")
                if i >= 20 and failed / i > 0.5:
                    for f in futures:
                        f.cancel()
                    print(f"\nStopped early to save credit: {failed} of the first {i} complaints came back empty.")
                    print("Send this output to Claude.")
                    raise SystemExit(1)
                if i >= 20 and not warned and empty / i > 0.5:
                    warned = True
                    print("  Note: the model is still reasoning before answering, so each complaint uses more tokens.")
                    print("  The texts are fine. Press Ctrl+C to stop if you want to check the cost first.")
        except APIStatusError as err:
            for f in futures:
                f.cancel()
            print(f"\nStopped: the API returned {err.status_code}: {err.message}")
            print("Everything finished so far is cached. Fix the problem, then run the same command again.")
            raise SystemExit(1)
    return results


def save(out_dir: Path, world: dict, chosen: list[dict], results: dict[str, dict], meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(c["ring_id"] for c in world["complaints"] if c["ring_id"])
    failed = [c["complaint_id"] for c in chosen if results[c["complaint_id"]]["stats"]["failed"]]
    stats_all = [results[c["complaint_id"]]["stats"] for c in chosen]
    chosen = [c for c in chosen if c["complaint_id"] not in failed]  # never save an empty complaint

    with (out_dir / "complaints.jsonl").open("w", encoding="utf-8") as f:
        for c in chosen:
            row = {k: c[k] for k in ("complaint_id", "filed_date", "district", "channel", "amount_npr", "split")}
            row["text"] = results[c["complaint_id"]]["text"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "ground_truth.jsonl").open("w", encoding="utf-8") as f:
        for c in chosen:
            row = {k: c[k] for k in ("complaint_id", "ring_id", "scam_type", "incident_date", "amount_npr",
                                     "mentions", "names", "flags", "split")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rings = [ring_to_dict(r, counts.get(r.ring_id, 0)) for r in world["rings"]]
    (out_dir / "rings.json").write_text(json.dumps(rings, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = stats_all
    summary = {
        **meta,
        "n_complaints": len(chosen),
        "n_rings": len(world["rings"]),
        "by_scam_type": dict(Counter(c["scam_type"] for c in chosen)),
        "by_split": dict(Counter(c["split"] for c in chosen)),
        "noise": sum(c["flags"]["noise"] for c in chosen),
        "script_only": sum(c["flags"]["script_only"] for c in chosen),
        "duplicates": sum(c["flags"]["duplicate_of"] is not None for c in chosen),
        "with_typo": sum(c["flags"]["has_typo"] for c in chosen),
        "shared_mule": sum(c["flags"]["shared_mule"] for c in chosen),
        "ring_sizes": dict(sorted(counts.items())),
        "generation": {
            "model_calls": sum(s["calls"] for s in stats),
            "cached_calls": sum(s["cached_calls"] for s in stats),
            "prompt_tokens": sum(s["prompt_tokens"] for s in stats),
            "completion_tokens": sum(s["completion_tokens"] for s in stats),
            "empty_first_answer": sum(s["empty"] for s in stats),
            "retried": sum(s["retried"] for s in stats),
            "repaired": sum(s["repaired"] for s in stats),
            "failed_and_skipped": len(failed),
            "with_extra_numbers": sum(bool(s["extra_numbers"]) for s in stats),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def show_examples(chosen: list[dict], results: dict[str, dict], k: int) -> None:
    for c in chosen[:k]:
        print("\n" + "-" * 72)
        flags = [name for name, v in c["flags"].items() if v]
        print(f"{c['complaint_id']} | ring {c['ring_id'] or 'none (one-off)'} | {c['scam_type']}"
              f"{' | ' + ', '.join(flags) if flags else ''}")
        print("-" * 72)
        print(results[c["complaint_id"]]["text"])
        print("Identifiers (truth): " + "; ".join(f"{m['type']}={m['value']}" for m in c["mentions"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=0, help="write only N varied complaints (0 = all)")
    p.add_argument("--writer", choices=["llm", "template"], default="llm")
    p.add_argument("--model", default="super", help="model tier (nano/super/ultra) or a full model ID")
    p.add_argument("--workers", type=int, default=8, help="parallel model calls")
    p.add_argument("--complaints", type=int, default=1500)
    p.add_argument("--rings", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="output folder")
    p.add_argument("--show", type=int, default=None, help="print N example complaints")
    args = p.parse_args()

    model = MODELS.get(args.model, args.model)
    out_dir = Path(args.out or ("data/synthetic_sample" if args.sample else "data/synthetic"))

    print(f"Building hidden rings and fact sheets (seed {args.seed})...")
    world = build_world(n_complaints=args.complaints, n_rings=args.rings, seed=args.seed)
    chosen = pick_sample(world["complaints"], args.sample) if args.sample else world["complaints"]
    print(f"  {len(world['rings'])} rings, {len(world['complaints'])} complaints in total; writing {len(chosen)}")

    who = "template writer (no model calls)" if args.writer == "template" else model
    print(f"Writing complaint texts with {who}...")
    results = write_all(chosen, args.writer, model, args.workers)

    summary = save(out_dir, world, chosen, results,
                   {"seed": args.seed, "writer": args.writer, "model": None if args.writer == "template" else model})

    g = summary["generation"]
    print(f"\nSaved to {out_dir}/")
    print(f"  complaints: {summary['n_complaints']}  | by split: {summary['by_split']}")
    print(f"  one-off (noise): {summary['noise']} | story-only links: {summary['script_only']} | "
          f"duplicates: {summary['duplicates']} | typos: {summary['with_typo']} | "
          f"shared mule account: {summary['shared_mule']}")
    if args.writer == "llm":
        print(f"  model calls: {g['model_calls']} ({g['cached_calls']} from cache) | "
              f"tokens: {g['prompt_tokens']:,} in, {g['completion_tokens']:,} out")
        print(f"  empty first answers: {g['empty_first_answer']} | retried: {g['retried']} | "
              f"repaired with a 'Details' line: {g['repaired']} | failed and skipped: {g['failed_and_skipped']} | "
              f"texts with extra long numbers: {g['with_extra_numbers']}")

    k = args.show if args.show is not None else (3 if args.sample else 0)
    if k:
        show_examples([c for c in chosen if results[c["complaint_id"]]["text"]], results, k)
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())
