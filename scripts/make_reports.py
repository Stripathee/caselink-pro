"""Write a verified case report for each ring.

  python scripts/make_reports.py --top 3          # the 3 rings with the largest losses
  python scripts/make_reports.py                  # every ring
  python scripts/make_reports.py --rings RING-01,RING-07

Writer: Nemotron 3 Ultra. Verifier: Nemotron 3 Super. Reports are saved as
Markdown and JSON in data/reports/. Every model call is cached.
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
from caselink.evaluation import load_dataset, names_match  # noqa: E402
from caselink.reports import load_inputs, render_markdown, write_report  # noqa: E402
from caselink.verify import verify_report  # noqa: E402


def truth_check(report: dict, ring: dict, truth: dict, true_rings: dict) -> dict:
    """Do the leads and names in the kept claims really belong to this ring? (synthetic truth)"""
    from collections import Counter

    rid = Counter(truth[m]["ring_id"] for m in ring["members"] if truth[m]["ring_id"]).most_common(1)[0][0]
    t = true_rings[rid]
    values = set(t["phones"]) | {p["value"] for p in t["payments"]} | {w["value"].lower() for w in t["web"]}
    names = [n["value"] for n in t["names"]] + [p.get("holder") for p in t["payments"] if p.get("holder")]
    checked = correct = 0
    for c in report["claims"]:
        if c["status"] != "kept" or c["section"] != "key_leads" or not c.get("lead"):
            continue
        lead = c["lead"]
        digits = "".join(ch for ch in lead if ch.isdigit())
        checked += 1
        if (len(digits) >= 9 and (digits in values or digits[3:] in values)) or lead.lower() in values \
                or any(v in lead.lower() for v in values if len(v) > 6) or any(names_match(lead, n) for n in names):
            correct += 1
    return {"true_ring": rid, "key_leads_checked": checked, "key_leads_correct": correct}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=0, help="only the N rings with the largest losses (0 = all)")
    p.add_argument("--rings", default="", help="comma-separated ring IDs, e.g. RING-01,RING-04")
    p.add_argument("--writer", default="ultra", help="model tier or ID that writes the reports")
    p.add_argument("--verifier", default="super", help="model tier or ID that checks claims ('none' to skip)")
    p.add_argument("--thinking", choices=["off", "low", "on"], default="low")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    writer = MODELS.get(args.writer, args.writer)
    verifier = None if args.verifier == "none" else MODELS.get(args.verifier, args.verifier)
    thinking = None if args.thinking == "on" else args.thinking
    data, extracted, complaints = load_inputs()
    rings = data["rings"]
    if args.rings:
        wanted = {r.strip() for r in args.rings.split(",")}
        rings = [r for r in rings if r["ring"] in wanted]
    elif args.top:
        rings = rings[:args.top]
    _, truth = load_dataset("data/synthetic")
    true_rings = {r["ring_id"]: r for r in json.loads(Path("data/synthetic/rings.json").read_text(encoding="utf-8"))}

    print(f"Writing {len(rings)} reports: writer {writer}, verifier {verifier or 'none'}, thinking {args.thinking}")
    out = Path("data/reports")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def job(ring: dict) -> dict:
        rep = write_report(ring, extracted, complaints, writer, thinking)
        rep = verify_report(rep, ring, extracted, complaints, verifier, thinking)
        rep["truth_check"] = truth_check(rep, ring, truth, true_rings)
        (out / f"{ring['ring']}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / f"{ring['ring']}.md").write_text(render_markdown(rep, ring), encoding="utf-8")
        return rep

    reports = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, r): r["ring"] for r in rings}
        try:
            for fut in as_completed(futures):
                rep = fut.result()
                reports.append(rep)
                v = rep["verification"]
                flag = "" if rep["writer"]["parsed"] else "  (writer answer unreadable!)"
                print(f"  {rep['ring']}: {v['claims_written']} claims, {v['removed_by_rules']} removed by rules, "
                      f"{v['removed_by_verifier']} by verifier, {v['kept']} kept{flag}")
        except APIStatusError as err:
            for f in futures:
                f.cancel()
            print(f"\nStopped: the API returned {err.status_code}: {err.message}")
            print("Finished reports are cached. Fix the problem and run the same command again.")
            return 1

    written = sum(r["verification"]["claims_written"] for r in reports)
    kept = sum(r["verification"]["kept"] for r in reports)
    by_rules = sum(r["verification"]["removed_by_rules"] for r in reports)
    by_verifier = sum(r["verification"]["removed_by_verifier"] for r in reports)
    leads = sum(r["truth_check"]["key_leads_checked"] for r in reports)
    leads_ok = sum(r["truth_check"]["key_leads_correct"] for r in reports)
    tok_in = sum(r["writer"]["prompt_tokens"] + r["verification"]["verifier"].get("prompt_tokens", 0) for r in reports)
    tok_out = sum(r["writer"]["completion_tokens"] + r["verification"]["verifier"].get("completion_tokens", 0)
                  for r in reports)
    print(f"\nClaims written: {written}; removed by evidence rules: {by_rules}; removed by verifier: {by_verifier}; "
          f"kept: {kept}")
    if written:
        print(f"Share of written claims that passed every check: {kept / written:.1%}")
    if leads:
        print(f"Key leads in kept claims that truly belong to the ring (synthetic truth): {leads_ok}/{leads}")
    print(f"New tokens: {tok_in:,} in, {tok_out:,} out ({time.perf_counter() - start:.0f}s)")

    summary = {"writer": writer, "verifier": verifier, "thinking": args.thinking, "reports": len(reports),
               "claims_written": written, "removed_by_rules": by_rules, "removed_by_verifier": by_verifier,
               "kept": kept, "key_leads_checked": leads, "key_leads_correct": leads_ok,
               "tokens_in": tok_in, "tokens_out": tok_out}
    Path("results").mkdir(exist_ok=True)
    Path("results/reports_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved data/reports/RING-XX.md and .json, and results/reports_eval.json")
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())
