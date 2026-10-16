"""Claim verifier: nothing reaches an investigator without checked evidence.

Two layers:
1. Rules (free, exact). Every claim must cite evidence; cited complaints must
   belong to the ring; cited facts must exist; every identifier in a claim must
   be a lead of this ring or appear in a cited complaint; every number in a
   claim must appear in a cited fact or complaint (the writer may not do its
   own arithmetic).
2. Nemotron Super reads each surviving claim next to the text of the evidence
   it cites and judges whether that evidence really says it.

Claims that fail either layer are removed from the report and listed in an
appendix, so a reviewer can see exactly what was cut and why.
"""
from __future__ import annotations

import re
from typing import Any

from caselink import llm

_ID_PATTERNS = [
    re.compile(r"\+?\d[\d\s\-]{7,}\d"),  # phone, wallet and account numbers
    re.compile(r"@[A-Za-z][A-Za-z0-9_]{3,31}"),  # Telegram handles
    re.compile(r"[a-z0-9\-]+\.example(?:/[\w\-/]*)?", re.IGNORECASE),  # links
    re.compile(r"\bT[1-9A-HJ-NP-Za-km-z0]{33}\b"),  # crypto addresses
]
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d,])|(?<![\w.,])\d+(?![\d,])")
_DATE = re.compile(r"\b20\d{2}-\d{2}(?:-\d{2})?\b")
_CID = re.compile(r"CMP-\d{5}")


def _digits(s: str) -> str:
    d = re.sub(r"\D", "", s)
    return d[3:] if d.startswith("977") and len(d) >= 12 else d


def _numbers(text: str) -> set[int]:
    text = _DATE.sub(" ", text)
    for p in _ID_PATTERNS:
        text = p.sub(" ", text)
    text = _CID.sub(" ", text)
    return {int(n.replace(",", "")) for n in _NUMBER.findall(text)}


def rule_check(claim: dict[str, Any], ring: dict, facts: dict[str, dict], extracted: dict,
               complaints: dict) -> list[str]:
    problems: list[str] = []
    cited = claim["evidence"]
    if not cited:
        return ["cites no evidence"]
    cids = [c for c in cited if c.startswith("CMP-")]
    fids = [c for c in cited if re.fullmatch(r"F\d+", c)]
    members = set(ring["members"])
    if any(c not in members for c in cids):
        problems.append("cites a complaint outside this ring")
    if any(f not in facts for f in fids):
        problems.append("cites a fact that does not exist")
    if len(cids) + len(fids) < len(cited):
        problems.append("cites something that is neither a complaint nor a fact")

    sources = [facts[f]["text"] for f in fids if f in facts]
    sources += [complaints[c]["text"] for c in cids if c in complaints]
    amounts = {int(extracted[c].get("amount_lost_npr") or complaints[c]["amount_npr"] or 0)
               for c in cids if c in complaints}
    source_text = " ".join(sources)
    source_digits = {_digits(m.group()) for p in _ID_PATTERNS[:1] for m in p.finditer(source_text)}
    ring_values = {lead["value"].lower() for lead in ring["identifiers"]}

    text = claim["text"] + " " + claim.get("lead", "")
    for m in _ID_PATTERNS[0].finditer(text):
        d = _digits(m.group())
        if len(d) >= 9 and d not in ring_values and d not in source_digits:
            problems.append(f"identifier {m.group().strip()} is not a lead of this ring")
    for p in _ID_PATTERNS[1:]:
        for m in p.finditer(text):
            v = m.group().lower()
            if v not in ring_values and v not in source_text.lower():
                problems.append(f"identifier {m.group()} is not a lead of this ring")

    allowed = set().union(*(_numbers(s) for s in sources)) | amounts if sources else amounts
    for n in _numbers(text):
        if n >= 10 and n not in allowed:
            problems.append(f"number {n:,} is not in the cited evidence")
    return problems


VERIFY_SYSTEM = ("You check claims in police case reports against their cited evidence. You are strict: a claim "
                 "is supported only if the cited evidence states it or directly implies it. "
                 "Reply with one JSON object and nothing else.")


def _source_block(cited: list[str], facts: dict[str, dict], complaints: dict, limit: int = 900) -> str:
    out = []
    for c in cited:
        if c in facts:
            out.append(f"{c}: {facts[c]['text']}")
        elif c in complaints:
            t = complaints[c]["text"].replace("\n", " ")
            out.append(f"{c}: \"{t[:limit]}{' ...' if len(t) > limit else ''}\"")
    return "\n".join(out)


def llm_check(claims: list[dict[str, Any]], facts: dict[str, dict], complaints: dict, model: str,
              thinking: str | None = "low") -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Verdict per claim id: supported, partly or not_supported, with a short reason."""
    if not claims:
        return {}, {"calls": 0}
    blocks = []
    for c in claims:
        what = f"{c['lead']}: {c['text']}" if c.get("lead") else c["text"]
        blocks.append(f"CLAIM {c['id']}: {what}\nEVIDENCE:\n{_source_block(c['evidence'], facts, complaints)}")
    prompt = ("For each claim, decide whether its evidence supports it.\n"
              "- supported: every part of the claim is stated in or directly implied by the evidence\n"
              "- partly: some part is supported, some part is not\n"
              "- not_supported: the evidence does not support it\n"
              "Recommendations (next steps) are supported when the evidence shows the lead or fact they act on.\n\n"
              'Return {"verdicts": [{"id": "C1", "verdict": "supported", "reason": "under 15 words"}]}\n\n'
              + "\n\n".join(blocks))
    data, stats = llm.chat_json(model, [{"role": "system", "content": VERIFY_SYSTEM},
                                        {"role": "user", "content": prompt}],
                                thinking=thinking, max_tokens=3000)
    verdicts = {}
    for v in (data or {}).get("verdicts") or []:
        if isinstance(v, dict) and v.get("id"):
            verdict = str(v.get("verdict", "")).strip().lower().replace(" ", "_")
            verdicts[str(v["id"])] = {"verdict": verdict if verdict in ("supported", "partly", "not_supported")
                                      else "unclear", "reason": str(v.get("reason", ""))[:200]}
    stats["parsed"] = data is not None
    return verdicts, stats


def verify_report(report: dict[str, Any], ring: dict, extracted: dict, complaints: dict, model: str | None,
                  thinking: str | None = "low") -> dict[str, Any]:
    """Annotate every claim with its checks; keep only fully supported claims."""
    facts = {f["id"]: f for f in report["facts"]}
    for c in report["claims"]:
        c["rule_problems"] = rule_check(c, ring, facts, extracted, complaints)
    passed = [c for c in report["claims"] if not c["rule_problems"]]
    verdicts, stats = (llm_check(passed, facts, complaints, model, thinking) if model else
                       ({c["id"]: {"verdict": "supported", "reason": "not checked by a model"} for c in passed},
                        {"calls": 0}))
    for c in report["claims"]:
        if c["rule_problems"]:
            c["status"], c["reason"] = "removed", "; ".join(c["rule_problems"])
        else:
            v = verdicts.get(c["id"], {"verdict": "unclear", "reason": "the verifier gave no verdict"})
            c["verifier"] = v["verdict"]
            c["status"] = "kept" if v["verdict"] == "supported" else "removed"
            c["reason"] = v["reason"]
    n = len(report["claims"])
    report["verification"] = {
        "claims_written": n,
        "removed_by_rules": sum(bool(c["rule_problems"]) for c in report["claims"]),
        "removed_by_verifier": sum(c["status"] == "removed" and not c["rule_problems"] for c in report["claims"]),
        "kept": sum(c["status"] == "kept" for c in report["claims"]),
        "supported_before_checks": round(sum(c.get("verifier") == "supported" for c in report["claims"]) / n, 4) if n else None,
        "verifier": {"model": model, **stats},
    }
    return report
