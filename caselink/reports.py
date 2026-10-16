"""Case reports: Nemotron Ultra writes, code and Nemotron Super check.

Division of labour, so the report can be trusted:
- Python computes every number (victims, losses, dates, how often each lead
  appears, when each phone was in use). These become numbered facts F1, F2...
- Nemotron Ultra reads the facts and a set of the ring's complaints and writes
  the investigative narrative: how the scam works, the strongest leads, and
  concrete next steps. Every claim must cite complaint IDs or fact IDs.
- The verifier (verify.py) removes any claim whose citations, identifiers or
  numbers do not check out, then asks Nemotron Super whether the cited
  complaints really support each remaining claim.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from caselink import llm
from caselink.synth import SCAM_LABELS

MAX_EVIDENCE = 40
EXCERPT_CHARS = 700

_TYPE_WORDS = {"number": "phone/wallet", "phone": "phone", "wallet": "wallet", "bank_account": "bank account",
               "url": "link", "telegram": "Telegram", "crypto_address": "crypto address",
               "facebook_page": "Facebook page"}


def load_inputs(rings_path: str | Path = "data/rings/rings.json",
                extracted_path: str | Path = "data/extracted/cascade_nano.jsonl",
                data: str | Path = "data/synthetic") -> tuple[dict, dict[str, dict], dict[str, dict]]:
    rings = json.loads(Path(rings_path).read_text(encoding="utf-8"))
    extracted = {}
    with open(extracted_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            extracted[r["complaint_id"]] = r
    complaints = {}
    with open(Path(data) / "complaints.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            complaints[r["complaint_id"]] = r
    return rings, extracted, complaints


def _date(cid: str, extracted: dict, complaints: dict) -> str:
    return extracted[cid].get("incident_date") or complaints[cid]["filed_date"]


def _amount(cid: str, extracted: dict, complaints: dict) -> int:
    return extracted[cid].get("amount_lost_npr") or complaints[cid]["amount_npr"] or 0


def build_facts(ring: dict, extracted: dict, complaints: dict) -> list[dict[str, Any]]:
    """Every number in the report, computed here and given an ID."""
    ms = ring["members"]
    repeat = {b for _, b in ring["possible_duplicates"]}
    counted = [m for m in ms if m not in repeat]
    amounts = sorted(_amount(m, extracted, complaints) for m in counted)
    facts: list[dict[str, Any]] = []

    def add(text: str, **data: Any) -> None:
        facts.append({"id": f"F{len(facts) + 1}", "text": text, **data})

    add(f"{ring['victims']} victims across {ring['complaints']} complaints; "
        f"{len(ring['possible_duplicates'])} possible repeat filings by the same victim.",
        victims=ring["victims"], complaints=ring["complaints"])
    add(f"Total reported loss NPR {ring['total_loss_npr']:,} (repeat filings counted once); "
        f"median loss per victim NPR {amounts[len(amounts) // 2]:,}; largest single loss NPR {amounts[-1]:,}.",
        total=ring["total_loss_npr"], median=amounts[len(amounts) // 2], largest=amounts[-1])
    add(f"Active from {ring['active_from']} to {ring['active_to']} (incident dates).")
    share = int(round((ring["scam_type_share"] or 0) * 100))
    add(f"Scam type: {SCAM_LABELS.get(ring['scam_type'], ring['scam_type'])} ({share}% of complaints).")

    months = Counter(_date(m, extracted, complaints)[:7] for m in ms)
    add("Complaints by month: " + ", ".join(f"{k}: {v}" for k, v in sorted(months.items())) + ".",
        by_month=dict(sorted(months.items())))
    districts = Counter(complaints[m]["district"] for m in ms)
    add("Victims' districts: " + ", ".join(f"{d} {n}" for d, n in districts.most_common(6)) + ".")

    # When each lead was in use: shows number rotation over time.
    seen = defaultdict(list)
    for m in ms:
        for i in extracted[m]["identifiers"]:
            t = "number" if i["type"] in ("phone", "wallet") else i["type"]
            seen[f"{t}:{i['value']}"].append(_date(m, extracted, complaints))
    kinds = Counter(_TYPE_WORDS.get(x["type"], x["type"]) for x in ring["identifiers"])
    add("Distinct leads: " + ", ".join(f"{n} {k}{'s' if n > 1 else ''}" for k, n in kinds.most_common())
        + f" ({len(ring['identifiers'])} in total).", lead_counts=dict(kinds))
    for lead in ring["identifiers"][:12]:
        dates = sorted(seen.get(lead["key"], []))
        kind = _TYPE_WORDS.get(lead["type"], lead["type"])
        span = f", first seen {dates[0]}, last seen {dates[-1]}" if dates else ""
        add(f"Lead: {kind} {lead['value']} appears in {lead['complaints']} complaints{span}.",
            lead=lead["value"], lead_type=lead["type"], count=lead["complaints"])
    for n in ring["names"][:6]:
        pct = round(100 * n["mentions"] / ring["complaints"])
        add(f"Name used: \"{n['name']}\" appears in {n['mentions']} of {ring['complaints']} complaints ({pct}%).",
            name=n["name"])
    for mule in ring["shared_mules"]:
        add(f"Shared mule: {mule['identifier'].split(':', 1)[1]} is also used by {', '.join(mule['also_used_by'])}.")
    if ring["linked_by_story"]:
        add(f"{len(ring['linked_by_story'])} complaints share no identifier with the ring and were linked "
            f"by the link predictor (probabilities "
            + ", ".join(f"{x['complaint']} {x['probability']:.2f}" for x in ring["linked_by_story"]) + ").")
    return facts


def select_evidence(ring: dict, extracted: dict, complaints: dict, limit: int = MAX_EVIDENCE,
                    seed: int = 0) -> list[str]:
    """A small set of complaints that still covers every lead, every name and the whole period."""
    ms = sorted(ring["members"], key=lambda m: _date(m, extracted, complaints))
    chosen: list[str] = []

    def take(m: str) -> None:
        if m not in chosen and len(chosen) < limit:
            chosen.append(m)

    take(ms[0])
    take(ms[-1])
    for x in ring["linked_by_story"]:
        take(x["complaint"])
    for m in sorted(ms, key=lambda m: -_amount(m, extracted, complaints))[:3]:
        take(m)
    # Greedy cover: every lead and name appears in at least one chosen complaint.
    wanted = {lead["key"] for lead in ring["identifiers"]} | {n["name"].lower() for n in ring["names"]}

    def covers(m: str) -> set[str]:
        keys = {("number" if i["type"] in ("phone", "wallet") else i["type"]) + ":" + i["value"]
                for i in extracted[m]["identifiers"]}
        names = {n.lower() for n in extracted[m].get("people", []) + extracted[m].get("organisations", [])}
        return (keys | names) & wanted

    for m in chosen:
        wanted -= covers(m)
    while wanted and len(chosen) < limit:
        best = max(ms, key=lambda m: len(covers(m) & wanted))
        gain = covers(best) & wanted
        if not gain:
            break
        take(best)
        wanted -= gain
    rest = [m for m in ms if m not in chosen]
    random.Random(seed).shuffle(rest)
    for m in sorted(rest[: max(0, limit - len(chosen))], key=lambda m: _date(m, extracted, complaints)):
        take(m)
    return sorted(chosen, key=lambda m: _date(m, extracted, complaints))


def evidence_line(cid: str, extracted: dict, complaints: dict) -> str:
    e, c = extracted[cid], complaints[cid]
    leads = "; ".join(f"{_TYPE_WORDS.get(i['type'], i['type'])} {i['value']}" for i in e["identifiers"]) or "none"
    text = c["text"].replace("\n", " ")
    if len(text) > EXCERPT_CHARS:
        text = text[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " ..."
    return (f"[{cid}] {_date(cid, extracted, complaints)}, {c['district']}, NPR {_amount(cid, extracted, complaints):,}. "
            f"Leads: {leads}.\n  Victim wrote: \"{text}\"")


SYSTEM_PROMPT = (
    "You are a senior cybercrime investigator writing a case report for police colleagues. "
    "You are precise, you never speculate beyond the evidence, and every claim you make cites its evidence. "
    "Reply with one JSON object and nothing else."
)

INSTRUCTIONS = """Write a case report on this suspected scam ring.

Use only the facts and complaints below. Every item must cite evidence IDs: complaint IDs (e.g. "CMP-00123")
and/or fact IDs (e.g. "F3"). Do not do arithmetic: when you need a number, cite the fact that states it.
Copy phone numbers, account numbers and other identifiers exactly as they appear.

Return this JSON object:
{
  "title": "short descriptive title of the ring",
  "summary": {"text": "2-3 sentences: who is targeted, how, and the scale", "evidence": [...]},
  "modus_operandi": [{"text": "one step of how the scam works", "evidence": [...]}],  // 3 to 6 steps, in order
  "key_leads": [{"lead": "the identifier or name", "why": "why it matters for the investigation", "evidence": [...]}],  // 3 to 6
  "recommended_actions": [{"text": "a concrete next step, e.g. which records to request from which telecom, bank or wallet provider", "evidence": [...]}],  // 3 to 6
  "open_questions": ["what the evidence does not yet establish"]  // 1 to 3
}"""


def build_prompt(ring: dict, facts: list[dict], evidence: list[str], extracted: dict, complaints: dict) -> str:
    lines = [INSTRUCTIONS, "", f"RING {ring['ring']}", "", "FACTS (computed from all complaints in the ring):"]
    lines += [f"{f['id']}: {f['text']}" for f in facts]
    lines += ["", f"COMPLAINTS ({len(evidence)} of {ring['complaints']}, chosen to cover every lead and the whole period):"]
    lines += [evidence_line(cid, extracted, complaints) for cid in evidence]
    return "\n".join(lines)


def _as_claims(items: Any, text_key: str = "text") -> list[dict[str, Any]]:
    out = []
    if isinstance(items, dict):
        items = [items]
    for it in items or []:
        if isinstance(it, str):
            out.append({"text": it, "evidence": []})
        elif isinstance(it, dict):
            ev = it.get("evidence") or []
            ev = [ev] if isinstance(ev, str) else [str(x).strip() for x in ev if isinstance(x, (str, int))]
            text = it.get(text_key) or it.get("text") or ""
            claim = {"text": str(text).strip(), "evidence": ev}
            if "lead" in it:
                claim["lead"] = str(it["lead"]).strip()
                claim["text"] = str(it.get("why") or text).strip()
            out.append(claim)
    return [c for c in out if c["text"]]


def write_report(ring: dict, extracted: dict, complaints: dict, model: str,
                 thinking: str | None = "low") -> dict[str, Any]:
    """Ask the model for the report and normalise its answer into claims."""
    facts = build_facts(ring, extracted, complaints)
    evidence = select_evidence(ring, extracted, complaints)
    prompt = build_prompt(ring, facts, evidence, extracted, complaints)
    data, stats = llm.chat_json(model, [{"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": prompt}],
                                thinking=thinking, max_tokens=10000, temperature=0.2)  # room to reason, then write
    data = data or {}
    claims: list[dict[str, Any]] = []
    for section, key in (("summary", "summary"), ("modus_operandi", "modus_operandi"),
                         ("key_leads", "key_leads"), ("recommended_actions", "recommended_actions")):
        for c in _as_claims(data.get(key)):
            claims.append({"id": f"C{len(claims) + 1}", "section": section, **c})
    questions = [q for q in (data.get("open_questions") or []) if isinstance(q, str)]
    return {"ring": ring["ring"], "title": str(data.get("title") or f"Scam ring {ring['ring']}"),
            "facts": facts, "evidence_shown": evidence, "claims": claims, "open_questions": questions,
            "writer": {"model": model, "parsed": bool(data), **stats}}


# ---------------------------------------------------------------------------
# Markdown for investigators
# ---------------------------------------------------------------------------

def _cite(ev: list[str]) -> str:
    return " " + " ".join(f"[{e}]" for e in ev) if ev else ""


def render_markdown(report: dict[str, Any], ring: dict) -> str:
    kept = [c for c in report["claims"] if c["status"] == "kept"]
    by = defaultdict(list)
    for c in kept:
        by[c["section"]].append(c)
    v = report["verification"]
    lines = [
        f"# {ring['ring']}: {report['title']}",
        "",
        "> Decision support only. Generated from synthetic complaints. Every statement below cites the",
        "> complaints (CMP-...) or computed facts (F...) it rests on; investigators make every decision.",
        "",
        "| Victims | Complaints | Total loss (NPR) | Active | Scam type |",
        "| --- | --- | --- | --- | --- |",
        f"| {ring['victims']} | {ring['complaints']} | {ring['total_loss_npr']:,} | {ring['active_from']} to "
        f"{ring['active_to']} | {SCAM_LABELS.get(ring['scam_type'], ring['scam_type'])} |",
        "",
    ]
    if by["summary"]:
        lines += ["## Summary", ""] + [c["text"] + _cite(c["evidence"]) for c in by["summary"]] + [""]
    if by["modus_operandi"]:
        lines += ["## How the scam works", ""]
        lines += [f"{i}. {c['text']}{_cite(c['evidence'])}" for i, c in enumerate(by["modus_operandi"], 1)] + [""]
    if by["key_leads"]:
        lines += ["## Key leads", "", "| Lead | Why it matters | Evidence |", "| --- | --- | --- |"]
        lines += [f"| {c.get('lead', '')} | {c['text']} | {' '.join(c['evidence'])} |" for c in by["key_leads"]]
        lines += [""]
    lines += ["## All leads", "", "| Type | Value | Complaints |", "| --- | --- | --- |"]
    lines += [f"| {_TYPE_WORDS.get(x['type'], x['type'])} | {x['value']} | {x['complaints']} |"
              for x in ring["identifiers"]] + [""]
    if ring["shared_mules"]:
        lines += ["**Shared mule accounts:** " + "; ".join(
            f"{m['identifier'].split(':', 1)[1]} also used by {', '.join(m['also_used_by'])}"
            for m in ring["shared_mules"]), ""]
    if by["recommended_actions"]:
        lines += ["## Recommended next steps", ""]
        lines += [f"{i}. {c['text']}{_cite(c['evidence'])}" for i, c in enumerate(by["recommended_actions"], 1)]
        lines += [""]
    if report["open_questions"]:
        lines += ["## Open questions", ""] + [f"- {q}" for q in report["open_questions"]] + [""]
    lines += ["## Computed facts", ""] + [f"- **{f['id']}** {f['text']}" for f in report["facts"]] + [""]
    if ring["possible_duplicates"]:
        lines += ["**Possible repeat filings (counted once):** "
                  + ", ".join(f"{a} / {b}" for a, b in ring["possible_duplicates"]), ""]
    removed = [c for c in report["claims"] if c["status"] == "removed"]
    lines += ["## Verification", "",
              f"{v['claims_written']} claims written by {report['writer']['model']}; "
              f"{v['removed_by_rules']} removed by the evidence rules, {v['removed_by_verifier']} removed by "
              f"the verifier ({v['verifier'].get('model') or 'none'}); {v['kept']} kept.", ""]
    if removed:
        lines += ["| Removed claim | Reason |", "| --- | --- |"]
        lines += [f"| {c.get('lead', '') + ': ' if c.get('lead') else ''}{c['text']} | {c['reason']} |" for c in removed]
        lines += [""]
    lines += [f"Members: {', '.join(ring['members'])}", ""]
    return "\n".join(lines)
