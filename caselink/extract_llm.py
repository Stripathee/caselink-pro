"""Tier 2 extraction: Nemotron reads the complaint and returns structured facts.

Rules (Tier 1) are near-perfect on numbers, accounts, links and handles, but
cannot read names: the alias a scammer used, the fake agency, app, platform
or courier, or a Facebook page mentioned without the word "Facebook". Nemotron
also labels the scam type and the amount lost.

Nothing Nemotron returns is trusted blindly: every identifier and name must
actually appear in the complaint text, or it is dropped.
"""
from __future__ import annotations

import json
import re
from typing import Any

from openai import BadRequestError

from caselink import llm
from caselink.extract_rules import Hit, extract, normalize, normalize_name, normalize_number
from caselink.synth import SCAM_LABELS, SCAM_TYPES

SYSTEM_PROMPT = (
    "You extract structured facts from cybercrime complaints for police investigators. "
    "Reply with one JSON object and nothing else."
)

ID_TYPES = ["phone", "wallet", "bank_account", "url", "telegram", "crypto_address", "facebook_page"]

_SCAM_LIST = "\n".join(f'  "{k}": {v}' for k, v in SCAM_LABELS.items())

INSTRUCTIONS = f"""Read the complaint and return this JSON object:

{{
  "scam_type": one of the keys below, or "other",
  "amount_lost_npr": total money the victim lost, as an integer, or null,
  "incident_date": "YYYY-MM-DD" if the date is stated, else null,
  "identifiers": [{{"type": ..., "value": ...}}],
  "people": [names or aliases the scammers used],
  "organisations": [agency, company, app, platform or courier names the scammers used],
  "modus_operandi": one sentence, at most 20 words, on how the scam worked
}}

Scam types:
{_SCAM_LIST}

Identifier types: phone (a number the scammer called or messaged from), wallet (a wallet ID money was
sent to, e.g. eSewa, Khalti, IME Pay), bank_account, url, telegram, crypto_address, facebook_page
(the page name).

Rules:
- Copy every identifier and name exactly as it appears in the complaint.
- Never invent anything that is not in the complaint.
- Do not list the victim or the victim's family in "people".
- Do not list real government bodies the scammers pretended to be from in "organisations".
- Facebook page names go in identifiers as facebook_page, not in organisations."""

_json_mode_supported = True


def _parse_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def ask_model(text: str, model: str, thinking: str | None = "off") -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """One extraction call. Returns (parsed JSON or None, call stats)."""
    global _json_mode_supported
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{INSTRUCTIONS}\n\nComplaint:\n\"\"\"\n{text}\n\"\"\""}]
    kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": 900, "thinking": thinking}
    if _json_mode_supported:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        record = llm.chat(model, messages, **kwargs)
    except BadRequestError:
        # This model or endpoint does not accept JSON mode: plain prompting works too.
        _json_mode_supported = False
        kwargs.pop("response_format", None)
        record = llm.chat(model, messages, **kwargs)
    data = _parse_json(record["content"])
    if data is None and not record["content"].strip():
        # Spent the budget reasoning: try again with more room.
        record = llm.chat(model, messages, **{**kwargs, "max_tokens": 3000})
        data = _parse_json(record["content"])
    stats = {"cached": record["cached"], "prompt_tokens": record["prompt_tokens"] or 0,
             "completion_tokens": record["completion_tokens"] or 0}
    return data, stats


# ---------------------------------------------------------------------------
# Validation: keep only what is really in the text
# ---------------------------------------------------------------------------

def _in_text(type_: str, value: str, text: str) -> bool:
    if not value:
        return False
    if type_ in ("phone", "wallet", "bank_account"):
        digits = normalize_number(value)
        return len(digits) >= 7 and digits in re.sub(r"\D", "", text)
    return normalize_name(value) in normalize_name(text)


def clean_llm_output(data: dict[str, Any] | None, text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"scam_type": None, "amount_lost_npr": None, "incident_date": None,
                           "identifiers": [], "people": [], "organisations": [], "modus_operandi": None,
                           "dropped": 0}
    if not data:
        return out
    st = data.get("scam_type")
    out["scam_type"] = st if st in SCAM_TYPES else ("other" if st else None)
    amount = data.get("amount_lost_npr")
    if isinstance(amount, (int, float)) and amount > 0:
        out["amount_lost_npr"] = int(amount)
    elif isinstance(amount, str) and re.sub(r"\D", "", amount):
        out["amount_lost_npr"] = int(re.sub(r"\D", "", amount))
    date = data.get("incident_date")
    if isinstance(date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        out["incident_date"] = date
    for item in data.get("identifiers") or []:
        if not isinstance(item, dict):
            continue
        t, v = item.get("type"), str(item.get("value") or "")
        if t in ID_TYPES and _in_text(t, v, text):
            out["identifiers"].append({"type": t, "value": normalize(t, v), "written": v})
        else:
            out["dropped"] += 1
    for key in ("people", "organisations"):
        for name in data.get(key) or []:
            if isinstance(name, str) and name.strip() and _in_text("name", name, text):
                out[key].append(name.strip())
            elif name:
                out["dropped"] += 1
    mo = data.get("modus_operandi")
    out["modus_operandi"] = mo.strip() if isinstance(mo, str) else None
    return out


# ---------------------------------------------------------------------------
# Merge rules and model
# ---------------------------------------------------------------------------

def merge(rule_hits: list[Hit], llm_out: dict[str, Any]) -> list[dict[str, Any]]:
    """Union of both, one entry per identifier. Rules decide the label unless unsure."""
    merged: dict[str, dict[str, Any]] = {}

    def key(t: str, v: str) -> str:
        return f"number:{v}" if t in ("phone", "wallet") else f"{t}:{v}"

    for h in rule_hits:
        merged[key(h.type, h.value)] = {**h.to_dict(), "source": "rules"}
    for item in llm_out["identifiers"]:
        k = key(item["type"], item["value"])
        if k not in merged:
            merged[k] = {**item, "confidence": 0.8, "source": "nemotron"}
        elif merged[k]["confidence"] < 0.9 and merged[k]["type"] != item["type"]:
            merged[k].update(type=item["type"], source="rules+nemotron")
    return list(merged.values())


def extract_complaint(text: str, model: str, mode: str = "cascade") -> dict[str, Any]:
    """mode: "rules" (free), "cascade" (rules + Nemotron), or "llm" (Nemotron only)."""
    hits = extract(text) if mode in ("rules", "cascade") else []
    result: dict[str, Any] = {"identifiers": [{**h.to_dict(), "source": "rules"} for h in hits],
                              "people": [], "organisations": [], "scam_type": None,
                              "amount_lost_npr": None, "incident_date": None, "modus_operandi": None,
                              "llm_called": False, "stats": {"prompt_tokens": 0, "completion_tokens": 0,
                                                             "cached": False, "dropped": 0}}
    if mode == "rules":
        return result
    data, stats = ask_model(text, model)
    cleaned = clean_llm_output(data, text)
    result.update({k: cleaned[k] for k in ("people", "organisations", "scam_type", "amount_lost_npr",
                                           "incident_date", "modus_operandi")})
    result["identifiers"] = merge(hits, cleaned) if mode == "cascade" else [
        {**i, "confidence": 0.8, "source": "nemotron"} for i in cleaned["identifiers"]]
    result["llm_called"] = True
    result["parse_failed"] = data is None
    result["stats"] = {**stats, "dropped": cleaned["dropped"]}
    return result
