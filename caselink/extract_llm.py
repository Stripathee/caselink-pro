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

from datetime import date, timedelta

from caselink import llm
from caselink.extract_rules import (ACCOUNT_RE, CRYPTO_RE, PHONE_RE, TELEGRAM_RE, URL_RE, Hit, extract,
                                    name_in_text, normalize, normalize_name, normalize_number)
from caselink.registry import is_generic_role, is_legitimate_entity
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
  "people": [names or aliases the scammers used, and the holder name of any bank account money went to],
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
    if type_ in ("name", "facebook_page"):
        return name_in_text(value, text)
    return normalize_name(value) in normalize_name(text)


_FORMATS = {
    "phone": PHONE_RE, "wallet": PHONE_RE, "bank_account": ACCOUNT_RE,
    "url": URL_RE, "telegram": TELEGRAM_RE, "crypto_address": CRYPTO_RE,
}


def valid_format(type_: str, value: str) -> bool:
    """An identifier must look like its type: a Telegram handle starts with @, a URL has a domain."""
    value = value.strip()
    if type_ in _FORMATS:
        return bool(_FORMATS[type_].fullmatch(value))
    if type_ == "facebook_page":
        words = value.split()
        return 1 <= len(words) <= 8 and not re.search(r"\d{5,}|@|https?://", value) and not is_legitimate_entity(value)
    return False


def plausible_date(value: Any, filed_date: str | None) -> str | None:
    """Keep a date only if it is real and falls in the year before the complaint was filed.

    Victims often omit the year or use the Nepali calendar (2083 BS is 2026 AD);
    a model then guesses a year. A wrong date would corrupt the ring timeline,
    so an implausible date is dropped and the filing date is used instead.
    """
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        d = date.fromisoformat(value)
    except ValueError:
        return None
    if filed_date:
        filed = date.fromisoformat(filed_date)
        if not (filed - timedelta(days=366) <= d <= filed):
            return None
    return value


def clean_name(name: Any, text: str, identifier_values: set[str]) -> str | None:
    """A scammer alias or fake organisation, or None if it is not a lead."""
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    if not _in_text("name", name, text):
        return None  # not in the complaint: invented
    if is_legitimate_entity(name) or is_generic_role(name):
        return None  # a real bank, wallet, platform or agency, or just a job title
    if URL_RE.fullmatch(name) or normalize_name(name) in identifier_values:
        return None  # already captured as an identifier (link or page)
    return name


def clean_llm_output(data: dict[str, Any] | None, text: str, filed_date: str | None = None) -> dict[str, Any]:
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
    out["incident_date"] = plausible_date(data.get("incident_date"), filed_date)

    named = {normalize_name(n) for key in ("people", "organisations")
             for n in (data.get(key) or []) if isinstance(n, str)}
    for item in data.get("identifiers") or []:
        if not isinstance(item, dict):
            continue
        t, v = item.get("type"), str(item.get("value") or "").strip()
        ok = t in ID_TYPES and _in_text(t, v, text) and valid_format(t, v)
        if t == "facebook_page" and normalize_name(v) in named:
            ok = False  # the model itself calls this a person or company, not a page
        if ok:
            out["identifiers"].append({"type": t, "value": normalize(t, v), "written": v})
        else:
            out["dropped"] += 1

    id_values = {i["value"] for i in out["identifiers"]}
    seen: set[str] = set()
    for key in ("people", "organisations"):
        for raw in data.get(key) or []:
            name = clean_name(raw, text, id_values)
            if name and normalize_name(name) not in seen:
                out[key].append(name)
                seen.add(normalize_name(name))
            elif raw:
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
    numeric = ("phone", "wallet", "bank_account")

    def key(t: str, v: str) -> str:
        return f"number:{v}" if t in numeric else f"{t}:{v}"

    for h in rule_hits:
        merged[key(h.type, h.value)] = {**h.to_dict(), "source": "rules"}
    for item in llm_out["identifiers"]:
        k = key(item["type"], item["value"])
        if k in merged:
            m = merged[k]
            # Rules are unsure whether a number is a phone or a wallet: let the model decide.
            if (m["confidence"] < 0.9 and m["type"] in ("phone", "wallet")
                    and item["type"] in ("phone", "wallet") and m["type"] != item["type"]):
                m.update(type=item["type"], source="rules+nemotron")
            continue
        if item["type"] == "facebook_page":
            # The same page read slightly differently: the model judges where a name ends
            # better than a pattern does ("Dravelle Second Hand Market Please").
            same = [rk for rk, m in merged.items() if m["type"] == "facebook_page"
                    and (item["value"] in m["value"] or m["value"] in item["value"])]
            for rk in same:
                del merged[rk]
            merged[k] = {**item, "confidence": 0.9, "source": "rules+nemotron" if same else "nemotron"}
            continue
        merged[k] = {**item, "confidence": 0.8, "source": "nemotron"}
    return list(merged.values())


def extract_complaint(text: str, model: str, mode: str = "cascade", filed_date: str | None = None) -> dict[str, Any]:
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
    cleaned = clean_llm_output(data, text, filed_date)
    result.update({k: cleaned[k] for k in ("people", "organisations", "scam_type", "amount_lost_npr",
                                           "incident_date", "modus_operandi")})
    result["identifiers"] = merge(hits, cleaned) if mode == "cascade" else [
        {**i, "confidence": 0.8, "source": "nemotron"} for i in cleaned["identifiers"]]
    result["llm_called"] = True
    result["parse_failed"] = data is None
    result["raw"] = data  # the model's answer before validation, kept for error analysis
    result["stats"] = {**stats, "dropped": cleaned["dropped"]}
    return result
