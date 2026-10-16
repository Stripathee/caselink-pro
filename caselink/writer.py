"""Turns a complaint fact sheet into victim-written text.

Two writers:
- write_with_llm: Nemotron writes the complaint (the real dataset).
- write_with_template: plain sentences, no model call (offline tests, zero cost).

Every identifier in the fact sheet must appear in the text exactly as written,
otherwise our ground truth would be wrong. The LLM writer checks this, retries
once, and as a last resort appends the missing details as a final line.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from caselink import llm
from caselink.synth import SCAM_LABELS

STORY_LABELS = {
    "agency": "Recruitment agency name",
    "recruiter": "Recruiter's name",
    "country": "Country of the job",
    "job": "Job offered",
    "promised_salary_npr": "Promised monthly salary (NPR)",
    "found_via": "How the victim found the offer",
    "fees_paid": "Fees the victim paid, in order",
    "ending": "What happened in the end",
    "app": "Loan app name",
    "recovery_agent": "Recovery agent's name",
    "mechanism": "How the scam worked",
    "pressure": "How they pressured the victim",
    "impersonated": "Who the caller pretended to be",
    "officer_name": "Name the caller used",
    "pretext": "What the caller claimed",
    "demand": "What they demanded",
    "tactic": "Pressure tactic",
    "platform": "Investment platform name",
    "hook": "How the victim got involved",
    "early_profit": "Early profits",
    "blocked": "How withdrawals were blocked",
    "persona": "Name the scammer used",
    "persona_role": "Who the scammer claimed to be",
    "met_on": "Where they met",
    "gift": "What was supposedly being sent",
    "courier": "Courier company named",
    "fee": "Fee demanded",
    "item": "Item for sale",
    "seller": "Seller's name",
    "advance_reason": "Why payment was asked in advance",
}

SYSTEM_PROMPT = (
    "You write realistic, fictional cybercrime complaints in the voice of victims in Nepal. "
    "They are used to test software that helps police investigators link related cases. "
    "Every detail you are given is fictional."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LONG_NUMBER = re.compile(r"\+?\d[\d\- ]{7,}\d")


def describe_mention(m: dict[str, Any]) -> str:
    kind = m["type"]
    if kind == "phone":
        return "Phone number the scammer called or messaged from"
    if kind == "wallet":
        return f"{m.get('provider', 'Wallet')} wallet ID the money was sent to"
    if kind == "bank_account":
        return f"Bank account the money was deposited to ({m.get('bank')}, account holder {m.get('holder')})"
    if kind == "crypto_address":
        return f"{m.get('network', 'Crypto')} address the money was sent to"
    if kind == "url":
        return "Website or app link the scammer used"
    if kind == "telegram":
        return "Telegram handle or group"
    if kind == "facebook_page":
        return "Facebook page name"
    return kind


def _story_lines(story: dict[str, Any]) -> list[str]:
    lines = []
    for key, value in story.items():
        label = STORY_LABELS.get(key, key.replace("_", " ").capitalize())
        if isinstance(value, list):
            value = ", ".join(value)
        elif isinstance(value, int):
            value = f"{value:,}"
        lines.append(f"- {label}: {value}")
    return lines


def _natural_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {d:%B %Y}"


def build_prompt(c: dict[str, Any], missing_note: list[str] | None = None) -> str:
    s = c["style"]
    voice = ("a relative writing on behalf of the victim" if s["writer"] == "relative"
             else "the victim, in first person")
    lines = [
        "Write one cybercrime complaint as it would be submitted to the police.",
        "",
        "Facts:",
        f"- Scam type: {SCAM_LABELS[c['scam_type']]}",
        f"- When it happened: {_natural_date(c['incident_date'])}",
        f"- Total money lost: NPR {c['amount_npr']:,}",
        *_story_lines(c["story"]),
        "",
        "Identifiers. Copy each one into the complaint exactly as written here, character for character:",
        *[f"- {describe_mention(m)}: {m['written']}" for m in c["mentions"]],
        "",
        "Writer:",
        f"- Voice: {voice}",
        f"- Background (mention only if it fits naturally): a {s['age']}-year-old {s['role']}",
        f"- Mood: {s['emotion']}",
        f"- English level: {s['english']}",
        f"- Length: about {s['words']} words",
    ]
    if s["nepali_words"]:
        lines.append("- Use two or three romanized Nepali words or short phrases where a real person would, "
                     "such as words of address, money words or exclamations. Pick them yourself.")
    lines += [
        "",
        "Rules:",
        "- Plain text only: no title, no headings, no bullet points, no placeholders like [name].",
        "- Do not open by stating your age or job. Start the way real people do: with what happened, "
        "a request for help, or a short greeting to the police.",
        "- Write the date the way this person naturally would, not as YYYY-MM-DD.",
        "- Do not invent any other phone numbers, account numbers, wallet IDs, web addresses or handles.",
        "- Do not use full names for the victim or family members.",
        "- Do not mention that this is fictional or synthetic.",
    ]
    if missing_note:
        lines += ["", "Your previous draft left out these identifiers. Include each one exactly as written:",
                  *[f"- {w}" for w in missing_note]]
    return "\n".join(lines)


def missing_identifiers(text: str, c: dict[str, Any]) -> list[str]:
    return [m["written"] for m in c["mentions"] if m["written"] not in text]


def extra_numbers(text: str, c: dict[str, Any]) -> list[str]:
    """Long numbers in the text that are not identifiers we gave (possible hallucinations)."""
    allowed = {re.sub(r"\D", "", m["written"]) for m in c["mentions"]}
    found = []
    for match in _LONG_NUMBER.findall(text):
        digits = re.sub(r"\D", "", match)
        if len(digits) >= 9 and not any(digits in a or a in digits for a in allowed):
            found.append(match)
    return found


def _clean(text: str) -> str:
    return _THINK.sub("", text).strip()


def write_with_llm(c: dict[str, Any], model: str) -> dict[str, Any]:
    stats = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "cached_calls": 0,
             "empty": False, "retried": False, "repaired": False, "failed": False}

    def ask(prompt: str, temperature: float, max_tokens: int = 1200) -> str:
        record = llm.chat(
            model,
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            thinking="off",  # writing a complaint needs no reasoning: faster and cheaper
        )
        stats["calls"] += 1
        if record["cached"]:
            stats["cached_calls"] += 1
        else:
            stats["prompt_tokens"] += record["prompt_tokens"] or 0
            stats["completion_tokens"] += record["completion_tokens"] or 0
        return _clean(record["content"])

    text = ask(build_prompt(c), temperature=0.9)
    if not text:
        # The model used its whole budget reasoning. Give it room to finish.
        stats["empty"] = True
        text = ask(build_prompt(c), temperature=0.9, max_tokens=4000)
    if not text:
        stats["failed"] = True
        stats["extra_numbers"] = []
        return {"text": "", "stats": stats}

    missing = missing_identifiers(text, c)
    if missing:
        stats["retried"] = True
        retry = ask(build_prompt(c, missing_note=missing), temperature=0.7, max_tokens=4000 if stats["empty"] else 1200)
        if retry and len(missing_identifiers(retry, c)) < len(missing):
            text = retry
        missing = missing_identifiers(text, c)
    if missing:
        stats["repaired"] = True
        text = text.rstrip() + "\n\nDetails: " + "; ".join(missing)
    stats["extra_numbers"] = extra_numbers(text, c)
    return {"text": text, "stats": stats}


def write_with_template(c: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, model-free complaint text. Used for tests and offline runs."""
    st = c["story"]
    parts = [f"On {c['incident_date']} I was cheated and lost NPR {c['amount_npr']:,}."]
    t = c["scam_type"]
    if t == "fake_foreign_job":
        parts.append(f"I saw an offer for a {st.get('job', 'job')} job in {st.get('country', 'abroad')} "
                     f"through {st.get('found_via', 'social media')}.")
        if "agency" in st:
            parts.append(f"The agency was called {st['agency']}.")
        if "recruiter" in st:
            parts.append(f"The person handling it said his name was {st['recruiter']}.")
        if st.get("fees_paid"):
            parts.append(f"I paid for: {', '.join(st['fees_paid'])}.")
        parts.append(f"In the end they {st.get('ending', 'disappeared')}.")
    elif t == "instant_loan_app":
        parts.append(f"I installed a loan app{(' called ' + st['app']) if 'app' in st else ''} "
                     f"after {st.get('found_via', 'an ad')}. They {st.get('mechanism', '')}.")
        parts.append(f"They also {st.get('pressure', 'harassed me')}.")
    elif t == "digital_arrest":
        parts.append(f"A caller claimed to be {st.get('impersonated', 'an officer')}"
                     f"{(' named ' + st['officer_name']) if 'officer_name' in st else ''}. "
                     f"He said {st.get('pretext', '')} and demanded {st.get('demand', 'money')}. "
                     f"He {st.get('tactic', 'threatened me')}.")
    elif t == "crypto_investment":
        parts.append(f"I joined through {st.get('hook', 'an ad')}"
                     f"{(' and used a platform called ' + st['platform']) if 'platform' in st else ''}. "
                     f"It {st.get('early_profit', '')}, then {st.get('blocked', 'blocked my money')}.")
    elif t == "romance":
        who = st.get("persona", "someone")
        parts.append(f"I met {who}, who said they were {st.get('persona_role', 'abroad')}, on "
                     f"{st.get('met_on', 'social media')}. They promised {st.get('gift', 'a gift')} and a courier"
                     f"{(' called ' + st['courier']) if 'courier' in st else ''} asked for a "
                     f"{st.get('fee', 'fee')}.")
    elif t == "marketplace":
        parts.append(f"I tried to buy {st.get('item', 'an item')} online"
                     f"{(' from ' + st['seller']) if 'seller' in st else ''}. They asked for "
                     f"{st.get('advance_reason', 'an advance')} and then {st.get('ending', 'disappeared')}.")
    for m in c["mentions"]:
        parts.append(f"{describe_mention(m).split(' (')[0]}: {m['written']}.")
    text = " ".join(p for p in parts if p.strip())
    return {"text": text, "stats": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
                                    "cached_calls": 0, "empty": False, "retried": False,
                                    "repaired": False, "failed": False, "extra_numbers": []}}
