"""Tier 1 extraction: free, instant, rule-based.

Finds the hard identifiers investigators link cases on: phone numbers, wallet
IDs, bank accounts, web addresses, Telegram handles, crypto addresses and
Facebook page names. Everything is normalised to a canonical form so the same
identifier written differently ("985-555-012", "+977985555012") becomes one key.

Each hit also carries a confidence. The cascade sends low-confidence complaints
to Nemotron (Tier 2) and accepts the rest as they are.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_SEP = r"[\s\-\.]?"
# Nepali mobile style: 9xx xxx xxx(x), optional +977 / 977 prefix.
PHONE_RE = re.compile(
    rf"(?<![\w+])(?:\+?977{_SEP})?9\d{{2}}{_SEP}\d{{3}}{_SEP}\d{{3,4}}(?![\w])"
)
# Bank accounts: 16 digits, plain or in groups of four; or any 11-18 digit run.
ACCOUNT_RE = re.compile(r"(?<![\w])(?:\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}|\d{11,18})(?![\w])")
URL_RE = re.compile(
    r"(?<![\w@.\-])(?:https?://)?(?:www\.)?(?:[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?\.)+"
    r"(?:example|com|net|org|np|io|app|xyz|info|co|top|site|online|link|biz)"
    r"(?:/[A-Za-z0-9_\-/]*)?",
    re.IGNORECASE,
)
TELEGRAM_RE = re.compile(r"(?<![\w.])@[A-Za-z][A-Za-z0-9_]{3,31}\b")
CRYPTO_RE = re.compile(r"(?<![\w])(?:T[1-9A-HJ-NP-Za-km-z0]{33}|0x[a-fA-F0-9]{40})(?![\w])")

# Facebook page names: a run of Title Case words next to "Facebook page".
# Only the cue words ignore case; the name itself must be Title Case.
_TITLE_RUN = r"[\"“']?((?:[A-Z][A-Za-z0-9&]*)(?:[ \t]+(?:[A-Z][A-Za-z0-9&]*|&)){1,6})[\"”']?"
_FB = r"(?i:facebook|fb)[ \t]+(?i:page)"
FB_AFTER_RE = re.compile(
    rf"(?:{_FB}|(?i:facebook)['’]s(?:[ \t]+(?i:page))?|(?i:page)[ \t]+(?i:called|named|name|titled)|"
    rf"(?i:page)[ \t]+(?i:on[ \t]+facebook)[ \t]+(?i:called|named))"
    rf"(?:[ \t]+(?i:called|named|titled|name))?[ \t]*[:\-]?[ \t]*{_TITLE_RUN}"
)
FB_BEFORE_RE = re.compile(rf"{_TITLE_RUN}[ \t]+(?:{_FB}|(?i:page[ \t]+on[ \t]+facebook))")
_NOT_NAMES = {"i", "the", "a", "an", "my", "their", "his", "her", "one", "this"}

# Words that suggest a number is where money went (wallet) or who called (phone).
WALLET_CUES = re.compile(
    r"wallet|e-?sewa|khalti|ime\s?-?pay|sent\s+(?:the\s+)?(?:money|amount|npr|rs|paisa)|transferred|"
    r"paid|payment|deposit|send\s+(?:money|npr|rs|paisa)|sent\s+npr|to\s+(?:his|their|her)\s+(?:wallet|esewa|khalti)",
    re.IGNORECASE,
)
PHONE_CUES = re.compile(
    r"call|phone|number|whatsapp|viber|imo|messag|sms|text|contact|rang|dial|mobile",
    re.IGNORECASE,
)
ACCOUNT_CUES = re.compile(r"account|a/c|bank|acc\.?\s*no", re.IGNORECASE)
# Text between two numbers that makes them one list: ", ", " and ", " or ", " / ".
LIST_JOIN = re.compile(r"[\s,;/&]*(?:and|or|,|&|/)?[\s,;/&]*\(?", re.IGNORECASE)


@dataclass
class Hit:
    type: str
    value: str  # canonical form, used as the linking key
    written: str  # exactly as it appears in the text
    start: int
    end: int
    confidence: float
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.type}:{self.value}"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "value": self.value, "written": self.written,
                "confidence": round(self.confidence, 2), **self.extra}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_number(s: str) -> str:
    digits = re.sub(r"\D", "", s)
    if digits.startswith("977") and len(digits) >= 12:
        digits = digits[3:]
    return digits


def normalize_url(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    return s.rstrip("/.")


def normalize_handle(s: str) -> str:
    return s.strip().lower()


def normalize_name(s: str) -> str:
    s = re.sub(r"[\"“”'’]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


_NAME_FILLER = {"mr", "mrs", "ms", "dr", "sir", "dai", "didi", "ji", "pvt", "ltd", "private", "limited",
                "co", "inc", "the", "app", "page"}


def name_tokens(s: str) -> list[str]:
    """Distinctive words of a name: no punctuation, titles or company suffixes."""
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _NAME_FILLER]


def name_in_text(name: str, text: str) -> bool:
    """Every distinctive word of the name appears in the text ('Mr. Adhikari' ~ 'Mr Adhikari')."""
    tokens = name_tokens(name)
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return bool(tokens) and all(t in words for t in tokens)


def normalize(type_: str, written: str) -> str:
    if type_ in ("phone", "wallet", "bank_account", "number"):
        return normalize_number(written)
    if type_ == "url":
        return normalize_url(written)
    if type_ == "telegram":
        return normalize_handle(written)
    if type_ == "facebook_page":
        return normalize_name(written)
    return written.strip()


# ---------------------------------------------------------------------------
# Phone or wallet? Read the words around the number.
# ---------------------------------------------------------------------------

def _nearest(pattern: re.Pattern, text: str, start: int, end: int, before: int = 60, after: int = 30) -> float:
    """Distance (in characters) from the number to the nearest cue word, or inf."""
    best = float("inf")
    left = text[max(0, start - before):start]
    for m in pattern.finditer(left):
        best = min(best, len(left) - m.end())
    right = text[end:end + after]
    for m in pattern.finditer(right):
        best = min(best, m.start() + 25)  # cues after the number count a little less
    return best


def classify_number(text: str, start: int, end: int) -> tuple[str, float, str | None]:
    """Return (type, confidence, wallet provider)."""
    window = text[max(0, start - 60):start].lower()
    provider = None
    for name, pat in (("eSewa", r"e-?sewa"), ("Khalti", r"khalti"), ("IME Pay", r"ime\s?-?pay")):
        if re.search(pat, window):
            provider = name
    d_wallet = _nearest(WALLET_CUES, text, start, end)
    d_phone = _nearest(PHONE_CUES, text, start, end)
    if d_wallet == d_phone == float("inf"):
        return "phone", 0.5, None
    if d_wallet < d_phone:
        margin = d_phone - d_wallet
        return "wallet", 0.95 if margin > 15 else 0.7, provider
    margin = d_wallet - d_phone
    return "phone", 0.95 if margin > 15 else 0.7, None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _overlaps(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    return any(start < e and s < end for s, e in taken)


def extract(text: str) -> list[Hit]:
    """All identifiers found in one complaint, in order of appearance."""
    hits: list[Hit] = []
    taken: list[tuple[int, int]] = []

    def add(hit: Hit) -> None:
        hits.append(hit)
        taken.append((hit.start, hit.end))

    for m in CRYPTO_RE.finditer(text):
        add(Hit("crypto_address", m.group(), m.group(), m.start(), m.end(), 0.95))

    for m in URL_RE.finditer(text):
        if _overlaps(m.start(), m.end(), taken):
            continue
        written = m.group().rstrip(".")
        add(Hit("url", normalize_url(written), written, m.start(), m.start() + len(written), 0.9))

    for m in TELEGRAM_RE.finditer(text):
        if not _overlaps(m.start(), m.end(), taken):
            add(Hit("telegram", normalize_handle(m.group()), m.group(), m.start(), m.end(), 0.9))

    # Accounts before phones, so a 16-digit account never splits into phone numbers.
    for m in ACCOUNT_RE.finditer(text):
        if _overlaps(m.start(), m.end(), taken):
            continue
        digits = normalize_number(m.group())
        if PHONE_RE.fullmatch(m.group()):  # e.g. +977 followed by a mobile number
            continue
        near_account = _nearest(ACCOUNT_CUES, text, m.start(), m.end()) < 60
        add(Hit("bank_account", digits, m.group(), m.start(), m.end(), 0.95 if near_account else 0.6))

    previous: Hit | None = None
    for m in PHONE_RE.finditer(text):
        if _overlaps(m.start(), m.end(), taken):
            continue
        between = text[previous.end:m.start()] if previous else ""
        if previous and LIST_JOIN.fullmatch(between):
            # "from 983555292 and 970555886": the second number is the same kind as the first.
            kind, conf, provider = previous.type, previous.confidence, previous.extra.get("provider")
        else:
            kind, conf, provider = classify_number(text, m.start(), m.end())
        extra = {"provider": provider} if kind == "wallet" and provider else {}
        hit = Hit(kind, normalize_number(m.group()), m.group(), m.start(), m.end(), conf, extra)
        add(hit)
        previous = hit

    for pattern in (FB_AFTER_RE, FB_BEFORE_RE):
        for m in pattern.finditer(text):
            name = m.group(1)
            if _overlaps(m.start(1), m.end(1), taken) or name.split()[0].lower() in _NOT_NAMES:
                continue
            add(Hit("facebook_page", normalize_name(name), name, m.start(1), m.end(1), 0.85))

    # One hit per identifier per complaint, keeping the most confident reading.
    best: dict[str, Hit] = {}
    for h in sorted(hits, key=lambda h: h.start):
        k = h.value if h.type in ("phone", "wallet") else h.key()
        if k not in best or h.confidence > best[k].confidence:
            best[k] = h
    return sorted(best.values(), key=lambda h: h.start)


def needs_llm(hits: list[Hit], text: str) -> bool:
    """Should Tier 2 (Nemotron) look at this complaint?"""
    if not hits:
        return True
    if any(h.confidence < 0.75 for h in hits):
        return True
    # A Facebook page is mentioned but we could not read its name.
    mentions_page = re.search(r"(?:facebook|fb)\s+page", text, re.IGNORECASE)
    return bool(mentions_page and not any(h.type == "facebook_page" for h in hits))
