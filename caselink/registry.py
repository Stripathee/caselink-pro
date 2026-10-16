"""Known legitimate entities: never treated as scammer names.

Scammers pay through real wallets and banks and pretend to be real agencies,
so these names appear in almost every complaint without being leads. In
production this list would be loaded from official registers (for example
Nepal Rastra Bank's list of licensed banks and payment providers); here it
holds the real providers and agencies plus the invented banks of the
synthetic world.
"""
from __future__ import annotations

import re

from caselink.synth import BANKS

PAYMENT_PROVIDERS = ["eSewa", "Khalti", "IME Pay", "Fonepay", "ConnectIPS", "mobile banking"]

GOVERNMENT_BODIES = [
    "Nepal Police", "Cyber Bureau", "Nepal Police Cyber Bureau", "Cyber Crime Bureau",
    "Central Investigation Bureau", "CIB", "Department of Customs", "Customs Office", "Customs",
    "Tribhuvan International Airport", "Nepal Telecom", "Ncell", "Nepal Rastra Bank", "Interpol",
    "Department of Foreign Employment", "Supreme Court", "District Court",
]

PLATFORMS = ["Facebook", "Facebook Marketplace", "Instagram", "TikTok", "WhatsApp", "Viber", "Telegram",
             "IMO", "YouTube", "Messenger", "Google", "SMS"]

# Words that describe a role rather than name a person.
GENERIC_WORDS = {
    "mr", "mrs", "ms", "miss", "dr", "sir", "madam", "dai", "didi", "bhai", "ji", "sathi",
    "inspector", "officer", "dsp", "sp", "police", "agent", "recruiter", "seller", "buyer", "caller",
    "scammer", "man", "woman", "lady", "person", "someone", "businessman", "businesswoman",
    "doctor", "engineer", "army", "soldier", "captain", "manager", "customer", "support", "team",
    "canadian", "american", "british", "uk", "us", "usa", "foreign", "foreigner", "indian",
    "telecom", "security", "customs", "bank", "staff", "executive", "representative",
    # Romanized Nepali courtesy and everyday words that are not names.
    "namaste", "namaskar", "dhanyabad", "dhanyabaad", "dhanyawad", "dhanbad", "hajur", "bahini",
    "aama", "buwa", "paisa", "ramro", "garnuhos", "sahayog", "kripaya",
}

_LEGIT = {re.sub(r"[^a-z0-9]", "", n.lower()) for n in PAYMENT_PROVIDERS + GOVERNMENT_BODIES + PLATFORMS + BANKS}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_legitimate_entity(name: str) -> bool:
    """A real provider, bank, platform or government body (or a near copy of one)."""
    k = _key(name)
    if not k:
        return True
    return k in _LEGIT or any(k == x or (len(x) >= 5 and x in k and len(k) - len(x) <= 6) for x in _LEGIT)


def is_generic_role(name: str) -> bool:
    """'officer', 'telecom security officer', 'Canadian businesswoman': no actual name inside."""
    tokens = re.findall(r"[A-Za-z]+", name)
    proper = [t for t in tokens
              if t[0].isupper() and t.lower() not in GENERIC_WORDS and _key(t) not in _LEGIT]
    return not proper  # e.g. "CIB officer": an agency plus a job title, no person
