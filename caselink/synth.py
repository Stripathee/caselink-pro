"""Synthetic cybercrime complaint world.

This module builds the hidden truth: scam rings, their identifiers, and one
fact sheet per complaint. No model is called here, so it is free, instant and
fully reproducible from a seed. scripts/generate_data.py then asks Nemotron to
turn each fact sheet into victim-written text.

Every identifier is fictional by construction, so none can point at a real
person or business:
- Phone numbers and wallet IDs have 9 digits with a 555 block. Real Nepali
  mobile numbers have 10 digits, so these can never be dialled.
- Web addresses use the reserved .example top-level domain (RFC 2606).
- Crypto addresses contain the character 0, which base58 never uses, so they
  can never be valid addresses.
- Bank, agency, app and platform names are invented words; account numbers
  are random.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

WINDOW_START = date(2026, 1, 1)
WINDOW_END = date(2026, 9, 30)

SCAM_TYPES = [
    "fake_foreign_job",
    "instant_loan_app",
    "digital_arrest",
    "crypto_investment",
    "romance",
    "marketplace",
]

SCAM_LABELS = {
    "fake_foreign_job": "Fake foreign employment offer",
    "instant_loan_app": "Instant loan app fraud",
    "digital_arrest": "Police or official impersonation ('digital arrest')",
    "crypto_investment": "Crypto or investment fraud",
    "romance": "Romance scam",
    "marketplace": "Online marketplace fraud",
}

# Range of money lost per victim, in NPR.
LOSS_RANGES = {
    "fake_foreign_job": (40_000, 600_000),
    "instant_loan_app": (5_000, 80_000),
    "digital_arrest": (50_000, 1_500_000),
    "crypto_investment": (20_000, 2_000_000),
    "romance": (30_000, 800_000),
    "marketplace": (5_000, 150_000),
}

# How victims of each scam type usually paid.
PAYMENT_MIX = {
    "fake_foreign_job": {"wallet": 0.5, "bank_account": 0.5},
    "instant_loan_app": {"wallet": 0.8, "bank_account": 0.2},
    "digital_arrest": {"wallet": 0.3, "bank_account": 0.7},
    "crypto_investment": {"crypto_address": 0.5, "bank_account": 0.25, "wallet": 0.25},
    "romance": {"wallet": 0.4, "bank_account": 0.6},
    "marketplace": {"wallet": 0.7, "bank_account": 0.3},
}

# Online identifiers each scam type uses, with the chance a victim mentions one.
WEB_MIX = {
    "fake_foreign_job": {"facebook_page": 0.5},
    "instant_loan_app": {"url": 0.6},
    "digital_arrest": {},
    "crypto_investment": {"url": 0.6, "telegram": 0.5},
    "romance": {},
    "marketplace": {"facebook_page": 0.7},
}

# Chance a victim mentions the number the scammer called or messaged from.
PHONE_MENTION = {
    "fake_foreign_job": 0.9,
    "instant_loan_app": 0.85,
    "digital_arrest": 0.95,
    "crypto_investment": 0.5,
    "romance": 0.5,
    "marketplace": 0.85,
}

PAYMENT_MENTION = 0.8
NAME_MENTION = 0.7

WALLET_PROVIDERS = ["eSewa", "Khalti", "IME Pay"]

# Invented bank names: not real Nepali banks.
BANKS = ["Zentaro Bank", "Kalvex Bank", "Morivan Finance", "Trelisa Bikas Bank", "Ovanta Bank"]

DISTRICTS = {
    "Kathmandu": 30, "Lalitpur": 8, "Bhaktapur": 5, "Kaski": 8, "Chitwan": 7,
    "Morang": 6, "Sunsari": 5, "Rupandehi": 7, "Jhapa": 6, "Dhanusha": 4,
    "Banke": 4, "Kailali": 5, "Parsa": 5,
}
CHANNELS = {"online complaint portal": 50, "walk-in at police office": 35, "hotline call": 15}

FIRST_NAMES = ["Bikash", "Suresh", "Ramesh", "Anil", "Prakash", "Sunil", "Kiran", "Deepak",
               "Rajan", "Nabin", "Sagar", "Manoj", "Pooja", "Anita", "Sabina", "Rekha"]
SURNAMES = ["Sharma", "Thapa", "Gurung", "Adhikari", "Karki", "Shrestha", "Rai", "Bhandari"]
WESTERN_PERSONAS = ["James Walker", "Daniel Morris", "Michael Grant", "Robert Hayes",
                    "David Collins", "Steven Price", "Emily Carter", "Sophia Bennett"]

# Coined words for organisation names, so no name matches a real business.
COINED = ["Zentara", "Velmora", "Quorix", "Talvane", "Marivo", "Orlenta", "Brivax", "Kestra",
          "Novamir", "Dravelle", "Synthra", "Lumora", "Pexalon", "Ravenna", "Tovaris", "Ulmera",
          "Vexium", "Corvana", "Halvex", "Sorenta"]

BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

VICTIM_ROLES = ["student", "shopkeeper", "teacher", "driver", "nurse", "farmer", "returnee migrant worker",
                "office assistant", "housewife", "retired government employee", "hotel worker", "small business owner"]
EMOTIONS = ["angry", "desperate", "embarrassed", "calm and formal", "confused", "worried about family"]
ENGLISH_LEVELS = ["basic, with grammar mistakes", "simple", "fluent"]


@dataclass
class Ring:
    ring_id: str | None  # None for one-off (noise) scams
    scam_type: str
    script: dict[str, Any]
    phones: list[str]
    payments: list[dict[str, Any]]  # each: {type, value, ...details}
    web: list[dict[str, Any]]  # each: {type, value}
    start: date
    end: date
    loss_range: tuple[int, int]
    names: list[dict[str, str]] = field(default_factory=list)  # aliases and organisation names


class IdFactory:
    """Makes unique, fictional identifiers."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: set[str] = set()

    def _unique(self, make) -> str:
        for _ in range(10_000):
            value = make()
            if value not in self.used:
                self.used.add(value)
                return value
        raise RuntimeError("Ran out of unique identifiers.")

    def phone(self) -> str:
        r = self.rng
        return self._unique(lambda: f"{r.choice(['96', '97', '98'])}{r.randint(0, 9)}555{r.randint(0, 999):03d}")

    def wallet(self) -> dict[str, Any]:
        return {"type": "wallet", "value": self.phone(), "provider": self.rng.choice(WALLET_PROVIDERS)}

    def bank_account(self) -> dict[str, Any]:
        r = self.rng
        number = self._unique(lambda: str(r.randint(1, 9)) + "".join(str(r.randint(0, 9)) for _ in range(15)))
        holder = f"{r.choice(FIRST_NAMES)} {r.choice(SURNAMES)[0]}."
        return {"type": "bank_account", "value": number, "bank": r.choice(BANKS), "holder": holder}

    def crypto_address(self) -> dict[str, Any]:
        r = self.rng

        def make() -> str:
            body = [r.choice(BASE58) for _ in range(33)]
            body[r.randint(4, 28)] = "0"  # never valid base58
            return "T" + "".join(body)

        return {"type": "crypto_address", "value": self._unique(make), "network": "USDT (TRC20)"}

    def org_name(self, suffixes: list[str]) -> str:
        r = self.rng
        return self._unique(lambda: f"{r.choice(COINED)} {r.choice(suffixes)}")

    def url(self, stem: str) -> dict[str, str]:
        r = self.rng
        tail = r.choice(["", "/app", "/login", "/download", "/apply"])
        return {"type": "url", "value": self._unique(lambda: f"{stem}-{r.randint(10, 99)}.example{tail}")}

    def telegram(self, stem: str) -> dict[str, str]:
        r = self.rng
        return {"type": "telegram", "value": self._unique(lambda: f"@{stem}_{r.randint(1000, 9999)}")}

    def facebook_page(self, suffixes: list[str]) -> dict[str, str]:
        return {"type": "facebook_page", "value": self.org_name(suffixes)}


def _slug(name: str) -> str:
    return name.split()[0].lower()


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _alias(rng: random.Random) -> str:
    return rng.choice([f"{rng.choice(FIRST_NAMES)} sir", f"{rng.choice(FIRST_NAMES)} dai",
                       rng.choice(FIRST_NAMES), f"Mr. {rng.choice(SURNAMES)}"])


# ---------------------------------------------------------------------------
# Scripts: how each ring operates. Victims of the same ring describe the same
# story beats, which is what lets us link complaints that share no identifier.
# ---------------------------------------------------------------------------

def _make_script(scam_type: str, ids: IdFactory) -> tuple[dict[str, Any], list[dict[str, str]], list[dict]]:
    """Return (script, names, web identifiers) for a new ring."""
    r = ids.rng
    web: list[dict] = []

    if scam_type == "fake_foreign_job":
        agency = ids.org_name(["Overseas Pvt. Ltd.", "Manpower Services", "Global Recruitment", "Placement Agency"])
        recruiter = _alias(r)
        script = {
            "agency": agency,
            "recruiter": recruiter,
            "country": r.choice(["Malaysia", "Qatar", "Dubai (UAE)", "Saudi Arabia", "Japan", "South Korea",
                                 "Poland", "Romania", "Croatia", "Portugal"]),
            "job": r.choice(["security guard", "factory worker", "delivery driver", "hotel housekeeping",
                             "construction helper", "caregiver", "warehouse packer", "farm worker"]),
            "promised_salary_npr": r.choice([90_000, 120_000, 150_000, 200_000, 250_000]),
            "found_via": r.choice(["a Facebook page", "a TikTok video", "a Viber group", "a post a friend shared"]),
            "fee_stages": r.sample(["registration fee", "medical test fee", "visa processing fee",
                                    "air ticket booking", "work permit fee", "insurance deposit"], k=3),
            "ending": r.choice(["kept saying the visa was delayed, then stopped answering",
                                "switched off the phone after the last payment",
                                "the office address turned out to be fake"]),
        }
        names = [{"type": "organisation", "value": agency}, {"type": "person_alias", "value": recruiter}]
        web.append(ids.facebook_page(["Jobs Abroad", "Foreign Jobs Nepal", "Work Visa Info", "Overseas Careers"]))

    elif scam_type == "instant_loan_app":
        app = ids.org_name(["Loan", "Cash", "Rin", "Paisa"])
        agent = _alias(r)
        script = {
            "app": app,
            "recovery_agent": agent,
            "found_via": r.choice(["a Facebook ad", "a link in an SMS", "a YouTube ad"]),
            "mechanism": r.choice(["asked for a processing fee before releasing the loan, then never released it",
                                   "released a small loan, then demanded several times more in repayment",
                                   "kept adding late fees every day even after repayment"]),
            "pressure": r.choice(["called people in my contact list and said I was a fraud",
                                  "sent threatening messages every day",
                                  "threatened to file a fake police case against me"]),
        }
        names = [{"type": "organisation", "value": app}, {"type": "person_alias", "value": agent}]
        web.append(ids.url(_slug(app) + "-loan"))

    elif scam_type == "digital_arrest":
        officer = f"{r.choice(['Inspector', 'Officer', 'DSP'])} {r.choice(SURNAMES)}"
        script = {
            "impersonated": r.choice(["an officer from the Nepal Police Cyber Bureau",
                                      "a customs officer at Tribhuvan International Airport",
                                      "an officer from the Central Investigation Bureau",
                                      "a telecom security officer"]),
            "officer_name": officer,
            "pretext": r.choice(["a parcel in my name contained illegal items",
                                 "my SIM card was used for money laundering",
                                 "my bank account was linked to a fraud case",
                                 "my national ID was used to open fake accounts"]),
            "demand": r.choice(["a refundable verification deposit", "a security deposit to clear my name",
                                "a fine to avoid immediate arrest"]),
            "tactic": r.choice(["kept me on a video call for hours", "told me not to tell my family",
                                "showed me a fake arrest warrant on WhatsApp"]),
        }
        names = [{"type": "person_alias", "value": officer}]

    elif scam_type == "crypto_investment":
        platform = ids.org_name(["Exchange", "Trade", "Capital", "Markets"])
        script = {
            "platform": platform,
            "hook": r.choice(["a Telegram group sharing daily trading signals",
                              "an Instagram ad promising 3% daily returns",
                              "a person who befriended me on Facebook and taught me trading"]),
            "early_profit": r.choice(["showed small profits at first and let me withdraw once",
                                      "my dashboard showed my money doubling in two weeks"]),
            "blocked": r.choice(["asked for a 20% tax before any withdrawal",
                                 "froze my account and asked for an unlock fee",
                                 "said my account was flagged and I must pay a verification fee"]),
        }
        names = [{"type": "organisation", "value": platform}]
        web.append(ids.url(_slug(platform) + "-trade"))
        web.append(ids.telegram(_slug(platform) + "_signals"))

    elif scam_type == "romance":
        persona = r.choice(WESTERN_PERSONAS)
        courier = ids.org_name(["Courier", "Logistics", "Parcel Services", "Express Cargo"])
        script = {
            "persona": persona,
            "persona_role": r.choice(["a doctor working in the UK", "a US army officer", "an engineer on an oil ship",
                                      "a businesswoman in Canada"]),
            "met_on": r.choice(["Facebook", "Instagram", "a dating app"]),
            "gift": r.choice(["a parcel with gifts and dollars", "a box with jewellery and a phone",
                              "money for our wedding"]),
            "courier": courier,
            "fee": r.choice(["customs clearance fee", "anti-money-laundering certificate fee", "delivery tax"]),
        }
        names = [{"type": "person_alias", "value": persona}, {"type": "organisation", "value": courier}]

    elif scam_type == "marketplace":
        seller = _alias(r)
        script = {
            "item": r.choice(["a second-hand motorbike", "an iPhone", "a laptop", "a scooter",
                              "a DSLR camera", "gold jewellery"]),
            "seller": seller,
            "advance_reason": r.choice(["advance payment for delivery", "a booking deposit",
                                        "half payment before dispatch"]),
            "ending": r.choice(["blocked me after the payment", "sent a fake courier receipt and disappeared",
                                "kept asking for more charges and never delivered"]),
        }
        names = [{"type": "person_alias", "value": seller}]
        web.append(ids.facebook_page(["Bike Bazaar", "Gadget Deals", "Second Hand Market", "Online Sale"]))

    else:
        raise ValueError(f"Unknown scam type: {scam_type}")

    return script, names, web


def _make_payments(scam_type: str, ids: IdFactory, counts: dict[str, int]) -> list[dict[str, Any]]:
    payments = []
    for kind in PAYMENT_MIX[scam_type]:
        for _ in range(counts.get(kind, 0)):
            if kind == "wallet":
                payments.append(ids.wallet())
            elif kind == "bank_account":
                payments.append(ids.bank_account())
            else:
                payments.append(ids.crypto_address())
    return payments


def _active_window(rng: random.Random) -> tuple[date, date]:
    span = (WINDOW_END - WINDOW_START).days
    length = rng.randint(45, 200)
    start = WINDOW_START + timedelta(days=rng.randint(0, max(0, span - length)))
    return start, min(WINDOW_END, start + timedelta(days=length))


def make_ring(ring_id: str | None, scam_type: str, ids: IdFactory, small: bool = False) -> Ring:
    r = ids.rng
    script, names, web = _make_script(scam_type, ids)
    if small:  # one-off scam: minimal identifiers
        phones = [ids.phone() for _ in range(r.randint(1, 2))]
        counts = {k: 1 for k in PAYMENT_MIX[scam_type]}
        start = WINDOW_START + timedelta(days=r.randint(0, (WINDOW_END - WINDOW_START).days))
        end = start
    else:
        phones = [ids.phone() for _ in range(r.randint(2, 6))]
        counts = {k: r.randint(1, 3) for k in PAYMENT_MIX[scam_type]}
        start, end = _active_window(r)
    lo, hi = LOSS_RANGES[scam_type]
    # Each ring works a narrower band inside the scam type's range.
    a = math.exp(r.uniform(math.log(lo), math.log(hi)))
    b = math.exp(r.uniform(math.log(lo), math.log(hi)))
    loss = (int(min(a, b)), int(max(a, b)) + 1000)
    return Ring(ring_id, scam_type, script, phones, _make_payments(scam_type, ids, counts), web,
                start, end, loss, names)


# ---------------------------------------------------------------------------
# How identifiers are written by victims (formatting noise and typos).
# ---------------------------------------------------------------------------

def written_number(value: str, rng: random.Random) -> str:
    """A 9-digit phone or wallet ID, written the way a victim might type it."""
    a, b, c = value[:3], value[3:6], value[6:]
    style = rng.choices(["plain", "dash", "space", "cc", "cc_dash"], weights=[40, 25, 15, 12, 8])[0]
    return {
        "plain": value,
        "dash": f"{a}-{b}-{c}",
        "space": f"{a} {b} {c}",
        "cc": f"+977{value}",
        "cc_dash": f"+977-{a}-{b}-{c}",
    }[style]


def written_account(value: str, rng: random.Random) -> str:
    if rng.random() < 0.6:
        return value
    return " ".join(value[i:i + 4] for i in range(0, len(value), 4))


def typo_number(value: str, rng: random.Random, used: set[str]) -> str | None:
    """Change one of the last three digits, avoiding any real identifier."""
    for _ in range(20):
        pos = rng.randint(len(value) - 3, len(value) - 1)
        digit = str(rng.choice([d for d in range(10) if str(d) != value[pos]]))
        candidate = value[:pos] + digit + value[pos + 1:]
        if candidate not in used:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Complaints
# ---------------------------------------------------------------------------

def _round_amount(x: float) -> int:
    return max(500, int(round(x / 500.0)) * 500)


def _victim_story(ring: Ring, rng: random.Random) -> dict[str, Any]:
    story = dict(ring.script)
    if ring.scam_type == "fake_foreign_job":
        k = rng.randint(1, len(story["fee_stages"]))
        story["fees_paid"] = story.pop("fee_stages")[:k]
    return story


def _style(rng: random.Random) -> dict[str, Any]:
    return {
        "writer": "relative" if rng.random() < 0.1 else "victim",
        "role": rng.choice(VICTIM_ROLES),
        "age": rng.randint(19, 68),
        "emotion": rng.choice(EMOTIONS),
        "english": rng.choice(ENGLISH_LEVELS),
        "nepali_words": rng.random() < 0.4,
        "words": rng.choice([80, 120, 170, 230]),
    }


def _pick_phones(ring: Ring, t: float, rng: random.Random) -> list[str]:
    """Rings rotate numbers over time, so pick numbers near this point in time."""
    n = len(ring.phones)
    centre = t * (n - 1)
    weights = [math.exp(-abs(i - centre)) for i in range(n)]
    k = 2 if (n > 1 and rng.random() < 0.3) else 1
    chosen: list[str] = []
    while len(chosen) < k:
        p = rng.choices(ring.phones, weights=weights)[0]
        if p not in chosen:
            chosen.append(p)
    return chosen


def _mention_number(kind: str, value: str, rng: random.Random, ids: IdFactory, **extra) -> dict[str, Any]:
    typo = None
    if rng.random() < 0.03:
        typo = typo_number(value, rng, ids.used)
        if typo:
            ids.used.add(typo)  # never reuse a mistyped number as a real identifier
    written = written_number(typo or value, rng)
    return {"type": kind, "value": value, "written": written, "typo": typo is not None, **extra}


def _mention_payment(payment: dict[str, Any], rng: random.Random, ids: IdFactory) -> dict[str, Any]:
    extra = {k: v for k, v in payment.items() if k not in ("type", "value")}
    if payment["type"] == "wallet":
        return _mention_number("wallet", payment["value"], rng, ids, **extra)
    if payment["type"] == "bank_account":
        return {"type": "bank_account", "value": payment["value"],
                "written": written_account(payment["value"], rng), "typo": False, **extra}
    return {"type": payment["type"], "value": payment["value"], "written": payment["value"], "typo": False, **extra}


def make_complaint(ring: Ring, rng: random.Random, ids: IdFactory, script_only: bool = False) -> dict[str, Any]:
    span = max(0, (ring.end - ring.start).days)
    offset = rng.randint(0, span) if span else 0
    t = offset / span if span else 0.5
    incident = ring.start + timedelta(days=offset)
    filed = min(WINDOW_END + timedelta(days=30), incident + timedelta(days=rng.randint(0, 30)))
    lo, hi = ring.loss_range
    amount = _round_amount(math.exp(rng.uniform(math.log(lo), math.log(hi))))

    mentions: list[dict[str, Any]] = []
    names: list[dict[str, str]] = []
    story = _victim_story(ring, rng)

    if script_only:
        # Only a one-time number: the story is the only link back to the ring.
        burner = ids.phone()
        mentions.append(_mention_number("phone", burner, rng, ids, burner=True))
        for key in ("agency", "recruiter", "app", "recovery_agent", "officer_name", "platform",
                    "persona", "courier", "seller"):
            story.pop(key, None)
    else:
        if rng.random() < PHONE_MENTION[ring.scam_type]:
            for p in _pick_phones(ring, t, rng):
                mentions.append(_mention_number("phone", p, rng, ids))
        if ring.payments and rng.random() < PAYMENT_MENTION:
            kind = _weighted(rng, {k: v for k, v in PAYMENT_MIX[ring.scam_type].items()
                                   if any(p["type"] == k for p in ring.payments)})
            options = [p for p in ring.payments if p["type"] == kind]
            mentions.append(_mention_payment(rng.choice(options), rng, ids))
        for w in ring.web:
            if rng.random() < WEB_MIX[ring.scam_type].get(w["type"], 0):
                mentions.append({"type": w["type"], "value": w["value"], "written": w["value"], "typo": False})
        for n in ring.names:
            if rng.random() < NAME_MENTION:
                names.append(dict(n))
        if not mentions and ring.phones:  # every complaint names at least one lead
            mentions.append(_mention_number("phone", rng.choice(ring.phones), rng, ids))

    # Drop story fields for names the victim did not mention.
    mentioned = {n["value"] for n in names}
    for key, value in list(story.items()):
        if any(value == n["value"] for n in ring.names) and value not in mentioned:
            story.pop(key)

    return {
        "ring_id": ring.ring_id,
        "scam_type": ring.scam_type,
        "incident_date": incident.isoformat(),
        "filed_date": filed.isoformat(),
        "district": _weighted(rng, DISTRICTS),
        "channel": _weighted(rng, CHANNELS),
        "amount_npr": amount,
        "story": story,
        "mentions": mentions,
        "names": names,
        "style": _style(rng),
        "flags": {"noise": ring.ring_id is None, "script_only": script_only, "duplicate_of": None,
                  "has_typo": any(m["typo"] for m in mentions), "shared_mule": False},
    }


def _ring_sizes(total: int, n_rings: int, rng: random.Random, minimum: int = 8) -> list[int]:
    """Heavy-tailed: a few big rings and many small ones."""
    weights = [1 / (i + 1) ** 0.9 for i in range(n_rings)]
    scale = total / sum(weights)
    sizes = [max(minimum, round(w * scale)) for w in weights]
    sizes[0] += total - sum(sizes)
    rng.shuffle(sizes)
    return sizes


def build_world(
    n_complaints: int = 1500,
    n_rings: int = 25,
    noise_share: float = 0.15,
    duplicate_share: float = 0.03,
    script_only_share: float = 0.06,
    n_shared_mules: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Build rings and complaint fact sheets. Pure Python, reproducible from the seed."""
    rng = random.Random(seed)
    ids = IdFactory(rng)

    types = (SCAM_TYPES * math.ceil(n_rings / len(SCAM_TYPES)))[:n_rings]
    rng.shuffle(types)
    rings = [make_ring(f"R{i + 1:02d}", t, ids) for i, t in enumerate(types)]

    # Hard case: one money-mule bank account used by two different rings.
    bank_rings = [r for r in rings if "bank_account" in PAYMENT_MIX[r.scam_type]]
    mule_values: set[str] = set()
    for _ in range(n_shared_mules):
        a, b = rng.sample(bank_rings, 2)
        mule = ids.bank_account()
        mule["shared_by"] = sorted([a.ring_id, b.ring_id])
        a.payments.append(dict(mule))
        b.payments.append(dict(mule))
        mule_values.add(mule["value"])

    n_noise = round(n_complaints * noise_share)
    n_dupes = round(n_complaints * duplicate_share)
    n_ring_base = n_complaints - n_noise - n_dupes
    sizes = _ring_sizes(n_ring_base, n_rings, rng)

    complaints: list[dict[str, Any]] = []
    for ring, size in zip(rings, sizes):
        for _ in range(size):
            complaints.append(make_complaint(ring, rng, ids, script_only=rng.random() < script_only_share))

    for _ in range(n_noise):
        one_off = make_ring(None, rng.choice(SCAM_TYPES), ids, small=True)
        complaints.append(make_complaint(one_off, rng, ids))

    # Hard case: the same victim files twice, in different words.
    originals = rng.sample([c for c in complaints if not c["flags"]["noise"]], n_dupes)
    duplicates = []
    for orig in originals:
        dup = copy.deepcopy(orig)
        dup["filed_date"] = (date.fromisoformat(orig["filed_date"]) + timedelta(days=rng.randint(3, 20))).isoformat()
        dup["channel"] = _weighted(rng, CHANNELS)
        dup["style"] = _style(rng)
        dup["_original"] = orig
        duplicates.append(dup)

    all_complaints = complaints + duplicates
    rng.shuffle(all_complaints)
    for i, c in enumerate(all_complaints, start=1):
        c["complaint_id"] = f"CMP-{i:05d}"
    for c in all_complaints:
        if "_original" in c:
            c["flags"]["duplicate_of"] = c.pop("_original")["complaint_id"]
        c["flags"]["shared_mule"] = any(m["value"] in mule_values for m in c["mentions"])

    _assign_splits(all_complaints, rng)
    return {"seed": seed, "rings": rings, "complaints": all_complaints}


def _assign_splits(complaints: list[dict[str, Any]], rng: random.Random) -> None:
    """About 2/3 train, 1/6 validation, 1/6 test. Duplicates follow their original."""
    by_id = {c["complaint_id"]: c for c in complaints}
    for c in complaints:
        if c["flags"]["duplicate_of"] is None:
            x = rng.random()
            c["split"] = "train" if x < 2 / 3 else ("val" if x < 5 / 6 else "test")
    for c in complaints:
        if c["flags"]["duplicate_of"] is not None:
            c["split"] = by_id[c["flags"]["duplicate_of"]]["split"]


def ring_to_dict(ring: Ring, n_complaints: int) -> dict[str, Any]:
    return {
        "ring_id": ring.ring_id,
        "scam_type": ring.scam_type,
        "active_from": ring.start.isoformat(),
        "active_to": ring.end.isoformat(),
        "loss_range_npr": list(ring.loss_range),
        "n_complaints": n_complaints,
        "phones": ring.phones,
        "payments": ring.payments,
        "web": ring.web,
        "names": ring.names,
        "script": ring.script,
    }
