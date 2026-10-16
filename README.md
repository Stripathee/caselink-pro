# CaseLink Pro

**Finds the scam rings hiding inside a cybercrime complaint backlog.**

CaseLink Pro reads fraud complaints in bulk, extracts the phone numbers, wallets,
accounts and URLs scammers use, links complaints that share them into scam rings,
and writes an evidence-backed case report for each ring, with every claim cited.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/)
(Best Apps and Agents track) with NVIDIA Nemotron models served on Nebius Token Factory.

> **Status:** work in progress (Oct 2026).
>
> **All data in this project is synthetic.** No real people, cases or identifiers are used.
> CaseLink Pro is decision support: it organizes evidence, and investigators make every decision.

## Why

Police cybercrime units handle complaints one at a time. The same scammer's phone
number or wallet can appear in dozens of complaints without anyone connecting them,
so the biggest rings, total losses and victim counts stay invisible.

## How it uses Nemotron on Nebius Token Factory

| Step | Model (Token Factory ID) | Why this model |
| --- | --- | --- |
| Reading every complaint: names, scam type, amount, date | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | Fast and cheap enough to read all 1,500 complaints |
| Writing the synthetic complaints; verifier agent | `nvidia/nemotron-3-super-120b-a12b` | Good balance of quality and cost |
| Ring case reasoning | `nvidia/Nemotron-3-Ultra-550b-a55b` | Strongest reasoning, used only once per ring |

Reasoning is switched off (`enable_thinking: false`) where it adds nothing, such as writing
and extraction, which cut output tokens per complaint by more than 4x.

## Results at a glance

Measured against the hidden truth of the synthetic dataset. Ring results are on the
held-out test split, which was never used for any design or tuning decision.

| | Result |
| --- | --- |
| Identifiers extracted (all 1,500 complaints) | 98.6% F1 (99.1% precision, 98.0% recall) |
| Scammer aliases and fake organisations | 95.6% F1 |
| Scam type / amount lost / incident date | 96.0% / 98.1% / 99.8% correct |
| Ring complaints placed in the right ring | 99.0% (plain matching: 94.0%) |
| Complaints with no shared identifier, linked by story | 10 of 12 (plain matching: 1 of 12) |
| One-off scams wrongly attached to a ring | 2 of 44 |
| Rings wrongly merged by a shared mule account | 0 (plain matching: 1) |
| Ring recovery, adjusted Rand index | 0.985 (plain matching: 0.926) |

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# open .env and paste your NEBIUS_API_KEY

python scripts/check_setup.py
```

## Synthetic data

Real complaint data is private, so CaseLink Pro ships its own realistic test world:
about 1,500 complaints from 25 hidden scam rings across six scam types (fake foreign jobs,
instant loan apps, "digital arrest" impersonation, crypto investment, romance and
marketplace fraud), plus one-off scams as noise. The true ring behind every complaint
is recorded, so every result can be measured.

```bash
python scripts/generate_data.py --sample 12   # quick look: 12 varied complaints
python scripts/generate_data.py               # full dataset
```

How it is built:

1. **Hidden rings (plain Python, seeded).** Each ring gets its own phones, wallets, bank
   accounts, web identifiers and a modus operandi. Rings rotate phone numbers over time.
2. **Fact sheets.** Each complaint mentions only some of its ring's identifiers, written the
   way victims write them (dashes, spaces, +977, occasional typos).
3. **Victim voice (Nemotron 3 Super).** Nemotron writes each complaint from its fact sheet,
   with varied writers, emotions and English levels, some with romanized Nepali words.
   Every identifier is checked to appear verbatim; misses are retried, then repaired.
4. **Hard cases on purpose.** A money-mule account shared by two rings, complaints linked
   only by the story (one-time numbers), duplicate filings, and mistyped numbers.

**Fictional by construction.** Phone numbers and wallet IDs have 9 digits with a 555 block
(real Nepali mobiles have 10), web addresses use the reserved `.example` domain, crypto
addresses contain a character base58 never uses, and bank, agency and app names are invented.

Output in `data/synthetic/`: `complaints.jsonl` (what the system sees), `ground_truth.jsonl`
and `rings.json` (the hidden answers, used only for evaluation), and `summary.json`.

## Extraction

Investigators link cases on identifiers, so extraction comes first. It runs in tiers:

1. **Rules (free, instant).** Patterns for phone numbers, wallet IDs, bank accounts, links,
   Telegram handles, crypto addresses and Facebook page names, normalised so that
   `985-555-012` and `+977985555012` become the same key. Phone vs wallet is decided from
   the surrounding words ("called from" vs "sent to eSewa").
2. **Nemotron 3 Nano.** Reads what rules cannot: scammer aliases, fake agency, app,
   platform and courier names, pages mentioned without the word "Facebook", the scam type
   and the amount lost. Its answer is validated: anything not present in the complaint
   text is dropped, so the model cannot invent a lead.

```bash
python scripts/run_extraction.py --mode rules                 # free
python scripts/run_extraction.py --mode cascade               # rules + Nemotron Nano, all complaints
```

Every model answer is checked before it is trusted: an identifier must match its format
(a Telegram handle starts with `@`, a link has a domain), real banks, wallets, platforms
and government agencies are never treated as scammer names, and a date outside the year
before filing is dropped rather than guessed. Rules plus Nemotron Nano on all 1,500
complaints (about 0.98M input and 0.27M output tokens):

| Identifier | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Phone number | 99.5% | 99.4% | 99.4% |
| Wallet ID | 98.8% | 98.7% | 98.7% |
| Bank account | 100% | 100% | 100% |
| Link / URL | 99.0% | 100% | 99.5% |
| Telegram handle | 100% | 100% | 100% |
| Crypto address | 100% | 100% | 100% |
| Facebook page name | 96.6% | 88.1% | 92.1% |
| **All** | **99.1%** | **98.0%** | **98.6%** |

Rules alone, on the held-out test split (245 complaints, never used for tuning):

| Identifier | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Phone number | 99.6% | 98.9% | 99.2% |
| Wallet ID | 98.2% | 99.1% | 98.6% |
| Bank account | 100% | 100% | 100% |
| Link / URL | 100% | 100% | 100% |
| Telegram handle | 100% | 100% | 100% |
| Crypto address | 100% | 100% | 100% |
| Facebook page name | 98.1% | 82.5% | 89.7% |
| **All** | **99.3%** | **97.3%** | **98.3%** |

Phone and wallet numbers are found with 100% precision and 99.7% recall regardless of label,
which is what ring linking depends on.

## Finding rings

```bash
python scripts/find_rings.py     # free: no model calls, about 2 minutes
```

1. **Link.** Complaints and identifiers form a graph. Complaints that share a phone, wallet,
   account, link, handle or page fall into the same group.
2. **Split shared mules.** Mule accounts are rented to more than one gang. When a single
   identifier is the only bridge between two sizeable groups that tell different stories
   (TF-IDF similarity of their complaints, identifiers masked), it is flagged as a shared
   mule and the groups become separate rings. A ring that rotates its phone numbers keeps
   telling the same story, so it is not split. Same-ring halves score at least 0.46; the
   detected mule's two sides scored 0.16.
3. **Link predictor.** Some victims only ever saw a one-time number. A gradient-boosted
   classifier scores each (lone complaint, candidate ring) pair on story similarity, scam
   type, matching or conflicting aliases and agency names, near-identical numbers (typos),
   timing and amount, and attaches the complaint only when confident. It is trained on the
   train split; each training complaint is also shown with its true ring hidden, so the
   model learns to say no to a similar story from an unknown gang. The attach threshold is
   chosen on the validation split by a rule fixed in advance: at most 5% of one-off scams
   wrongly attached, then as many story-only complaints linked as possible.

| Test split | Ring complaints correct | Story-only linked | One-offs wrongly attached | Rings merged | ARI |
| --- | --- | --- | --- | --- | --- |
| Plain matching | 94.0% | 1 / 12 | 0 / 44 | 1 | 0.926 |
| + shared-mule splitting | 94.5% | 1 / 12 | 0 / 44 | 0 | 0.937 |
| **+ link predictor (CaseLink)** | **99.0%** | **10 / 12** | **2 / 44** | **0** | **0.985** |

Validation split, for comparison: 97.8% correct, 14 / 19 story-only linked, 1 / 37 one-offs
wrongly attached, 0 merged, ARI 0.981. The story-only counts are small, so treat those
percentages as indicative.

CaseLink finds 25 rings (the hidden truth has 25). Each ring profile in
`data/rings/rings.json` lists victims and complaints, total losses, scam type, active
period, every lead ranked by how many complaints mention it, the aliases used, shared mule
accounts, which complaints were linked by story and with what probability, and possible
duplicate filings (identical leads, amount and date), which are flagged for an officer and
counted once in losses and victims.

## Case reports

```bash
python scripts/make_reports.py --top 3    # verified reports for the 3 costliest rings
```

Each ring gets an investigation report in `data/reports/` (Markdown and JSON), written so
that nothing reaches an investigator without checked evidence:

1. **Python computes every number** (victims, losses, dates, how often each lead appears and
   when each phone was in use) as numbered facts F1, F2, and so on.
2. **Nemotron 3 Ultra** reads those facts and up to 40 of the ring's complaints, chosen so
   every lead, every alias and the whole active period are covered, and writes how the scam
   works, the key leads and concrete next steps. Every claim must cite complaint IDs or
   fact IDs.
3. **Evidence rules** remove any claim that cites nothing, cites a complaint outside the
   ring, names an identifier that is not one of the ring's leads, or states a number that is
   not in its cited evidence.
4. **Nemotron 3 Super** reads each remaining claim next to the evidence it cites and keeps
   only claims the evidence fully supports. Removed claims are listed with the reason.

## Project structure

```
caselink/        core library: model client, synthetic world, extraction, ring finding, link predictor
scripts/         command-line tools: setup check, data generation, extraction, ring finding
data/            synthetic dataset, extraction outputs, ring profiles
results/         evaluation results as JSON
```

## License

[MIT](LICENSE)
