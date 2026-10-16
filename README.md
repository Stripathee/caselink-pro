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
| Messy-field extraction | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | Fast and cheap for medium-confidence complaints |
| Synthetic data, verifier agent | `nvidia/nemotron-3-super-120b-a12b` | Good balance of quality and cost |
| Teacher labels, ring case reasoning | `nvidia/Nemotron-3-Ultra-550b-a55b` | Strongest reasoning, used only where it matters |

Most complaints never reach a large model: free regex and a small student classifier
handle confident cases first. (Measured results will be added here.)

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

## Project structure

```
caselink/        core library: model client, config, synthetic world, complaint writer
scripts/         command-line tools: setup check, data generation, evaluations
data/            synthetic datasets
```

## License

[MIT](LICENSE)
