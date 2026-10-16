"""Model IDs. Confirmed on Nebius Token Factory on Oct 15, 2026.

Each can be overridden in .env (for example MODEL_NANO=...), but you do not
need to set anything: an empty or missing value falls back to the default.
"""
import os

from dotenv import load_dotenv

load_dotenv()

_DEFAULTS = {
    # Fast, cheap: messy-field extraction and medium-confidence complaints.
    "nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    # Mid-size: synthetic complaint writing and the verifier agent.
    "super": "nvidia/nemotron-3-super-120b-a12b",
    # Strongest reasoning: teacher labels and ring case reports.
    "ultra": "nvidia/Nemotron-3-Ultra-550b-a55b",
    # Vision-language: reading screenshots of chats and payment receipts (stretch goal).
    "vl": "nvidia/Nemotron-3.5-Super-VL",
    # Not yet evaluated: candidate fast tier.
    "lightning": "nvidia/Nemotron-3_5-Lightning",
}

MODELS = {
    tier: (os.getenv(f"MODEL_{tier.upper()}") or default)
    for tier, default in _DEFAULTS.items()
}
