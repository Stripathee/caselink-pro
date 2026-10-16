"""Day 1 setup check.

1. Lists the NVIDIA Nemotron models your Token Factory account can use.
2. Sends one test question to a Nemotron model.
3. Sends it again to prove the disk cache works (second call is free).

Run from the project root:  python scripts/check_setup.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import APIStatusError, AuthenticationError  # noqa: E402

from caselink.config import MODELS  # noqa: E402
from caselink.llm import chat, get_client  # noqa: E402

QUESTION = "In one sentence, what is a money mule account in an online fraud case?"


def main() -> int:
    print("Step 1/3: listing models on Nebius Token Factory...")
    try:
        all_ids = sorted(m.id for m in get_client().models.list())
    except AuthenticationError:
        print("  Your API key was rejected. Check NEBIUS_API_KEY in .env.")
        return 1
    except RuntimeError as err:
        print(f"  {err}")
        return 1

    nemotron = [m for m in all_ids if "nemotron" in m.lower()]
    if not nemotron:
        print(f"  Connected, but no Nemotron models found among {len(all_ids)} models.")
        return 1

    print(f"  Found {len(nemotron)} Nemotron models:")
    for model_id in nemotron:
        print(f"    - {model_id}")

    print("\n  Models CaseLink Pro is configured to use:")
    for tier, model_id in MODELS.items():
        status = "available" if model_id in nemotron else "NOT FOUND on your account"
        print(f"    {tier:<10} {model_id}  ({status})")

    test_model = MODELS["super"] if MODELS["super"] in nemotron else nemotron[0]
    print(f"\nStep 2/3: asking {test_model} a test question...")
    try:
        first = chat(test_model, [{"role": "user", "content": QUESTION}], max_tokens=512)
    except APIStatusError as err:
        print(f"  The call failed ({err.status_code}): {err.message}")
        return 1

    print(f"  Answer: {first['content'].strip() or '(empty, see reasoning field)'}")
    print(
        f"  Tokens: {first['prompt_tokens']} in, {first['completion_tokens']} out"
        f" | {first['latency_s']}s | cached={first['cached']}"
    )

    print("\nStep 3/3: asking again (should come from the cache, costing nothing)...")
    second = chat(test_model, [{"role": "user", "content": QUESTION}], max_tokens=512)
    print(f"  cached={second['cached']}")

    print("\nSetup works.")
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    raise SystemExit(main())
