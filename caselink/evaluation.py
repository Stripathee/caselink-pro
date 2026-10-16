"""Scoring helpers shared by the evaluation scripts."""
from __future__ import annotations

import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from caselink.extract_rules import normalize, normalize_name

ID_TYPES = ["phone", "wallet", "bank_account", "url", "telegram", "crypto_address", "facebook_page"]


def load_dataset(folder: str | Path) -> tuple[dict[str, dict], dict[str, dict]]:
    folder = Path(folder)
    complaints = {}
    with (folder / "complaints.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            complaints[row["complaint_id"]] = row
    truth = {}
    with (folder / "ground_truth.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            truth[row["complaint_id"]] = row
    return complaints, truth


def truth_keys(row: dict[str, Any]) -> set[tuple[str, str]]:
    """Identifiers as written in the text (a mistyped number is scored as written)."""
    return {(m["type"], normalize(m["type"], m["written"])) for m in row["mentions"]}


def score_extraction(pred: dict[str, set[tuple[str, str]]], truth: dict[str, dict],
                     ids: Iterable[str]) -> dict[str, Any]:
    """Precision, recall and F1 per identifier type, plus number-level scores."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    num_tp = num_fp = num_fn = 0
    for cid in ids:
        p, t = pred.get(cid, set()), truth_keys(truth[cid])
        for k in p & t:
            tp[k[0]] += 1
        for k in p - t:
            fp[k[0]] += 1
        for k in t - p:
            fn[k[0]] += 1
        # Numbers regardless of phone/wallet label: what linking actually needs.
        pn = {v for ty, v in p if ty in ("phone", "wallet")}
        tn = {v for ty, v in t if ty in ("phone", "wallet")}
        num_tp += len(pn & tn)
        num_fp += len(pn - tn)
        num_fn += len(tn - pn)

    def prf(a: int, b: int, c: int) -> dict[str, float]:
        precision = a / (a + b) if a + b else 1.0
        recall = a / (a + c) if a + c else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                "support": a + c}

    by_type = {t: prf(tp[t], fp[t], fn[t]) for t in ID_TYPES}
    overall = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    numbers = prf(num_tp, num_fp, num_fn)
    return {"by_type": by_type, "overall": overall, "numbers_any_label": numbers}


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "support": tp + fn}


def names_match(a: str, b: str) -> bool:
    """Same name, allowing titles ("Kiran" vs "Kiran sir") and small spelling slips."""
    x, y = normalize_name(a), normalize_name(b)
    if not x or not y:
        return False
    return x == y or x in y or y in x or SequenceMatcher(None, x, y).ratio() >= 0.85


def score_names(pred: dict[str, list[str]], truth: dict[str, dict], ids: Iterable[str]) -> dict[str, float]:
    """Scammer aliases and organisation names, matched loosely."""
    tp = fp = fn = 0
    for cid in ids:
        gold = [n["value"] for n in truth[cid]["names"]]
        guess = pred.get(cid, [])
        tp_here = sum(any(names_match(g, p) for p in guess) for g in gold)
        tp += tp_here
        fn += len(gold) - tp_here
        fp += sum(not any(names_match(g, p) for g in gold) for p in guess)
    return prf(tp, fp, fn)
