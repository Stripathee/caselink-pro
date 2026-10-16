"""Link predictor: attach complaints that share no identifier with anyone.

Some victims only ever saw a one-time number, so plain matching cannot link
them. They still tell their ring's story: the same job in the same country, the
same fee sequence, the same pretext, often the same alias, around the same time.
A classifier scores each (orphan complaint, candidate ring) pair from those
signals and attaches the orphan only when it is confident.

Training uses the train split only. Each training ring complaint is turned into
an artificial orphan (its own identifiers and story removed from the ring's
profile), which gives thousands of realistic examples; real one-off scams from
the train split teach the model when NOT to attach.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics.pairwise import cosine_similarity

from caselink.evaluation import names_match
from caselink.rings import Case, StoryModel

FEATURES = [
    "story_vs_ring", "story_best_member", "story_top3_members", "story_rank", "story_margin",
    "same_scam_type", "ring_type_share", "name_match", "name_conflict", "near_number",
    "days_outside_window", "amount_z", "ring_size_log",
]


@dataclass
class Ring:
    label: int
    members: list[str]
    centroid: Any
    scam_types: Counter
    names: list[str]
    numbers: set[str]
    start: date
    end: date
    log_amount_mean: float
    log_amount_std: float


def _date(s: str | None) -> date | None:
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def _numbers(c: Case) -> set[str]:
    return {k.split(":", 1)[1] for k in c.ids if k.startswith("number:")}


def one_digit_apart(a: str, b: str) -> bool:
    """A mistyped number: same length, exactly one digit different."""
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def build_ring(label: int, members: list[str], cases: dict[str, Case], story: StoryModel) -> Ring:
    vecs = story.vectors(members)
    dates = [d for d in (_date(cases[m].date) for m in members) if d]
    amounts = [math.log(cases[m].amount) for m in members if cases[m].amount]
    return Ring(
        label=label, members=members,
        centroid=np.asarray(vecs.mean(axis=0)),
        scam_types=Counter(cases[m].scam_type for m in members if cases[m].scam_type),
        names=[n for m in members for n in cases[m].names],
        numbers=set().union(*(_numbers(cases[m]) for m in members)),
        start=min(dates) if dates else date(2026, 1, 1),
        end=max(dates) if dates else date(2026, 12, 31),
        log_amount_mean=float(np.mean(amounts)) if amounts else 0.0,
        log_amount_std=float(np.std(amounts)) + 0.25 if amounts else 1.0,
    )


def pair_features(c: Case, rings: list[Ring], story: StoryModel) -> np.ndarray:
    """One feature row per candidate ring for complaint c."""
    v = story.vectors([c.cid])
    centroids = np.vstack([r.centroid for r in rings])
    sim_ring = cosine_similarity(v, centroids)[0]
    order = np.argsort(-sim_ring)
    rank = np.empty(len(rings))
    rank[order] = np.arange(1, len(rings) + 1)
    best, second = sim_ring[order[0]], (sim_ring[order[1]] if len(rings) > 1 else 0.0)
    d = _date(c.date)
    nums = _numbers(c)
    rows = []
    for i, r in enumerate(rings):
        member_sims = np.sort(cosine_similarity(v, story.vectors(r.members))[0])[::-1]
        total = sum(r.scam_types.values()) or 1
        top_type = r.scam_types.most_common(1)[0][0] if r.scam_types else None
        outside = 0.0
        if d:
            outside = max(0, (r.start - d).days, (d - r.end).days)
        amount_z = abs(math.log(c.amount) - r.log_amount_mean) / r.log_amount_std if c.amount else 0.0
        name_match = any(names_match(a, b) for a in c.names for b in r.names)
        # The complaint names an alias or agency this ring has never used: likely a different gang.
        name_conflict = bool(c.names) and bool(r.names) and not name_match
        rows.append([
            sim_ring[i],
            member_sims[0],
            member_sims[:3].mean(),
            rank[i],
            sim_ring[i] - (second if rank[i] == 1 else best),
            float(c.scam_type is not None and c.scam_type == top_type),
            r.scam_types.get(c.scam_type, 0) / total if c.scam_type else 0.0,
            float(name_match),
            float(name_conflict),
            float(any(one_digit_apart(a, b) for a in nums for b in r.numbers)),
            min(outside, 365) / 30.0,
            min(amount_z, 10.0),
            math.log(len(r.members)),
        ])
    return np.array(rows)


class LinkPredictor:
    def __init__(self, seed: int = 42):
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                    l2_regularization=1.0, random_state=seed)
        self.seed = seed

    # --- training ---------------------------------------------------------

    def training_rows(self, cases: dict[str, Case], labels: dict[str, int], truth: dict[str, dict],
                      story: StoryModel, min_ring: int = 3, max_examples: int = 600):
        """(features, target) from the train split only."""
        rng = random.Random(self.seed)
        members = defaultdict(list)
        for cid, lab in labels.items():
            members[lab].append(cid)
        ring_labels = [lab for lab, ms in members.items() if len(ms) >= min_ring]

        def true_ring(lab: int) -> str | None:
            train = [truth[m]["ring_id"] for m in members[lab] if cases[m].split == "train" and truth[m]["ring_id"]]
            return Counter(train).most_common(1)[0][0] if train else None

        ring_truth = {lab: true_ring(lab) for lab in ring_labels}
        train_ring_members = [c for c in cases.values() if c.split == "train" and truth[c.cid]["ring_id"]
                              and len(members[labels[c.cid]]) >= min_ring]
        real_orphans = [c for c in cases.values() if c.split == "train" and len(members[labels[c.cid]]) == 1]
        sample = rng.sample(train_ring_members, min(max_examples, len(train_ring_members))) + real_orphans

        all_rings = {lab: build_ring(lab, members[lab], cases, story) for lab in ring_labels}
        X, y = [], []
        for c in sample:
            # Rebuild this complaint's own ring without it, so it looks like an orphan.
            own = labels[c.cid]
            rings = [all_rings[lab] for lab in ring_labels if lab != own]
            rest = [m for m in members[own] if m != c.cid]
            if own in all_rings and len(rest) >= min_ring - 1:
                rings.append(build_ring(own, rest, cases, story))
            target = [int(truth[c.cid]["ring_id"] is not None and ring_truth[r.label] == truth[c.cid]["ring_id"])
                      for r in rings]
            feats = pair_features(c, rings, story)
            X.append(feats)
            y += target
            # The same complaint again with its true ring hidden: it now looks like a scam
            # from a gang we have never seen, and the right answer everywhere is "no".
            if truth[c.cid]["ring_id"] and rng.random() < 0.5:
                keep = [i for i, t in enumerate(target) if t == 0]
                if keep:
                    hidden = [rings[i] for i in keep]
                    X.append(pair_features(c, hidden, story))
                    y += [0] * len(hidden)
        return np.vstack(X), np.array(y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinkPredictor":
        self.model.fit(X, y)
        return self

    # --- use --------------------------------------------------------------

    def score_orphans(self, cases: dict[str, Case], labels: dict[str, int], story: StoryModel,
                      min_ring: int = 3) -> list[dict[str, Any]]:
        """Best candidate ring and probability for every complaint that sits alone."""
        members = defaultdict(list)
        for cid, lab in labels.items():
            members[lab].append(cid)
        rings = [build_ring(lab, ms, cases, story) for lab, ms in members.items() if len(ms) >= min_ring]
        out = []
        for lab, ms in members.items():
            if len(ms) != 1:
                continue
            c = cases[ms[0]]
            feats = pair_features(c, rings, story)
            p = self.model.predict_proba(feats)[:, 1]
            i = int(np.argmax(p))
            out.append({"complaint": c.cid, "ring_label": rings[i].label, "probability": float(p[i]),
                        "features": dict(zip(FEATURES, map(float, feats[i])))})
        return out


def attach(labels: dict[str, int], scored: list[dict[str, Any]], threshold: float) -> dict[str, int]:
    """New labels with every confident orphan moved into its predicted ring."""
    new = dict(labels)
    for s in scored:
        if s["probability"] >= threshold:
            new[s["complaint"]] = s["ring_label"]
    return new
