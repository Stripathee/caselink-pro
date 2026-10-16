"""Finding scam rings: link complaints that share leads, then split shared mules.

Step 1 - Link. Complaints and identifiers form a graph: a complaint is joined to
every phone, wallet, account, link, handle or page it mentions. Complaints that
share an identifier end up in the same connected group.

Step 2 - Split shared mules. Money-mule accounts are rented out to more than one
gang. When a single identifier is the only bridge between two sizeable groups
that tell different stories (different scam scripts), the groups are separate
rings and the identifier is flagged as a shared mule. A ring that simply rotates
its phone numbers keeps telling the same story, so it is not split.

Step 3 (caselink/linker.py) - Attach complaints that share no identifier with
anyone, using a trained link predictor over their story, names and timing.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from caselink.registry import is_generic_role, is_legitimate_entity

NUMERIC = ("phone", "wallet")


@dataclass
class Case:
    cid: str
    text: str
    ids: list[str]  # identifier keys, e.g. "number:985555012", "bank_account:1234..."
    names: list[str]
    scam_type: str | None
    amount: int | None
    date: str | None  # incident date, falling back to the filing date
    split: str
    raw_ids: list[dict[str, Any]] = field(default_factory=list)


def id_key(ident: dict[str, Any]) -> str:
    """Phones and wallets share one key space: a mislabelled number still links."""
    t = "number" if ident["type"] in NUMERIC else ident["type"]
    return f"{t}:{ident['value']}"


def load_cases(extracted: str | Path, data: str | Path = "data/synthetic") -> dict[str, Case]:
    complaints = {}
    with open(Path(data) / "complaints.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            complaints[row["complaint_id"]] = row
    cases = {}
    with open(extracted, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c = complaints[r["complaint_id"]]
            names = [n for n in r.get("people", []) + r.get("organisations", [])
                     if not is_generic_role(n) and not is_legitimate_entity(n)]  # re-apply current filters
            cases[r["complaint_id"]] = Case(
                cid=r["complaint_id"], text=c["text"],
                ids=sorted({id_key(i) for i in r["identifiers"]}),
                names=names,
                scam_type=r.get("scam_type"),
                amount=r.get("amount_lost_npr") or c.get("amount_npr"),
                date=r.get("incident_date") or c["filed_date"],
                split=c["split"], raw_ids=r["identifiers"],
            )
    return cases


# ---------------------------------------------------------------------------
# Story similarity: what the complaint says once identifiers are masked out
# ---------------------------------------------------------------------------

_MASK = re.compile(r"\+?\d[\d\s\-]{6,}\d|@\w+|\S+\.example\S*|T[1-9A-Za-z0]{33}")


class StoryModel:
    """TF-IDF over complaint text with every identifier masked out."""

    def __init__(self, cases: dict[str, Case]):
        self.order = list(cases)
        self.index = {cid: i for i, cid in enumerate(self.order)}
        texts = [_MASK.sub(" ", cases[cid].text) + " " + " ".join(cases[cid].names) for cid in self.order]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True,
                                          stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def vectors(self, cids: list[str]):
        return self.matrix[[self.index[c] for c in cids]]

    def group_similarity(self, a: list[str], b: list[str]) -> float:
        """Cosine similarity of the two groups' average stories."""
        ca = np.asarray(self.vectors(a).mean(axis=0))
        cb = np.asarray(self.vectors(b).mean(axis=0))
        return float(cosine_similarity(ca, cb)[0, 0])

    def pair_similarity(self, a: str, b: str) -> float:
        return float(cosine_similarity(self.vectors([a]), self.vectors([b]))[0, 0])


# ---------------------------------------------------------------------------
# Step 1 and 2
# ---------------------------------------------------------------------------

def build_graph(cases: dict[str, Case]) -> nx.Graph:
    g = nx.Graph()
    for c in cases.values():
        g.add_node(c.cid, kind="complaint")
        for k in c.ids:
            g.add_node(k, kind="identifier")
            g.add_edge(c.cid, k)
    return g


def _complaints(nodes) -> list[str]:
    return [n for n in nodes if n.startswith("CMP-")]


def split_shared_mules(g: nx.Graph, story: StoryModel, cases: dict[str, Case],
                       min_size: int = 5, max_similarity: float = 0.35) -> list[dict[str, Any]]:
    """Remove identifiers that are the only bridge between two different stories.

    Returns one record per shared mule found. The graph is modified in place:
    the bridge node is removed, and re-linked later to every ring it served.
    """
    mules: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        for comp in list(nx.connected_components(g)):
            members = _complaints(comp)
            if len(members) < 2 * min_size:
                continue
            sub = g.subgraph(comp)
            best = None
            for node in nx.articulation_points(sub):
                if g.nodes[node]["kind"] != "identifier":
                    continue
                parts = [_complaints(p) for p in nx.connected_components(sub.subgraph(set(comp) - {node}))]
                parts = sorted([p for p in parts if len(p) >= min_size], key=len, reverse=True)
                if len(parts) < 2:
                    continue
                sim = story.group_similarity(parts[0], parts[1])
                types = [Counter(cases[c].scam_type for c in p).most_common(1)[0][0] for p in parts[:2]]
                if sim < max_similarity and (best is None or sim < best[1]):
                    best = (node, sim, parts, types)
            if best:
                node, sim, parts, types = best
                mules.append({"identifier": node, "story_similarity": round(sim, 3),
                              "sides": [len(p) for p in parts], "scam_types": types,
                              "complaints": sorted(n for n in g.neighbors(node))})
                g.remove_node(node)
                changed = True
                break
    return mules


def clusters_from_graph(g: nx.Graph) -> dict[str, int]:
    label = {}
    for i, comp in enumerate(nx.connected_components(g)):
        for n in _complaints(comp):
            label[n] = i
    return label


def find_rings(cases: dict[str, Case], story: StoryModel | None = None, split_mules: bool = True,
               **kwargs) -> tuple[dict[str, int], list[dict[str, Any]], StoryModel]:
    story = story or StoryModel(cases)
    g = build_graph(cases)
    mules = split_shared_mules(g, story, cases, **kwargs) if split_mules else []
    return clusters_from_graph(g), mules, story


# ---------------------------------------------------------------------------
# Evaluation against the hidden truth
# ---------------------------------------------------------------------------

def evaluate(labels: dict[str, int], truth: dict[str, dict], ids: list[str] | None = None) -> dict[str, Any]:
    """How well predicted groups match the true rings, scored on `ids` (default: all)."""
    from sklearn.metrics import adjusted_rand_score

    ids = ids or list(labels)
    members = defaultdict(list)
    for cid, lab in labels.items():
        members[lab].append(cid)

    def attached_correctly(cid: str) -> bool:
        t = truth[cid]
        others = [m for m in members[labels[cid]] if m != cid and m != t["flags"]["duplicate_of"]
                  and truth[m]["flags"]["duplicate_of"] != cid]
        if not others or not t["ring_id"]:
            return False
        return Counter(truth[m]["ring_id"] for m in others).most_common(1)[0][0] == t["ring_id"]

    ring_ids = [c for c in ids if truth[c]["ring_id"]]
    story_only = [c for c in ring_ids if truth[c]["flags"]["script_only"]]
    noise = [c for c in ids if not truth[c]["ring_id"]]
    merged = []
    for lab, ms in members.items():
        counts = Counter(truth[m]["ring_id"] for m in ms if truth[m]["ring_id"])
        if len([n for n in counts.values() if n >= 3]) >= 2:
            merged.append(sorted(r for r, n in counts.items() if n >= 3))
    y_true = [truth[c]["ring_id"] or c for c in ids]
    y_pred = [labels[c] for c in ids]
    return {
        "n": len(ids),
        "ring_complaints_correct": sum(map(attached_correctly, ring_ids)),
        "ring_complaints": len(ring_ids),
        "story_only_linked": sum(map(attached_correctly, story_only)),
        "story_only": len(story_only),
        "noise_wrongly_attached": sum(len(members[labels[c]]) > 1 for c in noise),
        "noise": len(noise),
        "rings_wrongly_merged": merged,
        "ari": round(adjusted_rand_score(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# Ring summaries for investigators
# ---------------------------------------------------------------------------

def find_duplicates(members: list[str], cases: dict[str, Case]) -> list[tuple[str, str]]:
    """Possible repeat filings by the same victim: identical leads, amount and date.

    Flagged for an officer to confirm, never deleted, so losses are not
    counted twice.
    """
    pairs = []
    ms = sorted(members)
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            A, B = cases[a], cases[b]
            if A.ids and set(A.ids) == set(B.ids) and A.amount and A.amount == B.amount and A.date == B.date:
                pairs.append((a, b))
    return pairs


def _merge_names(names: list[str]) -> list[dict[str, Any]]:
    from caselink.evaluation import names_match

    groups: list[list[str]] = []
    for n in names:
        for g in groups:
            if names_match(n, g[0]):
                g.append(n)
                break
        else:
            groups.append([n])
    out = [{"name": Counter(g).most_common(1)[0][0], "mentions": len(g)} for g in groups]
    return sorted(out, key=lambda x: -x["mentions"])


def summarize_rings(labels: dict[str, int], cases: dict[str, Case], mules: list[dict[str, Any]],
                    linked: dict[str, float], min_size: int = 3) -> tuple[list[dict[str, Any]], list[str]]:
    """Ring profiles, largest losses first, plus the complaints left on their own."""
    members = defaultdict(list)
    for cid, lab in labels.items():
        members[lab].append(cid)
    rings, alone = [], []
    for lab, ms in members.items():
        if len(ms) < min_size:
            alone += ms
            continue
        dups = find_duplicates(ms, cases)
        repeat = {b for _, b in dups}
        counted = [m for m in ms if m not in repeat]
        ids = Counter(k for m in ms for k in cases[m].ids)
        labels_seen = defaultdict(Counter)
        for m in ms:
            for i in cases[m].raw_ids:
                labels_seen[id_key(i)][i["type"]] += 1
        dates = sorted(cases[m].date for m in ms if cases[m].date)
        types = Counter(cases[m].scam_type for m in ms if cases[m].scam_type)
        rings.append({
            "label": lab,
            "complaints": len(ms),
            "victims": len(counted),
            "possible_duplicates": [list(p) for p in dups],
            "total_loss_npr": int(sum(cases[m].amount or 0 for m in counted)),
            "scam_type": types.most_common(1)[0][0] if types else None,
            "scam_type_share": round(types.most_common(1)[0][1] / sum(types.values()), 2) if types else None,
            "active_from": dates[0] if dates else None,
            "active_to": dates[-1] if dates else None,
            "identifiers": [{"key": k, "type": labels_seen[k].most_common(1)[0][0] if labels_seen[k] else k.split(":")[0],
                             "value": k.split(":", 1)[1], "complaints": n} for k, n in ids.most_common()],
            "names": _merge_names([n for m in ms for n in cases[m].names])[:10],
            "shared_mules": [],
            "linked_by_story": sorted(({"complaint": m, "probability": round(linked[m], 3)}
                                       for m in ms if m in linked), key=lambda x: -x["probability"]),
            "members": sorted(ms),
        })
    rings.sort(key=lambda r: -r["total_loss_npr"])
    for i, r in enumerate(rings, start=1):
        r["ring"] = f"RING-{i:02d}"
    by_member = {m: r["ring"] for r in rings for m in r["members"]}
    for mule in mules:
        served = sorted({by_member[c] for c in mule["complaints"] if c in by_member})
        for r in rings:
            if r["ring"] in served:
                r["shared_mules"].append({"identifier": mule["identifier"],
                                          "also_used_by": [s for s in served if s != r["ring"]]})
    return rings, sorted(alone)
