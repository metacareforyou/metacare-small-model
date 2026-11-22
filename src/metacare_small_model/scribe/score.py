"""Precision and recall of extracted visit facts, for each kind of fact and each field (MET-1930).

A predicted item is matched to a right answer of the same kind whose name (or advice text, or follow up time) shares
at least half its words. A field is right when it says the same thing after normalising case, punctuation and
number words. Fields of unmatched items count against recall (right answers) or precision (predictions).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

KEY = {"advice": "text", "follow_up": "when"}
STOP = {"a", "an", "the", "of", "to", "for", "and", "at", "per", "in", "on", "with", "your", "my"}
WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split())}  # fmt: skip


def norm(value) -> str:
    text = str(value).lower().replace(",", "")
    text = re.sub(r"[^a-z0-9.%/ ]+", " ", text)
    return " ".join(WORDS.get(w, w) for w in text.split())


def words(value) -> set[str]:
    return {w for w in norm(value).split() if w not in STOP}


def similar(a, b) -> float:
    x, y = words(a), words(b)
    if not x or not y:
        return 0.0
    return len(x & y) / min(len(x), len(y))


def parse(text: str) -> list[dict] | None:
    """The items of a model's answer, or None when it is not the JSON asked for."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None
    return [i for i in data["items"] if isinstance(i, dict) and "kind" in i]


def match(gold: list[dict], pred: list[dict]) -> list[tuple[dict | None, dict | None]]:
    pairs, used = [], set()
    for g in gold:
        key = KEY.get(g["kind"], "name")
        best, score = None, 0.5
        for j, p in enumerate(pred):
            if j in used or p.get("kind") != g["kind"]:
                continue
            s = similar(g.get(key, ""), p.get(key, ""))
            if s >= score:
                best, score = j, s
        if best is None:
            pairs.append((g, None))
        else:
            used.add(best)
            pairs.append((g, pred[best]))
    pairs += [(None, p) for j, p in enumerate(pred) if j not in used]
    return pairs


class Tally:
    def __init__(self) -> None:
        self.n = defaultdict(lambda: {"right": 0, "said": 0, "predicted": 0})

    def add(self, name: str, gold, pred) -> None:
        row = self.n[name]
        g = gold is not None and gold != "" and gold != []
        p = pred is not None and pred != "" and pred != []
        row["said"] += g
        row["predicted"] += p
        if g and p and (norm(gold) == norm(pred) or (name.endswith(".name") and similar(gold, pred) >= 0.5)):
            row["right"] += 1

    def table(self) -> dict:
        out = {}
        for name, r in sorted(self.n.items()):
            out[name] = {
                "precision": round(r["right"] / r["predicted"], 3) if r["predicted"] else None,
                "recall": round(r["right"] / r["said"], 3) if r["said"] else None,
                **r,
            }
        return out


def score(rows: list[tuple[str, str]]) -> dict:
    """rows: (right answer JSON, model output) pairs."""
    items, fields, lines = Tally(), Tally(), {"right": 0, "said": 0, "predicted": 0}
    unparsed = 0
    for gold_text, output in rows:
        gold = json.loads(gold_text)["items"]
        pred = parse(output)
        if pred is None:
            unparsed += 1
            pred = []
        for g, p in match(gold, pred):
            kind = (g or p)["kind"]
            items.add(kind, g and "x", p and "x")
            for field in sorted({*(g or {}), *(p or {})} - {"kind", "lines"}):
                fields.add(f"{kind}.{field}", (g or {}).get(field), (p or {}).get(field))
            gl = set((g or {}).get("lines") or [])
            pl = {n for n in (p or {}).get("lines") or [] if isinstance(n, int)}
            lines["said"] += len(gl)
            lines["predicted"] += len(pl)
            lines["right"] += len(gl & pl)
    return {
        "visits": len(rows),
        "json_failed": unparsed,
        "items": items.table(),
        "fields": fields.table(),
        "lines": {
            **lines,
            "precision": round(lines["right"] / max(lines["predicted"], 1), 3),
            "recall": round(lines["right"] / max(lines["said"], 1), 3),
        },  # fmt: skip
    }
