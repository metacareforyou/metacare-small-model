"""Turns written visits into training pairs for the extraction model (MET-1930).

The model reads a numbered transcript and writes one JSON object: the visit's facts in fixed fields, each with the
transcript lines it came from. The right answer comes from the visit spec, never from a model, and the lines come
from where the script states each fact (the same patterns the generator checks).

A transcript line looks like the browser scribe's output, with generic speaker names:

    3 Speaker 1: We'll start lisinopril 10 mg once a day.

A visit can appear twice: as written ("clean") and as heard by Whisper from synthetic speech ("heard", see
speech.py). Whole visit types are held out for the test split, so the score shows whether the model learned to
extract rather than to recognise sixteen visits.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from .checks import to_pattern

# Visit types never seen in training. Chosen to cover a medicine change, an order, a device and a follow up.
HOLDOUT_TEMPLATES = ("thyroid", "migraine", "knee_post_op")
VALIDATION_PERCENT = 5

KINDS = ("condition", "medication", "observation", "order", "device", "immunization", "advice", "follow_up")

SYSTEM = """You read a numbered transcript of a medical visit and write down the facts that were said, as one JSON object.
Write names, doses and values as said, in their usual written form. Do not add codes, diagnoses or anything that was not said.
Output: {"items":[...]} with one entry per fact, in the order first said. Each entry has "kind" and "lines" (the line numbers that state it), and:
- condition: name
- medication: name, dose, unit, frequency, route, action (start, continue, increase, decrease, stop), duration, reason
- observation: name, value, unit
- order: name
- device: name
- immunization: name
- advice: text
- follow_up: when
Leave out a field that was not said. Write only the JSON."""  # noqa: E501 (the prompt is kept word for word)


def parse_turns(script: str) -> list[tuple[str, str]]:
    """(speaker letter, text) for each `L|text` line."""
    out = []
    for row in script.split("\n"):
        m = re.match(r"^\s*([A-Z])\|(.*)$", row)
        if m and m.group(2).strip():
            out.append((m.group(1), m.group(2).strip()))
    return out


def evidence(spec: dict, turns: list[tuple[str, str]]) -> list[list[int]]:
    """For each spec item, the 1 based turn numbers that state it.

    A turn counts when it matches the item's first check (usually the name). If no turn does, the turns matching
    any of its checks are used.
    """
    out = []
    for item in spec["items"]:
        patterns = [to_pattern(str(v)) for v in item["check"]]
        if not patterns:  # a follow up has no check: its words are said as written
            patterns = [re.escape(str(v)) for v in item["fields"].values() if isinstance(v, str) and v]
        if not patterns:
            out.append([])
            continue
        first = [i + 1 for i, (_, text) in enumerate(turns) if re.search(patterns[0], text, re.I)]
        if not first:
            first = [i + 1 for i, (_, text) in enumerate(turns) if any(re.search(p, text, re.I) for p in patterns)]
        out.append(first)
    return out


def transcript(turns: list[tuple[str, str]]) -> str:
    """Numbered lines with speakers renamed Speaker 1, 2 ... in the order they first talk."""
    names: dict[str, str] = {}
    rows = []
    for n, (letter, text) in enumerate(turns, start=1):
        names.setdefault(letter, f"Speaker {len(names) + 1}")
        rows.append(f"{n} {names[letter]}: {text}")
    return "\n".join(rows)


def answer(spec: dict, lines: list[list[int]]) -> dict:
    """The right answer: spec items in the order first said, fields that were not set left out."""
    items = []
    for item, ln in zip(spec["items"], lines, strict=True):
        entry = {"kind": item["kind"].replace(" ", "_")}
        for key, value in item["fields"].items():
            if value is not None and value != "":
                entry[key] = str(value) if isinstance(value, int | float) else value
        entry["lines"] = ln
        items.append(entry)
    items.sort(key=lambda e: e["lines"][0] if e["lines"] else 10**6)
    return {"items": items}


def split_of(spec: dict) -> str:
    if spec["template"] in HOLDOUT_TEMPLATES:
        return "test"
    bucket = int(hashlib.sha256(spec["id"].encode()).hexdigest(), 16) % 100
    return "validation" if bucket < VALIDATION_PERCENT else "train"


def pair(visit: dict, variant: str = "clean", heard: list[tuple[str, str]] | None = None, keep: list[int] | None = None) -> dict:
    """One training pair. For a heard variant, `heard` holds the Whisper turns and `keep` the written turn number
    (1 based) that each heard turn came from, so evidence found in the written script maps onto the heard one."""
    spec = visit["spec"]
    written = parse_turns(visit["script"])
    lines = evidence(spec, written)
    turns = written
    if heard is not None:
        assert keep is not None and len(keep) == len(heard)
        where = {old: new for new, old in enumerate(keep, start=1)}
        lines = [[where[n] for n in ln if n in where] for ln in lines]
        turns = heard
    return {
        "id": f"{spec['id']}-{variant}",
        "visit": spec["id"],
        "template": spec["template"],
        "variant": variant,
        "split": split_of(spec),
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": transcript(turns)}],
        "completion": [{"role": "assistant", "content": json.dumps(answer(spec, lines), separators=(",", ":"))}],
    }


def build(visits: Iterable[dict], heard: dict[str, dict] | None = None) -> list[dict]:
    """Pairs for every kept visit: always the written version, plus the heard version where there is one."""
    out = []
    for visit in visits:
        if not visit.get("ok"):
            continue
        out.append(pair(visit))
        h = (heard or {}).get(visit["id"])
        if h:
            out.append(pair(visit, f"heard-{h['whisper']}", [tuple(t) for t in h["turns"]], h["keep"]))
    return out


def main() -> None:
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description="Build extraction training pairs from written (and heard) visits.")
    ap.add_argument("--visits", required=True)
    ap.add_argument("--heard", help="JSONL from speech.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.visits) as f:
        visits = [json.loads(line) for line in f if line.strip()]
    heard = {}
    if args.heard:
        with open(args.heard) as f:
            heard = {row["id"]: row for row in map(json.loads, f)}
    rows = build(visits, heard)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(dict(Counter((r["split"], r["variant"].split("-")[0]) for r in rows)))


if __name__ == "__main__":
    main()
