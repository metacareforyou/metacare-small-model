"""Training records, evaluation items and the source register, with the rules every one of them must pass.

A training record is one JSON object per line:

    {"id": "en-0001", "lang": "en", "question": "...", "answer": "...", "source": "metacare-authored",
     "topic": "fever", "review": {"status": "approved", "by": "clinician:RN", "on": "2026-11-09"}}

An evaluation item is one JSON object per line:

    {"id": "eval-en-0001", "lang": "en", "question": "...", "category": "medication", "expect": "answer",
     "must_include": ["pharmacist"], "must_not": ["safe to take"], "source": "metacare-authored"}

The source register (data/sources.yaml) names where every record came from and its licence. A record whose source
is not registered, or whose licence is not on the allow list, is refused.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .emergency import is_emergency
from .screen import screen

LANGS = {"en", "es"}
REVIEW = {"unreviewed", "approved", "rejected"}
EXPECT = {"answer", "emergency"}
CATEGORIES = {"general", "medication", "mental_health", "pediatric", "pregnancy", "chronic", "nutrition", "emergency"}

# Licences whose text can be used to train and then publish open weights without a further grant. Adding one is a
# reviewed change: say why in the pull request. ShareAlike and non commercial licences are left out on purpose.
ALLOWED_LICENCES = {"Apache-2.0", "MIT", "CC0-1.0", "CC-BY-4.0", "US-Government-Work"}

# The answer style the system prompt asks for: at most five short plain sentences, no lists.
MAX_SENTENCES = 5
MAX_WORDS = 120
LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)

# Things the system prompt forbids the model to say. A training answer that says one teaches the opposite.
FORBIDDEN = [
    "safe to take",
    "safe to combine",
    "perfectly safe",
    "stop taking",
    "increase your dose",
    "double the dose",
    "you don't need to see",
    "no need to see a doctor",
    "es seguro tomar",
    "es seguro combinar",
    "deje de tomar",
    "deja de tomar",
    "aumente la dosis",
    "duplique la dosis",
    "no necesita ver",
]
CLINICIAN = re.compile(r"doctor|clinician|pharmacist|nurse|911|médic|medic|farmac|enfermer|clínic", re.I)

ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


@dataclass
class Problem:
    where: str
    message: str
    error: bool = True

    def __str__(self) -> str:
        return f"{'ERROR' if self.error else 'warn '} {self.where}: {self.message}"


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)
    count: int = 0

    @property
    def ok(self) -> bool:
        return not any(p.error for p in self.problems)

    def add(self, where: str, message: str, error: bool = True) -> None:
        self.problems.append(Problem(where, message, error))


def read_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, json.loads(line)


def load_sources(path: Path) -> dict[str, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {entry["id"]: entry for entry in data.get("sources", [])}


def check_sources(sources: dict[str, dict]) -> Report:
    report = Report(count=len(sources))
    for sid, entry in sources.items():
        for key in ("name", "url", "licence", "retrieved_on"):
            if not entry.get(key):
                report.add(f"source {sid}", f"missing {key}")
        if entry.get("licence") and entry["licence"] not in ALLOWED_LICENCES:
            report.add(f"source {sid}", f"licence {entry['licence']!r} is not on the allow list {sorted(ALLOWED_LICENCES)}")
    return report


def normalise(question: str) -> str:
    return re.sub(r"[^\w]+", " ", question.lower()).strip()


def sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def _common(where: str, item: dict, sources: dict[str, dict], seen: set[str], report: Report) -> None:
    if not ID.match(str(item.get("id", ""))):
        report.add(where, "id must be 3 to 64 lowercase letters, digits or hyphens")
    elif item["id"] in seen:
        report.add(where, f"duplicate id {item['id']}")
    seen.add(str(item.get("id")))
    if item.get("lang") not in LANGS:
        report.add(where, f"lang must be one of {sorted(LANGS)}")
    if not str(item.get("question", "")).strip():
        report.add(where, "question is empty")
    if item.get("source") not in sources:
        report.add(where, f"source {item.get('source')!r} is not in the source register")
    for name in ("question", "answer"):
        for finding in screen(str(item.get(name, ""))):
            report.add(where, f"personal data screen: {finding.kind} in {name} ({finding.text!r})")


def check_records(records: Iterable[tuple[int, dict]], sources: dict[str, dict], *, for_training: bool) -> Report:
    report, seen = Report(), set()
    for number, record in records:
        report.count += 1
        where = f"line {number} ({record.get('id', '?')})"
        _common(where, record, sources, seen, report)
        answer = str(record.get("answer", "")).strip()
        if not answer:
            report.add(where, "answer is empty")
        if is_emergency(str(record.get("question", ""))):
            report.add(where, "question reads like an emergency; the app answers those in code, so the model never learns them")
        if sentences(answer) > MAX_SENTENCES:
            report.add(where, f"answer has {sentences(answer)} sentences; the prompt asks for at most {MAX_SENTENCES}")
        if len(answer.split()) > MAX_WORDS:
            report.add(where, f"answer has {len(answer.split())} words; keep it under {MAX_WORDS}")
        if LIST_LINE.search(answer):
            report.add(where, "answer uses a list; the prompt asks for no lists")
        low = answer.lower()
        for phrase in FORBIDDEN:
            if phrase in low:
                report.add(where, f"answer says {phrase!r}, which the system prompt forbids")
        if answer and not CLINICIAN.search(answer):
            report.add(where, "answer does not say when to see a clinician", error=False)
        status = (record.get("review") or {}).get("status")
        if status not in REVIEW:
            report.add(where, f"review.status must be one of {sorted(REVIEW)}")
        elif for_training and status != "approved":
            report.add(where, f"review.status is {status!r}; only approved records are trained on")
    return report


def check_eval(items: Iterable[tuple[int, dict]], sources: dict[str, dict], training_questions: set[str]) -> Report:
    report, seen = Report(), set()
    for number, item in items:
        report.count += 1
        where = f"line {number} ({item.get('id', '?')})"
        _common(where, item, sources, seen, report)
        if item.get("expect") not in EXPECT:
            report.add(where, f"expect must be one of {sorted(EXPECT)}")
        if item.get("category") not in CATEGORIES:
            report.add(where, f"category must be one of {sorted(CATEGORIES)}")
        flagged = is_emergency(str(item.get("question", "")))
        if item.get("expect") == "emergency" and not flagged:
            report.add(where, "expect is emergency but the emergency filter does not catch it; add the wording to emergency.py")
        if item.get("expect") == "answer" and flagged:
            report.add(where, "the emergency filter catches this question, so the model would never see it; expect emergency")
        if normalise(str(item.get("question", ""))) in training_questions:
            report.add(where, "the same question is in the training data; an evaluation item must be unseen")
    return report


def split_of(record_id: str, validation_share: float = 0.1) -> str:
    """A stable split: the same id always lands in the same split, whatever order the file is in."""
    bucket = int(hashlib.sha256(record_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < validation_share else "train"
