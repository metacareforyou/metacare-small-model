"""Checks that a written script states every right answer (MET-1929).

A value such as "8.2" may be written as digits or as words ("eight point two"); names are matched
without regard to case. Values given as patterns (containing |, ., ?) are used as written.
"""

from __future__ import annotations

import re

ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SEP = r"[\s-]+"


def int_words(n: int) -> list[str]:
    """Ways to say a whole number aloud, as regex fragments."""
    if n < 20:
        return [ONES[n]]
    if n < 100:
        t, o = divmod(n, 10)
        return [TENS[t] + ("" if o == 0 else SEP + ONES[o])]
    if n < 1000:
        h, rest = divmod(n, 100)
        tail = int_words(rest) if rest else [""]
        out = [ONES[h] + SEP + "hundred" + ("" if not rest else f"(?:{SEP}and)?{SEP}" + t) for t in tail]
        if rest and h < 10 and n >= 110:  # "one forty two" style
            out += [ONES[h] + SEP + t for t in int_words(rest)] if rest >= 10 else []
        return out
    if n < 10000:
        th, rest = divmod(n, 1000)
        out = [
            ONES[th] + SEP + "thousand" + ("" if not rest else f"(?:{SEP}and)?{SEP}" + t)
            for t in (int_words(rest) if rest else [""])
        ]
        if n % 100 == 0 and n < 2000 or (1100 <= n < 2000):  # twelve hundred
            hundreds, rest2 = divmod(n, 100)
            if hundreds < 20:
                out.append(ONES[hundreds] + SEP + "hundred" + ("" if not rest2 else SEP + int_words(rest2)[0]))
        return out
    return []


def number_pattern(value: str) -> str:
    """A pattern matching a number written as digits or as words."""
    digits = re.escape(value)
    if re.fullmatch(r"\d{4,}", value):
        digits = re.escape(value[:-3]) + ",?" + re.escape(value[-3:])
    alts = [r"(?<![\d.])" + digits + r"(?!\d|\.\d)"]
    m = re.fullmatch(r"(\d+)(?:\.(\d+))?", value)
    if m:
        whole = int(m.group(1))
        frac = m.group(2)
        for w in int_words(whole):
            if frac is None:
                alts.append(r"\b" + w + r"\b")
            else:
                alts.append(r"\b" + w + SEP + "point" + "".join(SEP + ONES[int(c)] for c in frac) + r"\b")
    return "(?:" + "|".join(alts) + ")"


def to_pattern(value: str) -> str:
    if re.search(r"[|?\\]|\.\?|\.\*", value):
        return value
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return number_pattern(value)
    return re.escape(value)


def checks_for(spec: dict) -> list[tuple[str, str]]:
    """(item say text, pattern) for every value the script must state."""
    out = []
    for item in spec["items"]:
        for value in item["check"]:
            out.append((item["say"], to_pattern(str(value))))
    return out


def missing(script: str, spec: dict) -> list[str]:
    return [say for say, pattern in checks_for(spec) if not re.search(pattern, script, re.I)]


JARGON = re.compile(
    r"ServiceRequest|MedicationRequest|MedicationStatement|DeviceRequest|Observation\b|Immunization\b|FHIR|LETTER\|", re.I
)


def format_problems(script: str, spec: dict) -> list[str]:
    rows = [r.strip() for r in script.strip().split("\n") if r.strip()]
    problems = []
    letters = set(spec["speakers"])
    bad = [r for r in rows if not re.match(r"^[A-Z]\|.+", r) or r[0] not in letters]
    if bad:
        problems.append(f"{len(bad)} lines not in LETTER|text form with a known letter")
    if len(rows) < 10:
        problems.append(f"only {len(rows)} turns")
    used = {r[0] for r in rows if re.match(r"^[A-Z]\|", r)}
    if letters - used:
        problems.append("unused speaker " + ",".join(sorted(letters - used)))
    if JARGON.search(script):
        problems.append("FHIR words in the dialogue")
    if len(rows) - len(set(rows)) > 1:
        problems.append("repeated lines")
    return problems
