"""A personal data screen for training and evaluation text.

This repository must never hold anyone's health information. Every record is written for training (or taken from
a licensed public source) and is screened here before it is accepted. The screen is deliberately strict: a false
alarm costs a minute of editing, a miss puts a person's data into public weights forever.

What it catches: email addresses, telephone numbers, United States Social Security numbers, card numbers, dates
of birth, record numbers, street addresses, links that carry a query string, and a person introducing themselves
by name, in English and Spanish. What it cannot catch: a bare personal name in free text. That is why every record
also needs a human review before it is used (see docs/DATA.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_STREET = (
    r"street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|way|place|pl|terrace|ter|"
    r"calle|avenida|av|camino|carretera"
)

PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    "date_of_birth": re.compile(r"\b(?:dob|date of birth|born on|fecha de nacimiento|naci[oó] el|nac[ií] el)\b", re.I),
    "numeric_date": re.compile(r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-](?:19|20)?\d{2}(?!\d)"),
    "record_number": re.compile(
        r"\b(?:mrn|medical record(?: number)?|n[uú]mero de (?:historia|expediente)|patient id|member id)\b", re.I
    ),
    "street_address": re.compile(rf"\b\d{{1,6}}\s+(?:[A-Za-zÁÉÍÓÚáéíóúñÑ]+\s+){{1,4}}(?:{_STREET})\b\.?", re.I),
    "link_with_query": re.compile(r"https?://\S+\?\S+"),
    "self_named": re.compile(r"\b(?:my name is|i am called|me llamo|mi nombre es)\s+[A-ZÁÉÍÓÚÑ]", re.I),
}


@dataclass(frozen=True)
class Finding:
    kind: str
    text: str


def screen(text: str) -> list[Finding]:
    """Every match of every pattern; an empty list means the text passed."""
    return [Finding(kind, m.group(0)) for kind, pattern in PATTERNS.items() for m in pattern.finditer(text)]
