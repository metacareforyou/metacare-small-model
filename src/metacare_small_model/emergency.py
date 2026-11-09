"""Emergency wording, kept out of what the model learns to answer.

In the app, an emergency never reaches the model: metacare-gpt-frontend checks every message in code first and
answers a crisis or a medical emergency with a fixed message (src/lib/onDeviceChat.ts, src/safety/). The model is
therefore never trained to handle one, and a training record whose question reads like an emergency is refused, so
the model does not learn a soft answer where the app would have said "call 911".

This list is a training data filter, not the app's safety check, and it is not a substitute for it. It errs toward
refusing. Evaluation items with "expect": "emergency" test that the filter (and, in the app, the real check) routes
them; they are never sent to the model.
"""

from __future__ import annotations

import re
import unicodedata

PHRASES = {
    "en": [
        "chest pain",
        "can't breathe",
        "cannot breathe",
        "not breathing",
        "stopped breathing",
        "choking",
        "overdose",
        "overdosed",
        "took too many",
        "poisoned",
        "suicide",
        "suicidal",
        "kill myself",
        "end my life",
        "hurt myself",
        "self harm",
        "unconscious",
        "passed out",
        "won't wake up",
        "stroke",
        "face drooping",
        "slurred speech",
        "seizure",
        "severe bleeding",
        "bleeding a lot",
        "won't stop bleeding",
        "anaphylaxis",
        "throat closing",
        "heart attack",
    ],
    "es": [
        "dolor de pecho",
        "dolor en el pecho",
        "no puedo respirar",
        "no respira",
        "dejó de respirar",
        "se ahoga",
        "atragant",
        "sobredosis",
        "tomé demasiad",
        "envenen",
        "suicid",
        "matarme",
        "quitarme la vida",
        "hacerme daño",
        "inconsciente",
        "se desmayó",
        "no despierta",
        "derrame cerebral",
        "infarto",
        "convulsi",
        "sangrado abundante",
        "sangra mucho",
        "no para de sangrar",
        "anafilaxia",
        "se me cierra la garganta",
    ],
}


# A count of pills taken at once reads as a possible overdose. In the MET-1385 trial a model told someone who took
# 20 extra strength Tylenol that they did not need to do anything, which is why the app now decides these in code.
PATTERNS = [
    re.compile(r"\b(?:i|he|she|they|we|my \w+)?\s*(?:took|swallowed|ate|drank)\s+(?:\d{2,}|all|a whole bottle)\b"),
    re.compile(r"\b(?:me )?(?:tomé|tome|tomó|tragué|tragó|se tomó)\s+(?:\d{2,}|todas|todos|un frasco)\b"),
]


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return re.sub(r"\s+", " ", text)


def is_emergency(text: str) -> bool:
    folded = _fold(text)
    if any(_fold(phrase) in folded for phrases in PHRASES.values() for phrase in phrases):
        return True
    return any(pattern.search(folded) for pattern in PATTERNS)
