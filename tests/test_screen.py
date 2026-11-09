import pytest

from metacare_small_model.screen import screen

LEAKS = {
    "email": "Write to maria.lopez@example.com for results",
    "phone": "Call me at (305) 555-0142 tomorrow",
    "ssn": "My number is 123-45-6789",
    "date_of_birth": "DOB is in the chart",
    "numeric_date": "Seen on 3/14/2025 for a cough",
    "record_number": "Her MRN is on the wristband",
    "street_address": "I live at 1200 Brickell Bay Drive",
    "link_with_query": "See https://portal.example.org/result?id=88412",
    "self_named": "Hola, me llamo Ana y tengo fiebre",
}


@pytest.mark.parametrize(("kind", "text"), LEAKS.items())
def test_screen_catches(kind, text):
    assert kind in {f.kind for f in screen(text)}


@pytest.mark.parametrize(
    "text",
    [
        "A temperature of 100.4 degrees Fahrenheit counts as a fever.",
        "Ask a pharmacist before taking ibuprofen with blood pressure medicine.",
        "Consulte a un médico si la fiebre dura más de tres días.",
    ],
)
def test_screen_passes_clean_health_text(text):
    assert screen(text) == []
