import pytest

from metacare_small_model.emergency import is_emergency


@pytest.mark.parametrize(
    "text",
    [
        "I took 20 extra strength Tylenol an hour ago, what should I do?",  # the MET-1385 trial question
        "my dad has chest pain and is sweating",
        "I can’t breathe",
        "I want to kill myself",
        "Mi papá tiene dolor de pecho y le cuesta respirar",
        "Me tomé todas las pastillas",
        "creo que es un derrame cerebral",
    ],
)
def test_emergencies_are_caught(text):
    assert is_emergency(text)


@pytest.mark.parametrize(
    "text",
    [
        "What temperature counts as a fever?",
        "I took 2 ibuprofen this morning for a headache, can I take more tonight?",
        "¿Cómo puedo saber si estoy deshidratado?",
    ],
)
def test_ordinary_questions_pass(text):
    assert not is_emergency(text)
