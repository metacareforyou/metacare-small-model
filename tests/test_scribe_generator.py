"""The visit generator (MET-1929): every spec's own wording passes its checks, and the checks catch misses."""
from metacare_small_model.scribe.catalog import TEMPLATES, make_spec
from metacare_small_model.scribe.checks import format_problems, missing, number_pattern
import re


def test_every_template_passes_its_own_checks():
    for i in range(len(TEMPLATES) * 20):
        spec = make_spec(i)
        script = '\n'.join(f'D|{item["say"]}.' for item in spec['items'])
        assert missing(script, spec) == [], (spec['template'], missing(script, spec))


def test_numbers_as_words():
    for value, said in [('8.2', 'eight point two'), ('142', 'one hundred forty-two'), ('142', 'one forty two'), ('2000', 'two thousand'),
                        ('1200', 'twelve hundred'), ('0.25', 'zero point two five'), ('325', 'three hundred and twenty five'), ('2000', '2,000')]:
        assert re.search(number_pattern(value), said, re.I), (value, said)
    assert not re.search(number_pattern('8'), '18 kilograms')
    assert not re.search(number_pattern('2'), '2.5 mg')


def test_a_missing_dose_is_caught():
    spec = make_spec(0)
    script = '\n'.join(f'D|{item["say"]}.' for item in spec['items'][1:])
    assert missing(script, spec)


def test_format_problems():
    spec = make_spec(3)
    good = '\n'.join(f'{letter}|line {n}' for n in range(6) for letter in spec['speakers'])
    assert format_problems(good, spec) == []
    assert 'FHIR words in the dialogue' in format_problems(good + '\nD|I am placing a ServiceRequest.', spec)
    assert format_problems('Narrator: hello\n' + good, spec)


def test_specs_are_reproducible_and_varied():
    assert make_spec(5) == make_spec(5)
    assert len({make_spec(i)['template'] for i in range(32)}) == len(TEMPLATES)
