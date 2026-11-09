from pathlib import Path

from metacare_small_model.records import check_eval, check_records, check_sources, load_sources, read_jsonl, split_of

ROOT = Path(__file__).resolve().parents[1]
SOURCES = load_sources(ROOT / "data" / "sources.yaml")


def record(**overrides):
    base = {
        "id": "t-0001",
        "lang": "en",
        "question": "What helps a sore throat?",
        "answer": "Warm drinks, rest and throat lozenges help most sore throats. See a doctor if it lasts more than a week.",
        "source": "metacare-authored",
        "review": {"status": "approved"},
    }
    return base | overrides


def errors(rec, for_training=True):
    return [p.message for p in check_records([(1, rec)], SOURCES, for_training=for_training).problems if p.error]


def test_register_and_examples_are_valid():
    assert check_sources(SOURCES).ok
    examples = list(read_jsonl(ROOT / "data" / "examples" / "train.example.jsonl"))
    assert check_records(examples, SOURCES, for_training=False).ok


def test_good_record_passes():
    assert errors(record()) == []


def test_unreviewed_is_refused_for_training_only():
    rec = record(review={"status": "unreviewed"})
    assert any("only approved" in e for e in errors(rec))
    assert errors(rec, for_training=False) == []


def test_unknown_source_and_licence():
    assert any("source register" in e for e in errors(record(source="reddit")))
    report = check_sources({"x": {"name": "x", "url": "u", "licence": "CC-BY-NC-4.0", "retrieved_on": "2026-11-09"}})
    assert not report.ok


def test_emergency_question_is_refused():
    assert any("emergency" in e for e in errors(record(question="My husband has chest pain, what do I do?")))


def test_style_rules():
    long = " ".join(["This is a sentence."] * 6) + " See a doctor."
    assert any("sentences" in e for e in errors(record(answer=long)))
    assert any("list" in e for e in errors(record(answer="Try this:\n- rest\n- fluids\nSee a doctor.")))
    assert any("forbids" in e for e in errors(record(answer="It is safe to take both. See a doctor if worse.")))


def test_personal_data_is_refused():
    assert any("personal data" in e for e in errors(record(question="I am 45, email me at a@b.co")))


def test_duplicate_ids():
    report = check_records([(1, record()), (2, record())], SOURCES, for_training=True)
    assert any("duplicate" in p.message for p in report.problems)


def test_eval_items_and_overlap():
    items = list(read_jsonl(ROOT / "evalsets" / "health-qa-v0" / "items.jsonl"))
    assert check_eval(items, SOURCES, set()).ok
    leaked = {"is it okay to take tylenol and advil at the same time"}
    assert not check_eval(items, SOURCES, leaked).ok


def test_eval_emergency_must_be_caught():
    item = {
        "id": "e-0001",
        "lang": "en",
        "question": "What is a cold?",
        "category": "emergency",
        "expect": "emergency",
        "source": "metacare-authored",
    }
    assert not check_eval([(1, item)], SOURCES, set()).ok


def test_split_is_stable():
    assert split_of("en-0001") == split_of("en-0001")
    assert {split_of(f"id-{n}") for n in range(200)} == {"train", "validation"}
