import json

from metacare_small_model.scribe.catalog import make_spec
from metacare_small_model.scribe.pairs import HOLDOUT_TEMPLATES, build, pair, parse_turns, split_of
from metacare_small_model.scribe.score import score

SPEC = {
    "id": "v99999",
    "template": "hypertension",
    "items": [
        {
            "kind": "medication",
            "fields": {"name": "lisinopril", "dose": "10", "unit": "mg", "action": "start", "duration": None},
            "check": ["lisinopril", "10"],
        },  # fmt: skip
        {"kind": "observation", "fields": {"name": "blood pressure", "value": "152/94", "unit": "mmHg"}, "check": ["152"]},
        {"kind": "follow up", "fields": {"when": "in 4 weeks"}, "check": []},
    ],
}
SCRIPT = "D|Your blood pressure is 152 over 94.\nP|That's high.\nD|Let's start lisinopril 10 mg.\nD|See you in 4 weeks."


def test_pair_answer_has_items_in_order_said_with_lines():
    row = pair({"spec": SPEC, "script": SCRIPT})
    items = json.loads(row["completion"][0]["content"])["items"]
    assert [i["kind"] for i in items] == ["observation", "medication", "follow_up"]
    assert items[0]["lines"] == [1] and items[1]["lines"] == [3] and items[2]["lines"] == [4]
    assert "duration" not in items[1]
    assert row["prompt"][1]["content"].startswith("1 Speaker 1: Your blood pressure")
    assert "3 Speaker 1: Let's start" in row["prompt"][1]["content"]


def test_heard_variant_maps_lines_past_a_lost_turn():
    heard = [("D", "your blood pressure is 152 over 94"), ("D", "let's start lice in a pril 10 mg"), ("D", "see you in 4 weeks")]
    row = pair({"spec": SPEC, "script": SCRIPT}, "heard-tiny.en", heard, keep=[1, 3, 4])
    items = json.loads(row["completion"][0]["content"])["items"]
    assert [i["lines"] for i in items] == [[1], [2], [3]]
    assert items[1]["name"] == "lisinopril"  # the right answer stays the written name


def test_whole_templates_are_held_out():
    for i in range(64):
        spec = make_spec(i)
        assert (split_of(spec) == "test") == (spec["template"] in HOLDOUT_TEMPLATES)


def test_build_skips_failed_visits():
    assert build([{"ok": False, "spec": SPEC, "script": SCRIPT}]) == []
    assert parse_turns("D|hi\nnarration\nP|  \nP|yes") == [("D", "hi"), ("P", "yes")]


def test_score_perfect_and_wrong_answers():
    gold = pair({"spec": SPEC, "script": SCRIPT})["completion"][0]["content"]
    perfect = score([(gold, gold)])
    assert perfect["json_failed"] == 0
    assert all(v["precision"] == 1 and v["recall"] == 1 for v in perfect["fields"].values())
    wrong = json.loads(gold)
    wrong["items"][1]["dose"] = "twenty"
    wrong["items"].append({"kind": "condition", "name": "made up", "lines": [2]})
    s = score([(gold, json.dumps(wrong))])
    assert s["fields"]["medication.dose"]["recall"] == 0
    assert s["items"]["condition"]["precision"] == 0
    assert score([(gold, "not json")])["json_failed"] == 1


def test_suite_gold_covers_every_case_and_scores_perfectly():
    from metacare_small_model.scribe.suite import HERE, gold, rows

    cases = json.loads((HERE / "suite_cases.json").read_text())["cases"]
    right = gold()
    assert {c["id"] for c in cases} == set(right)
    for case in cases:
        for item in right[case["id"]]["items"]:
            assert item["lines"] and all(1 <= n <= len(case["lines"]) for n in item["lines"]), (case["id"], item)
    suite = rows(HERE / "suite_cases.json", HERE / "suite_heard.jsonl")
    assert len(suite) == 6 * 5  # written, and heard by four Whisper models
    s = score([(r["completion"][0]["content"], r["completion"][0]["content"]) for r in suite])
    assert s["json_failed"] == 0 and all(v["recall"] == 1 for v in s["items"].values())
