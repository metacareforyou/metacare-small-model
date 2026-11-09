import json
import shutil
from pathlib import Path

import metacare_small_model.cli as cli

ROOT = Path(__file__).resolve().parents[1]


def test_train_dry_run_needs_no_gpu(tmp_path, monkeypatch, capsys):
    for part in ("configs", "data"):
        shutil.copytree(ROOT / part, tmp_path / part)
    approved = [
        json.loads(line) | {"review": {"status": "approved"}}
        for line in (ROOT / "data" / "examples" / "train.example.jsonl").read_text().splitlines()
    ]
    (tmp_path / "data" / "train.jsonl").write_text("\n".join(json.dumps(r) for r in approved) + "\n")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["train", "--config", "configs/granite-h-350m.yaml", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["train_examples"] + plan["validation_examples"] == 4
    assert plan["base"].startswith("ibm-granite/granite-4.0-h-350m@")


def test_data_check_command(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert cli.main(["data", "check", "--records", "data/examples/train.example.jsonl", "--draft"]) == 0
    assert cli.main(["data", "check", "--records", "data/examples/train.example.jsonl"]) == 1
    assert cli.main(["eval", "check"]) == 0
