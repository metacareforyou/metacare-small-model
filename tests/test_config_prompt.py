from pathlib import Path

from metacare_small_model import config as cfg
from metacare_small_model.prompt import check, system_prompt

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_are_valid():
    for path in (ROOT / "configs").glob("*.yaml"):
        assert cfg.problems(cfg.load(path)) == [], path


def test_q4f16_refused_for_granite_hybrid(tmp_path):
    text = (ROOT / "configs" / "granite-h-350m.yaml").read_text().replace("dtype: q4", "dtype: q4f16")
    path = tmp_path / "c.yaml"
    path.write_text(text)
    assert any("q4f16" in p for p in cfg.problems(cfg.load(path)))


def test_branch_name_revision_refused(tmp_path):
    text = (ROOT / "configs" / "granite-h-350m.yaml").read_text()
    path = tmp_path / "c.yaml"
    path.write_text(text.replace("3b17b717b8f2f5d305b0a92c1491e239aeda19c8", "main"))
    assert any("40 character" in p for p in cfg.problems(cfg.load(path)))


def test_prompt_check_against_an_app_file(tmp_path):
    app = tmp_path / "onDeviceChat.ts"
    app.write_text(f"const MAX = 1;\nconst SYSTEM_PROMPT = `{system_prompt()}`;\n")
    assert check(app)
    app.write_text("const SYSTEM_PROMPT = `Be brief.`;\n")
    assert not check(app)
