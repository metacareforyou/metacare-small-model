import json
from pathlib import Path

from metacare_small_model import config as cfg
from metacare_small_model.manifest import IPHONE_MAX_BYTES, budget, build

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {"transformers": "4.3.1", "wasm": {"asyncify": {"path": "ort/x.wasm", "bytes": 26_861_777, "sha256": "0" * 64}}}


def export_dir(tmp_path, weight_bytes):
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_q4.onnx").write_bytes(b"graph")
    with (tmp_path / "onnx" / "model_q4.onnx_data").open("wb") as handle:
        handle.truncate(weight_bytes)  # a sparse file: the size counts, the disk stays empty
    return tmp_path


def test_manifest_has_the_app_shape(tmp_path):
    conf = cfg.load(ROOT / "configs" / "granite-h-350m.yaml")
    manifest = build(conf, export_dir(tmp_path, 1000), "a" * 40, RUNTIME)
    assert manifest["dtype"] == "q4"
    assert manifest["id"] == "granite-h-350m-health-v0-q4-aaaaaaa"
    assert [f["path"] for f in manifest["files"]][-2:] == ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data"]
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])
    assert manifest["mobile"] is True
    json.dumps(manifest)


def test_budget_refuses_an_iphone_sized_failure(tmp_path):
    conf = cfg.load(ROOT / "configs" / "granite-h-1b.yaml")
    manifest = build(conf, export_dir(tmp_path, 1_000_000_000), "b" * 40, RUNTIME)
    result = budget(manifest)
    assert manifest["mobile"] is False
    assert result["total_bytes"] >= IPHONE_MAX_BYTES
    assert len(result["problems"]) == 2


def test_budget_counts_the_runtime(tmp_path):
    conf = cfg.load(ROOT / "configs" / "granite-h-350m.yaml")
    manifest = build(conf, export_dir(tmp_path, IPHONE_MAX_BYTES - 26_000_000), "c" * 40, RUNTIME)
    assert manifest["mobile"] is False  # under the limit alone, over it once the runtime binary is counted
