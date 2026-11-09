"""Merge the adapter, export to ONNX, and quantize to 4 bit weights for Transformers.js in the browser.

The output folder has the layout Transformers.js asks for, the same as the onnx-community exports the app
already runs:

    config.json  generation_config.json  tokenizer.json  tokenizer_config.json
    onnx/model_q4.onnx  onnx/model_q4.onnx_data

Open question, stated plainly: the published Granite 4.0 H ONNX files come from onnx-community, and whether
`optimum` exports a fine tuned Granite hybrid (Mamba2 plus attention) model with its state cache in the shape
Transformers.js 4.3.1 expects has not been tested here yet. `msm export --check-against` compares the input and
output names of our export with the published one, so a mismatch fails here instead of on a phone.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .config import HALF_PRECISION_BROKEN, Config

TOKENIZER_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]


def refuse_broken_dtype(config: Config) -> None:
    if config.get("export.dtype") == "q4f16" and config.get("base.model_type") in HALF_PRECISION_BROKEN:
        raise SystemExit("q4f16 overflows on WebGPU for Granite hybrid models (answers come back as '!!!!'); use q4")


def merge(config: Config, adapter: Path, out: Path) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo, revision = config.get("base.repo"), config.get("base.revision")
    base = AutoModelForCausalLM.from_pretrained(repo, revision=revision, torch_dtype="float32")
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    target = out / "merged"
    merged.save_pretrained(target)
    AutoTokenizer.from_pretrained(repo, revision=revision).save_pretrained(target)
    return target


def to_onnx(merged: Path, out: Path) -> Path:
    raw = out / "onnx-fp32"
    subprocess.run(
        ["optimum-cli", "export", "onnx", "--model", str(merged), "--task", "text-generation-with-past", str(raw)],
        check=True,
    )
    return raw


def quantize_q4(raw: Path, out: Path, block_size: int = 32) -> Path:
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model = onnx.load(str(raw / "model.onnx"), load_external_data=True)
    quantizer = MatMulNBitsQuantizer(model, block_size=block_size, is_symmetric=True)
    quantizer.process()
    target = out / "onnx"
    target.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        quantizer.model.model,
        str(target / "model_q4.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_q4.onnx_data",
    )
    for name in TOKENIZER_FILES:
        if (raw / name).exists():
            shutil.copy2(raw / name, out / name)
    return out


def io_names(model_path: Path) -> dict[str, list[str]]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return {"inputs": [i.name for i in model.graph.input], "outputs": [o.name for o in model.graph.output]}


def compare(ours: Path, published: Path) -> list[str]:
    """Differences in graph inputs and outputs between our export and a published one that runs in the app."""
    a, b = io_names(ours), io_names(published)
    return [
        f"{side}: ours only {sorted(set(a[side]) - set(b[side]))}, published only {sorted(set(b[side]) - set(a[side]))}"
        for side in ("inputs", "outputs")
        if set(a[side]) != set(b[side])
    ]


def run(config: Config, adapter: Path, out: Path, check_against: Path | None = None) -> dict:
    refuse_broken_dtype(config)
    out.mkdir(parents=True, exist_ok=True)
    merged = merge(config, adapter, out)
    raw = to_onnx(merged, out)
    final = quantize_q4(raw, out / "final")
    report = {"dir": str(final), "dtype": config.get("export.dtype")}
    if check_against:
        report["io_differences"] = compare(final / "onnx" / "model_q4.onnx", check_against)
    (out / "export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
