"""Run configuration: one YAML file per model, in configs/."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SHA = re.compile(r"^[0-9a-f]{40}$")
DTYPES = {"q4", "q4f16", "q8", "fp16", "fp32"}
# Granite 4.0 H models are hybrid Mamba2 and attention. Their q4f16 export answers with a string of "!" on WebGPU
# (half precision overflow), so it is refused here; q4 (4 bit weights, full precision activations) answers well.
HALF_PRECISION_BROKEN = {"granitemoehybrid"}


@dataclass
class Config:
    path: Path
    raw: dict

    @property
    def name(self) -> str:
        return self.raw["name"]

    def get(self, dotted: str, default=None):
        node = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load(path: Path) -> Config:
    return Config(path, yaml.safe_load(path.read_text(encoding="utf-8")))


def problems(config: Config) -> list[str]:
    out = []
    for key in (
        "name",
        "base.repo",
        "base.revision",
        "base.licence",
        "base.model_type",
        "data.train",
        "data.sources",
        "lora.r",
        "training.epochs",
        "export.dtype",
        "manifest.cache_name",
        "manifest.disclosure.name",
    ):
        if config.get(key) in (None, ""):
            out.append(f"{config.path}: {key} is missing")
    if config.get("base.revision") and not SHA.match(str(config.get("base.revision"))):
        out.append(f"{config.path}: base.revision must be a full 40 character commit, not a branch name")
    dtype = config.get("export.dtype")
    if dtype and dtype not in DTYPES:
        out.append(f"{config.path}: export.dtype {dtype!r} is not one of {sorted(DTYPES)}")
    if dtype == "q4f16" and config.get("base.model_type") in HALF_PRECISION_BROKEN:
        out.append(f"{config.path}: q4f16 overflows on WebGPU for {config.get('base.model_type')}; use q4")
    return out
