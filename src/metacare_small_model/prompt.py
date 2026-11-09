"""The system prompt the model is trained with, which must be the one the app sends at run time.

A model fine tuned under one system prompt and run under another loses much of what the tuning bought, so the
prompt is a pinned file here and `msm prompt check` compares it with the app's source.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

# The app keeps its prompt in metacare-gpt-frontend, src/lib/onDeviceChat.ts, as a template literal.
APP_PROMPT = re.compile(r"const SYSTEM_PROMPT = `(.*?)`;", re.S)


def system_prompt() -> str:
    return files("metacare_small_model").joinpath("prompts/system.md").read_text(encoding="utf-8").strip()


def app_prompt(source: Path) -> str:
    match = APP_PROMPT.search(source.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"{source}: no `const SYSTEM_PROMPT = `...`;` found")
    return match.group(1).strip()


def check(source: Path) -> bool:
    return app_prompt(source) == system_prompt()
