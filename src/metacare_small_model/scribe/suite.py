"""The six visit scribe test visits as extraction test pairs, written and heard (MET-1928).

The visits, recordings and line timings live in metacare-gpt-frontend (src/pages/labs/scribe/suite). The right
answers are evalsets/scribe/suite_gold.json. Hearing cuts each recording at its line timings and transcribes each
piece with every browser Whisper model, so heard line n is written line n (as with perfect speaker separation):

    python -m metacare_small_model.scribe.suite hear --suite ../metacare-gpt-frontend/src/pages/labs/scribe/suite \
        --out evalsets/scribe/suite_heard.jsonl        # needs a GPU or patience, plus torch, transformers, ffmpeg

`rows()` gives test pairs in the same form as scribe.pairs, for finetune.py to score.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .pairs import SYSTEM, transcript

HERE = Path(__file__).resolve().parents[3] / "evalsets" / "scribe"


def gold(path: Path = HERE / "suite_gold.json") -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def answer(items: list[dict], where: dict[int, int] | None = None) -> str:
    out = []
    for item in items:
        lines = item["lines"] if where is None else [where[n] for n in item["lines"] if n in where]
        out.append({**{k: v for k, v in item.items() if k != "lines"}, "lines": lines})
    return json.dumps({"items": out}, separators=(",", ":"))


def rows(cases_path: Path, heard_path: Path | None = None, gold_path: Path = HERE / "suite_gold.json") -> list[dict]:
    """Test pairs: each visit as written, and as heard by each Whisper model when `heard_path` is given."""
    cases = {c["id"]: c for c in json.loads(Path(cases_path).read_text(encoding="utf-8"))["cases"]}
    right = gold(gold_path)
    out = []

    def add(cid: str, variant: str, turns: list[tuple[str, str]], where: dict[int, int] | None) -> None:
        out.append({
            "id": f"suite-{cid}-{variant}", "visit": cid, "template": cid, "variant": variant, "split": "suite",
            "seen_type": right[cid]["seen_type"],
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": transcript(turns)}],
            "completion": [{"role": "assistant", "content": answer(right[cid]["items"], where)}],
        })  # fmt: skip

    for cid, case in cases.items():
        add(cid, "clean", [(line["s"], line["t"]) for line in case["lines"]], None)
    if heard_path and Path(heard_path).exists():
        for h in map(json.loads, Path(heard_path).read_text(encoding="utf-8").splitlines()):
            where = {old: new for new, old in enumerate(h["keep"], start=1)}
            add(h["case"], f"heard-{h['whisper']}", [tuple(t) for t in h["turns"]], where)
    return out


def hear(suite: Path, out_path: Path) -> None:
    import numpy as np
    import torch
    from transformers import pipeline

    from .speech import WHISPER

    cases = json.loads((suite / "cases.json").read_text(encoding="utf-8"))["cases"]
    timings = json.loads((suite / "timings.json").read_text(encoding="utf-8"))
    audio = {}
    for case in cases:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(suite / f"scribeTrial-{case['id']}.mp3"), "-f", "f32le",
                              "-ac", "1", "-ar", "16000", "-"], check=True, capture_output=True).stdout  # fmt: skip
        audio[case["id"]] = np.frombuffer(raw, dtype=np.float32)
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if device == 0 else torch.float32
    with open(out_path, "w") as out:
        for name, (repo, _) in WHISPER.items():
            asr = pipeline("automatic-speech-recognition", model=repo, torch_dtype=dtype, device=device)
            for case in cases:
                wave, spans = audio[case["id"]], timings[case["id"]]
                assert len(spans) == len(case["lines"]), case["id"]
                pieces = [wave[int(a * 16000) : int(b * 16000) + 1600] for a, b, _ in spans]  # a tenth of a second after
                heard = asr([{"raw": p, "sampling_rate": 16000} for p in pieces], batch_size=8)
                turns, keep = [], []
                for n, ((_, _, speaker), h) in enumerate(zip(spans, heard, strict=True), start=1):
                    if h["text"].strip():
                        turns.append((speaker, h["text"].strip()))
                        keep.append(n)
                out.write(json.dumps({"case": case["id"], "whisper": name, "turns": turns, "keep": keep}) + "\n")
                print(name, case["id"], len(turns), "of", len(spans), "lines heard", flush=True)
            del asr


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    h = sub.add_parser("hear")
    h.add_argument("--suite", required=True, type=Path)
    h.add_argument("--out", default=HERE / "suite_heard.jsonl", type=Path)
    args = ap.parse_args()
    hear(args.suite, args.out)


if __name__ == "__main__":
    main()
