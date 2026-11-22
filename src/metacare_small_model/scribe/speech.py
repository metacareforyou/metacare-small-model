"""Hears each written visit the way the browser scribe would, for training on real transcription errors (MET-1930).

Every turn is spoken by an open text to speech model (Kokoro 82M, Apache 2.0) in a voice chosen for its speaker,
roughened with noise and sometimes a phone line, and transcribed by one of the Whisper models the browser offers.
A few turns get the wrong speaker, as diarization sometimes does. The stock test suite voices (ElevenLabs) are
never used here, so the test recordings stay unseen.

    python -m metacare_small_model.scribe.speech --visits visits.jsonl --out heard.jsonl

Needs a GPU and `pip install kokoro soundfile transformers torch torchaudio` plus espeak-ng. Resumes from --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time

from .pairs import parse_turns

# The browser's models (metacare-gpt-frontend src/localai/scribeManifest.*), as full weights, and how often each is
# used: the phone models most, since that is where a small extraction model matters most.
WHISPER = {
    "tiny.en": ("openai/whisper-tiny.en", 30),
    "base.en": ("openai/whisper-base.en", 35),
    "small.en": ("openai/whisper-small.en", 20),
    "turbo": ("openai/whisper-large-v3-turbo", 15),
}
VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_aoede", "af_kore", "af_nova", "af_river",
    "am_adam", "am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam", "am_onyx",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily", "bm_george", "bm_fable", "bm_lewis", "bm_daniel",
]  # fmt: skip
WRONG_SPEAKER = 0.04
PHONE_LINE = 0.2


def rng_for(visit_id: str, seed: int) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{seed}:{visit_id}".encode()).hexdigest(), 16))


def whisper_for(rng: random.Random) -> str:
    names = list(WHISPER)
    return rng.choices(names, weights=[WHISPER[n][1] for n in names])[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1930)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torchaudio.functional as AF
    from kokoro import KPipeline
    from transformers import pipeline

    with open(args.visits) as f:
        visits = [v for v in map(json.loads, f) if v.get("ok")]
    if args.limit:
        visits = visits[: args.limit]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    plan: dict[str, list[tuple[dict, random.Random]]] = {name: [] for name in WHISPER}
    for v in visits:
        if v["id"] in done:
            continue
        rng = rng_for(v["id"], args.seed)
        plan[whisper_for(rng)].append((v, rng))
    print({k: len(x) for k, x in plan.items()}, "to hear;", len(done), "already done", flush=True)

    speakers = {"a": KPipeline(lang_code="a"), "b": KPipeline(lang_code="b")}

    def speak(text: str, voice: str):
        parts = [audio for _, _, audio in speakers[voice[0]](text, voice=voice)]
        wave = torch.cat([torch.as_tensor(p).float().flatten() for p in parts]) if parts else torch.zeros(2400)
        return AF.resample(wave, 24000, 16000)

    def roughen(wave, rng: random.Random, phone: bool):
        if phone:
            wave = AF.highpass_biquad(AF.lowpass_biquad(wave, 16000, 3400.0), 16000, 300.0)
        snr = rng.uniform(12, 35)
        power = wave.pow(2).mean().clamp_min(1e-8)
        noise = torch.randn(wave.shape, generator=torch.Generator().manual_seed(rng.randrange(2**31)))
        return (wave + noise * (power / 10 ** (snr / 10)).sqrt()).clamp(-1, 1)

    started, n = time.time(), 0
    with open(args.out, "a") as out:
        for name, jobs in plan.items():
            if not jobs:
                continue
            asr = pipeline("automatic-speech-recognition", model=WHISPER[name][0], torch_dtype=torch.float16, device=0)
            for visit, rng in jobs:
                turns = parse_turns(visit["script"])
                letters = sorted({t[0] for t in turns})
                voice = dict(zip(letters, rng.sample(VOICES, len(letters)), strict=True))
                phone_letter = rng.choice(letters) if rng.random() < PHONE_LINE else None
                waves = [roughen(speak(text, voice[s]), rng, s == phone_letter).numpy().astype(np.float32) for s, text in turns]
                heard = asr([{"raw": w, "sampling_rate": 16000} for w in waves], batch_size=16)
                rows, keep = [], []
                for i, ((letter, _), h) in enumerate(zip(turns, heard, strict=True), start=1):
                    text = h["text"].strip()
                    if not text:
                        continue  # Whisper heard nothing: the line is lost, as it would be in the browser
                    if len(letters) > 1 and rng.random() < WRONG_SPEAKER:
                        letter = rng.choice([x for x in letters if x != letter])
                    rows.append((letter, text))
                    keep.append(i)
                out.write(json.dumps({"id": visit["id"], "whisper": name, "voices": voice, "phone": phone_letter,
                                      "turns": rows, "keep": keep}) + "\n")  # fmt: skip
                out.flush()
                n += 1
                if n % 25 == 0:
                    print(f"{n} heard, {(time.time() - started) / n:.1f} s each, model {name}", flush=True)
            del asr
            torch.cuda.empty_cache()
    print(f"done: {n} heard in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
