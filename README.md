# MetaCare small model

**Small, open models that run in a phone's browser, so what a person says or asks never leaves the phone.**

[![CI](https://github.com/metacareforyou/metacare-small-model/actions/workflows/ci.yml/badge.svg)](https://github.com/metacareforyou/metacare-small-model/actions/workflows/ci.yml)
[![Licence: Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-blue.svg)](LICENSE)

This repository holds the code to make, train, score and export two small models for
[Metacare.ai](https://metacare.ai), both fine tuned from IBM Granite 4.0 H 350M and exported to 4 bit ONNX files that
run in the browser with Transformers.js:

1. **MetaCare Ember, the visit scribe.** Its model reads the transcript of a medical visit and writes down the facts that were said
   (medicines and doses, results, problems, orders, advice, follow up) in fixed fields, each with the transcript lines
   it came from. Everything after that (fixing misheard drug names, codes, FHIR) is plain code in the app. English.
2. **The health question model.** It answers a person's health question in a few plain sentences. English and
   Spanish. Its tooling is built; no data has been reviewed and it has not been trained.

Code under Apache 2.0. Weights and evaluation sets will be published openly with model cards when they are ready.

> Status, November 22, 2026: the Ember model's first training run is under way. No trained weights are published
> yet and no scores are final. Read [Where things stand](#where-things-stand) before relying on anything.

---

## Where things stand

| Piece | State |
|---|---|
| Ember: visit generator, 16 visit types, every fact checked in the script | Built and tested; first run wrote 2,000 visits, 1,870 kept |
| Ember: synthetic speech and Whisper transcripts of each visit | Built; first run under way |
| Ember: training pairs, LoRA fine tune, scoring by field | Built; first training run under way |
| Ember: test suite of six recorded visits, with right answers | Built; transcripts by four Whisper models included |
| Export of a fine tuned Granite 4.0 H 350M to q4 ONNX | Built and checked on the base model (see [Export](#export-to-the-browser)) |
| Health model: data rules, emergency filter, prompt check, evaluation runner | Built and tested; never run on a trained model |
| Health model: training data | Four unreviewed examples only |
| Health model: evaluation set | Six seed items; clinicians have not written or rated any yet |
| Published weights | None yet |

## Why small models on the phone

Metacare.ai keeps a person's records on their own device, with no account. The app already has an on device path
([metacare-gpt-frontend](https://github.com/metacareforyou/metacare-gpt-frontend) is private; the relevant parts are
described here). It downloads a pinned model from MetaCare's own host, checks every file's size and SHA 256, runs it
in a web worker and never falls back to a server. The model has to fit a phone's browser:

| Model on the device | Size at q4, with runtime | What happened |
|---|---|---|
| SmolLM3 3B | about 2.1 GB | Refused on iPhone: iOS closed a page holding 1.5 GB |
| Granite 4.0 H 1B | about 1.0 GB | Would not install on an iPhone; Chrome on a test Mac refused to cache the single 1 GB file |
| Granite 4.0 H 350M | about 291 MB | Installs on an iPhone; the base model has answered on an iPhone 18 (November 9, 2026); answer time on phones is not yet measured |

The iPhone budget the app enforces is 600,000,000 bytes, counting the ONNX Runtime binary.

## MetaCare Ember: the visit scribe model

Ember records a visit, turns it into text on the device, and drafts the notes, the structured records and the order
suggestions for a clinician to review and sign. This model is the part that pulls the facts out of the transcript.

```
 scribe/catalog.py      16 visit types; each made up visit gets a spec: speakers, style, and the facts that must be said
        │
 scribe/generate.py     an open model writes the dialogue; scribe/checks.py keeps it only if every fact is said,
        │               the format holds and no FHIR words leak in (up to 3 attempts)
        │
 scribe/speech.py       each turn is spoken by Kokoro 82M in its own voice, with noise and sometimes a phone line,
        │               and transcribed by one of the Whisper models the browser offers
        │
 scribe/pairs.py        transcript (written, or as heard) ──► the right answer, taken from the spec, never a model
        │
 scribe/finetune.py     LoRA on Granite 4.0 H 350M, then scores held out visit types and the test suite
        │               with scribe/score.py (precision and recall for each kind of fact and each field)
        │
 scribe/export_onnx.py  the browser's q4 files, size against the iPhone budget, and a smoke generation
```

What matters, and why:

- **The right answers come from specs, not from a model.** A script writer model (Qwen3 14B, Apache 2.0, run with
  llama.cpp on MetaCare's own hardware) only writes the dialogue around facts fixed in advance, and the script is
  checked for every one of them. Writers whose licence limits training other models on their output are not used.
  On one rented L4, 2,000 visits took 157 minutes; 1,870 passed every check.
- **It trains on what Whisper actually hears.** Each visit is also spoken by an open text to speech model and
  transcribed by the browser's Whisper models (tiny, base, small, and large v3 turbo), with room noise, phone lines and
  a few turns given to the wrong speaker, so the model learns to read misheard words ("Levithyroxine", "fura semide").
- **Whole visit types are held out.** Thyroid, migraine and knee follow up visits never appear in training, so the
  score shows whether the model learned to extract rather than to recognise sixteen visits.
- **A separate test suite of six recorded visits** ([`evalsets/scribe`](evalsets/scribe)), read by stock voices,
  with hand written right answers for every fact and each visit as heard by four Whisper models. Four of the six are
  visit types the training data also has, so their scores are reported apart from the two that are not.
- **The model writes no codes and no FHIR.** It copies what was said into fixed fields. Codes and FHIR resources are
  added by tested code in the app, from code lists confirmed against MetaCare's terminology server, so the model has
  nothing to make up.

Run it (generation needs an OpenAI compatible server such as llama.cpp's; speech and training need a GPU):

```bash
python -m metacare_small_model.scribe.generate --out data/private/scribe/visits.jsonl --count 2000 \
    --server http://127.0.0.1:8099 --model <model and revision> --parallel 8
python -m metacare_small_model.scribe.speech --visits data/private/scribe/visits.jsonl --out data/private/scribe/heard.jsonl
python -m metacare_small_model.scribe.pairs --visits data/private/scribe/visits.jsonl \
    --heard data/private/scribe/heard.jsonl --out data/private/scribe/pairs.jsonl
python -m metacare_small_model.scribe.finetune --config configs/scribe/granite-h-350m-scribe.yaml \
    --pairs data/private/scribe/pairs.jsonl --baseline 100
```

Generated visits, transcripts, pairs and weights stay out of the repository (`data/private/` and `runs/` are
ignored); only the code, the configs and the test suite are here.

## The health question model

The goal is a fine tuned 350M model whose answers a clinician would be comfortable sending to a patient as written.

```
 data/train.jsonl ──► msm data check ──► msm train ──► runs/<name>/adapter
 data/sources.yaml        (licences, personal data,        (LoRA on Granite 4.0 H,
                           emergencies, answer style)        pinned revision)
                                                                  │
 evalsets/health-qa-v0 ──► msm eval run ◄─────────────────────────┤
                                 │                                │
                      msm eval sheet ──► clinicians rate ──► msm eval summarize
                                                                  │
                                                     export (see below)
                                                                  │
                                             msm manifest ──► chatManifest.<name>.json
                                             msm budget        (sizes, SHA 256, runtime;
                                                                 fits the iPhone or not)
```

- **Base models are pinned to a commit**, never a branch. `ibm-granite/granite-4.0-h-350m` and `-1b` are Apache 2.0.
- **Prompt and completion pairs** put the loss on the answer only. The prompt is the exact system prompt the app
  sends ([`prompts/system.md`](src/metacare_small_model/prompts/system.md)); check it still matches with
  `msm prompt --app <path to metacare-gpt-frontend>/src/lib/onDeviceChat.ts`.
- **Only approved records train.** `msm train` refuses to start if `msm data check` finds any error, and each record's
  id decides its split, so reordering the file changes nothing.
- **Evaluation is clinicians'.** Automatic checks (empty answers, too long, lists, forbidden phrases, missing advice
  on when to see a clinician, wrong language) are filters; clinicians rate accuracy, safety, harm if followed, plain
  language and whether they would send the answer as written. See [docs/EVALUATION.md](docs/EVALUATION.md).
- **Emergencies are decided in code, never by the model.** The app answers a crisis or an emergency with a fixed
  message, and [`emergency.py`](src/metacare_small_model/emergency.py) refuses any training question that reads like
  one, in English or Spanish. In a trial of an earlier on device model, the model told someone who had taken 20 extra
  strength Tylenol that they did not need to do anything; that is why.

## Export to the browser

```bash
python -m metacare_small_model.scribe.export_onnx --merged runs/<name>/merged --out runs/<name>/onnx \
    --runtime-from <path to metacare-gpt-frontend>/src/localai/chatManifest.granite-h-350m.json \
    --fp32-check --prompt @<prompt file> --max-new-tokens 200
msm manifest --config <config> --dir runs/<name>/onnx --revision <commit of the published weights> \
    --runtime-from <path to metacare-gpt-frontend>/src/localai/chatManifest.granite-h-350m.json --out <manifest>
msm budget <manifest>
```

- **`optimum` cannot export Granite 4.0 H.** `optimum-cli export onnx` stops with "a custom or unsupported
  architecture", so `msm export`, which calls it, does not work for these models. `scribe/export_onnx.py` instead
  writes the merged weights into the graph the app already runs
  (`onnx-community/granite-4.0-h-350m-ONNX` at commit `06653818`). Fine tuning changes weights, not layers. It works for
  any merged Granite 4.0 H 350M, the health model included.
- **Checked on the base model:** the output is byte for byte the published q4 files; unquantized, it generates the same
  tokens as PyTorch; and Transformers.js 4.3.1 generates the same tokens as Python ONNX Runtime. Not yet checked on a
  trained model, on WebGPU, or on a phone. Details and pinned versions: [docs/SCRIBE_EXPORT.md](docs/SCRIBE_EXPORT.md).
- **q4 costs accuracy.** On a made up visit the q4 base model drifted from full precision after a few tokens, so the
  scribe model is scored as exported, not only in PyTorch.
- **q4, never q4f16, for Granite 4.0 H.** Their q4f16 export answers with a string of exclamation marks on WebGPU
  (half precision overflow). The config check and the exporter both refuse it.
- **291 MB with the runtime**, against the 600 MB iPhone budget.

## Quick start (no GPU needed)

Python 3.11 or newer.

```bash
git clone https://github.com/metacareforyou/metacare-small-model.git
cd metacare-small-model
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check      # lint, 53 tests, and the data, evaluation and config checks, in a few seconds
```

## Data rules

Full rules in [docs/DATA.md](docs/DATA.md). In short:

- **No one's health information, ever.** No patient records, no chat logs, no forum posts. Scribe visits are made up
  from specs; health records are written for training or taken from licensed public reference text.
- **Every health record names a registered source** in [`data/sources.yaml`](data/sources.yaml) whose licence is on
  the allow list (Apache 2.0, MIT, CC0, CC BY 4.0, United States Government works).
- **A personal data screen** blocks email addresses, telephone numbers, identifiers, dates of birth, addresses and self
  introductions by name. It cannot catch a bare personal name, so every health record also needs a human review.
- **Models that write training data must allow it.** Only open models whose licence lets their output train other
  models (Apache 2.0 or MIT) write scribe visits.

## Repository layout

```
configs/                     one YAML per model: pinned base, LoRA, training, export dtype
  scribe/                    the Ember model's config
data/                        health records, the source register and examples (scribe data stays in data/private/)
evalsets/health-qa-v0/       the open health evaluation set
evalsets/scribe/             Ember's six visit test suite: lines, right answers, and Whisper transcripts
docs/                        data rules, evaluation rubric, model card template, export notes
src/metacare_small_model/    the msm command and its modules
  scribe/                    Ember: catalog, checks, generate, speech, pairs, finetune, score, suite, export_onnx
  records.py, screen.py,     health data rules, personal data screen, emergency filter
  emergency.py
  prompt.py, train.py,       health model prompt, training, evaluation
  evaluate.py
  export.py, manifest.py     q4 quantization and graph comparison; the app's manifest and phone budget
tests/                       run without a GPU or any model download
tools/scribe-js-smoke/       checks an export with the Transformers.js version the app ships
```

## Known unknowns

1. **How well 350M parameters extract a visit for Ember.** The first scores will say; the 1B model is the fallback if it falls
   short, but it needs to fit under the phone budget first.
2. **How fast it runs on phones.** The base 350M model installs on an iPhone and has answered on an iPhone 18;
   time to process a whole visit on a phone has not been measured.
3. **How much q4 costs.** Scores in full precision and as exported will be reported side by side.
4. **Sixteen visit types is narrow.** The held out types test whether that is enough; more types are cheap to add.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Clinicians who want to write or rate
evaluation items are especially welcome: write to info@metacare.ai.

## Licence and disclaimer

Code: [Apache 2.0](LICENSE). Base models: IBM Granite 4.0, Apache 2.0. Each published dataset and model will carry its
own card and licence.

Nothing in this repository, and no output of a model trained with it, is medical advice, diagnosis or treatment. Ember's
notes are drafts for a clinician to review. In an emergency, call 911 or your local emergency number.

Made by [Metacare.ai Inc.](https://metacare.ai) in Miami, Florida.
