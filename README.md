# MetaCare small model

**A small, open health model that answers in a phone's browser, so the question never leaves the phone.**

[![CI](https://github.com/metacareforyou/metacare-small-model/actions/workflows/ci.yml/badge.svg)](https://github.com/metacareforyou/metacare-small-model/actions/workflows/ci.yml)
[![Licence: Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-blue.svg)](LICENSE)

This repository holds everything needed to fine tune, evaluate and export the on device model behind
[Metacare.ai](https://metacare.ai): the data rules, the training configuration, an open evaluation set with a
clinician rubric, and the export to the 4 bit ONNX files that run in the browser with Transformers.js and WebGPU.
English and Spanish. Code under Apache 2.0; weights and evaluation sets will be released openly as they are made.

> Status: **scaffolding.** No model has been trained here yet. Read [Where things stand](#where-things-stand) before
> relying on anything.

---

## Why this exists

Metacare.ai keeps a person's medical record, nutrition, movement and sleep logs on their own device, with no
account. Its core features need no AI at all. When a person asks a health question, though, the answer today
comes from a large model on a server, so the question leaves the phone.

The app already has an on device path ([metacare-gpt-frontend](https://github.com/metacareforyou/metacare-gpt-frontend)
is private; the relevant parts are described below). It downloads a pinned model from MetaCare's own host, checks every
file's size and SHA 256, runs it in a web worker, and never falls back to a server. What it lacks is a model that both
**fits a phone's browser** and **answers health questions well**. Off the shelf small models do one or the other:

| Model on the device | Size at q4, with runtime | What happened |
|---|---|---|
| SmolLM3 3B | about 2.1 GB | Refused on iPhone: iOS closed a page holding 1.5 GB |
| Granite 4.0 H 1B | about 1.0 GB | Would not install on an iPhone; Chrome on a test Mac refused to cache the single 1 GB file |
| Granite 4.0 H 350M | about 265 MB | Installs on an iPhone; on a desktop browser with WebGPU it answered a Spanish translation in 15 seconds; on an iPhone 18 it has answered (November 9, 2026), while an earlier iPhone test timed out; answer time on phones is not yet measured |

The goal of this repository is a fine tuned model at the 350M size class (or a 1B model squeezed under the phone
budget) that a clinician would be comfortable sending to a patient as written.

## Where things stand

| Piece | State |
|---|---|
| Data format, source register, personal data screen, emergency filter | Built and tested |
| System prompt pinned to the app's, with a check that they match | Built and tested |
| LoRA fine tuning configuration for Granite 4.0 H 350M and 1B | Written; **not yet run** |
| Evaluation runner, automatic checks, clinician sheet and summary | Written; runner not yet run on a model |
| Export to ONNX and q4 quantization | Written; **export of a fine tuned Granite hybrid not yet verified** (see [Known unknowns](#known-unknowns)) |
| App manifest builder and phone budget check | Built and tested |
| Training data | Four unreviewed examples only |
| Evaluation set | Six seed items; clinicians have not written or rated any yet |
| Published weights | None yet |

## How it fits together

```
 data/train.jsonl ──► msm data check ──► msm train ──► runs/<name>/adapter
 data/sources.yaml        (licences, personal data,        (LoRA on Granite 4.0 H,
                           emergencies, answer style)        pinned revision)
                                                                  │
 evalsets/…/items.jsonl ──► msm eval run ◄────────────────────────┤
                                 │                                │
                      msm eval sheet ──► clinicians rate ──► msm eval summarize
                                                                  │
                                                     msm export (merge, ONNX, q4)
                                                                  │
                                             msm manifest ──► chatManifest.<name>.json
                                             msm budget        (sizes, SHA 256, runtime;
                                                                 fits the iPhone or not)
                                                                  │
                                              MetaCare model host ──► the app, in the browser
```

## Quick start (no GPU needed)

Python 3.11 or newer.

```bash
git clone https://github.com/metacareforyou/metacare-small-model.git
cd metacare-small-model
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

msm data check --records data/examples/train.example.jsonl --draft   # the examples pass as drafts
msm data check --records data/examples/train.example.jsonl           # and are refused for training: not yet reviewed
msm eval check                                                        # the evaluation set, and no overlap with training data
msm config configs/*.yaml                                             # pinned revisions, q4 for Granite hybrids
msm prompt                                                            # the system prompt the model is trained with
pytest                                                                # 42 tests, a second or two
```

`make check` runs the lint and the tests together.

## Training

On a machine with a GPU. A single 24 GB card should hold either model with LoRA at the batch sizes in the configs;
that is an estimate, not yet measured, so start with `--dry-run` and a short run.

```bash
pip install -e ".[train]"
msm train --config configs/granite-h-350m.yaml --dry-run   # checks the data and prints the plan
msm train --config configs/granite-h-350m.yaml             # writes runs/granite-h-350m-health-v0/adapter
```

What training does, and why:

- **Base models are pinned to a commit**, never a branch. `ibm-granite/granite-4.0-h-350m` and `-1b` are Apache 2.0,
  list Spanish among their languages, and IBM's model card for the H 1B base says it was trained from scratch on
  about 15 trillion tokens; neither card mentions pruning or distillation from a larger model.
- **LoRA adapters** (rank 16, all linear layers) keep a run cheap and the result easy to audit against the base.
- **Prompt and completion pairs** put the loss on the answer only. The prompt is the exact system prompt the app
  sends at run time ([`prompts/system.md`](src/metacare_small_model/prompts/system.md)). Check it is still the same
  with `msm prompt --app <path to metacare-gpt-frontend>/src/lib/onDeviceChat.ts`; a model tuned under one prompt
  and run under another loses much of what the tuning bought.
- **Only approved records train.** `msm train` refuses to start if `msm data check` finds any error.
- **A stable split**: each record's id decides whether it lands in training or validation, so reordering the file
  changes nothing.

## Evaluation

Automatic checks are cheap filters, never the verdict. The verdict is clinicians'.

```bash
msm eval run --model runs/granite-h-350m-health-v0/merged --out runs/eval/answers.jsonl
msm eval sheet --answers runs/eval/answers.jsonl --out runs/eval/sheet.csv     # one row per answer, rubric left blank
msm eval summarize --sheet runs/eval/sheet-completed.csv                       # totals overall and by category
```

- The runner uses the app's settings: the same system prompt, at most 320 new tokens, no sampling.
- **Emergency items are never sent to the model**, exactly as in the app, and are recorded as routed to the fixed
  message. They exist to prove the routing catches them.
- Automatic flags: empty answers, more than five sentences, lists, forbidden phrases (for example "safe to take"),
  a missing `must_include`, a present `must_not`, the wrong language, and no advice on when to see a clinician.
- Clinicians rate accuracy, safety, harm if followed, plain language, whether it sends the person to care when it
  should, and whether they would send the answer as written. The rubric and the proposed pass bar are in
  [docs/EVALUATION.md](docs/EVALUATION.md).

The evaluation set is open so anyone can check our numbers, or theirs, on the same questions.

## Export to the browser

```bash
pip install -e ".[export]"
msm export --config configs/granite-h-350m.yaml --adapter runs/granite-h-350m-health-v0/adapter \
  --out artifacts/granite-h-350m-health-v0 \
  --check-against <published onnx-community model_q4.onnx>       # fail here, not on a phone, if the graph differs
msm manifest --config configs/granite-h-350m.yaml --dir artifacts/granite-h-350m-health-v0/final \
  --revision <commit of the published weights> \
  --runtime-from <path to metacare-gpt-frontend>/src/localai/chatManifest.granite-h-350m.json \
  --out artifacts/chatManifest.granite-h-350m-health-v0.json
msm budget artifacts/chatManifest.granite-h-350m-health-v0.json
```

- **q4, never q4f16, for Granite 4.0 H.** These are hybrid Mamba2 and attention models; their q4f16 export answers
  with a string of exclamation marks on WebGPU (half precision overflow). The config check and the exporter both
  refuse it.
- **The output folder has the layout Transformers.js expects**: `config.json`, `generation_config.json`,
  `tokenizer.json`, `tokenizer_config.json`, `onnx/model_q4.onnx` and `onnx/model_q4.onnx_data`.
- **The manifest has the app's shape** (`src/localai/chatManifest.*.json`): every file's path, size and SHA 256, the
  disclosure the person reads before installing (name, maker, licence, link), and the runtime block copied from a
  manifest the app already ships.
- **The budget is the app's**: the total, counting the ONNX Runtime binary, must stay under 600,000,000 bytes to be
  offered on an iPhone, and no single file should reach the size a browser refused to cache. The manifest's
  `mobile` flag is set from that check.

## Safety by design

- **Emergencies are decided in code, never by the model.** The app checks every message first and answers a crisis
  or a medical emergency with a fixed message. In a trial of an earlier on device model, the model told someone who
  had taken 20 extra strength Tylenol that they did not need to do anything; that is why. Training data never teaches
  the model to answer an emergency: [`emergency.py`](src/metacare_small_model/emergency.py) refuses any training
  question that reads like one, in English or Spanish. It is a training data filter, not the app's safety check.
- **The answer style is the prompt's.** At most five short plain sentences, no lists, never call a medicine or a
  combination safe, never tell anyone to start, stop or change a dose, say "call 911 first" when someone may be in
  danger, and end with when to see a clinician. Training answers that break these rules are refused.
- **No model output is medical advice.** The app says so, and so does every model card here.

## Data rules

Full rules in [docs/DATA.md](docs/DATA.md). In short:

- **No one's health information, ever.** No patient records, no chat logs, no forum posts, nothing a person wrote
  about their own health. Records are written for training or taken from licensed public reference text.
- **Every record names a registered source** in [`data/sources.yaml`](data/sources.yaml), and the source's licence
  must be on the allow list (Apache 2.0, MIT, CC0, CC BY 4.0, United States Government works). Adding a licence is a
  reviewed change.
- **A strict personal data screen** blocks email addresses, telephone numbers, Social Security numbers, card numbers,
  dates of birth, record numbers, street addresses, links with query strings and self introductions by name. It
  cannot catch a bare personal name, so every record also needs a human review before it trains.
- **Evaluation items must be unseen**: `msm eval check` refuses an item whose question is in the training data.

## Repository layout

```
configs/                    one YAML per model: pinned base, LoRA, training, export dtype, manifest fields
data/sources.yaml           the source register, with licences
data/train.jsonl            approved training records (empty until review starts)
data/examples/              format examples; unreviewed, never trained on
evalsets/health-qa-v0/      the open evaluation set
docs/                       data rules, evaluation rubric, model card template
src/metacare_small_model/   the msm command and its modules
  records.py                schema and every rule a record or item must pass
  screen.py                 personal data screen
  emergency.py              emergency wording filter (English and Spanish)
  prompt.py                 the pinned system prompt and the check against the app
  train.py                  LoRA supervised fine tuning (heavy imports only when run)
  evaluate.py               generation, automatic checks, clinician sheet, summary
  export.py                 merge, ONNX export, q4 quantization, graph comparison
  manifest.py               the app's manifest and the phone budget
tests/                      run without a GPU or any model download
```

## Known unknowns

Stated plainly, so nobody discovers them on a phone:

1. **Exporting a fine tuned Granite hybrid.** The ONNX files the app runs were made by onnx-community. Whether
   `optimum` exports a fine tuned `granitemoehybrid` model with its Mamba state cache in the shape Transformers.js
   4.3.1 expects is untested. `msm export --check-against` compares graph inputs and outputs with the published file
   to catch a mismatch early; if it differs, the conversion needs onnx-community's method instead.
2. **How fast the 350M model answers on phones.** It installs on an iPhone and has answered on an iPhone 18, but an
   earlier iPhone test never returned an answer, and answer time has not been measured on any phone. The app stops an
   answer after five minutes and shows the reason, so a failure on a given phone says whether the problem is speed or
   memory.
3. **Whether 350M parameters are enough** for answers a clinician would send. That is what the evaluation is for.
4. **The 1B model on a phone** needs either a smaller quantization or the model split across files under the browser's
   cache limit.

## Roadmap

- [x] Scaffolding, data rules, evaluation format, export and manifest tooling
- [ ] Clinician reviewers named and the rubric's pass bar agreed
- [ ] First few hundred reviewed records in English and Spanish
- [ ] First 350M run, evaluated, and the export verified against the app on desktop browsers
- [ ] Answer time measured on iPhone and Android phones (the base 350M has answered once on an iPhone 18)
- [ ] Weights, recipe and evaluation set published openly with a model card
- [ ] The on device path offered in production after the clinician review (MET-1385)

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Clinicians who want to write or rate
evaluation items are especially welcome: write to info@metacare.ai.

## Licence and disclaimer

Code: [Apache 2.0](LICENSE). Base models: IBM Granite 4.0, Apache 2.0. Each published dataset and model carries its
own card and licence.

Nothing in this repository, and no answer from a model trained with it, is medical advice, diagnosis or treatment.
In an emergency, call 911 or your local emergency number.

Made by [Metacare.ai Inc.](https://metacare.ai) in Miami, Florida.
