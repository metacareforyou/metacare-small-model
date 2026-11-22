# Scribe model export to ONNX q4 (MET-1930)

`python -m metacare_small_model.scribe.export_onnx` turns the merged scribe model into the folder the app downloads,
reports its size against the iPhone budget and runs a greedy smoke generation. Only q4 (4 bit weights, float32
activations) is written; q4f16 answers "!!!!" on WebGPU for Granite hybrid models.

## Why not `msm export`

`optimum` 2.1.0 with `optimum-onnx` 0.1.0 has no ONNX configuration for `granitemoehybrid`. `optimum-cli export onnx`
on Granite 4.0 H stops with "Trying to export a granitemoehybrid model, that is a custom or unsupported architecture".

The script keeps the graph the app already runs, `onnx-community/granite-4.0-h-350m-ONNX` at commit
`06653818ebf090337d196df49a805d83ec0d211b`, and writes the merged weights into it. A LoRA merge changes weights, not
layers. On the untrained base model the output is byte for byte the published `onnx/model_q4.onnx` and
`onnx/model_q4.onnx_data` (sha256 `f2c517b2…` and `7b906467…`), which `--verify-reference` checks.

## Run it

```
pip install -e ".[train,export]"   # or the pinned versions below
python -m metacare_small_model.scribe.export_onnx \
    --merged runs/granite-h-350m-scribe-v0/merged \
    --out runs/granite-h-350m-scribe-v0/onnx \
    --runtime-from <metacare-gpt-frontend>/src/localai/chatManifest.granite-h-350m.json \
    --fp32-check --prompt @<a made up transcript prompt>.txt --max-new-tokens 200
```

It writes `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `onnx/model_q4.onnx`
and `onnx/model_q4.onnx_data`, the file list `msm manifest` pins, plus `export.json` with the report. It exits
non zero when the total is at or over 600,000,000 bytes or one file is over the browser cache limit.
`--fp32-check` also fills the unquantized reference graph (1.4 GB, deleted afterwards) and compares it with PyTorch.
It needs about 6 GB of memory and under a minute on an M1 Pro.

Transformers.js check, with the version the app ships: `tools/scribe-js-smoke` (`npm install`, then
`node smoke.mjs <out folder> [prompt file] [max new tokens]`; on macOS 13 prefix `ORT_WEB=1`, because the
onnxruntime-node 1.30.0 binary needs macOS 14).

## Measured on the untrained base model (2026-11-22)

| File | Bytes |
| --- | ---: |
| onnx/model_q4.onnx_data | 258,859,776 |
| onnx/model_q4.onnx | 1,849,840 |
| tokenizer.json | 3,143,980 |
| tokenizer_config.json | 24,272 |
| config.json | 2,023 |
| generation_config.json | 147 |
| Model files | 263,880,038 |
| ONNX Runtime binary (asyncify wasm, from the app manifest) | 26,861,777 |
| Total | 290,741,815 |

That is 309 MB under the 600 MB iPhone limit. The fine tuned model has the same tensor sizes, so only the tokenizer
files can change the total.

Greedy generation agreed as follows. The unquantized ONNX graph matched PyTorch float32 token for token, for the base
model and for a randomly perturbed copy standing in for a fine tuned one. Python onnxruntime q4 and Transformers.js
4.3.1 on onnxruntime-web 1.31.0-dev.20260914-8d85527a0 (WebAssembly) gave the same 66 tokens. q4 against PyTorch:
the same on a short question, but on a JSON extraction prompt the output diverged after 7 tokens with a worse findings
field. Score the scribe model on the q4 export, not only on PyTorch.

## Pinned versions used

Python 3.12.8: torch 2.9.1, transformers 4.57.3, tokenizers 0.22.2, safetensors 0.8.0, huggingface_hub 0.36.2,
onnx 1.19.1, onnxruntime 1.23.2, numpy 2.5.5 (optimum 2.1.0 and optimum-onnx 0.1.0 only to confirm the failure).
Node 24.19.0: @huggingface/transformers 4.3.1 (onnxruntime-web 1.31.0-dev.20260914-8d85527a0, onnxruntime-node 1.30.0).

```
pip install torch==2.9.1 transformers==4.57.3 tokenizers==0.22.2 safetensors==0.8.0 huggingface_hub==0.36.2 \
    onnx==1.19.1 onnxruntime==1.23.2 numpy==2.5.5 pyyaml
```
