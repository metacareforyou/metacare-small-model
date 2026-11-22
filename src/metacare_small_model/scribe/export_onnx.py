"""Export the merged scribe model to ONNX q4 for Transformers.js, report its size, and smoke test it (MET-1930).

    python -m metacare_small_model.scribe.export_onnx --merged runs/granite-h-350m-scribe-v0/merged \
        --out runs/granite-h-350m-scribe-v0/onnx [--runtime-from <app chatManifest.json>] [--fp32-check]

Why not `msm export`: `optimum` 2.1 with optimum-onnx 0.1.0 has no ONNX config for `granitemoehybrid` (it refuses
the model type), so the existing optimum path cannot export Granite 4.0 H. The onnx-community export that the app
already runs (onnx-community/granite-4.0-h-350m-ONNX at REFERENCE_REVISION) is a hand built graph: Mamba2 layers with
an If for prefill and decode, GroupQueryAttention for the four attention layers, past_conv / past_ssm /
past_key_values inputs, and in the q4 file MatMulNBits plus a GatherBlockQuantized embedding (4 bit weights, block 32,
asymmetric, float32 scales and activations).

A LoRA merge changes weights, not layers, so this keeps that graph and writes our weights into it:

1. Check the merged config has the reference architecture (every shape and multiplier in the graph comes from it).
2. Read the merged safetensors and build each tensor the graph names (transposes, the gate and up split of
   shared_mlp.input_linear, -exp(A_log), the dt_bias and D expansions).
3. Quantize every MatMul weight and the embedding to 4 bits, asymmetric, block 32, float32 scales (quantize_q4).
   Run on the untrained base model this reproduces the published model_q4.onnx and model_q4.onnx_data byte for
   byte (--verify-reference checks that).
4. Write onnx/model_q4.onnx (the reference graph with the weight derived inline tensors replaced) and
   onnx/model_q4.onnx_data (same offsets as the reference), plus config.json, generation_config.json,
   tokenizer.json and tokenizer_config.json, the file list the app manifest pins.
5. Report the download size against the iPhone budget, and greedy generate with onnxruntime and with PyTorch.

Only q4 is written. q4f16 overflows on WebGPU for Granite hybrid models (answers come back as "!!!!").
Needs: pip install -e ".[train,export]"; pinned versions and measured sizes are in docs/SCRIBE_EXPORT.md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from ..manifest import CACHE_ENTRY_MAX_BYTES, IPHONE_MAX_BYTES

REFERENCE_REPO = "onnx-community/granite-4.0-h-350m-ONNX"
REFERENCE_REVISION = "06653818ebf090337d196df49a805d83ec0d211b"
DTYPE = "q4"
BLOCK_SIZE = 32
FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
# Keys whose values are baked into the reference graph (shapes, constants, layer order). The merged config must agree.
ARCHITECTURE_KEYS = [
    "model_type",
    "hidden_size",
    "intermediate_size",
    "shared_intermediate_size",
    "num_hidden_layers",
    "layer_types",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "tie_word_embeddings",
    "embedding_multiplier",
    "attention_multiplier",
    "residual_multiplier",
    "logits_scaling",
    "rms_norm_eps",
    "position_embedding_type",
    "mamba_chunk_size",
    "mamba_d_conv",
    "mamba_d_head",
    "mamba_d_state",
    "mamba_expand",
    "mamba_n_groups",
    "mamba_n_heads",
    "mamba_conv_bias",
    "mamba_proj_bias",
    "num_local_experts",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
]
SMOKE_PROMPT = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]


# ---------------------------------------------------------------- reference files


def reference_file(name: str, cache: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REFERENCE_REPO, name, revision=REFERENCE_REVISION, cache_dir=cache))


def check_architecture(merged_config: dict, reference_config: dict) -> list[str]:
    return [
        f"{key}: merged {merged_config.get(key)!r}, reference graph {reference_config.get(key)!r}"
        for key in ARCHITECTURE_KEYS
        if merged_config.get(key) != reference_config.get(key)
    ]


# ---------------------------------------------------------------- weights


def load_weights(merged: Path) -> dict[str, np.ndarray]:
    import torch
    from safetensors import safe_open

    out: dict[str, np.ndarray] = {}
    for shard in sorted(merged.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key).to(torch.float32).numpy()
    if not out:
        raise SystemExit(f"{merged}: no .safetensors files")
    # The graph ties the output layer to the embedding (one quantized tensor for both). A merge that untied them
    # (for example a LoRA on lm_head) cannot be written into it.
    if "lm_head.weight" in out and not np.array_equal(out["lm_head.weight"], out["model.embed_tokens.weight"]):
        raise SystemExit(f"{merged}: lm_head.weight differs from the embedding; the reference graph needs them tied")
    return out


def _neg_exp(a_log: np.ndarray) -> np.ndarray:
    """A = -exp(A_log) in float32, computed by torch as the model's own forward does (numpy's exp differs by an ulp)."""
    import torch

    return (-torch.exp(torch.from_numpy(np.ascontiguousarray(a_log, dtype=np.float32)))).numpy()


def hf_tensor(name: str, hf: dict[str, np.ndarray], config: dict) -> np.ndarray:
    """The float32 tensor the reference graph calls `name`, built from the Hugging Face weights."""
    if name == "model.embed_tokens.weight":
        return hf[name]
    m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
    if not m:
        raise KeyError(name)
    i, rest = m.group(1), m.group(2)
    p = f"model.layers.{i}."
    heads, head_dim = config["mamba_n_heads"], config["mamba_d_head"]
    if rest in (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mamba.norm.weight",
        "mamba.conv1d.weight",
        "mamba.conv1d.bias",
    ):
        return hf[p + rest]
    if rest == "final_norm_layernorm.weight":
        return hf["model.norm.weight"]
    if rest == "mamba.conv1d.weight_squeezed":
        return hf[p + "mamba.conv1d.weight"][:, 0, :]
    if rest == "mamba.dt_bias_expanded_decode":
        return np.broadcast_to(hf[p + "mamba.dt_bias"][:, None], (heads, head_dim))
    if rest == "mamba.dt_bias_unsqueezed_prefill":
        return hf[p + "mamba.dt_bias"][None, :]
    if rest == "mamba.A_neg_exp_FLOAT":
        return _neg_exp(hf[p + "mamba.A_log"])
    if rest == "mamba.A_neg_exp_expanded_decode":
        return _neg_exp(hf[p + "mamba.A_log"])[None, :, None]
    if rest == "mamba.D_expanded_decode":
        return np.broadcast_to(hf[p + "mamba.D"][:, None], (heads, head_dim))
    if rest == "mamba.D_f32_unsq":
        return hf[p + "mamba.D"][:, None]
    mm = re.fullmatch(r"(mamba|attn|mlp)\.(\w+)\.MatMul\.weight", rest)
    if mm:  # ONNX MatMul weights are [in, out]; torch Linear weights are [out, in]
        block, proj = mm.groups()
        if block == "mamba":
            return hf[f"{p}mamba.{proj}.weight"].T
        if block == "attn":
            return hf[f"{p}self_attn.{proj}.weight"].T
        fused = hf[f"{p}shared_mlp.input_linear.weight"]  # gate rows then up rows: GraniteMoeHybridMLP chunk(2)
        half = config["shared_intermediate_size"]
        if proj == "gate_proj":
            return fused[:half].T
        if proj == "up_proj":
            return fused[half:].T
        if proj == "down_proj":
            return hf[f"{p}shared_mlp.output_linear.weight"].T
    raise KeyError(name)


def _pack_nibbles(values: np.ndarray) -> np.ndarray:
    """Pairs along the last axis into bytes, first value in the low nibble (MatMulNBits and GatherBlockQuantized)."""
    if values.shape[-1] % 2:
        values = np.concatenate([values, np.zeros((*values.shape[:-1], 1), values.dtype)], axis=-1)
    return (values[..., 0::2] | (values[..., 1::2] << 4)).astype(np.uint8)


def quantize_q4(weight_kn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4 bit asymmetric block quantization of a [K, N] float32 weight along K, in MatMulNBits layout.

    Blocks of 32 along K; scale = (max(x) - min(x)) / 15 (the range is not widened to include 0); zero point =
    round(-min / scale) and q = round(x / scale + zero point), both clipped to 0..15, rounding half to even. With these
    exact rules the untrained base model reproduces the published onnx-community model_q4.onnx_data byte for byte.
    ONNX Runtime's C++ quantize_matmul_4bits (behind MatMulNBitsQuantizer) widens the range to include 0 and rounds
    half away from zero; it differed from that file on about 0.1% of zero points, so it is not used.
    Returns packed [N, blocks, 16] uint8, scales [N * blocks] float32, zero points [N * ceil(blocks / 2)] uint8.
    """
    k, n = weight_kn.shape
    blocks = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
    w = np.ascontiguousarray(weight_kn.T, dtype=np.float32)
    if blocks * BLOCK_SIZE != k:
        w = np.pad(w, ((0, 0), (0, blocks * BLOCK_SIZE - k)))
    w = w.reshape(n, blocks, BLOCK_SIZE)
    low = w.min(axis=-1)
    high = w.max(axis=-1)
    scales = ((high - low) / np.float32(15)).astype(np.float32)
    safe = np.where(scales == 0, np.float32(1), scales)
    zero_points = np.clip(np.rint(-low / safe), 0, 15).astype(np.float32)
    q = np.clip(np.rint(w / safe[..., None] + zero_points[..., None]), 0, 15).astype(np.uint8)
    return _pack_nibbles(q), scales.reshape(-1), _pack_nibbles(zero_points.astype(np.uint8)).reshape(-1)


# ---------------------------------------------------------------- writing


def fill_template(template: Path, hf: dict[str, np.ndarray], config: dict, out_onnx: Path) -> dict:
    """Write `template` (a reference graph) with our weights to out_onnx and out_onnx + '_data'."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(template), load_external_data=False)
    data_name = out_onnx.name + "_data"
    quant_cache: dict[str, tuple] = {}
    blobs: list[tuple[int, int, bytes]] = []

    def float_tensor(name: str) -> np.ndarray:
        return np.ascontiguousarray(hf_tensor(name, hf, config), dtype=np.float32)

    def quant_parts(base: str):
        if base not in quant_cache:
            dotted = (
                "model.embed_tokens.weight"
                if base == "model_embed_tokens_weight"
                else re.sub(
                    r"^model_layers_(\d+)_(mamba|attn|mlp)_(\w+?)_MatMul_weight$", r"model.layers.\1.\2.\3.MatMul.weight", base
                )
            )
            weight = hf_tensor(dotted, hf, config)
            quant_cache[base] = quantize_q4(weight.T if dotted == "model.embed_tokens.weight" else weight)
        return quant_cache[base]

    for t in model.graph.initializer:
        if t.name.startswith("/model/"):
            continue  # constants from the config (multipliers, shapes, split sizes), checked by check_architecture
        q = re.fullmatch(r"(.+)_(quant_matmul|quant|scales|zp)", t.name)
        if q:
            packed, scales, zps = quant_parts(q.group(1))
            value = {"quant": packed, "quant_matmul": packed, "scales": scales, "zp": zps}[q.group(2)]
        else:
            value = float_tensor(t.name)
        raw = np.ascontiguousarray(value).tobytes()
        expected = int(np.prod(t.dims)) * (4 if t.data_type == onnx.TensorProto.FLOAT else 1)
        if t.data_type == onnx.TensorProto.FLOAT and value.dtype != np.float32:
            raise SystemExit(f"{t.name}: built {value.dtype}, graph wants float32")
        if t.data_type == onnx.TensorProto.FLOAT and int(np.prod(t.dims)) != value.size:
            raise SystemExit(f"{t.name}: built shape {value.shape}, graph wants {list(t.dims)}")
        ext = {e.key: e.value for e in t.external_data}
        if ext:
            offset, length = int(ext["offset"]), int(ext["length"])
            if len(raw) != length:
                raise SystemExit(f"{t.name}: built {len(raw)} bytes, graph wants {length}")
            blobs.append((offset, length, raw))
            for e in t.external_data:
                if e.key == "location":
                    e.value = data_name
        else:
            if len(raw) != expected:
                raise SystemExit(f"{t.name}: built {len(raw)} bytes, graph wants {expected}")
            t.CopyFrom(numpy_helper.from_array(value.reshape(tuple(t.dims)), t.name))

    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    with open(out_onnx.parent / data_name, "wb") as handle:
        position = 0
        for offset, length, raw in sorted(blobs):
            if offset < position:
                raise SystemExit("overlapping tensors in the reference data file")
            handle.write(b"\0" * (offset - position))
            handle.write(raw)
            position = offset + length
    onnx.save_model(model, str(out_onnx))
    return {"tensors_written": len(blobs), "data_bytes": position}


def write_side_files(merged: Path, reference_config: dict, out: Path) -> None:
    # config.json: the reference file, which carries transformers.js_config (external data) and matches the graph;
    # check_architecture has already proved the merged model has the same values.
    (out / "config.json").write_text(json.dumps(reference_config, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(merged / "generation_config.json", out / "generation_config.json")
    # tokenizer.json: the merged model's own, written without indentation (same content, smaller download).
    tokenizer = json.loads((merged / "tokenizer.json").read_text(encoding="utf-8"))
    (out / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # tokenizer_config.json: transformers 4.56+ saves the chat template as chat_template.jinja; put it inline too,
    # as the reference export does, so the browser's apply_chat_template finds it.
    tok_config = json.loads((merged / "tokenizer_config.json").read_text(encoding="utf-8"))
    jinja = merged / "chat_template.jinja"
    if "chat_template" not in tok_config and jinja.exists():
        tok_config["chat_template"] = jinja.read_text(encoding="utf-8")
    if "chat_template" not in tok_config:
        raise SystemExit(f"{merged}: no chat template in tokenizer_config.json or chat_template.jinja")
    (out / "tokenizer_config.json").write_text(json.dumps(tok_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- report and smoke test


def size_report(out: Path, runtime_bytes: int) -> dict:
    files = [out / name for name in FILES] + sorted((out / "onnx").glob(f"model_{DTYPE}.onnx*"))
    sizes = {p.relative_to(out).as_posix(): p.stat().st_size for p in files}
    model = sum(sizes.values())
    largest = max(sizes, key=sizes.get)
    problems = []
    if not runtime_bytes:
        problems.append("ONNX Runtime binary not counted; pass --runtime-from <app chatManifest json>")
    if model + runtime_bytes >= IPHONE_MAX_BYTES:
        problems.append(f"total {model + runtime_bytes:,} bytes is at or over the iPhone limit {IPHONE_MAX_BYTES:,}")
    if sizes[largest] >= CACHE_ENTRY_MAX_BYTES:
        problems.append(f"{largest} is {sizes[largest]:,} bytes, over the single cache entry limit")
    return {
        "files": sizes,
        "model_bytes": model,
        "runtime_bytes": runtime_bytes,
        "total_bytes": model + runtime_bytes,
        "iphone_max_bytes": IPHONE_MAX_BYTES,
        "headroom_bytes": IPHONE_MAX_BYTES - model - runtime_bytes,
        "problems": problems,
    }


def ort_generate(model_path: Path, out: Path, messages: list[dict], max_new_tokens: int) -> list[int]:
    """Greedy decoding with onnxruntime, feeding the cache the way Transformers.js does for this graph."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(out))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="np").astype(np.int64)
    mask = np.ones_like(ids)
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    conv_dim = config["mamba_expand"] * config["hidden_size"] + 2 * config["mamba_n_groups"] * config["mamba_d_state"]
    cache = {}
    for i, kind in enumerate(config["layer_types"]):
        if kind == "attention":
            for kv in ("key", "value"):
                cache[f"past_key_values.{i}.{kv}"] = np.zeros((1, config["num_key_value_heads"], 0, head_dim), np.float32)
        else:
            cache[f"past_conv.{i}"] = np.zeros((1, conv_dim, config["mamba_d_conv"]), np.float32)
            cache[f"past_ssm.{i}"] = np.zeros(
                (1, config["mamba_n_heads"], config["mamba_d_head"], config["mamba_d_state"]), np.float32
            )
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        feed = {"input_ids": ids, "attention_mask": mask, "num_logits_to_keep": np.array(1, np.int64)} | cache
        outputs = dict(zip(names, session.run(None, feed), strict=True))
        token = int(outputs["logits"][0, -1].argmax())
        tokens.append(token)
        if token == config["eos_token_id"]:
            break
        for name in cache:
            cache[name] = outputs[name.replace("past_key_values", "present").replace("past_", "present_")]
        ids = np.array([[token]], np.int64)
        mask = np.concatenate([mask, np.ones((1, 1), np.int64)], axis=1)
    return tokens


def torch_generate(merged: Path, messages: list[dict], max_new_tokens: int) -> list[int]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(merged))
    model = AutoModelForCausalLM.from_pretrained(str(merged), torch_dtype=torch.float32).eval()
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return out[0, ids.shape[1] :].tolist()


def compare_tokens(a: list[int], b: list[int]) -> dict:
    same = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    return {"identical": a == b, "matching_prefix_tokens": same, "lengths": [len(a), len(b)]}


def verify_reference(out: Path, cache: Path | None) -> dict:
    """For the untrained base model only: our q4 data file against the published one, tensor by tensor."""
    import onnx

    ref_graph = reference_file(f"onnx/model_{DTYPE}.onnx", cache)
    ref_data = np.memmap(reference_file(f"onnx/model_{DTYPE}.onnx_data", cache), dtype=np.uint8, mode="r")
    ours = np.memmap(out / "onnx" / f"model_{DTYPE}.onnx_data", dtype=np.uint8, mode="r")
    from onnx import numpy_helper

    ref_model = onnx.load(str(ref_graph), load_external_data=False)
    our_model = onnx.load(str(out / "onnx" / f"model_{DTYPE}.onnx"), load_external_data=False)
    differing = []
    for t, mine in zip(ref_model.graph.initializer, our_model.graph.initializer, strict=True):
        ext = {e.key: e.value for e in t.external_data}
        if ext:
            o, n = int(ext["offset"]), int(ext["length"])
            if not np.array_equal(ref_data[o : o + n], ours[o : o + n]):
                differing.append(t.name)
        elif t.name != mine.name or not np.array_equal(numpy_helper.to_array(t), numpy_helper.to_array(mine)):
            differing.append(t.name)
    nodes_same = [n.SerializeToString() for n in ref_model.graph.node] == [n.SerializeToString() for n in our_model.graph.node]
    return {
        "data_identical": len(ref_data) == len(ours) and not differing and nodes_same,
        "graph_nodes_identical": nodes_same,
        "differing_tensor_count": len(differing),
        "differing_tensors_first": differing[:10],
    }


# ---------------------------------------------------------------- main


def run(
    merged: Path,
    out: Path,
    runtime_from: Path | None = None,
    fp32_check: bool = False,
    verify: bool = False,
    max_new_tokens: int = 40,
    cache: Path | None = None,
    skip_torch: bool = False,
    prompt: str | None = None,
) -> dict:
    started = time.time()
    reference_config = json.loads(reference_file("config.json", cache).read_text(encoding="utf-8"))
    merged_config = json.loads((merged / "config.json").read_text(encoding="utf-8"))
    mismatches = check_architecture(merged_config, reference_config)
    if mismatches:
        raise SystemExit("merged model does not match the reference graph:\n  " + "\n  ".join(mismatches))

    hf = load_weights(merged)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "reference": f"{REFERENCE_REPO}@{REFERENCE_REVISION}",
        "dtype": DTYPE,
        "block_size": BLOCK_SIZE,
        "merged": str(merged),
        "out": str(out),
    }
    q4_path = out / "onnx" / f"model_{DTYPE}.onnx"
    report["write"] = fill_template(reference_file(f"onnx/model_{DTYPE}.onnx", cache), hf, merged_config, q4_path)
    write_side_files(merged, reference_config, out)

    runtime = 0
    if runtime_from:
        runtime = json.loads(runtime_from.read_text(encoding="utf-8"))["runtime"]["wasm"]["asyncify"]["bytes"]
    report["size"] = size_report(out, runtime)
    if verify:
        report["reference_check"] = verify_reference(out, cache)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(out))
    messages = [{"role": "user", "content": prompt}] if prompt else SMOKE_PROMPT
    smoke: dict = {"prompt": messages, "max_new_tokens": max_new_tokens}
    q4_tokens = ort_generate(q4_path, out, messages, max_new_tokens)
    smoke["onnx_q4"] = tokenizer.decode(q4_tokens, skip_special_tokens=True)
    if not skip_torch:
        torch_tokens = torch_generate(merged, messages, max_new_tokens)
        smoke["pytorch_fp32"] = tokenizer.decode(torch_tokens, skip_special_tokens=True)
        smoke["q4_vs_pytorch"] = compare_tokens(q4_tokens, torch_tokens)
    if fp32_check:  # the unquantized reference graph with our weights: should match PyTorch token for token
        check_dir = out / "fp32-check"
        fill_template(reference_file("onnx/model.onnx", cache), hf, merged_config, check_dir / "model.onnx")
        shutil.copy2(out / "config.json", check_dir / "config.json")
        fp32_tokens = ort_generate(check_dir / "model.onnx", out, messages, max_new_tokens)
        smoke["onnx_fp32"] = tokenizer.decode(fp32_tokens, skip_special_tokens=True)
        if not skip_torch:
            smoke["fp32_vs_pytorch"] = compare_tokens(fp32_tokens, torch_tokens)
        shutil.rmtree(check_dir)  # 1.4 GB; not part of the download
    report["smoke"] = smoke
    report["seconds"] = round(time.time() - started, 1)
    (out / "export.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--merged", type=Path, required=True, help="merged Hugging Face model folder (safetensors)")
    ap.add_argument("--out", type=Path, required=True, help="output folder, Transformers.js layout")
    ap.add_argument("--runtime-from", type=Path, help="an app chatManifest json, to count the ONNX Runtime binary")
    ap.add_argument("--fp32-check", action="store_true", help="also run the unquantized graph (1.4 GB scratch)")
    ap.add_argument(
        "--verify-reference",
        action="store_true",
        help="compare the q4 data with the published file; only meaningful for the untrained base model",
    )
    ap.add_argument("--prompt", help="user message for the smoke generation, or @file to read it from a file")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--skip-torch", action="store_true", help="skip the PyTorch comparison")
    ap.add_argument("--cache", type=Path, help="Hugging Face cache folder for the reference files")
    args = ap.parse_args(argv)
    prompt = args.prompt
    if prompt and prompt.startswith("@"):
        prompt = Path(prompt[1:]).read_text(encoding="utf-8")
    report = run(
        args.merged,
        args.out,
        args.runtime_from,
        args.fp32_check,
        args.verify_reference,
        args.max_new_tokens,
        args.cache,
        args.skip_torch,
        prompt,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    problems = report["size"]["problems"]
    if report.get("reference_check") and not report["reference_check"]["data_identical"]:
        problems = problems + ["q4 data differs from the published file"]
    return 1 if [p for p in problems if "not counted" not in p] else 0


if __name__ == "__main__":
    raise SystemExit(main())
