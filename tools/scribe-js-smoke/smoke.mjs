// node smoke.mjs <exported folder> [prompt file] [max new tokens]
// Prints the greedy token ids and text, to compare with the onnx_q4 tokens from export_onnx.py.
// On macOS 13 run with ORT_WEB=1 (see ort-web-shim.cjs).
import fs from "node:fs";
import Module, { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
if (process.env.ORT_WEB) {
  const shim = new URL("./ort-web-shim.cjs", import.meta.url).pathname;
  const resolve = Module._resolveFilename;
  Module._resolveFilename = function (request, ...rest) {
    return request === "onnxruntime-node" ? shim : resolve.call(this, request, ...rest);
  };
}
const { AutoTokenizer, AutoModelForCausalLM, env } = await import("@huggingface/transformers");

const [dir, promptFile, maxNew] = process.argv.slice(2);
const folder = path.resolve(dir);
env.allowRemoteModels = false;
env.localModelPath = `${path.dirname(folder)}/`;
if (process.env.ORT_WEB) {
  env.useWasmCache = false;
  env.backends.onnx.wasm.wasmPaths = `file://${path.dirname(require.resolve("onnxruntime-web"))}/`;
  env.backends.onnx.wasm.numThreads = 1;
}
const content = promptFile ? fs.readFileSync(promptFile, "utf8") : "What is the capital of France? Answer in one sentence.";
const name = path.basename(folder);
const tokenizer = await AutoTokenizer.from_pretrained(name);
const model = await AutoModelForCausalLM.from_pretrained(name, { dtype: "q4", device: "cpu" });
const inputs = tokenizer.apply_chat_template([{ role: "user", content }], { add_generation_prompt: true, return_dict: true });
const started = Date.now();
const output = await model.generate({ ...inputs, max_new_tokens: Number(maxNew || 40), do_sample: false });
const ids = Array.from(output.tolist()[0].slice(inputs.input_ids.dims[1]), Number);
console.log(JSON.stringify({ runtime: process.env.ORT_WEB ? "onnxruntime-web wasm" : "onnxruntime-node",
  ms: Date.now() - started, ids, text: tokenizer.decode(ids, { skip_special_tokens: true }) }));
