// Used only with ORT_WEB=1. Transformers.js in Node loads onnxruntime-node, whose 1.30.0 binary needs macOS 14.
// This hands it onnxruntime-web instead (the WebAssembly runtime the browser uses) and mounts the .onnx_data file
// beside the model, which the browser build does through session options.
const fs = require("node:fs");
const path = require("node:path");
const ort = require("onnxruntime-web");

const create = ort.InferenceSession.create.bind(ort.InferenceSession);
ort.InferenceSession.create = async (model, options = {}) => {
  if (typeof model === "string" && !options.externalData && fs.existsSync(`${model}_data`)) {
    options = { ...options, externalData: [{ path: path.basename(`${model}_data`), data: `${model}_data` }] };
  }
  return create(model, options);
};
module.exports = ort;
