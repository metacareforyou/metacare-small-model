"""Fine tunes Granite to extract visit facts, then scores it on visit types it never saw (MET-1930).

    python -m metacare_small_model.scribe.finetune --config configs/scribe/granite-h-350m-scribe.yaml \
        --pairs pairs.jsonl [--baseline 100]

Writes, under the config's output_dir: the LoRA adapter, the merged model (for the ONNX export), the test answers
and score.json with precision and recall for each kind and field, split by written and heard transcripts.
Needs a GPU and `pip install -e ".[train]"`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from .score import score
from .suite import HERE
from .suite import rows as suite_rows


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def predict(model, tokenizer, rows: list[dict], batch: int = 16, max_new_tokens: int = 768) -> list[str]:
    import torch

    tokenizer.padding_side = "left"
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        texts = [tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        out += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"predicted {len(out)}/{len(rows)}", flush=True)
    return out


def report(rows: list[dict], answers: list[str]) -> dict:
    """Scores for written transcripts, for all heard ones, and for each Whisper model. Test suite visits are scored
    apart, and split into kinds of visit that training also has (seen) and kinds it does not (held out)."""
    groups = defaultdict(list)
    for r, a in zip(rows, answers, strict=True):
        row = (r["completion"][0]["content"], a)
        prefix = ""
        if r["split"] == "suite":
            prefix = "suite " + ("seen " if r["seen_type"] else "held out ")
            groups["suite " + ("written" if r["variant"] == "clean" else "heard")].append(row)
        if r["variant"] == "clean":
            groups[prefix + "written"].append(row)
        else:
            groups[prefix + "heard"].append(row)
            if not prefix:
                groups[f"heard by {r['variant'].removeprefix('heard-')}"].append(row)
            else:
                groups[f"suite heard by {r['variant'].removeprefix('heard-')}"].append(row)
    return {k: score(v) for k, v in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--baseline", type=int, default=0, help="also score the untrained model on this many test pairs")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = load(args.pairs)
    split = defaultdict(list)
    for r in rows:
        split[r["split"]].append({"prompt": r["prompt"], "completion": r["completion"]} if r["split"] != "test" else r)
    suite = suite_rows(HERE / "suite_cases.json", HERE / "suite_heard.jsonl")
    print({k: len(v) for k, v in split.items()}, "suite", len(suite), flush=True)

    repo, revision = cfg["base"]["repo"], cfg["base"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    t = cfg["training"]
    output = Path(t["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    lengths = sorted(len(tokenizer.apply_chat_template(r["prompt"] + r["completion"], tokenize=True)) for r in split["train"])
    print(f"tokens per pair: median {lengths[len(lengths) // 2]}, max {lengths[-1]}, limit {t['max_length']}", flush=True)

    if args.baseline:
        base = AutoModelForCausalLM.from_pretrained(repo, revision=revision, torch_dtype=torch.bfloat16).to("cuda")
        test = split["test"][: args.baseline] + suite
        base_score = report(test, predict(base, tokenizer, test))
        (output / "baseline_score.json").write_text(json.dumps(base_score, indent=2))
        del base
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, torch_dtype=torch.bfloat16)
    # The 350M and 1B Granite 4.0 H models are dense: there are no experts to balance, and asking for the router loss
    # fails inside the trainer.
    model.config.output_router_logits = False
    lora = cfg["lora"]
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(split["train"]),
        eval_dataset=Dataset.from_list(split["validation"]),
        peft_config=LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),  # fmt: skip
        args=SFTConfig(
            output_dir=str(output),
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            per_device_train_batch_size=t["batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation"],
            max_length=t["max_length"],
            bf16=True,
            seed=t["seed"],
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            report_to=[],
            router_aux_loss_coef=0.0,  # TRL turns the router loss back on unless its weight is zero
        ),
    )
    trainer.train()
    trainer.save_model(str(output / "adapter"))
    (output / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(output / "merged"))
    tokenizer.save_pretrained(str(output / "merged"))

    merged.eval()
    test = split["test"] + suite
    answers = predict(merged, tokenizer, test)
    with open(output / "test_answers.jsonl", "w") as f:
        for r, a in zip(test, answers, strict=True):
            f.write(json.dumps({"id": r["id"], "answer": a, "right": r["completion"][0]["content"]}) + "\n")
    result = report(test, answers)
    (output / "score.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {"json_failed": v["json_failed"], "items": v["items"]} for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
