"""Supervised fine tuning with a LoRA adapter.

The heavy imports (torch, transformers, peft, trl) happen inside `run`, so the rest of the package and `--dry-run`
work on a laptop without them. Install them with `pip install -e ".[train]"` on the training machine.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .prompt import system_prompt
from .records import check_records, load_sources, read_jsonl, split_of


def examples(records: list[dict]) -> dict[str, list[dict]]:
    """Prompt and completion pairs in chat form, so the loss falls on the answer only."""
    prompt = system_prompt()
    out: dict[str, list[dict]] = {"train": [], "validation": []}
    for record in records:
        out[split_of(record["id"])].append(
            {
                "prompt": [{"role": "system", "content": prompt}, {"role": "user", "content": record["question"]}],
                "completion": [{"role": "assistant", "content": record["answer"]}],
            }
        )
    return out


def prepare(config: Config, root: Path) -> dict[str, list[dict]]:
    sources = load_sources(root / config.get("data.sources"))
    rows = list(read_jsonl(root / config.get("data.train")))
    report = check_records(rows, sources, for_training=True)
    if not report.ok:
        raise SystemExit("training data failed its checks; run `msm data check` for the list")
    return examples([record for _, record in rows])


def plan(config: Config, data: dict[str, list[dict]]) -> dict:
    return {
        "name": config.name,
        "base": f"{config.get('base.repo')}@{config.get('base.revision')}",
        "train_examples": len(data["train"]),
        "validation_examples": len(data["validation"]),
        "lora": config.get("lora"),
        "training": config.get("training"),
        "output": config.get("training.output_dir", f"runs/{config.name}"),
    }


def run(config: Config, root: Path, dry_run: bool = False) -> dict:
    data = prepare(config, root)
    summary = plan(config, data)
    if dry_run:
        return summary

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    repo, revision = config.get("base.repo"), config.get("base.revision")
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, torch_dtype="auto")
    train = config.get("training")
    output = Path(summary["output"])
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(data["train"]),
        eval_dataset=Dataset.from_list(data["validation"]) if data["validation"] else None,
        peft_config=LoraConfig(
            r=config.get("lora.r"),
            lora_alpha=config.get("lora.alpha"),
            lora_dropout=config.get("lora.dropout"),
            target_modules=config.get("lora.target_modules", "all-linear"),
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=str(output),
            num_train_epochs=train["epochs"],
            learning_rate=train["learning_rate"],
            per_device_train_batch_size=train["batch_size"],
            gradient_accumulation_steps=train.get("gradient_accumulation", 1),
            max_length=train.get("max_length", 1024),
            bf16=train.get("bf16", True),
            seed=train.get("seed", 42),
            logging_steps=10,
            eval_strategy="epoch" if data["validation"] else "no",
            save_strategy="epoch",
            report_to=[],
        ),
    )
    trainer.train()
    trainer.save_model(str(output / "adapter"))
    (output / "plan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
