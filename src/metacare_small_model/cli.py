"""`msm`: the one command for every step, from checking data to building the app's manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as cfg
from . import prompt as prompt_mod
from .records import check_eval, check_records, check_sources, load_sources, normalise, read_jsonl

ROOT = Path.cwd()


def _print_report(label: str, report) -> bool:
    for problem in report.problems:
        print(problem)
    errors = sum(p.error for p in report.problems)
    print(f"{label}: {report.count} checked, {errors} errors, {len(report.problems) - errors} warnings")
    return report.ok


def cmd_data_check(args) -> int:
    sources = load_sources(args.sources)
    ok = _print_report("sources", check_sources(sources))
    ok &= _print_report("records", check_records(read_jsonl(args.records), sources, for_training=not args.draft))
    return 0 if ok else 1


def cmd_eval_check(args) -> int:
    sources = load_sources(args.sources)
    seen = {normalise(r.get("question", "")) for _, r in read_jsonl(args.records)} if args.records.exists() else set()
    return 0 if _print_report("eval items", check_eval(read_jsonl(args.items), sources, seen)) else 1


def cmd_prompt(args) -> int:
    if args.app:
        same = prompt_mod.check(args.app)
        print("same as the app" if same else f"DIFFERENT from the app's prompt in {args.app}")
        return 0 if same else 1
    print(prompt_mod.system_prompt())
    return 0


def cmd_config_check(args) -> int:
    problems = [p for path in args.configs for p in cfg.problems(cfg.load(path))]
    print("\n".join(problems) or f"{len(args.configs)} configs ok")
    return 1 if problems else 0


def cmd_train(args) -> int:
    from .train import run

    conf = cfg.load(args.config)
    if problems := cfg.problems(conf):
        print("\n".join(problems))
        return 1
    print(json.dumps(run(conf, ROOT, dry_run=args.dry_run), indent=2))
    return 0


def cmd_eval_run(args) -> int:
    from .evaluate import generate

    print(json.dumps(generate(args.model, args.items, args.out)))
    return 0


def cmd_eval_sheet(args) -> int:
    from .evaluate import sheet

    print(f"{sheet(args.answers, args.out)} answers written to {args.out}")
    return 0


def cmd_eval_summarize(args) -> int:
    from .evaluate import summarize

    print(json.dumps(summarize(args.sheet), indent=2))
    return 0


def cmd_export(args) -> int:
    from .export import run

    print(json.dumps(run(cfg.load(args.config), args.adapter, args.out, args.check_against), indent=2))
    return 0


def cmd_manifest(args) -> int:
    from .manifest import budget, build, runtime_from

    runtime = runtime_from(args.runtime_from) if args.runtime_from else None
    manifest = build(cfg.load(args.config), args.dir, args.revision, runtime)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(budget(manifest), indent=2))
    return 0


def cmd_budget(args) -> int:
    from .manifest import budget

    result = budget(json.loads(args.manifest.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2))
    return 1 if result["problems"] else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="msm", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="training data").add_subparsers(dest="action", required=True)
    d = data.add_parser("check", help="schema, licences, personal data screen, emergency filter, answer style")
    d.add_argument("--records", type=Path, default=Path("data/train.jsonl"))
    d.add_argument("--sources", type=Path, default=Path("data/sources.yaml"))
    d.add_argument("--draft", action="store_true", help="allow records not yet approved by review")
    d.set_defaults(func=cmd_data_check)

    ev = sub.add_parser("eval", help="evaluation").add_subparsers(dest="action", required=True)
    e = ev.add_parser("check", help="check evaluation items, including overlap with the training data")
    e.add_argument("--items", type=Path, default=Path("evalsets/health-qa-v0/items.jsonl"))
    e.add_argument("--records", type=Path, default=Path("data/train.jsonl"))
    e.add_argument("--sources", type=Path, default=Path("data/sources.yaml"))
    e.set_defaults(func=cmd_eval_check)
    e = ev.add_parser("run", help="answer every item with a local model")
    e.add_argument("--model", required=True)
    e.add_argument("--items", type=Path, default=Path("evalsets/health-qa-v0/items.jsonl"))
    e.add_argument("--out", type=Path, required=True)
    e.set_defaults(func=cmd_eval_run)
    e = ev.add_parser("sheet", help="write the clinician rating sheet")
    e.add_argument("--answers", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.set_defaults(func=cmd_eval_sheet)
    e = ev.add_parser("summarize", help="totals from a completed clinician sheet")
    e.add_argument("--sheet", type=Path, required=True)
    e.set_defaults(func=cmd_eval_summarize)

    pr = sub.add_parser("prompt", help="show the system prompt, or compare it with the app's")
    pr.add_argument("--app", type=Path, help="path to metacare-gpt-frontend/src/lib/onDeviceChat.ts")
    pr.set_defaults(func=cmd_prompt)

    c = sub.add_parser("config", help="check run configurations")
    c.add_argument("configs", type=Path, nargs="+")
    c.set_defaults(func=cmd_config_check)

    t = sub.add_parser("train", help="fine tune a LoRA adapter")
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--dry-run", action="store_true", help="check the data and print the plan; needs no GPU")
    t.set_defaults(func=cmd_train)

    x = sub.add_parser("export", help="merge, export to ONNX and quantize to q4")
    x.add_argument("--config", type=Path, required=True)
    x.add_argument("--adapter", type=Path, required=True)
    x.add_argument("--out", type=Path, required=True)
    x.add_argument("--check-against", type=Path, help="a published model_q4.onnx that runs in the app")
    x.set_defaults(func=cmd_export)

    m = sub.add_parser("manifest", help="build the app's manifest for an exported model")
    m.add_argument("--config", type=Path, required=True)
    m.add_argument("--dir", type=Path, required=True)
    m.add_argument("--revision", required=True, help="the commit of the published model repository")
    m.add_argument("--runtime-from", type=Path, help="an app chatManifest.*.json to copy the runtime block from")
    m.add_argument("--out", type=Path, required=True)
    m.set_defaults(func=cmd_manifest)

    b = sub.add_parser("budget", help="check a manifest against the iPhone and browser cache limits")
    b.add_argument("manifest", type=Path)
    b.set_defaults(func=cmd_budget)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
