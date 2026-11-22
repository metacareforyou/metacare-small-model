"""Writes made up visit scripts from specs with a local, open model and checks each one (MET-1929).

    python -m metacare_small_model.scribe.generate --out data/private/scribe/visits.jsonl \
        --count 2000 --server http://127.0.0.1:8099 --model qwen3-14b-q4_k_m@530227a7 --parallel 8

Each line of the output holds the spec, its right answers, the script, the model and revision,
the seed and how many attempts it took. A script is kept only if it states every right answer,
keeps to the format and uses no FHIR words; otherwise it is written again (up to --attempts).
Lines already in the output are skipped, so a stopped run resumes where it left off.

The script writer must be an open model whose licence allows its outputs to train other models
(Apache 2.0 or MIT; never Claude, Llama or Gemma). See MET-1929.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import threading
import time
import urllib.request

from .catalog import make_spec
from .checks import format_problems, missing

PROMPT = """Write a realistic, made up medical visit as a dialogue. Everyone in it is fictional; first names are fine, no surnames of real people.
Reason for the visit: {reason}.
Speakers: {speakers}
During the visit these must all be said clearly, in natural speech (numbers may be digits or words):
{facts}
Do not mention any other medicine, dose, test result, diagnosis or order.
Style: {style}.
Format: one line per turn, as LETTER|what they say, using only the letters {letters}. No narration, no stage directions, no headings, nothing else. About {turns} turns."""

RETRY = """

Your last version left out or garbled these, so make sure each one is said clearly this time:
{missing}"""


def prompt_for(spec: dict, rng_turns: int, missing_says: list[str] | None = None) -> str:
    text = PROMPT.format(reason=spec['reason'], speakers=json.dumps(spec['speakers']), style=spec['style'],
                         facts='\n'.join('- ' + item['say'] for item in spec['items']), letters=', '.join(spec['speakers']), turns=rng_turns)
    if missing_says:
        text += RETRY.format(missing='\n'.join('- ' + s for s in missing_says))
    return text


def ask(server: str, prompt: str, seed: int, timeout: int = 900) -> dict:
    body = json.dumps({'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.8, 'top_p': 0.9, 'seed': seed,
                       'max_tokens': 2500, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
    req = urllib.request.Request(server.rstrip('/') + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def write_one(index: int, args) -> dict:
    spec = make_spec(index, args.seed)
    turns = 18 + (index * 7) % 18
    miss: list[str] = []
    tokens = 0
    started = time.time()
    for attempt in range(1, args.attempts + 1):
        seed = args.seed * 1000 + index * 10 + attempt
        out = ask(args.server, prompt_for(spec, turns, miss or None), seed)
        script = out['choices'][0]['message']['content'].strip()
        tokens += out['usage']['completion_tokens']
        miss = missing(script, spec)
        problems = format_problems(script, spec)
        if not miss and not problems:
            return dict(id=spec['id'], ok=True, attempts=attempt, spec=spec, script=script, model=args.model, seed=seed,
                        tokens=tokens, seconds=round(time.time() - started, 1))
        miss = miss or [f'(format: {p})' for p in problems]
    return dict(id=spec['id'], ok=False, attempts=args.attempts, spec=spec, script=script, model=args.model, seed=seed,
                missing=miss, problems=problems, tokens=tokens, seconds=round(time.time() - started, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--count', type=int, default=20)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--server', default='http://127.0.0.1:8099')
    ap.add_argument('--model', required=True, help='model and revision, recorded with each script')
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--attempts', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1929)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)['id'] for line in f if line.strip()}
    todo = [i for i in range(args.start, args.start + args.count) if f'v{i:05d}' not in done]
    print(f'{len(done)} already written, {len(todo)} to write', flush=True)

    lock = threading.Lock()
    stats = dict(ok=0, failed=0, attempts=0, tokens=0)
    started = time.time()
    with open(args.out, 'a') as out, cf.ThreadPoolExecutor(args.parallel) as pool:
        for future in cf.as_completed([pool.submit(write_one, i, args) for i in todo]):
            try:
                row = future.result()
            except Exception as error:  # a server hiccup: report and carry on; a rerun fills the gap
                print('error:', error, flush=True)
                continue
            with lock:
                out.write(json.dumps(row) + '\n')
                out.flush()
                stats['ok' if row['ok'] else 'failed'] += 1
                stats['attempts'] += row['attempts']
                stats['tokens'] += row['tokens']
                n = stats['ok'] + stats['failed']
                if n % 10 == 0 or n == len(todo):
                    wall = time.time() - started
                    print(f"{n}/{len(todo)} written, {stats['ok']} kept, {stats['failed']} failed, "
                          f"{stats['attempts'] / n:.2f} attempts each, {stats['tokens'] / wall:.0f} tokens/s overall, {wall / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
