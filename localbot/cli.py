"""Run experiments without the UI, e.g. from a script or overnight.

    python -m localbot.cli sets
    python -m localbot.cli add-set questions.jsonl --name "My set"
    python -m localbot.cli run "My set" --set top_k=3 --set rerank=false
    python -m localbot.cli run "My set" --sweep top_k=2,3,4 --passages-only
    python -m localbot.cli runs
    python -m localbot.cli report <run id or name>

Results appear in the UI as well (Experiments, Evaluate).
"""
import argparse
import json
import sys
import time

from . import experiments, storage
from .config import SCHEMA_BY_KEY


def _value(key, raw):
    spec = SCHEMA_BY_KEY.get(key)
    if not spec:
        raise SystemExit(f"Unknown setting {key!r}. Known: {', '.join(SCHEMA_BY_KEY)}")
    if spec["type"] == "bool":
        return raw.lower() in ("1", "true", "yes", "on")
    return raw


def _find_set(ref):
    sets = experiments.list_sets()
    match = [s for s in sets if s["id"] == ref or s["name"] == ref] or [s for s in sets if ref.lower() in s["name"].lower()]
    if len(match) != 1:
        raise SystemExit(f"Question set {ref!r} not found or ambiguous. Sets: {', '.join(s['name'] for s in sets) or 'none'}")
    return match[0]


def _find_run(ref):
    runs = experiments.list_runs()
    match = [r for r in runs if r["id"] == ref] or [r for r in runs if ref.lower() in r["name"].lower()]
    if not match:
        raise SystemExit(f"Run {ref!r} not found")
    return match[0]


def _print_summary(run):
    s = run["summary"]
    print(f"\n{run['name']}  [{run['status']}]  {run['total']} questions")
    for key in ("good_rate", "auto_correct", "fact_coverage", "context_recall", "refusal_accuracy", "offtopic_blocked",
                "false_refusals", "generation_misses", "flagged", "latency_s", "graded"):
        if s.get(key) is not None:
            v = s[key]
            print(f"  {key:<18} {v:.3f}" if isinstance(v, float) else f"  {key:<18} {v}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m localbot.cli", description="LocalBot experiments")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sets", help="list question sets")
    a = sub.add_parser("add-set", help="add a question set from a .jsonl file")
    a.add_argument("file")
    a.add_argument("--name")
    r = sub.add_parser("run", help="run a question set and wait for the result")
    r.add_argument("question_set")
    r.add_argument("--name", default="")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="setting override (repeatable)")
    r.add_argument("--sweep", metavar="KEY=V1,V2,...", help="one run per value")
    r.add_argument("--passages-only", action="store_true", help="skip answers; score retrieved passages only")
    r.add_argument("--limit", type=int)
    r.add_argument("--variant", action="append", help="only these variants (repeatable)")
    sub.add_parser("runs", help="list runs")
    rep = sub.add_parser("report", help="print a run's summary as JSON")
    rep.add_argument("run")
    args = p.parse_args(argv)
    storage.init()

    if args.cmd == "sets":
        for s in experiments.list_sets():
            print(f"{s['id']}  {s['name']:<30} {s['count']:>4} questions  {s['variants']}")
    elif args.cmd == "add-set":
        with open(args.file, encoding="utf-8") as f:
            items, errors = experiments.parse_jsonl(f.read())
        if errors:
            raise SystemExit("\n".join(errors))
        sid = experiments.create_set(args.name or args.file.rsplit("/", 1)[-1].rsplit(".", 1)[0], items)
        print(f"Added {len(items)} questions as set {sid}")
    elif args.cmd == "runs":
        for run in experiments.list_runs():
            s = run["summary"]
            print(f"{run['id']}  {run['status']:<11} {run['name']:<40} auto={s.get('auto_correct')}  recall={s.get('context_recall')}")
    elif args.cmd == "report":
        print(json.dumps(experiments.get_run(_find_run(args.run)["id"])["summary"], indent=2))
    elif args.cmd == "run":
        qs = _find_set(args.question_set)
        overrides = {}
        for kv in args.set:
            k, _, v = kv.partition("=")
            overrides[k] = _value(k, v)
        values = [None]
        if args.sweep:
            key, _, raw = args.sweep.partition("=")
            values = [(key, _value(key, v.strip())) for v in raw.split(",") if v.strip()]
        ids = []
        for kv in values:
            ov = dict(overrides, **({kv[0]: kv[1]} if kv else {}))
            name = args.name or ", ".join(f"{k}={v}" for k, v in ov.items()) or "current settings"
            if kv and args.name:
                name = f"{args.name} · {kv[0]}={kv[1]}"
            try:
                ids.append(experiments.create_run(name, qs["id"], ov, args.passages_only, args.variant, args.limit))
            except ValueError as e:
                raise SystemExit(str(e))
        experiments.start_worker()
        while True:
            runs = [experiments.get_run(i) for i in ids]
            line = "  ".join(f"{r['name']}: {r['done']}/{r['total']} {r['status']}" for r in runs)
            print("\r" + line[:160], end="", flush=True)
            if all(r["status"] not in ("queued", "running") for r in runs):
                break
            time.sleep(2)
        for r in runs:
            _print_summary(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
