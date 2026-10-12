"""Command-line interface.

    whichtest list [--status ready]        tests in the taxonomy and whether they are implemented
    whichtest show <test_id>               what the taxonomy says about a test
    whichtest resolve key=value ...        design answers → spec template (JSON)
    whichtest ask [--out spec.json]        interactive wizard through the taxonomy → spec
    whichtest tree [--out file.md]         the question tree as Markdown (for the skill)
    whichtest check spec.json              validate a spec against the taxonomy
    whichtest run spec.json data.csv       run the test, print the report (--json for machine output)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .registry import available, run
from .spec import Design, Spec, SpecError
from .taxonomy import TaxonomyError, load_taxonomy


def _load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".csv", ".txt"):
        return pd.read_csv(p)
    if suf == ".tsv":
        return pd.read_csv(p, sep="\t")
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".json":
        return pd.read_json(p)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    raise SystemExit(f"unsupported data format: {suf}")


def _load_spec(path: str) -> Spec:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    try:
        return Spec(**raw)
    except Exception as e:  # pydantic ValidationError
        raise SystemExit(f"invalid spec {path}: {e}") from None


def cmd_list(args: argparse.Namespace) -> int:
    tax = load_taxonomy()
    impl = set(available())
    rows = [t for t in tax.tests.values() if not args.status or t.status == args.status]
    width = max(len(t.id) for t in rows)
    for t in rows:
        mark = "yes" if t.id in impl else "no "
        print(f"{t.id:<{width}}  {t.status:<9} {mark}  {t.name}")
    n_impl = sum(t.id in impl for t in rows)
    print(f"\n{len(rows)} tests, {n_impl} implemented", file=sys.stderr)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    tax = load_taxonomy()
    try:
        t = tax.test(args.test_id)
    except TaxonomyError as e:
        raise SystemExit(str(e)) from None
    print(f"{t.name} [{t.id}]  status: {t.status}  family: {t.family}")
    print(f"  when: {t.when}")
    print("  checks:")
    for c in t.checks:
        print(f"    - {c.key}: {c.text}")
    if t.fallback:
        print(f"  fallback: {t.fallback.text} → {t.fallback.to or t.fallback.technique}")
    print(f"  call: {t.call}")
    print(f"  cross-check: {t.xcall}\n    {t.cross_why}")
    if t.post:
        print(f"  post hoc: {t.post}")
    print(f"  report: {t.report}")
    if t.watch:
        print(f"  watch: {t.watch}")
    print("  reached by:")
    for p in tax.paths_to(t.id):
        print("    " + ", ".join(f"{k}={v}" for k, v in p.items()))
    impl = t.id in available()
    print(f"  implemented: {'yes' if impl else 'no'}")
    if impl:
        from .registry import PROCEDURES

        proc = PROCEDURES[t.id]
        if proc.requires:
            print(f"  variables required: {', '.join(proc.requires)}")
        if proc.decisions:
            print(f"  decisions required (spec.decisions): {', '.join(proc.decisions)}")
    return 0


def _spec_template(spec: Spec) -> dict:
    d = json.loads(spec.to_json())
    d["variables"] = {k: v for k, v in d.get("variables", {}).items() if v not in ([], None)}
    return d


def cmd_resolve(args: argparse.Namespace) -> int:
    answers = {}
    for kv in args.answers:
        if "=" not in kv:
            raise SystemExit(f"expected key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        answers[k] = v
    try:
        spec = Spec.from_design(answers)
    except (SpecError, TaxonomyError) as e:
        raise SystemExit(str(e)) from None
    except Exception as e:
        raise SystemExit(f"bad design: {e}") from None
    print(json.dumps(_spec_template(spec), indent=2, ensure_ascii=False))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    tax = load_taxonomy()
    answers: dict[str, str] = {}
    print("Answer each question with a number. Ctrl-C to quit.\n", file=sys.stderr)
    while (q := tax.next_question(answers)) is not None:
        print(f"{q.q}", file=sys.stderr)
        if q.help:
            print(f"  ({q.help})", file=sys.stderr)
        for i, o in enumerate(q.options, 1):
            print(f"  {i}. {o.label}" + (f" — {o.sub}" if o.sub else ""), file=sys.stderr)
        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                return 1
            if raw.isdigit() and 1 <= int(raw) <= len(q.options):
                answers[q.key] = q.options[int(raw) - 1].val
                break
            if raw in [o.val for o in q.options]:
                answers[q.key] = raw
                break
            print(f"  pick 1–{len(q.options)}", file=sys.stderr)
        print(file=sys.stderr)
    spec = Spec.from_design(Design(**answers))
    info = tax.test(spec.test)
    print(f"→ {info.name} [{spec.test}] (status: {info.status})", file=sys.stderr)
    print(f"  {info.when}", file=sys.stderr)
    if info.status == "judgment":
        print("  This test needs explicit human decisions in spec.decisions before it will run.", file=sys.stderr)
    out = json.dumps(_spec_template(spec), indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"  spec written to {args.out}; fill in variables, then: whichtest run {args.out} data.csv", file=sys.stderr)
    else:
        print(out)
    return 0


def render_tree(tax=None) -> str:
    """The question tree as Markdown: one line per option, leaves name the test and its status."""
    from .registry import PROCEDURES, _load

    _load()
    tax = tax or load_taxonomy()
    lines = ["# Taxonomy questions", "", "Generated by `whichtest tree`; do not edit by hand.", ""]

    def walk(nid: str, depth: int) -> None:
        q = tax.questions[nid]
        ind = "  " * depth
        lines.append(f"{ind}- **{q.key}**: {q.q}")
        for o in q.options:
            if o.is_leaf:
                t = tax.test(o.test_id)
                dec = PROCEDURES.get(t.id).decisions if t.id in PROCEDURES else ()
                extra = f"; decisions: {', '.join(dec)}" if dec else ""
                lines.append(f"{ind}  - `{o.val}` → **{t.name}** (`{t.id}`, {t.status}{extra}): {o.label}")
            else:
                lines.append(f"{ind}  - `{o.val}`: {o.label}")
                walk(o.to, depth + 2)

    walk(tax.root, 0)
    lines += ["", "## Tests", ""]
    for t in tax.tests.values():
        dec = PROCEDURES.get(t.id).decisions if t.id in PROCEDURES else ()
        lines.append(f"- `{t.id}` **{t.name}** ({t.status}): {t.when} Checks: {', '.join(t.check_keys)}."
                     + (f" Fallback: {t.fallback.to or t.fallback.technique}." if t.fallback else "")
                     + (f" Decisions required: {', '.join(dec)}." if dec else ""))
    return "\n".join(lines) + "\n"


def cmd_tree(args: argparse.Namespace) -> int:
    out = render_tree()
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"written to {args.out}", file=sys.stderr)
    else:
        print(out, end="")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    try:
        spec = spec.complete()
    except SpecError as e:
        print(f"INVALID: {e}")
        return 1
    print(f"OK: {spec.test} ({spec.status}); implemented: {'yes' if spec.test in available() else 'no'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    data = _load_data(args.data) if args.data else None
    try:
        result = run(spec, data)
    except (SpecError, TaxonomyError, NotImplementedError) as e:
        raise SystemExit(f"error: {e}") from None
    if args.json:
        print(result.to_json())
    else:
        print(result.summary())
    if args.out:
        Path(args.out).write_text(result.to_json() + "\n", encoding="utf-8")
    return 2 if result.flagged and args.strict else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="whichtest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"whichtest {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list tests")
    p.add_argument("--status", choices=["ready", "helper", "judgment"])
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="describe a test")
    p.add_argument("test_id")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("resolve", help="design answers → spec template")
    p.add_argument("answers", nargs="+", metavar="key=value")
    p.set_defaults(fn=cmd_resolve)

    p = sub.add_parser("ask", help="interactive wizard → spec")
    p.add_argument("--out", help="write the spec to this file")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("tree", help="print the question tree as Markdown")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_tree)

    p = sub.add_parser("check", help="validate a spec file")
    p.add_argument("spec")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("run", help="run a spec on a dataset")
    p.add_argument("spec")
    p.add_argument("data", nargs="?", help="CSV/TSV/parquet/JSON/xlsx; optional for table-only tests")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--out", help="also write the JSON result to this file")
    p.add_argument("--strict", action="store_true", help="exit 2 if the result is flagged for review")
    p.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
