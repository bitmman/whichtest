"""Load and walk ``taxonomy.json``, the decision tree that maps a study design to a test.

The JSON file at the repository root is the source of truth. This module never
modifies it; it only reads it and answers questions about it:

* which question comes next given the answers so far,
* which test a complete set of answers leads to,
* what the taxonomy says about a test (name, status, checks, fallback, ...).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "Taxonomy",
    "Question",
    "Option",
    "TestInfo",
    "load_taxonomy",
    "TaxonomyError",
]


class TaxonomyError(ValueError):
    """Raised when a design does not describe a path through the tree."""


@dataclass(frozen=True)
class Option:
    val: str
    to: str
    label: str = ""
    sub: str = ""

    @property
    def is_leaf(self) -> bool:
        return self.to.startswith("L:")

    @property
    def test_id(self) -> str | None:
        return self.to[2:] if self.is_leaf else None


@dataclass(frozen=True)
class Question:
    id: str
    key: str
    q: str
    help: str
    options: tuple[Option, ...]

    def option(self, val: str) -> Option:
        for o in self.options:
            if o.val == val:
                return o
        raise TaxonomyError(
            f"{val!r} is not an answer to {self.key!r}; choose one of "
            f"{[o.val for o in self.options]}"
        )


@dataclass(frozen=True)
class Check:
    key: str
    text: str
    code: str | None = None


@dataclass(frozen=True)
class Fallback:
    text: str
    to: str | None = None  # another test id in the tree
    technique: str | None = None  # a technique that is not in the tree
    code: str | None = None


@dataclass(frozen=True)
class TestInfo:
    id: str
    name: str
    family: str
    status: str  # ready | helper | judgment
    when: str
    checks: tuple[Check, ...]
    fallback: Fallback | None
    run: str
    call: str
    cross_code: str
    cross_why: str
    xcall: str
    report: str
    post: str | None = None
    watch: str | None = None

    @property
    def check_keys(self) -> list[str]:
        return [c.key for c in self.checks]


@dataclass
class Taxonomy:
    version: Any
    description: str
    root: str
    status_levels: dict[str, str]
    questions: dict[str, Question]
    tests: dict[str, TestInfo]
    _paths: dict[str, list[dict[str, str]]] = field(default_factory=dict, repr=False)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Taxonomy":
        questions = {
            nid: Question(
                id=nid,
                key=n["key"],
                q=n["q"],
                help=n.get("help", ""),
                options=tuple(
                    Option(val=o["val"], to=o["to"], label=o.get("label", ""), sub=o.get("sub", ""))
                    for o in n["opts"]
                ),
            )
            for nid, n in raw["nodes"].items()
        }
        tests = {}
        for tid, t in raw["tests"].items():
            fb = t.get("fb")
            fallback = (
                Fallback(text=fb.get("t", ""), to=fb.get("to"), technique=fb.get("k"), code=fb.get("c"))
                if fb
                else None
            )
            tests[tid] = TestInfo(
                id=tid,
                name=t["name"],
                family=t["family"],
                status=t["status"],
                when=t.get("when", ""),
                checks=tuple(Check(key=c["k"], text=c["t"], code=c.get("c")) for c in t.get("checks", [])),
                fallback=fallback,
                run=t.get("run", ""),
                call=t.get("call", ""),
                cross_code=t["cross"]["c"],
                cross_why=t["cross"]["why"],
                xcall=t.get("xcall", ""),
                report=t.get("report", ""),
                post=t.get("post"),
                watch=t.get("watch"),
            )
        tax = cls(
            version=raw.get("version"),
            description=raw.get("description", ""),
            root=raw["root"],
            status_levels=dict(raw.get("status_levels", {})),
            questions=questions,
            tests=tests,
        )
        tax._validate()
        return tax

    def _validate(self) -> None:
        for q in self.questions.values():
            for o in q.options:
                if o.is_leaf:
                    if o.test_id not in self.tests:
                        raise TaxonomyError(f"{q.id}.{o.val} points to unknown test {o.test_id!r}")
                elif o.to not in self.questions:
                    raise TaxonomyError(f"{q.id}.{o.val} points to unknown node {o.to!r}")
        reachable = {p_test for p_test, _ in self.paths()}
        missing = set(self.tests) - reachable
        if missing:
            raise TaxonomyError(f"tests not reachable from the root: {sorted(missing)}")
        for t in self.tests.values():
            if t.status not in self.status_levels:
                raise TaxonomyError(f"{t.id}: unknown status {t.status!r}")
            if t.fallback and t.fallback.to and t.fallback.to not in self.tests:
                raise TaxonomyError(f"{t.id}: fallback points to unknown test {t.fallback.to!r}")

    # -- walking ---------------------------------------------------------
    def next_question(self, answers: dict[str, str]) -> Question | None:
        """The next unanswered question for ``answers`` (keyed by question ``key``).

        Returns ``None`` once the answers reach a leaf.
        """
        node = self.questions[self.root]
        while True:
            if node.key not in answers:
                return node
            opt = node.option(answers[node.key])
            if opt.is_leaf:
                return None
            node = self.questions[opt.to]

    def resolve(self, answers: dict[str, str]) -> str:
        """Follow ``answers`` from the root to a leaf and return the test id.

        Keys that are not on the path are ignored; a missing key raises.
        """
        node = self.questions[self.root]
        while True:
            if node.key not in answers:
                raise TaxonomyError(
                    f"design is incomplete: {node.key!r} is not set ({node.q}). "
                    f"Options: {[o.val for o in node.options]}"
                )
            opt = node.option(answers[node.key])
            if opt.is_leaf:
                return opt.test_id  # type: ignore[return-value]
            node = self.questions[opt.to]

    def path_keys(self, answers: dict[str, str]) -> list[str]:
        """The question keys actually consulted when resolving ``answers``."""
        keys: list[str] = []
        node = self.questions[self.root]
        while True:
            keys.append(node.key)
            opt = node.option(answers[node.key])
            if opt.is_leaf:
                return keys
            node = self.questions[opt.to]

    def paths(self) -> Iterator[tuple[str, dict[str, str]]]:
        """Every (test_id, answers) pair in the tree, in tree order."""

        def walk(nid: str, acc: dict[str, str]) -> Iterator[tuple[str, dict[str, str]]]:
            node = self.questions[nid]
            for o in node.options:
                a = {**acc, node.key: o.val}
                if o.is_leaf:
                    yield o.test_id, a  # type: ignore[misc]
                else:
                    yield from walk(o.to, a)

        yield from walk(self.root, {})

    def paths_to(self, test_id: str) -> list[dict[str, str]]:
        if test_id not in self.tests:
            raise TaxonomyError(f"unknown test {test_id!r}")
        if not self._paths:
            for tid, a in self.paths():
                self._paths.setdefault(tid, []).append(a)
        return list(self._paths.get(test_id, []))

    def test(self, test_id: str) -> TestInfo:
        try:
            return self.tests[test_id]
        except KeyError:
            raise TaxonomyError(f"unknown test {test_id!r}; known: {sorted(self.tests)}") from None

    def by_status(self, status: str) -> list[str]:
        return [t.id for t in self.tests.values() if t.status == status]


def _candidate_files() -> list[Path]:
    here = Path(__file__).resolve()
    cands = [here.parents[2] / "taxonomy.json"]  # dev checkout: <repo>/taxonomy.json
    try:
        pkg = resources.files("whichtest") / "taxonomy.json"  # installed wheel
        cands.append(Path(str(pkg)))
    except Exception:  # pragma: no cover
        pass
    return cands


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    """Load the taxonomy. With no ``path``, find the repository copy or the packaged copy."""
    if path is None:
        for cand in _candidate_files():
            if cand.is_file():
                path = cand
                break
        else:
            raise FileNotFoundError("taxonomy.json not found; pass a path explicitly")
    with open(path, encoding="utf-8") as fh:
        return Taxonomy.from_dict(json.load(fh))
