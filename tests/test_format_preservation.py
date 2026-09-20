"""Format-preservation mutation tests.

These tests pin the exact rendered bytes of tomlkit documents while they are
mutated.  They focus on the three areas where style-preserving TOML libraries
are easiest to break without changing any semantic result:

* dotted keys that share prefixes with explicit ``[table]`` headers,
* arrays of tables (AoT) with nested sub-tables and element boundaries,
* out-of-order tables (a child header declared before / after its parent),
  including trailing comments, blank lines and CRLF documents.

The boundary matrix is data-driven from
``tests/fixtures/format_preservation/boundary_matrix.yaml`` so each scenario
can be run in isolation (``pytest tests/test_format_preservation.py -k ...``),
and property tests below it exercise the same invariants on generated
documents with deterministic seeds.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from tomlkit import dumps
from tomlkit import parse
from tomlkit import table as _table
from tomlkit.items import AoT as _AoT
from tomlkit.toml_document import TOMLDocument as _Document

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "format_preservation"

# A readable placeholder in the YAML fixtures: the loader turns it into a real
# carriage return so diffs stay reviewable and git does not mangle CR bytes.
CRLF_PLACEHOLDER = "CRLF"

def _expand_markers(text: str) -> str:
    """Turn the readable ``CRLF`` end-of-line marker into real CR bytes.

    Only a marked line ending is rewritten; an unmarked final line keeps no
    synthetic newline (YAML block scalars already encode the final break).
    A single space in front of the marker is treated as separator for the
    readable syntax and dropped (``"# c CRLF"`` -> ``"# c\r\n"``).
    """
    return re.sub(rf" ?{CRLF_PLACEHOLDER}(\n|$)", "\r\n", text)

@dataclass(frozen=True)
class BoundaryCase:
    raw: dict[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.raw["id"])

    @property
    def description(self) -> str:
        return str(self.raw.get("description", ""))

    @property
    def toml(self) -> str:
        text = self.raw["toml"]
        return _expand_markers(text)

    @property
    def noop(self) -> bool:
        return bool(self.raw.get("noop", False))

    @property
    def ops(self) -> list[dict[str, Any]]:
        return list(self.raw.get("ops", []))

    @property
    def expect_bytes(self) -> str | None:
        if "expect_bytes" not in self.raw:
            return None
        return _expand_markers(self.raw["expect_bytes"])

    @property
    def changed_lines(self) -> list[int]:
        return list(self.raw.get("changed_lines", []))

    @property
    def contains(self) -> list[str]:
        return list(self.raw.get("contains", []))

    @property
    def absent(self) -> list[str]:
        return list(self.raw.get("absent", []))

    @property
    def reserializes(self) -> bool:
        return bool(self.raw.get("reserializes", True))

def _load_boundary_cases() -> list[BoundaryCase]:
    with open(FIXTURE_DIR / "boundary_matrix.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [BoundaryCase(case) for case in data["cases"]]

CASES = _load_boundary_cases()

def _walk(doc: Any, path: list[str]) -> Any:
    cur = doc
    for part in path:
        cur = cur[part]
    return cur

def _apply_op(doc: Any, op: dict[str, Any]) -> None:
    kind = op["op"]
    path: list[str] = list(op.get("path", []))

    if kind == "update":
        parent = _walk(doc, path[:-1])
        parent[path[-1]] = op["value"]
    elif kind == "delete":
        parent = _walk(doc, path[:-1])
        del parent[path[-1]]
    elif kind == "reinsert":
        parent = _walk(doc, path[:-1])
        key = path[-1]
        value = parent.pop(key)
        parent[key] = value
    elif kind in ("aot_pop", "aot_insert"):
        aot = _walk(doc, path)
        if kind == "aot_pop":
            popped = aot.pop(op["index"])
            _POPPED[id(doc)] = popped
        else:
            aot.insert(op["index"], _POPPED.pop(id(doc)))
    elif kind == "aot_swap":
        aot = _walk(doc, path)
        a, b = op["a"], op["b"]
        held = aot[a]
        aot[a] = aot[b]
        aot[b] = held
    elif kind == "append":
        parent = _walk(doc, path)
        parent[op["key"]] = op["value"]
    else:  # pragma: no cover - guarded by the YAML author, not runtime users
        raise AssertionError(f"unknown op in fixture: {kind!r}")

# Stores a popped AoT element across the paired aot_pop / aot_insert ops of a
# single case. Keyed by document id so a failure never leaks state into the
# next case.
_POPPED: dict[int, Any] = {}

def _assert_unchanged_regions(original: str, rendered: str, allowed: list[int]) -> None:
    """Every line outside ``allowed`` (1-based) must be byte-identical."""
    old_lines = original.splitlines(keepends=True)
    new_lines = rendered.splitlines(keepends=True)
    if len(old_lines) != len(new_lines):
        pytest.fail(
            "line count changed while only line(s) "
            f"{allowed} were allowed to change:\n"
            f"--- original ---\n{original!r}\n--- rendered --- \n{rendered!r}"
        )
    mismatched = [
        i
        for i, (a, b) in enumerate(zip(old_lines, new_lines), start=1)
        if a != b and i not in allowed
    ]
    if mismatched:
        details = "\n".join(
            f"line {i}: {old_lines[i - 1]!r} -> {new_lines[i - 1]!r}"
            for i in mismatched
        )
        pytest.fail(
            "byte drift outside the allowed mutation region "
            f"(allowed lines {allowed}):\n{details}"
        )
    for line_no in allowed:
        assert old_lines[line_no - 1] != new_lines[line_no - 1], (
            f"fixture claims line {line_no} changes, but it did not; "
            "update changed_lines to describe the real mutation region"
        )

@pytest.mark.parametrize(
    "case", CASES, ids=[case.case_id for case in CASES]
)
def test_boundary_matrix(case: BoundaryCase) -> None:
    original = case.toml

    # A pure parse -> dumps cycle must always be byte-identical first; this is
    # the baseline every mutation assertion relies on.
    baseline = dumps(parse(original))
    assert baseline == original, (
        "no-op round trip is already lossy:\n"
        f"--- original ---\n{original!r}\n--- rendered --- \n{baseline!r}"
    )

    doc = parse(original)
    for op in case.ops:
        _apply_op(doc, op)
    rendered = dumps(doc)

    if case.expect_bytes is not None:
        assert rendered == case.expect_bytes, (
            f"[{case.case_id}] {case.description}\n"
            f"--- expected ---\n{case.expect_bytes!r}\n"
            f"--- rendered ---\n{rendered!r}"
        )
    elif case.changed_lines:
        _assert_unchanged_regions(original, rendered, case.changed_lines)

    for needle in case.contains:
        assert needle in rendered, (
            f"[{case.case_id}] expected {needle!r} in output:\n{rendered!r}"
        )
    for needle in case.absent:
        assert needle not in rendered, (
            f"[{case.case_id}] did not expect {needle!r} in output:\n{rendered!r}"
        )

    if case.reserializes:
        reparsed = dumps(parse(rendered))
        assert reparsed == rendered, (
            f"[{case.case_id}] output is not a fixed point:\n"
            f"--- rendered ---\n{rendered!r}\n--- reparsed ---\n{reparsed!r}"
        )

# ---------------------------------------------------------------------------
# Generative property tests
#
# The boundary matrix pins hand-written dangerous shapes; these tests build
# randomised but deterministically seeded documents that combine the same
# features (dotted keys, nested AoT tables, child-before-parent headers,
# comments, blank lines and CRLF) and assert structural invariants.
# Seeds are fixed integers, so failures reproduce exactly: no sleeps, network
# or machine-specific state are involved.
# ---------------------------------------------------------------------------

import random

def _generate_document(
    seed: int, eol: str
) -> tuple[str, list[tuple[list[Any], str]], int]:
    """Return ``(text, scalar_targets, aot_element_count)``.

    A scalar target is ``(access_path, line_prefix)`` where ``access_path``
    walks the parsed document to the value and ``line_prefix`` is the exact
    prefix of the source line that renders it.
    """
    rng = random.Random(seed)
    lines: list[str] = []
    targets: list[tuple[list[Any], str]] = []

    def emit(line: str) -> None:
        lines.append(line)

    for i in range(rng.randint(1, 3)):
        prefix = f"root{i}"
        emit(f"{prefix} = {rng.randint(0, 9)} # root-comment-{i}")
        targets.append(([prefix], prefix + " = "))

    node_count = rng.randint(2, 4)
    for i in range(node_count):
        prefix = f"shared.node{i}.leaf"
        emit(f"{prefix} = {i} # dotted-comment-{i}")
        targets.append((["shared", f"node{i}", "leaf"], prefix + " = "))

    emit("[ooo.child1]")
    emit("v = 1 # child-one")
    emit("[ooo]")
    emit("w = 2 # parent-between-children")
    emit("[ooo.child2]")
    emit("v = 3 # child-two")
    targets.append((["ooo", "child1", "v"], "v = 1 # child-one"))
    targets.append((["ooo", "w"], "w = 2 # parent-between-children"))
    targets.append((["ooo", "child2", "v"], "v = 3 # child-two"))

    element_count = rng.randint(2, 4)
    for i in range(element_count):
        emit(f"[[items]] # element-header-{i}")
        emit(f'name = "n{i}" # element-name-{i}')
        emit("[items.meta]")
        emit(f"k = {i} # element-meta-{i}")
        targets.append((["items", i, "name"], f'name = "n{i}"'))
        targets.append((["items", i, "meta", "k"], f"k = {i}"))
        if i < element_count - 1 and rng.choice([True, False]):
            emit("")

    text = eol.join(lines) + eol
    return text, targets, element_count

SEEDS = list(range(16))
EOLS = ["\n", "\r\n"]

@pytest.mark.parametrize("eol", EOLS, ids=["lf", "crlf"])
@pytest.mark.parametrize("seed", SEEDS)
def test_generated_noop_roundtrip_is_byte_identical(seed: int, eol: str) -> None:
    text, _, _ = _generate_document(seed, eol)
    assert dumps(parse(text)) == text

@pytest.mark.parametrize("eol", EOLS, ids=["lf", "crlf"])
@pytest.mark.parametrize("seed", SEEDS)
def test_generated_scalar_update_only_touches_target_line(
    seed: int, eol: str
) -> None:
    text, targets, _ = _generate_document(seed, eol)
    original_lines = text.splitlines(keepends=True)

    for path, line_prefix in targets:
        doc = parse(text)
        parent = _walk(doc, path[:-1])
        old_value = _walk(doc, path)
        new_value: Any = f"v-{seed}-{line_prefix.strip()}"
        if isinstance(old_value, int):
            new_value = 90000 + seed * 17 + len(line_prefix)
        parent[path[-1]] = new_value

        rendered = dumps(doc)
        new_lines = rendered.splitlines(keepends=True)
        assert len(new_lines) == len(original_lines), (
            f"seed={seed} path={path}\n{rendered!r}"
        )
        changed = [
            i
            for i, (a, b) in enumerate(zip(original_lines, new_lines), start=1)
            if a != b
        ]
        matching = [
            i
            for i, line in enumerate(original_lines, start=1)
            if line.startswith(line_prefix)
        ]
        assert len(matching) == 1, (path, line_prefix, matching)
        assert changed == matching, (
            f"seed={seed} eol={eol!r} path={path} "
            f"changed={changed} expected={matching}"
        )
        # Comment text and the document's line ending survive the value swap.
        rendered_line = new_lines[matching[0] - 1]
        assert rendered_line.endswith(eol)
        assert "#" in rendered_line
        assert original_lines[matching[0] - 1].split("#", 1)[1] == \
            rendered_line.split("#", 1)[1]

@pytest.mark.parametrize("eol", EOLS, ids=["lf", "crlf"])
@pytest.mark.parametrize("seed", SEEDS)
def test_generated_aot_permutation_keeps_each_elements_trivia(
    seed: int, eol: str
) -> None:
    text, _, element_count = _generate_document(seed, eol)
    doc = parse(text)
    aot = doc["items"]
    held = [aot[i] for i in range(element_count)]
    for i, element in enumerate(reversed(held)):
        aot[i] = element
    rendered = dumps(doc)

    name_lines = [f'name = "n{i}"' for i in range(element_count)]
    positions = [rendered.index(line) for line in name_lines]
    assert positions == sorted(positions, reverse=True)
    for i in range(element_count):
        assert rendered.count(f"# element-header-{i}") == 1
        assert rendered.count(f"# element-name-{i}") == 1
        assert rendered.count(f"# element-meta-{i}") == 1
        assert f"[items.meta]{eol}k = {i} # element-meta-{i}" in rendered
    # No element is rendered twice and the permuted document is a fixed point.
    assert rendered.count("[[items]]") == element_count
    assert dumps(parse(rendered)) == rendered

@pytest.mark.parametrize("eol", EOLS, ids=["lf", "crlf"])
@pytest.mark.parametrize("seed", SEEDS)
def test_generated_delete_dotted_or_scalar_removes_exactly_one_line(
    seed: int, eol: str
) -> None:
    text, targets, _ = _generate_document(seed, eol)
    # Only remove lines that are unique by their rendered prefix; the two
    # ``v = ...`` out-of-order lines happen to be unique through comments, but
    # deletion is asserted per full line text anyway below.
    selected = [
        (path, prefix)
        for path, prefix in targets
        if path[0] in ("root0",) or path[0] == "shared"
    ]
    for path, line_prefix in selected[: 2 if selected else 0]:
        doc = parse(text)
        parent = _walk(doc, path[:-1])
        del parent[path[-1]]
        rendered = dumps(doc)

        original_lines = text.splitlines(keepends=True)
        matching = [
            (i, line)
            for i, line in enumerate(original_lines)
            if line.startswith(line_prefix)
        ]
        assert len(matching) == 1
        removed_idx, _removed_line = matching[0]
        expected = (
            original_lines[:removed_idx] + original_lines[removed_idx + 1 :]
        )
        assert rendered == "".join(expected), (
            f"seed={seed} path={path}\n"
            f"--- expected ---\n{''.join(expected)!r}\n"
            f"--- rendered ---\n{rendered!r}"
        )
        assert dumps(parse(rendered)) == rendered

@pytest.mark.parametrize("eol", EOLS, ids=["lf", "crlf"])
@pytest.mark.parametrize("seed", SEEDS)
def test_generated_out_of_order_region_is_byte_stable_under_distant_edit(
    seed: int, eol: str
) -> None:
    # Editing a root scalar (which lives in the root value region) must never
    # reorder or re-render the split [ooo.child1] / [ooo] / [ooo.child2]
    # declarations, nor the AoT block following them.
    text, _, _ = _generate_document(seed, eol)
    doc = parse(text)
    doc["root0"] = -1
    rendered = dumps(doc)

    region_anchor = "[ooo.child1]"
    region_start = rendered.index(region_anchor)
    original_start = text.index(region_anchor)
    assert rendered[region_start:] == text[original_start:], (
        f"seed={seed}\n--- tail expected ---\n{text[original_start:]!r}\n"
        f"--- tail rendered ---\n{rendered[region_start:]!r}"
    )

# ---------------------------------------------------------------------------
# Programmatic-build paths
#
# The matrix above mutates parsed documents.  Two dangerous paths only exist
# when items are built in code and inserted into parsed/built containers: the
# blank-line inheritance of an AoT element inserted in the middle, and the
# insertion boundary in front of a dotted super-table that gained a header.
# Both are pinned here with exact bytes plus a semantic reparse assertion.
# ---------------------------------------------------------------------------

def _new_aot_document(element_values: list[int]) -> tuple[_Document, _AoT]:
    doc = _Document()
    doc["root"] = 0
    aot = _AoT([], parsed=False)
    for value in element_values:
        element = _table()
        element["x"] = value
        element._is_aot_element = True
        aot.append(element)
    doc["p"] = aot
    return doc, aot

def test_built_aot_insert_inherits_blank_line_before_element() -> None:
    doc, aot = _new_aot_document([0, 1, 2])

    inserted = _table()
    inserted["x"] = 9
    inserted._is_aot_element = True
    aot.insert(1, inserted)

    expected = (
        "root = 0\n"
        "\n"
        "[[p]]\n"
        "x = 0\n"
        "\n"
        "[[p]]\n"
        "x = 9\n"
        "\n"
        "[[p]]\n"
        "x = 1\n"
        "\n"
        "[[p]]\n"
        "x = 2\n"
    )
    rendered = dumps(doc)
    assert rendered == expected, repr(rendered)
    reparsed = parse(rendered)
    assert dumps(reparsed) == rendered
    assert [element["x"] for element in reparsed["p"]] == [0, 9, 1, 2]

def test_scalar_after_dotted_super_table_with_new_header_stays_at_root() -> None:
    # Regression for the "dotted super table gains a child header, then a
    # scalar is appended" boundary: the scalar must render before the header
    # region, or it gets semantically captured by the new table on reparse.
    doc = parse("a.b = 1 # dotted-original\n")
    doc["a"]["c"] = {}
    doc["z"] = 2

    rendered = dumps(doc)
    assert rendered == (
        "z = 2\n"
        "\n"
        "a.b = 1 # dotted-original\n"
        "\n"
        "[a.c]\n"
    ), repr(rendered)
    reparsed = parse(rendered)
    assert reparsed.unwrap() == {"a": {"b": 1, "c": {}}, "z": 2}
    assert "z" in reparsed
    assert "z" not in reparsed["a"]["c"]
    # The original dotted line keeps its trivia while the region moves.
    assert "a.b = 1 # dotted-original" in rendered
