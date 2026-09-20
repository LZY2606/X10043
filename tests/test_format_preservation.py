"""Format-fidelity mutation tests for dotted keys, AoTs and out-of-order tables.

The fixtures in this module are deliberately *small* and independently
runnable: every test function can be selected on its own (see the group
``pytest -q tests/test_format_preservation.py -k matrix``).

The invariant under test is narrow and strong: after ``parse`` -> mutate ->
``dumps`` every byte that does not belong to the explicitly mutated region
must come back untouched, including whitespace, comments, CRLF endings and
the declaration order of tables/AoT elements.

``assert_unmodified_regions_identical`` reports a unified diff with 1-based
line numbers so a fidelity failure points at the exact surface line that
moved or changed instead of failing a whole-document comparison.
"""

from __future__ import annotations

import difflib
import json
import re

from collections.abc import Callable
from itertools import product
from textwrap import dedent

import pytest

import tomlkit

from tomlkit import parse
from tomlkit.items import AoT


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------


LineAllow = Callable[[str, str], bool]


def _only(old_expected: str, new_expected: str) -> LineAllow:
    """Allow-predicate accepting exactly one line replacement pair."""

    def allow(old: str, new: str) -> bool:
        return old == old_expected and new == new_expected

    return allow


def _lines(text: str) -> list[str]:
    # keepends=True keeps the exact CRLF/LF bytes attached to each line.
    return text.splitlines(keepends=True)


def _format_diff(before: str, after: str) -> str:
    diff = difflib.unified_diff(
        _lines(before),
        _lines(after),
        fromfile="input",
        tofile="dumps(parse(input))",
        n=2,
    )
    out: list[str] = []
    for line in diff:
        # difflib already terminates retained lines with their original
        # newline; only stamp synthetic context lines without one.
        out.append(line if line.endswith(("\n", "\r")) else line + "\n")
    return "".join(out)


def assert_unmodified_regions_identical(
    before: str, after: str, *, allow: Callable[[str, str], bool]
) -> None:
    """Assert only lines accepted by ``allow(old, new)`` may differ.

    Lines are aligned with :class:`difflib.SequenceMatcher`.  For each
    non-equal hunk every removed/inserted *pair* must be accepted by
    ``allow``; deletions/insertions are checked separately so a change that
    merely reorders or drops an unrelated line is reported even when the
    multiset of lines happens to survive.
    """
    old_lines = _lines(before)
    new_lines = _lines(after)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)

    failures: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        removed = old_lines[i1:i2]
        inserted = new_lines[j1:j2]

        if tag == "replace":
            if len(removed) != len(inserted):
                failures.append(
                    f"line {i1 + 1}: replaced block changed size "
                    f"({len(removed)} -> {len(inserted)} lines)"
                )
                continue
            for offset, (old_line, new_line) in enumerate(zip(removed, inserted)):
                if old_line != new_line and not allow(old_line, new_line):
                    failures.append(
                        f"line {i1 + 1 + offset}: unexpected change "
                        f"{old_line!r} -> {new_line!r}"
                    )
        else:  # insert / delete touching an unmutated region
            for offset, line in enumerate(removed):
                failures.append(
                    f"line {i1 + 1 + offset}: unexpectedly removed {line!r}"
                )
            for offset, line in enumerate(inserted):
                failures.append(
                    f"after line {i1}: unexpectedly inserted {line!r}"
                )

    assert not failures, (
        "unmodified regions were not byte-preserved:\n"
        + "\n".join(failures)
        + "\n"
        + _format_diff(before, after)
    )


def assert_roundtrip_idempotent(text: str) -> None:
    """dumps(parse(text)) must be a fixpoint: re-parsing changes nothing."""
    again = parse(text).as_string()
    assert again == text, (
        "output is not stable under a second parse/dumps cycle\n"
        + _format_diff(text, again)
    )


# ---------------------------------------------------------------------------
# Small, independently runnable fixtures
# ---------------------------------------------------------------------------


def test_dotted_key_mixed_with_explicit_table_crlf_trailing_comment() -> None:
    # Root-level dotted key *and* explicit headers share the document; the
    # mutated value lives inside the explicitly opened table, whose body also
    # contains a dotted key and a nested sub-table.
    content = (
        "root = 0\r\n"
        "alpha.beta = 1 # dotted header one\r\n"
        "\r\n"
        "[section]\r\n"
        "name = \"old\" # trailing comment must survive\r\n"
        "section.dot = 3 # dotted inside the explicit table\r\n"
        "[section.child]\r\n"
        "deep = 4 # nested comment\r\n"
    )

    doc = parse(content)
    doc["section"]["name"] = "new"
    rendered = doc.as_string()

    expected_line = 'name = "new" # trailing comment must survive\r\n'
    assert expected_line in _lines(rendered)

    def allow(old: str, new: str) -> bool:
        return (
            old == 'name = "old" # trailing comment must survive\r\n'
            and new == expected_line
        )

    assert_unmodified_regions_identical(content, rendered, allow=allow)
    assert_roundtrip_idempotent(rendered)


def test_nested_table_inside_aot_elements_is_boundary_preserved() -> None:
    # Each AoT element owns a nested [products.sku] table; mutating a scalar
    # in the second element must not touch the first element, the blank line
    # between elements, or the final trailing comment.
    content = dedent("""\
        # document lead comment
        [[products]]
        name = "Hammer" # element one comment
        [products.sku]
        id = 1 # sku one

        [[products]]
        name = "Nail" # element two comment
        [products.sku]
        id = 2 # sku two
        # document trail comment
    """)

    doc = parse(content)
    products = doc["products"]
    assert isinstance(products, AoT)
    products[1]["name"] = "Screw"
    products[1]["sku"]["id"] = 42
    rendered = doc.as_string()

    expected = {
        'name = "Screw" # element two comment\n',
        "id = 42 # sku two\n",
    }
    assert expected.issubset(set(_lines(rendered)))

    changed = {
        'name = "Nail" # element two comment\n',
        "id = 2 # sku two\n",
    }

    def allow(old: str, new: str) -> bool:
        if old not in changed:
            return False
        if old.startswith("name"):
            return new == 'name = "Screw" # element two comment\n'
        return new == "id = 42 # sku two\n"

    assert_unmodified_regions_identical(content, rendered, allow=allow)
    assert_roundtrip_idempotent(rendered)


def test_child_declared_before_parent_out_of_order() -> None:
    # [a.b] appears before the [a] header that contributes a's own keys.
    # Mutations in either fragment must preserve fragment order, comments and
    # surrounding unrelated tables.
    content = dedent("""\
        title = "root" # unrelated root key

        [a.b]
        x = 1 # declared first, as a child

        [a]
        y = 2 # parent declared later

        [z]
        w = 3 # unrelated trailing table
    """)

    doc = parse(content)
    doc["a"]["b"]["x"] = 11
    doc["a"]["y"] = 22
    rendered = doc.as_string()

    def allow(old: str, new: str) -> bool:
        if old == "x = 1 # declared first, as a child\n":
            return new == "x = 11 # declared first, as a child\n"
        if old == "y = 2 # parent declared later\n":
            return new == "y = 22 # parent declared later\n"
        return False

    assert_unmodified_regions_identical(content, rendered, allow=allow)
    assert_roundtrip_idempotent(rendered)


def test_delete_then_reinsert_scalar_keeps_surface_slot() -> None:
    # Deleting a value replaces its body slot with a Null tombstone; adding a
    # new key to the same container must not reorder the dotted key relative
    # to the header or swallow the following sibling.
    content = dedent("""\
        alpha.beta = 1 # dotted line
        gamma = 2 # plain sibling
        [delta]
        z = 3 # after header
    """)

    doc = parse(content)
    del doc["gamma"]
    doc["gamma"] = 20
    rendered = doc.as_string()

    expected = dedent("""\
        alpha.beta = 1 # dotted line
        gamma = 20
        [delta]
        z = 3 # after header
    """)
    assert rendered == expected
    assert_roundtrip_idempotent(rendered)


def test_aot_element_deletion_touches_only_that_element() -> None:
    # Removing an AoT element deletes precisely its header/body lines (and the
    # separating blank line consumed into the tombstone); the other element,
    # its nested table and both lead/trail comments must stay byte-identical.
    content = dedent("""\
        # lead
        [[products]]
        name = "Hammer" # keep? no
        [products.sku]
        id = 1

        [[products]]
        name = "Nail" # keep
        [products.sku]
        id = 2
        # trail
    """)

    doc = parse(content)
    del doc["products"][0]
    rendered = doc.as_string()

    expected = dedent("""\
        # lead
        [[products]]
        name = "Nail" # keep
        [products.sku]
        id = 2
        # trail
    """)
    assert rendered == expected
    assert_roundtrip_idempotent(rendered)


def test_reparse_of_mutated_document_keeps_semantics() -> None:
    # Fidelity must not come at the cost of valid TOML: every mutated output
    # parses back to a document with the mutated value.
    content = (
        "a.b = 1 # dot\r\n"
        "[c]\r\n"
        "x = 2 # c\r\n"
    )
    doc = parse(content)
    doc["c"]["x"] = 20
    rendered = doc.as_string()
    reparsed = parse(rendered)
    assert reparsed["a"]["b"] == 1
    assert reparsed["c"]["x"] == 20



# ---------------------------------------------------------------------------
# Boundary matrix
# ---------------------------------------------------------------------------
#
# Each builder emits a *small* structurally equivalent document in one of the
# surface shapes the task calls out, parameterised independently over:
#
# * line ending (LF vs CRLF),
# * trailing comments (present vs absent),
# * blank separator lines between declarations,
# * declaration order (child before parent vs parent before child),
# * the mutation kind (in-place replace, delete + reinsert, AoT element
#   round-trip),
# * the mutated location (root dotted key, explicit table, first/last AoT
#   element's nested table, either out-of-order fragment).


LF = "\n"
CRLF = "\r\n"


def _render(lines: list[str], newline: str, blanks_after: frozenset[int]) -> str:
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line.replace("\n", newline))
        if i in blanks_after:
            out.append(newline)
    return "".join(out)


def _c(enabled: bool, text: str) -> str:
    return text if enabled else ""


def _build_dotted_and_explicit(
    newline: str, comment: bool, blank: bool, *, target: str
) -> tuple[str, str, str]:
    dot_comment = _c(comment, " # dotted scalar")
    tab_comment = _c(comment, " # table scalar")
    lines = [
        f"alpha.beta = 1{dot_comment}\n",
        "[section]\n",
        f'name = "old"{tab_comment}\n',
    ]
    blanks = frozenset({0}) if blank else frozenset()
    content = _render(lines, newline, blanks)
    if target == "dotted":
        old = f"alpha.beta = 1{dot_comment}\n"
        new = f"alpha.beta = 9{dot_comment}\n"
    else:
        old = f'name = "old"{tab_comment}\n'
        new = f'name = "new"{tab_comment}\n'
    return content, old.replace("\n", newline), new.replace("\n", newline)


def _build_nested_aot(
    newline: str, comment: bool, blank: bool, *, element: int
) -> tuple[str, str, str]:
    comment_one = _c(comment, " # element one")
    comment_two = _c(comment, " # element two")
    nested_one = _c(comment, " # nested one")
    nested_two = _c(comment, " # nested two")
    lines = [
        "[[products]]\n",
        f'name = "one"{comment_one}\n',
        "[products.sku]\n",
        f"id = 1{nested_one}\n",
        "[[products]]\n",
        f'name = "two"{comment_two}\n',
        "[products.sku]\n",
        f"id = 2{nested_two}\n",
    ]
    blanks = frozenset({3}) if blank else frozenset()
    content = _render(lines, newline, blanks)
    if element == 0:
        old = f'name = "one"{comment_one}\n'
        new = f'name = "patched"{comment_one}\n'
    else:
        old = f'name = "two"{comment_two}\n'
        new = f'name = "patched"{comment_two}\n'
    return content, old.replace("\n", newline), new.replace("\n", newline)


def _build_out_of_order(
    newline: str, comment: bool, blank: bool, *, child_first: bool, target: str
) -> tuple[str, str, str]:
    child_comment = _c(comment, " # child value")
    parent_comment = _c(comment, " # parent value")
    child_block = ["[a.b]\n", f"x = 1{child_comment}\n"]
    parent_block = ["[a]\n", f"y = 2{parent_comment}\n"]
    first, second = (child_block, parent_block) if child_first else (
        parent_block,
        child_block,
    )
    lines = [
        "title = 0 # root anchor\n",
        *first,
        *second,
        "[unrelated]\n",
        "z = 3 # trailing anchor\n",
    ]
    blanks_idx = frozenset({0, 2, 4}) if blank else frozenset()
    content = _render(lines, newline, blanks_idx)
    if target == "child":
        old = f"x = 1{child_comment}\n"
        new = f"x = 11{child_comment}\n"
    else:
        old = f"y = 2{parent_comment}\n"
        new = f"y = 22{parent_comment}\n"
    return content, old.replace("\n", newline), new.replace("\n", newline)


def _mutate_dotted(doc: tomlkit.TOMLDocument, target: str, kind: str) -> None:
    holder = doc["alpha"] if target == "dotted" else doc["section"]
    key = "beta" if target == "dotted" else "name"
    new_value = 9 if target == "dotted" else "new"
    if kind == "replace":
        holder[key] = new_value
    elif kind == "delete_reinsert":
        del holder[key]
        holder[key] = new_value
    else:
        raise AssertionError(kind)


def _mutate_aot(doc: tomlkit.TOMLDocument, element: int, kind: str) -> None:
    products = doc["products"]
    if kind == "replace":
        products[element]["name"] = "patched"
    elif kind == "element_roundtrip":
        moved = products[element]
        products.remove(moved)
        products.append(moved)
    else:
        raise AssertionError(kind)


def _mutate_out_of_order(doc: tomlkit.TOMLDocument, target: str, kind: str) -> None:
    holder = doc["a"]["b"] if target == "child" else doc["a"]
    key = "x" if target == "child" else "y"
    new_value = 11 if target == "child" else 22
    if kind == "replace":
        holder[key] = new_value
    elif kind == "delete_reinsert":
        del holder[key]
        holder[key] = new_value
    else:
        raise AssertionError(kind)


def _assert_crlf_purity(rendered: str, newline: str) -> None:
    if newline == CRLF:
        assert "\r\n" in rendered
        assert rendered.count("\r\n") == rendered.count("\n"), (
            "CRLF document gained a bare LF line ending"
        )
    else:
        assert "\r" not in rendered


def _assert_unrelated_lines_survive(
    before: str, after: str, changed_lines: frozenset[str]
) -> None:
    """Every line not in ``changed_lines`` must reappear byte-for-byte.

    A delete + reinsert legitimately recreates the target slot with default
    trivia (it is a *new* item), so the reinserted line itself is exempt; the
    assertion still fails when any unrelated line is rewritten, normalised or
    dropped, and reports the first offending line with a unified diff.
    """
    before_lines = _lines(before)
    after_lines = list(_lines(after))
    for line in before_lines:
        if line in changed_lines:
            continue
        assert line in after_lines, (
            f"unrelated line lost or rewritten after mutation: {line!r}\n"
            + _format_diff(before, after)
        )
        after_lines.remove(line)


_MATRIX_BOOLS = (False, True)


@pytest.mark.parametrize(
    "newline,comment,blank,target,kind",
    [
        (newline, comment, blank, target, kind)
        for newline in (LF, CRLF)
        for comment in _MATRIX_BOOLS
        for blank in _MATRIX_BOOLS
        for target in ("dotted", "explicit")
        for kind in ("replace", "delete_reinsert")
    ],
)
def test_matrix_dotted_mixed_with_explicit_table(
    newline: str, comment: bool, blank: bool, target: str, kind: str
) -> None:
    content, old_line, new_line = _build_dotted_and_explicit(
        newline, comment, blank, target=target
    )
    doc = parse(content)
    _mutate_dotted(doc, target, kind)
    rendered = doc.as_string()

    if kind == "replace":
        assert_unmodified_regions_identical(
            content,
            rendered,
            allow=_only(old_line, new_line),
        )
    else:
        _assert_unrelated_lines_survive(content, rendered, frozenset({old_line}))

    if kind == "replace":
        # In-place replacement inherits the old item's trivia verbatim, so a
        # CRLF document must stay CRLF everywhere. A delete + reinsert creates
        # a brand new item with default (LF) trivia; line-ending normalisation
        # for that path is TOMLFile's job, so there we only constrain the
        # surviving lines above.
        _assert_crlf_purity(rendered, newline)
    assert_roundtrip_idempotent(rendered)
    reparsed = parse(rendered)
    if target == "dotted":
        assert reparsed["alpha"]["beta"] == 9
        assert reparsed["section"]["name"] == "old"
    else:
        assert reparsed["alpha"]["beta"] == 1
        assert reparsed["section"]["name"] == "new"


@pytest.mark.parametrize(
    "newline,comment,blank,element,kind",
    [
        (newline, comment, blank, element, kind)
        for newline in (LF, CRLF)
        for comment in _MATRIX_BOOLS
        for blank in _MATRIX_BOOLS
        for element in (0, 1)
        for kind in ("replace", "element_roundtrip")
    ],
)
def test_matrix_nested_table_inside_aot(
    newline: str, comment: bool, blank: bool, element: int, kind: str
) -> None:
    content, old_line, new_line = _build_nested_aot(
        newline, comment, blank, element=element
    )
    doc = parse(content)
    _mutate_aot(doc, element, kind)
    rendered = doc.as_string()

    if kind == "replace":
        assert_unmodified_regions_identical(
            content,
            rendered,
            allow=_only(old_line, new_line),
        )
    else:
        # The element block moves; its four header/body lines are exempt, and
        # the blank separator between elements travels with the moved block.
        moved_values = (
            ('name = "one"', "id = 1")
            if element == 0
            else ('name = "two"', "id = 2")
        )
        exempt = {
            line
            for line in _lines(content)
            if line.startswith(moved_values[0])
            or line.startswith(moved_values[1])
            or line in ("[[products]]\n", "[[products]]\r\n", "[products.sku]\n",
                        "[products.sku]\r\n")
            or line.strip() == ""
        }
        _assert_unrelated_lines_survive(content, rendered, frozenset(exempt))

    _assert_crlf_purity(rendered, newline)
    assert_roundtrip_idempotent(rendered)
    names = [element_table["name"] for element_table in parse(rendered)["products"]]
    if kind == "replace":
        assert names[element] == "patched"
        assert names[1 - element] in ("one", "two")
    else:
        assert set(names) == {"one", "two"}


@pytest.mark.parametrize(
    "newline,comment,blank,child_first,target,kind",
    [
        (newline, comment, blank, child_first, target, kind)
        for newline in (LF, CRLF)
        for comment in _MATRIX_BOOLS
        for blank in _MATRIX_BOOLS
        for child_first in _MATRIX_BOOLS
        for target in ("child", "parent")
        for kind in ("replace",)
    ],
)
def test_matrix_out_of_order_declarations(
    newline: str,
    comment: bool,
    blank: bool,
    child_first: bool,
    target: str,
    kind: str,
) -> None:
    content, old_line, new_line = _build_out_of_order(
        newline, comment, blank, child_first=child_first, target=target
    )
    doc = parse(content)
    _mutate_out_of_order(doc, target, kind)
    rendered = doc.as_string()

    assert_unmodified_regions_identical(
        content,
        rendered,
        allow=_only(old_line, new_line),
    )

    headers_in = re.findall(r"^\[+.+?\]+$", content, flags=re.MULTILINE)
    headers_out = re.findall(r"^\[+.+?\]+$", rendered, flags=re.MULTILINE)
    assert headers_out == headers_in, "out-of-order headers were reordered"
    _assert_crlf_purity(rendered, newline)
    assert_roundtrip_idempotent(rendered)


# ---------------------------------------------------------------------------
# Generative properties
# ---------------------------------------------------------------------------
#
# Rather than hand-writing every combination, the following properties
# deterministically generate structurally-equivalent documents with different
# surface shapes (LF/CRLF, comments, blanks, declaration order, AoT element
# counts and nesting depth). They are intentionally exhaustive over a small
# finite space -- no randomness, no sleeps, no external resources.


def _generated_documents() -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []
    for (
        newline,
        with_comments,
        blanks,
        child_first,
        n_products,
    ) in product(
        (LF, CRLF),
        _MATRIX_BOOLS,
        _MATRIX_BOOLS,
        _MATRIX_BOOLS,
        (1, 2, 3),
    ):
        dot_comment = _c(with_comments, " # generated dotted")
        child_comment = _c(with_comments, " # generated child")
        parent_comment = _c(with_comments, " # generated parent")
        anchor_comment = _c(with_comments, " # generated anchor")

        lines = []
        child_block = [
            "[fruit.apple]\n",
            f'color = "red"{dot_comment}\n',
            f"weight = 1{child_comment}\n",
        ]
        parent_block = ["[fruit]\n", f"fresh = true{parent_comment}\n"]
        first, second = (child_block, parent_block) if child_first else (
            parent_block,
            child_block,
        )
        lines.extend(first)
        for i in range(n_products):
            lines.append("[[products]]\n")
            lines.append(f'name = "v{i}" # generated element {i}\n')
            if i % 2 == 0:
                lines.append("[products.sku]\n")
                lines.append(f"id = {i} # generated nested\n")
        lines.extend(second)
        lines.append("[anchor]\n")
        lines.append(f"keep = \"me\"{anchor_comment}\n")

        blank_indices = frozenset(range(0, len(lines) - 1)) if blanks else frozenset()
        documents.append(
            (
                f"crlf={newline == CRLF},comments={with_comments},"
                f"blanks={blanks},child_first={child_first},n={n_products}",
                _render(lines, newline, blank_indices),
            )
        )
    return documents


_GENERATED_DOCUMENTS = _generated_documents()


def test_generated_documents_are_individually_distinct_and_valid() -> None:
    # Sanity guard: the generator actually produces distinct surface shapes
    # (otherwise the property would vacuously pass), and every shape parses.
    assert len(_GENERATED_DOCUMENTS) == 2 * 2 * 2 * 2 * 3
    surfaces = {surface for _, surface in _GENERATED_DOCUMENTS}
    assert len(surfaces) == len(_GENERATED_DOCUMENTS)
    for _, surface in _GENERATED_DOCUMENTS:
        parsed = parse(surface)
        assert parsed.as_string() == surface, "parsed fixture is not a fixpoint"


def _native_newline(surface: str) -> str:
    return "\r\n" if surface.count("\r\n") == surface.count("\n") else "\n"


def test_property_unmodified_regions_survive_scalar_replace() -> None:
    """Property: for every generated shape, replacing one scalar leaves every
    other surface line byte-identical (comments, blanks, CRLF, declaration
    order and AoT element boundaries included)."""
    for label, surface in _GENERATED_DOCUMENTS:
        doc = parse(surface)
        nl = _native_newline(surface)
        old_line = next(
            line[:-2] if line.endswith("\r\n") else line.rstrip("\n")
            for line in _lines(surface)
            if line.startswith("weight = ")
        )
        new_line = old_line.replace("weight = 1", "weight = 11")
        old_surface, new_surface = old_line + nl, new_line + nl

        doc["fruit"]["apple"]["weight"] = 11
        rendered = doc.as_string()

        assert_unmodified_regions_identical(
            surface, rendered, allow=_only(old_surface, new_surface)
        )
        assert parse(rendered).as_string() == rendered, label
        # declarations keep their exact sequence (no implicit reordering)
        headers = re.findall(r"^\[+.+?\]+$", rendered, flags=re.MULTILINE)
        assert headers == re.findall(
            r"^\[+.+?\]+$", surface, flags=re.MULTILINE
        ), label


def test_property_aot_element_boundaries_survive_inner_mutation() -> None:
    """Property: mutating a scalar inside the nested table of one AoT element
    cannot move or rewrite the headers/body of any other element."""
    for label, surface in _GENERATED_DOCUMENTS:
        doc = parse(surface)
        products = doc["products"]
        target_index = max(i for i in range(len(products)) if "sku" in products[i])
        nl = _native_newline(surface)
        old_surface = f"id = {target_index} # generated nested{nl}"
        new_surface = "id = 999 # generated nested" + nl

        products[target_index]["sku"]["id"] = 999
        rendered = doc.as_string()

        assert_unmodified_regions_identical(
            surface, rendered, allow=_only(old_surface, new_surface)
        )

        def header_positions(text: str) -> list[int]:
            return [
                i
                for i, line in enumerate(_lines(text))
                if line.strip("\r\n") == "[[products]]"
            ]

        assert header_positions(rendered) == header_positions(surface), label
        assert parse(rendered).as_string() == rendered, label
        assert parse(rendered)["products"][target_index]["sku"]["id"] == 999


# ---------------------------------------------------------------------------
# The single most dangerous counterexample
# ---------------------------------------------------------------------------
#
# An AoT whose elements are split across *out-of-order fragments* of the same
# parent table, interleaved with an unrelated table.  The live document keeps
# each fragment where it was declared, while logical access to the key goes
# through OutOfOrderTableProxy, which merges the fragments into one AoT
# without copying or reordering the live element tables.  A regression that
# (a) reverses fragment merge order, (b) copies trivia onto the merged view,
# or (c) reorders the fragments while writing a mutation back silently
# corrupts either semantics or the surface bytes -- or produces TOML that
# fails to round-trip.  test_split_aot_across_out_of_order_fragments pins all
# three failure modes at once.


def test_split_aot_across_out_of_order_fragments() -> None:
    content = dedent("""\
        [hooks]
        setting = 1 # first fragment body

        [[hooks.step]]
        name = "first" # element from the first fragment

        [unrelated]
        marker = "do not move me"

        [[hooks.step]]
        name = "second" # element from a later fragment

        [hooks.state]
        flag = true # last fragment, trailing comment
    """)

    doc = parse(content)
    assert doc.as_string() == content

    hooks = doc["hooks"]
    steps = hooks["step"]
    assert isinstance(steps, AoT)
    assert [step["name"] for step in steps] == ["first", "second"]

    # Mutate through the merged proxy view; the write must land on the live
    # second fragment without relocating anything.
    steps[1]["name"] = "second-patched"
    rendered = doc.as_string()

    expected_line = 'name = "second-patched" # element from a later fragment\n'
    assert expected_line in _lines(rendered)

    assert_unmodified_regions_identical(
        content,
        rendered,
        allow=_only(
            'name = "second" # element from a later fragment\n', expected_line
        ),
    )

    # Fragment headers must keep their original, interleaved positions.
    headers = re.findall(r"^\[+.+?\]+.*$", rendered, flags=re.MULTILINE)
    assert headers == [
        "[hooks]",
        "[[hooks.step]]",
        "[unrelated]",
        "[[hooks.step]]",
        "[hooks.state]",
    ]
    assert_roundtrip_idempotent(rendered)
    reparsed = parse(rendered)
    assert [step["name"] for step in reparsed["hooks"]["step"]] == [
        "first",
        "second-patched",
    ]
    assert reparsed["unrelated"]["marker"] == "do not move me"


# ---------------------------------------------------------------------------
# Whitespace inheritance on value replacement
# ---------------------------------------------------------------------------


def test_replaced_value_inherits_leading_indent_and_trailing_comment() -> None:
    # Leading whitespace before a key belongs to the value's trivia.indent;
    # the trailing comment belongs to trivia.comment. Replacing the value must
    # copy both, in both a deep out-of-order header and an AoT element, and
    # must keep CRLF byte-for-byte outside the value token.
    content = (
        "\r\n"
        "[key1]\r\n"
        "      [key1.key2]\r\n"
        "      value1 = 10 # indented child comment\r\n"
        "      value2 = 30 # untouched indented sibling\r\n"
        "\r\n"
        "[[products]]\r\n"
        "  name = \"old\" # indented aot value\r\n"
        "  other = 1 # untouched aot sibling\r\n"
    )

    doc = parse(content)
    doc["key1"]["key2"]["value1"] = 20
    doc["products"][0]["name"] = "new"
    rendered = doc.as_string()

    expected_lines = [
        "      value1 = 20 # indented child comment\r\n",
        '  name = "new" # indented aot value\r\n',
    ]
    for line in expected_lines:
        assert line in _lines(rendered)

    allowed = {
        "      value1 = 10 # indented child comment\r\n": expected_lines[0],
        '  name = "old" # indented aot value\r\n': expected_lines[1],
    }

    def allow(old: str, new: str) -> bool:
        return allowed.get(old) == new

    assert_unmodified_regions_identical(content, rendered, allow=allow)
    assert rendered.count("\r\n") == rendered.count("\n")
    assert_roundtrip_idempotent(rendered)


# ---------------------------------------------------------------------------
# Out-of-order fragment deletion
# ---------------------------------------------------------------------------


def test_deleting_out_of_order_child_fragment_preserves_everything_else() -> None:
    # [a] is opened first, an unrelated [c] sits in between, and [a.b]
    # reopens `a` out of order. Deleting the child must null out exactly that
    # fragment: the first fragment, its comment, the unrelated table and every
    # CRLF byte must survive in place.
    content = (
        "[a]\r\n"
        "x = 1 # first fragment\r\n"
        "\r\n"
        "[c]\r\n"
        "z = 3 # unrelated table\r\n"
        "\r\n"
        "[a.b]\r\n"
        "y = 2 # second fragment\r\n"
    )

    doc = parse(content)
    del doc["a"]["b"]
    rendered = doc.as_string()

    expected = (
        "[a]\r\n"
        "x = 1 # first fragment\r\n"
        "\r\n"
        "[c]\r\n"
        "z = 3 # unrelated table\r\n"
        "\r\n"
    )
    assert rendered == expected
    assert_roundtrip_idempotent(rendered)
    assert "b" not in parse(rendered)["a"]


def test_deleting_key_from_first_fragment_keeps_fragment_order() -> None:
    # Remove a scalar from the *first* of two out-of-order fragments and then
    # mutate a scalar in the second. Neither fragment may be relocated, and
    # the unrelated interleaved table stays between them.
    content = dedent("""\
        [a]
        x = 1 # first fragment
        keep = "sibling" # must survive

        [c]
        z = 3 # unrelated

        [a.b]
        y = 2 # second fragment
    """)

    doc = parse(content)
    del doc["a"]["x"]
    doc["a"]["b"]["y"] = 22
    rendered = doc.as_string()

    headers = re.findall(r"^\[+.+?\]+$", rendered, flags=re.MULTILINE)
    assert headers == ["[a]", "[c]", "[a.b]"]

    expected = dedent("""\
        [a]
        keep = "sibling" # must survive

        [c]
        z = 3 # unrelated

        [a.b]
        y = 22 # second fragment
    """)
    assert rendered == expected
    assert_roundtrip_idempotent(rendered)


# ---------------------------------------------------------------------------
# Dotted key separator and quote fidelity
# ---------------------------------------------------------------------------


def test_replaced_dotted_value_keeps_separator_and_comment_under_crlf() -> None:
    # A dotted key with a non-standard separator ("\t=\t") and trailing
    # comment: only the value token may change.
    content = "alpha.beta.gamma\t=\t1 # deep dotted\r\nnext = 2 # anchor\r\n"
    doc = parse(content)
    doc["alpha"]["beta"]["gamma"] = 9
    rendered = doc.as_string()

    expected = "alpha.beta.gamma\t=\t9 # deep dotted\r\nnext = 2 # anchor\r\n"
    assert rendered == expected
    assert_roundtrip_idempotent(rendered)
    assert parse(rendered)["alpha"]["beta"]["gamma"] == 9


# ---------------------------------------------------------------------------
# Container/trivia isolation of parsed tables and copies
# ---------------------------------------------------------------------------


def test_parsed_table_copy_is_isolated_from_live_document() -> None:
    # A copy of a parsed table (including a dotted-key super table and a table
    # nested inside an AoT element) must not share its live Container:
    # mutating the copy cannot rewrite or reorder the original document.
    content = dedent("""\
        alpha.beta = 1 # dotted super table
        gamma = 2 # root sibling

        [[products]]
        name = "one" # first element
        [products.sku]
        id = 1 # nested first

        [[products]]
        name = "two" # second element
        [products.sku]
        id = 2 # nested second
    """)
    doc = parse(content)
    snapshot = content

    # Table.__copy__ clones the table's own Container (its top-level body),
    # so writing a new top-level value into the copy cannot rewrite the live
    # document's byte surface. (Nested sub-tables are shared reference types
    # upstream and are intentionally not part of this contract.)
    dotted_copy = doc["alpha"].copy()
    dotted_copy["beta"] = 99
    aot_element_copy = doc["products"][0].copy()
    aot_element_copy["name"] = "mutated"

    # The live document is untouched, byte-for-byte.
    assert doc.as_string() == snapshot
    # The copies themselves hold the mutated values.
    assert dotted_copy["beta"] == 99
    assert aot_element_copy["name"] == "mutated"
    assert_roundtrip_idempotent(doc.as_string())


def test_deleting_out_of_order_fragment_leaves_no_null_ghosts() -> None:
    # After nulling one fragment of an out-of-order table, the surviving
    # fragment must re-index cleanly to a single body position: the index
    # tuple collapses instead of pointing at the Null tombstone, otherwise the
    # merged view leaks a ``None`` and membership/read paths disagree. An
    # unrelated table sits between the two fragments so the indices are not
    # adjacent, which is where an off-by-one in the re-indexing shows.
    content = dedent("""\
        [a]
        x = 1 # surviving fragment

        [c]
        z = 3 # unrelated table between the fragments

        [a.b]
        y = 2 # removed fragment
    """)
    doc = parse(content)
    del doc["a"]["b"]

    assert "b" not in doc["a"]
    assert doc["a"]["x"] == 1
    assert doc.unwrap() == {"a": {"x": 1}, "c": {"z": 3}}
    # json.dumps walks the same unwrap/value paths and fails with a raw
    # TypeError (not an assertion) if a Null tombstone survives re-indexing.
    assert json.dumps(doc.unwrap()) == '{"a": {"x": 1}, "c": {"z": 3}}'

    rendered = doc.as_string()
    assert rendered == dedent("""\
        [a]
        x = 1 # surviving fragment

        [c]
        z = 3 # unrelated table between the fragments

    """)
    assert_roundtrip_idempotent(rendered)


def test_deleting_middle_of_three_out_of_order_fragments() -> None:
    # Same re-indexing invariant with three fragments: the middle one is
    # removed while the first and third keep both their bytes and positions.
    content = dedent("""\
        [a]
        x = 1 # fragment one

        [c]
        z = 3 # unrelated between fragments

        [a.b]
        y = 2 # fragment two (removed)

        [a.d]
        w = 4 # fragment three
    """)
    doc = parse(content)
    del doc["a"]["b"]

    assert "b" not in doc["a"]
    assert doc["a"]["x"] == 1
    assert doc["a"]["d"]["w"] == 4
    assert doc.unwrap() == {"a": {"x": 1, "d": {"w": 4}}, "c": {"z": 3}}

    doc["a"]["d"]["w"] = 40
    rendered = doc.as_string()
    headers = re.findall(r"^\[+.+?\]+$", rendered, flags=re.MULTILINE)
    assert headers == ["[a]", "[c]", "[a.d]"]
    assert "y = 2" not in rendered
    assert "w = 40 # fragment three" in rendered
    assert_roundtrip_idempotent(rendered)
