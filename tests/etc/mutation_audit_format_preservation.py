"""Standalone mutation audit for the format-preservation tests.

Run from the repository root::

    python3 tests/etc/mutation_audit_format_preservation.py

For every fixed mutation in the catalogue below this script patches the
product source in place, runs the targeted test file and the pre-existing
suite separately, prints how many tests fail in each group, and then
restores the file. Exit code is non-zero if any mutation is *not* caught by
the new tests or the working tree fails to restore.

It is deterministic: no randomness, no sleeps, no network access, no
absolute paths, and mutations are located by source anchors rather than by
fixture or test names.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER = REPO_ROOT / "tomlkit" / "container.py"
ITEMS = REPO_ROOT / "tomlkit" / "items.py"

NEW_TESTS = ["tests/test_format_preservation.py"]

MUTATIONS: list[tuple[str, Path, str, str]] = [
    (
        "M1 drop trailing-comment/comment_ws inheritance on replacement",
        CONTAINER,
        """            if not isinstance(value, (Whitespace, AoT)):
                value.trivia.indent = v.trivia.indent
                value.trivia.comment_ws = value.trivia.comment_ws or v.trivia.comment_ws
                value.trivia.comment = value.trivia.comment or v.trivia.comment
                value.trivia.trail = v.trivia.trail""",
        """            if not isinstance(value, (Whitespace, AoT)):
                value.trivia.indent = v.trivia.indent
                value.trivia.trail = v.trivia.trail""",
    ),
    (
        "M2 reverse split-AoT fragment merge order",
        CONTAINER,
        "merged = AoT([*existing.body, *item.body], parsed=True)",
        "merged = AoT([*item.body, *existing.body], parsed=True)",
    ),
    (
        "M3 drop leading-indent inheritance on replacement",
        CONTAINER,
        """            if not isinstance(value, (Whitespace, AoT)):
                value.trivia.indent = v.trivia.indent
                value.trivia.comment_ws = value.trivia.comment_ws or v.trivia.comment_ws
                value.trivia.comment = value.trivia.comment or v.trivia.comment
                value.trivia.trail = v.trivia.trail""",
        """            if not isinstance(value, (Whitespace, AoT)):
                value.trivia.comment_ws = value.trivia.comment_ws or v.trivia.comment_ws
                value.trivia.comment = value.trivia.comment or v.trivia.comment
                value.trivia.trail = v.trivia.trail""",
    ),
    (
        "M4 null the wrong fragment index in out-of-order _remove_at",
        CONTAINER,
        """        if isinstance(index, tuple):
            index_list = list(index)
            index_list.remove(idx)""",
        """        if isinstance(index, tuple):
            index_list = list(index)
            index_list.pop(0)""",
    ),
    (
        "M5 Table copy shares its live Container",
        ITEMS,
        """    def __copy__(self) -> Table:
        return type(self)(
            self._value.copy(),""",
        """    def __copy__(self) -> Table:
        return type(self)(
            self._value,""",
    ),
]


def _run_pytest(args: list[str]) -> tuple[int, int, int]:
    import re
    import shutil

    commands = [[sys.executable, "-m", "pytest", "-q", *args]]
    pytest_bin = shutil.which("pytest")
    if pytest_bin:
        commands.append([pytest_bin, "-q", *args])
    output = ""
    returncode = 1
    for command in commands:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        output = proc.stdout + proc.stderr
        returncode = proc.returncode
        # An interpreter/plugin that dies before collection yields no pytest
        # summary at all; in that case fall back to another interpreter.
        if re.search(r"\d+ (passed|failed|error)", output):
            break

    failed = passed = 0
    m = re.search(r"(\d+) failed", output)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))
    return returncode, failed, passed


def main() -> int:
    originals = {path: path.read_text() for path in {CONTAINER, ITEMS}}
    exit_code = 0
    try:
        for name, path, anchor, replacement in MUTATIONS:
            text = originals[path]
            if anchor not in text:
                print(f"!! could not locate mutation anchor for {name}")
                exit_code = 2
                continue
            path.write_text(text.replace(anchor, replacement))

            _, new_failed, _ = _run_pytest(NEW_TESTS)
            _, old_failed, old_passed = _run_pytest(
                ["--ignore=tests/test_format_preservation.py"]
            )
            caught = new_failed > 0
            if not caught:
                exit_code = 1
            print(
                f"[{'CAUGHT' if caught else 'MISSED'}] {name}: "
                f"new_tests_failed={new_failed}, "
                f"old_suite_failed={old_failed} (old_suite_passed={old_passed})"
            )

            path.write_text(originals[path])
    finally:
        for path, text in originals.items():
            path.write_text(text)

    rc, failed, passed = _run_pytest([])
    print(f"restored full suite: rc={rc}, failed={failed}, passed={passed}")
    if rc != 0:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
