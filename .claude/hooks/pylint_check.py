"""Claude Code hook: pylint checks every Python file Claude writes.

Runs after Write, Edit and MultiEdit, in the background (``asyncRewake`` in
.claude/settings.json): Claude carries on while pylint works, and is woken up only when there
is something to fix (exit code 2). It reports

* errors anywhere in the file, because an edit can break code it didn't touch, and
* every other message on the lines Claude just wrote: all of them for a Write, the new text
  for an Edit or MultiEdit. A one-line fix doesn't turn into a clean-up of the whole file.

Pylint needs the project's environment to resolve imports, so this script (standard library
only, like the docstring hook) runs it with a Python that has it installed: $PYLINT_PYTHON,
the project's .venv, or ~/.venvs/telco-recommender. The configuration is [tool.pylint] in
pyproject.toml. Files outside the project are skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
VENVS = (PROJECT / ".venv", Path.home() / ".venvs" / "telco-recommender")
# Pytest hands fixtures to tests as arguments named like module-level functions, and some are
# requested only for their side effects; pylint reads both as mistakes. `make pylint` agrees.
TEST_DISABLE = "redefined-outer-name,unused-argument"
SHOWN = 20  # messages listed back to Claude; the rest are counted


def pylint_python() -> str | None:
    """A Python interpreter with pylint installed, or None if there is none to be found."""
    if override := os.environ.get("PYLINT_PYTHON"):
        return override
    layouts = (
        ("Scripts/python.exe", "Lib/site-packages/pylint"),
        ("bin/python", "lib/python3*/site-packages/pylint"),
    )
    for venv in VENVS:
        for python, package in layouts:
            if (venv / python).is_file() and any(venv.glob(package)):
                return str(venv / python)
    return None


def written_lines(tool: str, tool_input: dict, source: str) -> list[tuple[int, int]] | None:
    """The line ranges Claude wrote (1-based, inclusive); None means the whole file."""
    if tool == "Write":
        return None
    ranges = []
    for edit in tool_input.get("edits") or [tool_input]:  # MultiEdit has a list
        new = edit.get("new_string") or ""
        if not new.strip():
            continue  # a deletion: only errors can come of it, and those are always reported
        start = source.find(new)
        while start != -1:  # every place the new text now appears; a few too many is fine
            first = source.count("\n", 0, start) + 1
            ranges.append((first, first + new.count("\n")))
            start = source.find(new, start + len(new))
    return ranges


def run_pylint(python: str, path: Path) -> list[dict]:
    """Pylint's messages for one file, as the dictionaries of its JSON output."""
    command = [
        python,
        "-m",
        "pylint",
        str(path),
        f"--rcfile={PROJECT / 'pyproject.toml'}",
        "--output-format=json",
        "--score=n",
        "--persistent=n",
    ]
    if path.name.startswith("test_") or path.name == "conftest.py":
        command.append(f"--disable={TEST_DISABLE}")
    result = subprocess.run(
        command,
        cwd=PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    try:
        return json.loads(result.stdout or "[]")
    except ValueError:  # pylint itself failed: a usage error or a crash, not a finding
        raise RuntimeError(result.stderr.strip() or result.stdout.strip()) from None


def relevant(messages: list[dict], ranges: list[tuple[int, int]] | None) -> list[dict]:
    """Errors anywhere in the file; every other message only on the lines Claude wrote."""
    return [
        message
        for message in messages
        if message["type"] in ("error", "fatal")
        or ranges is None
        or any(first <= message["line"] <= last for first, last in ranges)
    ]


def main() -> int:
    """Read the hook event from stdin; exit 2 with pylint's findings for Claude to fix."""
    try:
        event = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return 0
    tool, tool_input = event.get("tool_name", ""), event.get("tool_input") or {}
    path = Path(tool_input.get("file_path") or "")
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or PROJECT).resolve()
    if path.suffix != ".py" or not path.is_file() or not path.resolve().is_relative_to(root):
        return 0
    sys.stderr.reconfigure(encoding="utf-8")

    python = pylint_python()
    if python is None:
        print(
            "pylint hook: no Python with pylint installed. Run `pip install -r requirements.txt` "
            "in .venv or ~/.venvs/telco-recommender, or set PYLINT_PYTHON.",
            file=sys.stderr,
        )
        return 1  # shown to the user; doesn't hold Claude up
    try:
        messages = run_pylint(python, path)
    except (OSError, RuntimeError) as error:
        print(f"pylint hook: pylint could not run: {error}", file=sys.stderr)
        return 1
    source = path.read_text(encoding="utf-8", errors="replace")
    found = relevant(messages, written_lines(tool, tool_input, source))
    if not found:
        return 0

    shown = path.resolve().relative_to(root)
    lines = "\n".join(
        f"  line {m['line']}: {m['symbol']} ({m['message-id']}) {m['message']}"
        for m in found[:SHOWN]
    )
    more = f"\n  ... and {len(found) - SHOWN} more" if len(found) > SHOWN else ""
    print(
        f"pylint found {len(found)} problem(s) in {shown} (errors anywhere in the file, other "
        f"messages on the lines you wrote):\n{lines}{more}\n"
        "Fix each one. If a message is wrong for this code, silence it on that line with "
        "`# pylint: disable=<symbol>` and a short reason.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
