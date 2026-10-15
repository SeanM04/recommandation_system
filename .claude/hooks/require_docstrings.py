"""Claude Code hook: every Python function Claude writes must have a docstring.

Runs after Write, Edit and MultiEdit (a PostToolUse hook, see .claude/settings.json). It
checks the functions whose ``def`` line Claude has just written: every function for a
Write, the functions inside the new text for an Edit or MultiEdit. Older functions
elsewhere in the file are left alone. Any without a docstring are reported back to
Claude (exit code 2), which then adds them before carrying on.

Exempt: functions named ``test_*``, whose names already say what they check.
Standard library only, so any Python 3.8+ on the PATH can run it.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

EXEMPT_PREFIXES = ("test_",)  # set to () to require docstrings on tests too


def written_lines(tool: str, tool_input: dict, source: str) -> list[tuple[int, int]] | None:
    """The line ranges Claude wrote (1-based, inclusive); None means the whole file."""
    if tool == "Write":
        return None
    ranges = []
    for edit in tool_input.get("edits") or [tool_input]:  # MultiEdit has a list
        new = edit.get("new_string") or ""
        if "def " not in new:
            continue
        start = source.find(new)
        while start != -1:  # every place the new text now appears; a few too many is fine
            first = source.count("\n", 0, start) + 1
            ranges.append((first, first + new.count("\n")))
            start = source.find(new, start + len(new))
    return ranges


def undocumented(tree: ast.AST, ranges: list[tuple[int, int]] | None) -> list[tuple[int, str]]:
    """(line, qualified name) of each function in ``ranges`` that lacks a docstring."""
    missing = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = prefix + getattr(child, "name", "")
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                written = ranges is None or any(a <= child.lineno <= b for a, b in ranges)
                exempt = child.name.startswith(EXEMPT_PREFIXES)
                if written and not exempt and not ast.get_docstring(child):
                    missing.append((child.lineno, name))
                visit(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return missing


def main() -> int:
    """Read the hook event from stdin; exit 2 with a message if docstrings are missing."""
    try:
        event = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return 0
    tool, tool_input = event.get("tool_name", ""), event.get("tool_input") or {}
    path = Path(tool_input.get("file_path") or "")
    if path.suffix != ".py" or not path.is_file():
        return 0
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return 0  # not this hook's business: the linter and tests report those
    missing = undocumented(tree, written_lines(tool, tool_input, source))
    if not missing:
        return 0

    root = os.environ.get("CLAUDE_PROJECT_DIR")
    try:
        shown = path.resolve().relative_to(Path(root).resolve()) if root else path
    except ValueError:
        shown = path
    lines = "\n".join(f"  line {line}: def {name}" for line, name in missing)
    sys.stderr.reconfigure(encoding="utf-8")
    print(
        f"Docstrings missing in {shown} (every function you write needs one):\n{lines}\n"
        "Add a docstring to each: one line on what it does, and why when that isn't "
        "obvious from the code.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
