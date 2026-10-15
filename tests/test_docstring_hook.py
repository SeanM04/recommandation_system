"""The Claude Code hook in .claude/hooks that requires a docstring on every new function."""

import json
import subprocess
import sys

from recommender.config import PROJECT_ROOT

HOOK = PROJECT_ROOT / ".claude" / "hooks" / "require_docstrings.py"
SOURCE = '''def old_helper(x):
    return x + 1


class Store:
    def get(self, key):
        """Fetch one value."""
        return key


def new_function(a):
    def inner():
        return a

    return inner()


def test_adds_up():
    assert new_function(1) == 1
'''
NEW_TEXT = "def new_function(a):\n    def inner():\n        return a"


def run_hook(tool: str, tool_input: dict) -> subprocess.CompletedProcess:
    """Send one hook event to the script, the way Claude Code does, and return the result."""
    event = json.dumps({"tool_name": tool, "tool_input": tool_input})
    return subprocess.run(
        [sys.executable, str(HOOK)], input=event, capture_output=True, text=True, check=False
    )


def test_a_written_file_must_document_every_function_but_tests(tmp_path) -> None:
    path = tmp_path / "sample.py"
    path.write_text(SOURCE, encoding="utf-8")
    result = run_hook("Write", {"file_path": str(path)})
    assert result.returncode == 2
    assert "def old_helper" in result.stderr and "def new_function.inner" in result.stderr
    assert "Store.get" not in result.stderr and "test_adds_up" not in result.stderr


def test_an_edit_is_checked_only_where_it_wrote_functions(tmp_path) -> None:
    path = tmp_path / "sample.py"
    path.write_text(SOURCE, encoding="utf-8")
    result = run_hook("Edit", {"file_path": str(path), "new_string": NEW_TEXT})
    assert result.returncode == 2
    assert "def new_function" in result.stderr and "old_helper" not in result.stderr
    body_only = run_hook("Edit", {"file_path": str(path), "new_string": "    return x + 1"})
    assert body_only.returncode == 0


def test_documented_code_and_other_files_pass(tmp_path) -> None:
    path = tmp_path / "fine.py"
    path.write_text('def double(x):\n    """Twice x."""\n    return 2 * x\n', encoding="utf-8")
    assert run_hook("Write", {"file_path": str(path)}).returncode == 0
    assert run_hook("Write", {"file_path": str(tmp_path / "notes.md")}).returncode == 0
    broken = tmp_path / "broken.py"
    broken.write_text("def oops(:\n", encoding="utf-8")
    assert run_hook("Write", {"file_path": str(broken)}).returncode == 0  # the linter's job
