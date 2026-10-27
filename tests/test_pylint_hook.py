"""The Claude Code hook in .claude/hooks that runs pylint on every Python file Claude writes."""

import json
import os
import subprocess
import sys

from recommender.config import PROJECT_ROOT

HOOK = PROJECT_ROOT / ".claude" / "hooks" / "pylint_check.py"
SOURCE = '''"""A module with one old problem and one new function."""


def old_total(values):
    """Sum the values."""
    leftover = 3
    return sum(values)


def double(x):
    """Twice x."""
    return 2 * x
'''


def run_hook(tool: str, tool_input: dict, project) -> subprocess.CompletedProcess:
    """Send one hook event to the script the way Claude Code does, with ``project`` as the
    project directory and this interpreter's pylint."""
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project), "PYLINT_PYTHON": sys.executable}
    event = json.dumps({"tool_name": tool, "tool_input": tool_input})
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=event,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_a_written_file_reports_every_problem(tmp_path) -> None:
    path = tmp_path / "sample.py"
    path.write_text(SOURCE, encoding="utf-8")
    result = run_hook("Write", {"file_path": str(path)}, tmp_path)
    assert result.returncode == 2
    assert "line 6: unused-variable" in result.stderr


def test_an_edit_reports_its_own_lines_and_errors_anywhere(tmp_path) -> None:
    path = tmp_path / "sample.py"
    path.write_text(SOURCE, encoding="utf-8")
    new_text = 'def double(x):\n    """Twice x."""\n    return 2 * x'
    assert (
        run_hook("Edit", {"file_path": str(path), "new_string": new_text}, tmp_path).returncode == 0
    )
    path.write_text(SOURCE + "\n\nprint(undefined_name)\n", encoding="utf-8")
    result = run_hook("Edit", {"file_path": str(path), "new_string": new_text}, tmp_path)
    assert result.returncode == 2
    assert "undefined-variable" in result.stderr and "unused-variable" not in result.stderr


def test_files_outside_the_project_and_other_files_are_skipped(tmp_path) -> None:
    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    outside = elsewhere / "sample.py"
    outside.write_text(SOURCE, encoding="utf-8")
    assert run_hook("Write", {"file_path": str(outside)}, project).returncode == 0
    assert run_hook("Write", {"file_path": str(project / "notes.md")}, project).returncode == 0
