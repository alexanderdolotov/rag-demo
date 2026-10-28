"""
The README against the project: every file it names exists, every `ask_stories.py` command it shows is accepted by the real argument parser,
and every test file is mentioned. Pure checks, no services. (The hard-question tables are checked in test_hard_questions.py.)
"""
import re
import shlex
from pathlib import Path

import pytest

import ask_stories as ask

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
SKIP = {".env"}                                              # git-ignored, so absent in a fresh clone (the README says to create it)


def project_files():
    return [p for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts and "__pycache__" not in p.parts
            and ".pytest_cache" not in p.parts]


def named_files():
    """Backticked names in the README that look like files or folders of this project."""
    for token in re.findall(r"`([^`\n]+)`", README):
        if re.fullmatch(r"[\w./-]+\.(py|md|json|txt|ini)|[\w./-]+/", token) and token not in SKIP and "<" not in token:
            yield token


def test_every_file_and_folder_the_readme_names_exists():
    files = project_files()
    names = {p.name for p in files}
    missing = []
    for token in set(named_files()):
        path = ROOT / token
        if not (path.exists() or (token.endswith("/") is False and token in names) or any(p.name == token.rstrip("/") for p in files)):
            missing.append(token)
    assert not missing, f"the README names files that do not exist: {sorted(missing)}"


def test_every_test_file_is_mentioned_in_the_readme():
    mentioned = set(named_files())
    unmentioned = [p.name for p in (ROOT / "test").glob("test_*.py") if p.name not in mentioned and f"test/{p.name}" not in mentioned]
    assert not unmentioned, f"test files the README never mentions: {unmentioned}"


def ask_commands():
    in_code = False
    for line in README.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        elif in_code and line.startswith("python ask_stories.py"):
            yield shlex.split(line.split("#")[0])[2:]


def test_the_readme_shows_the_main_ways_to_run_the_app():
    shown = [" ".join(args) for args in ask_commands()]
    for must in ("", "--no-llm", "--no-rerank", "--no-rewrite", "--judge", "--debug"):
        assert must in shown, f"no `python ask_stories.py {must}` example in the README"


@pytest.mark.parametrize("args", list(ask_commands()), ids=lambda a: " ".join(a) or "(no arguments)")
def test_every_ask_stories_command_in_the_readme_is_accepted_by_the_real_parser(args):
    ask.make_parser().parse_args(args)                       # an unknown or misspelt flag exits the parser with an error
