"""python manage.py ..., against a throwaway SQLite database."""

import socket

import pytest

from recommender import cli
from recommender.accounts import Accounts
from recommender.db import make_engine


@pytest.fixture
def database(tmp_path, monkeypatch) -> str:
    url = f"sqlite:///{(tmp_path / 'manage.db').as_posix()}"
    monkeypatch.setenv("RECOMMENDER_DATABASE_URL", url)
    return url


def test_migrate_creates_the_tables_and_loads_the_customers(database, capsys) -> None:
    cli.main(["migrate"])
    out = capsys.readouterr().out
    assert "users" in out and "options" in out and "invocations" in out
    assert "Customers: 5,000" in out
    assert "createsuperuser" in out  # no accounts yet
    cli.main(["migrate"])  # safe to rerun
    assert "Customers: 5,000" in capsys.readouterr().out


def test_createsuperuser_without_prompts(database, monkeypatch, capsys) -> None:
    monkeypatch.setenv("RECOMMENDER_SUPERUSER_PASSWORD", "a-long-password")
    cli.main(["createsuperuser", "--noinput"])
    assert "Created admin 'admin'" in capsys.readouterr().out
    user = Accounts(make_engine(database)).authenticate("admin", "a-long-password")
    assert user is not None and user.is_admin


def test_createuser_asks_for_the_password_twice(database, monkeypatch, capsys) -> None:
    answers = iter(["viewer-password", "viewer-password", "one", "two"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    cli.main(["createuser", "naledi"])
    assert "Created viewer 'naledi'" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="differ"):
        cli.main(["createuser", "bongani"])
    cli.main(["users"])
    assert "naledi" in capsys.readouterr().out


def test_createapikey_prints_a_working_key(database, capsys) -> None:
    cli.main(["createapikey", "--label", "mobile app"])
    key = next(word for word in capsys.readouterr().out.split() if word.startswith("rk_"))
    assert Accounts(make_engine(database)).api_key_valid(key)


def test_runserver_refuses_a_port_in_use(database) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        with pytest.raises(SystemExit, match=f"Port {port} is already in use"):
            cli.main(["runserver", "--api-port", str(port), "--no-dashboard"])
