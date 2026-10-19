import pytest

from conftest import postgres_schema
from recommender.accounts import AccountError, Accounts, User, hash_password, verify_password
from recommender.db import make_engine


@pytest.fixture(params=["sqlite", "postgres"])
def accounts(request, tmp_path):
    """Accounts on a throwaway SQLite file, and on Postgres when configured."""
    if request.param == "sqlite":
        engine = make_engine(f"sqlite:///{(tmp_path / 'accounts.db').as_posix()}")
        yield Accounts(engine)
        engine.dispose()
    else:
        with postgres_schema("pytest_accounts") as url:
            engine = make_engine(url)
            yield Accounts(engine)
            engine.dispose()


def test_passwords_are_salted_hashes_checked_exactly() -> None:
    first, second = hash_password("correct horse"), hash_password("correct horse")
    assert first != second  # a fresh salt every time
    assert "correct horse" not in first
    assert verify_password("correct horse", first)
    assert not verify_password("correct horsE", first)
    assert not verify_password("correct horse", "not-a-hash")


def test_sign_in_with_the_right_password_only(accounts: Accounts) -> None:
    accounts.create_user("Thandi", "s3cret-pass", role="admin")
    assert accounts.authenticate("thandi", "s3cret-pass") == User("thandi", "admin")
    assert accounts.authenticate(" THANDI ", "s3cret-pass") == User("thandi", "admin")
    assert accounts.authenticate("thandi", "wrong-pass") is None
    assert accounts.authenticate("nobody", "s3cret-pass") is None
    [row] = accounts.list_users()
    assert row["last_login_at"] is not None


@pytest.mark.parametrize(
    ("username", "password", "role", "message"),
    [
        ("ab", "long-enough", "viewer", "username"),
        ("has space", "long-enough", "viewer", "username"),
        ("sipho", "short", "viewer", "password"),
        ("sipho", "long-enough", "owner", "role"),
    ],
)
def test_invalid_accounts_are_refused(accounts, username, password, role, message) -> None:
    with pytest.raises(AccountError, match=message):
        accounts.create_user(username, password, role)
    assert accounts.user_count() == 0


def test_usernames_are_unique(accounts: Accounts) -> None:
    accounts.create_user("lerato", "first-password")
    with pytest.raises(AccountError, match="taken"):
        accounts.create_user("Lerato", "other-password")


def test_change_password_and_remove(accounts: Accounts) -> None:
    accounts.create_user("lerato", "first-password")
    accounts.set_password("lerato", "second-password")
    assert accounts.authenticate("lerato", "first-password") is None
    assert accounts.authenticate("lerato", "second-password") == User("lerato", "viewer")
    with pytest.raises(AccountError, match="no user"):
        accounts.set_password("nobody", "whatever-123")
    accounts.delete_user("lerato")
    assert accounts.user_count() == 0


def test_api_keys_are_stored_hashed_and_can_be_revoked(accounts: Accounts) -> None:
    key = accounts.create_api_key("mobile app")
    assert key.startswith("rk_") and len(key) > 40
    assert accounts.api_key_valid(key)
    assert not accounts.api_key_valid(key + "x")
    assert not accounts.api_key_valid(None)
    [row] = accounts.list_api_keys()
    assert row["label"] == "mobile app" and row["active"]
    assert key not in str(row)  # only the first characters, to name it
    accounts.revoke_api_key(row["key_id"])
    assert not accounts.api_key_valid(key)
