"""Accounts: who may sign in to the dashboard, and which API keys may call the API.

Passwords are stored as salted scrypt hashes (Python's standard library, no extra
dependency) and checked in constant time. API keys are random 32-byte tokens shown once
at creation; only their SHA-256 hash is stored, which is safe for random keys of that
length. Both tables live in the same database as everything else.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, Engine, String, Table, func, select, update
from sqlalchemy.exc import IntegrityError

from recommender.db import metadata

ROLES = ("admin", "viewer")  # admins may change the model and manage users
MIN_PASSWORD_LENGTH = 8
USERNAME = re.compile(r"^[a-z0-9_.-]{3,64}$")
API_KEY_PREFIX = "rk_"

users = Table(
    "users",
    metadata,
    Column("username", String(64), primary_key=True),
    Column("password_hash", String(256), nullable=False),
    Column("role", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_login_at", DateTime(timezone=True)),
)

api_keys = Table(
    "api_keys",
    metadata,
    Column("key_id", String(16), primary_key=True),  # the key's first characters, to name it
    Column("key_hash", String(64), nullable=False, unique=True),
    Column("label", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("active", Boolean, nullable=False, default=True),
)

# scrypt cost: 16 MB of memory and ~50 ms per check, which makes guessing slow.
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


class AccountError(ValueError):
    """An invalid username, password or role, or a username that is taken."""


@dataclass(frozen=True)
class User:
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        """Whether this user may change the model and manage accounts."""
        return self.role == "admin"


def hash_password(password: str) -> str:
    """A salted scrypt hash of ``password``, with the scrypt settings stored alongside."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    encoded = (base64.b64encode(part).decode() for part in (salt, digest))
    return "scrypt${}${}${}${}${}".format(*_SCRYPT.values(), *encoded)


def verify_password(password: str, stored: str) -> bool:
    """Whether ``password`` matches the stored hash, compared in constant time."""
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    candidate = hashlib.scrypt(
        password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p)
    )
    return hmac.compare_digest(candidate, base64.b64decode(digest))


def _hash_key(key: str) -> str:
    """The SHA-256 of an API key: enough for long random keys, unlike passwords."""
    return hashlib.sha256(key.encode()).hexdigest()


class Accounts:
    """Users and API keys, in the ``users`` and ``api_keys`` tables."""

    def __init__(self, engine: Engine) -> None:
        """Accounts in ``engine``'s database, creating the two tables if missing."""
        self.engine = engine
        for table in (users, api_keys):
            table.create(engine, checkfirst=True)

    # --- users ----------------------------------------------------------------------

    def user_count(self) -> int:
        """How many accounts exist (0 means the first admin is still to be created)."""
        with self.engine.connect() as connection:
            return connection.execute(select(func.count()).select_from(users)).scalar_one()

    def create_user(self, username: str, password: str, role: str = "viewer") -> User:
        """Add an account after checking the username, password length and role."""
        username = username.strip().lower()
        if not USERNAME.match(username):
            raise AccountError("username: 3-64 characters, lowercase letters, digits, . _ -")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AccountError(f"password: at least {MIN_PASSWORD_LENGTH} characters")
        if role not in ROLES:
            raise AccountError(f"role: one of {', '.join(ROLES)}")
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    users.insert().values(
                        username=username,
                        password_hash=hash_password(password),
                        role=role,
                        created_at=datetime.now(UTC),
                    )
                )
        except IntegrityError:
            raise AccountError(f"username {username!r} is taken") from None
        return User(username, role)

    def authenticate(self, username: str, password: str) -> User | None:
        """The user, if the password matches; records the time of the sign-in."""
        username = username.strip().lower()
        with self.engine.connect() as connection:
            row = connection.execute(
                select(users.c.password_hash, users.c.role).where(users.c.username == username)
            ).first()
        if row is None:
            hash_password(password)  # same work as a real check, so timing reveals nothing
            return None
        if not verify_password(password, row.password_hash):
            return None
        with self.engine.begin() as connection:
            connection.execute(
                update(users)
                .where(users.c.username == username)
                .values(last_login_at=datetime.now(UTC))
            )
        return User(username, row.role)

    def set_password(self, username: str, password: str) -> None:
        """Give an existing account a new password."""
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AccountError(f"password: at least {MIN_PASSWORD_LENGTH} characters")
        with self.engine.begin() as connection:
            changed = connection.execute(
                update(users)
                .where(users.c.username == username.strip().lower())
                .values(password_hash=hash_password(password))
            ).rowcount
        if not changed:
            raise AccountError(f"no user {username!r}")

    def delete_user(self, username: str) -> None:
        """Remove an account (nothing happens if it doesn't exist)."""
        with self.engine.begin() as connection:
            connection.execute(users.delete().where(users.c.username == username))

    def list_users(self) -> list[dict]:
        """Every account with its role, creation time and last sign-in; never a password."""
        query = select(users.c.username, users.c.role, users.c.created_at, users.c.last_login_at)
        with self.engine.connect() as connection:
            return [dict(row._mapping) for row in connection.execute(query.order_by("username"))]

    # --- API keys -------------------------------------------------------------------

    def create_api_key(self, label: str) -> str:
        """A new key. It is returned once and never stored, so show it to the user now."""
        key = API_KEY_PREFIX + secrets.token_urlsafe(32)
        with self.engine.begin() as connection:
            connection.execute(
                api_keys.insert().values(
                    key_id=key[:11],
                    key_hash=_hash_key(key),
                    label=label.strip() or "unnamed",
                    created_at=datetime.now(UTC),
                    active=True,
                )
            )
        return key

    def api_key_valid(self, key: str | None) -> bool:
        """Whether ``key`` is an active key: looked up by its hash, never stored in clear."""
        if not key or not key.startswith(API_KEY_PREFIX):
            return False
        query = select(api_keys.c.key_id).where(
            api_keys.c.key_hash == _hash_key(key), api_keys.c.active.is_(True)
        )
        with self.engine.connect() as connection:
            return connection.execute(query).first() is not None

    def revoke_api_key(self, key_id: str) -> None:
        """Switch off one key, by the first characters that name it."""
        with self.engine.begin() as connection:
            connection.execute(
                update(api_keys).where(api_keys.c.key_id == key_id).values(active=False)
            )

    def list_api_keys(self) -> list[dict]:
        """Every key's name, label, creation time and status; never the key itself."""
        query = select(
            api_keys.c.key_id, api_keys.c.label, api_keys.c.created_at, api_keys.c.active
        )
        with self.engine.connect() as connection:
            return [dict(row._mapping) for row in connection.execute(query.order_by("created_at"))]
