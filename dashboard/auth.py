"""Sign-in for the dashboard: accounts live in the ``users`` table of the project database.

The first time the dashboard opens on a database with no users, it asks for the admin
account instead (``python manage.py createsuperuser`` does the same from a terminal).
A sign-in lasts for the browser session; reloading the page asks again.
"""

from __future__ import annotations

import time

import streamlit as st

from recommender.accounts import MIN_PASSWORD_LENGTH, AccountError, Accounts, User
from shared import engine

MAX_ATTEMPTS = 5  # failed sign-ins before this browser session waits
LOCKOUT_S = 30


@st.cache_resource
def accounts() -> Accounts:
    return Accounts(engine())


def current_user() -> User | None:
    return st.session_state.get("user")


def _sign_in(user: User) -> None:
    st.session_state["user"] = user
    st.session_state.pop("failed_sign_ins", None)
    st.rerun()


def _first_admin() -> None:
    st.info("No accounts yet. Create the admin account to start.", icon=":material/person_add:")
    with st.form("first_admin"):
        username = st.text_input("Username", value="admin")
        password = st.text_input(
            f"Password (at least {MIN_PASSWORD_LENGTH} characters)", type="password"
        )
        confirm = st.text_input("Repeat the password", type="password")
        if st.form_submit_button("Create admin and sign in", type="primary"):
            if password != confirm:
                st.error("The passwords differ.")
                return
            # Only possible while the table is empty: re-check, in case someone was quicker.
            if accounts().user_count():
                st.error("An admin was created meanwhile. Sign in instead.")
                return
            try:
                _sign_in(accounts().create_user(username, password, role="admin"))
            except AccountError as error:
                st.error(str(error))


def _sign_in_form() -> None:
    failures = st.session_state.get("failed_sign_ins", [])
    recent = [t for t in failures if time.time() - t < LOCKOUT_S]
    if len(recent) >= MAX_ATTEMPTS:
        st.error(f"Too many failed sign-ins. Try again in {LOCKOUT_S} seconds.")
        return
    with st.form("sign_in"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            user = accounts().authenticate(username, password)
            if user:
                _sign_in(user)
            st.session_state["failed_sign_ins"] = recent = [*recent, time.time()]
            if len(recent) >= MAX_ATTEMPTS:
                st.error(f"Too many failed sign-ins. Try again in {LOCKOUT_S} seconds.")
            else:
                st.error("Wrong username or password.")


def login_gate() -> User:
    """The signed-in user. Until someone signs in, shows the sign-in page and stops."""
    user = current_user()
    if user is not None:
        return user
    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.title("Telco Dynamic Recommender")
        st.caption("Real-time offer recommendations that learn from every response.")
        if accounts().user_count() == 0:
            _first_admin()
        else:
            _sign_in_form()
    st.stop()


def sidebar_account(user: User) -> None:
    with st.sidebar:
        st.caption(f"Signed in as **{user.username}** · {user.role}")
        if st.button("Sign out", icon=":material/logout:"):
            st.session_state.pop("user", None)
            st.rerun()


def require_admin() -> User:
    """Stop the page unless an admin is signed in (pages are also hidden from viewers)."""
    user = current_user()
    if user is None or not user.is_admin:
        st.error("Only admins can use this page.")
        st.stop()
    return user
