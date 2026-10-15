"""Users and API keys: who can sign in, and which keys may call the API (admins only)."""

import pandas as pd
import streamlit as st

from auth import accounts, require_admin
from recommender.accounts import MIN_PASSWORD_LENGTH, ROLES, AccountError
from shared import settings

me = require_admin()
store = accounts()

st.title("Users and API keys")
st.caption(
    "Stored in the project database: the users and api_keys tables. Passwords and keys "
    "are kept only as hashes."
)

# --- users ---------------------------------------------------------------------------------
st.subheader("Users")
users = pd.DataFrame(
    store.list_users(), columns=["username", "role", "created_at", "last_login_at"]
)
st.dataframe(
    users.rename(columns={"created_at": "created", "last_login_at": "last sign-in"}),
    hide_index=True,
    width="stretch",
)

add, manage = st.columns(2)
with add, st.form("add_user", clear_on_submit=True):
    st.markdown("**Add a user**")
    username = st.text_input("Username")
    password = st.text_input(
        f"Password (at least {MIN_PASSWORD_LENGTH} characters)", type="password"
    )
    role = st.radio(
        "Role",
        ROLES,
        index=ROLES.index("viewer"),
        horizontal=True,
        help="Admins can also use Try it (which changes the model) and this page.",
    )
    if st.form_submit_button("Add user", type="primary"):
        try:
            created = store.create_user(username, password, role)
            st.toast(f"Added {created.username} ({created.role}).")
            st.rerun()
        except AccountError as error:
            st.error(str(error))

with manage, st.form("manage_user", clear_on_submit=True):
    st.markdown("**Change a password or remove a user**")
    who = st.selectbox("User", users["username"].tolist())
    new_password = st.text_input("New password", type="password")
    change, remove = st.columns(2)
    if change.form_submit_button("Change password"):
        try:
            store.set_password(who, new_password)
            st.success(f"Password changed for {who}.")
        except AccountError as error:
            st.error(str(error))
    if remove.form_submit_button("Remove user"):
        if who == me.username:
            st.error("You can't remove yourself while signed in.")
        else:
            store.delete_user(who)
            st.rerun()

# --- API keys ------------------------------------------------------------------------------
st.subheader("API keys")
if settings().require_api_key:
    st.caption(
        "The API requires a key on every call but /health (RECOMMENDER_REQUIRE_API_KEY "
        "is on). Callers send it in the X-API-Key header."
    )
else:
    st.caption(
        "The API accepts calls without a key. Set RECOMMENDER_REQUIRE_API_KEY=true in "
        ".env and restart it to require one."
    )

keys = pd.DataFrame(store.list_api_keys(), columns=["key_id", "label", "created_at", "active"])
if keys.empty:
    st.write("No API keys yet.")
else:
    st.dataframe(
        keys.rename(columns={"key_id": "key starts with", "created_at": "created"}),
        hide_index=True,
        width="stretch",
    )

new_key, revoke = st.columns(2)
with new_key, st.form("new_key", clear_on_submit=True):
    label = st.text_input("Label", placeholder="e.g. mobile app, dashboard")
    if st.form_submit_button("Create key", type="primary"):
        st.session_state["new_api_key"] = store.create_api_key(label)
if key := st.session_state.pop("new_api_key", None):
    new_key.success("Copy this key now; it is not stored and won't be shown again.")
    new_key.code(key, language=None)

active = keys.loc[keys["active"].astype(bool), "key_id"].tolist() if not keys.empty else []
with revoke, st.form("revoke_key"):
    chosen = st.selectbox("Revoke a key", active, placeholder="No active keys")
    if st.form_submit_button("Revoke", disabled=not active):
        store.revoke_api_key(chosen)
        st.rerun()
