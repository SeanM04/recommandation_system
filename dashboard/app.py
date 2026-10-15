"""Telco Dynamic Recommender dashboard.

Run with:  python manage.py runserver   (the API and this dashboard together)
      or:  make dashboard               (streamlit run dashboard/app.py)
Everyone signs in first; accounts live in the project database. Viewers see the results
and the live log; admins can also use Try it (which changes the model) and manage users.
"""

import streamlit as st

from auth import login_gate, sidebar_account

st.set_page_config(
    page_title="Telco Dynamic Recommender",
    page_icon=":material/insights:",
    layout="wide",
)

user = login_gate()

pages = [
    st.Page("views/live.py", title="Live view", icon=":material/sensors:", default=True),
    st.Page("views/comparison.py", title="Algorithm comparison", icon=":material/leaderboard:"),
    st.Page("views/beliefs.py", title="Explore vs exploit", icon=":material/stacked_line_chart:"),
    st.Page("views/shift.py", title="Behaviour shift", icon=":material/swap_horiz:"),
]
if user.is_admin:
    pages += [
        st.Page("views/try_it.py", title="Try it", icon=":material/touch_app:"),
        st.Page("views/users.py", title="Users and API keys", icon=":material/manage_accounts:"),
    ]
navigation = st.navigation(pages)
sidebar_account(user)
navigation.run()
