"""Telco Dynamic Recommender dashboard.

Run with:  make dashboard   (streamlit run dashboard/app.py)
Run make simulate first: it produces the results and live events these pages show.
"""

import streamlit as st

st.set_page_config(
    page_title="Telco Dynamic Recommender",
    page_icon=":material/insights:",
    layout="wide",
)

navigation = st.navigation(
    [
        st.Page("views/live.py", title="Live view", icon=":material/sensors:", default=True),
        st.Page("views/comparison.py", title="Algorithm comparison", icon=":material/leaderboard:"),
        st.Page(
            "views/beliefs.py", title="Explore vs exploit", icon=":material/stacked_line_chart:"
        ),
        st.Page("views/shift.py", title="Behaviour shift", icon=":material/swap_horiz:"),
        st.Page("views/try_it.py", title="Try it", icon=":material/touch_app:"),
    ]
)
navigation.run()
