"""Research results & project status page of Indic Entity Lab."""

import os
import sys

import streamlit as st

# Streamlit runs pages with the main script's folder on sys.path, but add it
# explicitly so the page also works when run on its own.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_style import APP_CSS  # noqa: E402
import research_dashboard  # noqa: E402

st.set_page_config(
    page_title="Research Results · Indic Entity Lab",
    page_icon="📊",
    layout="wide",
)

st.markdown(APP_CSS, unsafe_allow_html=True)

# The page list (Streamlit's multipage navigation) above this links back to
# the NER workflow on app.py.
with st.sidebar:
    st.markdown("## 🌐 Indic Entity Lab")
    st.caption("Research results and project status")

research_dashboard.render()
