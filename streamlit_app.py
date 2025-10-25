# streamlit_app.py
# ------------------------------------------------------------------
# Home page for the NOF dashboards.
# Provides: simple tiles to navigate, short descriptions, and a
# global toggle to remember filters across pages.
# ------------------------------------------------------------------

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NOF Dashboards", page_icon="📊", layout="wide")

# ---- Global UI prefs & state ------------------------------------
if "remember_filters" not in st.session_state:
    st.session_state.remember_filters = True  # default ON

st.markdown("""
<style>
.tile {
  border: 1px solid #E5E7EB; border-radius: 16px; padding: 16px;
  background: #FFFFFF; box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.tile h3 { margin: 0 0 6px 0; }
.tile p { margin: 0 0 12px 0; color: #4B5563; }
</style>
""", unsafe_allow_html=True)

st.title("NOF Dashboards")

# ---- Remember filters toggle ------------------------------------
st.checkbox("Remember filters across pages", key="remember_filters")

st.write("")  # small spacer

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="tile">
          <h3>Quarterly Rankings</h3>
          <p>Explore provider performance by quarter, domain and metric. Bar chart by rank, sticky metric panel, and KPIs.</p>
        </div>
        """, unsafe_allow_html=True
    )
    # Streamlit 1.31+ has st.page_link; fall back to a button if older
    try:
        st.page_link("pages/1_Quarterly_Rankings.py", label="Open Quarterly Dashboard", icon="📈")
    except Exception:
        st.link_button("Open Quarterly Dashboard", "./pages/1_Quarterly_Rankings.py")

with col2:
    st.markdown(
        """
        <div class="tile">
          <h3>Monthly Rankings</h3>
          <p>Monthly view with KPIs, rank deltas vs previous month, 12-month trend, and Region vs National comparison.</p>
        </div>
        """, unsafe_allow_html=True
    )
    try:
        st.page_link("pages/2_Monthly_Rankings.py", label="Open Monthly Dashboard", icon="📅")
    except Exception:
        st.link_button("Open Monthly Dashboard", "./pages/2_Monthly_Rankings.py")

st.write("")
st.caption("Tip: you can toggle ‘Remember filters across pages’ any time on the Home page.")
