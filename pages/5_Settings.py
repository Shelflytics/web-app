import streamlit as st
from utils.auth import require_auth, logout_button, setup_2fa_ui
from utils.components import app_header, hide_default_pages_nav

st.set_page_config(page_title="Settings • Admin", page_icon="⚙️", layout="wide")
hide_default_pages_nav()
require_auth()
with st.sidebar:
    st.page_link("pages/1_Home.py", label="🏠 Home")
    st.page_link("pages/2_SKUs.py", label="📦 SKUs")
    st.page_link("pages/3_Outlets.py", label="🏬 Outlets")
    st.page_link("pages/4_SKU_Recommender.py", label="🤖 Recommender")
    st.page_link("pages/6_Routes.py", label="🗺️ Routes")
    st.page_link("pages/5_Settings.py", label="⚙️ Settings")
    st.page_link("pages/7_Merchandisers.py", label="🧑‍🤝‍🧑 Merchandisers")
    st.page_link("pages/sku_detection.py", label="👁️ Detector")
    logout_button()

app_header("Settings")
st.subheader("Two-Factor Authentication")
setup_2fa_ui()

st.divider()
st.subheader("Appearance")
st.caption("Theme colours are defined in .streamlit/config.toml. Change primaryColor to switch between company green (#74C043) and the logo light blue (#57B3E8).")