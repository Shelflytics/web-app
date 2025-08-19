import streamlit as st
from utils.auth import require_auth, logout_button, setup_2fa_ui
from utils.components import app_header, hide_default_pages_nav

st.set_page_config(page_title="Settings • Admin", page_icon="⚙️", layout="wide")
hide_default_pages_nav()
require_auth()

with st.sidebar:
    st.image("assets/shelflytics_logo_transparent_white.png")

    st.page_link("pages/1_Home.py", label="🏠 Home")
    logout_button()

    st.divider()
    st.markdown("**U.S.**")
    st.page_link("pages/2_SKUs.py", label="📦 SKUs")
    st.page_link("pages/3_Outlets.py", label="🏬 Outlets")
    st.page_link("pages/4_SKU_Recommender.py", label="🎁 SKU Recommender")
    st.page_link("pages/6_Routes.py", label="🗺️ Route Optimiser")
    st.page_link("pages/7_Merchandisers.py", label="🧑‍🤝‍🧑 Merchandisers")

    st.divider()
    st.markdown("**China**")
    st.page_link("pages/chatbot_page.py", label="💬 Chatbot")
    st.page_link("pages/predict_page.py", label="📈 Predict Item Performance")

    st.divider()
    st.page_link("pages/sku_detection.py", label="👁️ Detector") 
    st.page_link("pages/policy_faq.py", label="❓ Policy FAQ")
    st.page_link("pages/5_Settings.py", label="⚙️ Settings")

app_header("Settings")
st.subheader("Two-Factor Authentication")
setup_2fa_ui()

st.divider()
st.subheader("Appearance")
st.caption("Theme colours are defined in .streamlit/config.toml. Change primaryColor to switch between company green (#74C043) and the logo light blue (#57B3E8).")