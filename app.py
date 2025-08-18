import streamlit as st
from utils.auth import login_ui, setup_2fa_ui, is_authenticated
from utils.components import hide_default_pages_nav, hide_entire_sidebar

st.set_page_config(page_title="SKU Admin Portal", page_icon="📦", layout="wide", initial_sidebar_state="collapsed")

# Always hide the built-in Pages nav (we render our own later)
hide_default_pages_nav()

# Initialize flag once
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not is_authenticated():
    # Entire sidebar hidden pre-login
    hide_entire_sidebar()
    tab1, tab2 = st.tabs(["Login", "Setup 2FA"])
    with tab1: login_ui()
    with tab2: setup_2fa_ui()
else:
    # st.success("You're logged in ✔")
    # st.write("Use the sidebar or the buttons below to navigate.")
    st.switch_page("pages/1_Home.py")
    
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     if st.button("🏠 Home / Dashboard"):
    #         st.switch_page("pages/1_Home.py")
    # with col2:
    #     if st.button("📦 Manage SKUs"):
    #         st.switch_page("pages/2_SKUs.py")
    # with col3:
    #     if st.button("🏬 Outlets"):
    #         st.switch_page("pages/3_Outlets.py")

    # col4, col5 = st.columns(2)
    # with col4:
    #     if st.button("🤖 SKU Recommender (ML)"):
    #         st.switch_page("pages/4_SKU_Recommender.py")
    # with col5:
    #     if st.button("⚙️ Settings"):
    #         st.switch_page("pages/5_Settings.py")
