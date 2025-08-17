import streamlit as st
from utils.auth import require_auth, logout_button
from utils.components import app_header, kpi_tile, pill, hide_default_pages_nav

st.set_page_config(page_title="Home • SKU Admin", page_icon="🏠", layout="wide")
hide_default_pages_nav()
require_auth()

with st.sidebar:
    st.page_link("pages/1_Home.py", label="🏠 Home")
    st.page_link("pages/2_SKUs.py", label="📦 SKUs")
    st.page_link("pages/3_Outlets.py", label="🏬 Outlets")
    st.page_link("pages/4_SKU_Recommender.py", label="🤖 Recommender")
    st.page_link("pages/5_Settings.py", label="⚙️ Settings")
    logout_button()

app_header("Admin Dashboard")
pill("Note: Dashboard widgets are placeholders for your future code.")

# Quick actions that actually navigate
a,b,c,d = st.columns(4)
with a:
    if st.button("📦 SKUs"):
        st.switch_page("pages/2_SKUs.py")
with b:
    if st.button("🏬 Outlets"):
        st.switch_page("pages/3_Outlets.py")
with c:
    if st.button("🤖 Recommender"):
        st.switch_page("pages/4_SKU_Recommender.py")
with d:
    if st.button("⚙️ Settings"):
        st.switch_page("pages/5_Settings.py")

# KPI placeholders …
k1, k2, k3, k4 = st.columns(4)
with k1: kpi_tile("Total SKUs", "—", "Connect DB to populate")
with k2: kpi_tile("Active Outlets", "—")
with k3: kpi_tile("Low-Stock SKUs", "—")
with k4: kpi_tile("Open Tickets", "—")

st.divider()
st.subheader("📊 Dashboard Area")
st.info("TODO: Add charts, tables, and real metrics here.")
