import streamlit as st
import streamlit.components.v1 as components
from utils.db import fetch_sales_superstore
from utils.auth import require_auth, logout_button
from utils.components import app_header, kpi_tile, pill, hide_default_pages_nav
from streamlit.components.v1 import iframe

st.set_page_config(page_title="Home • SKU Admin", page_icon="🏠", layout="wide")
hide_default_pages_nav()
require_auth()

# one-time welcome message only immediately after login
if st.session_state.pop("just_logged_in", False):
    st.success("Welcome back, sales manager!")

with st.sidebar:
    st.image("assets\shelflytics_logo_transparent_white.png")
    st.page_link("pages/1_Home.py", label="🏠 Home")
    st.page_link("pages/2_SKUs.py", label="📦 SKUs")
    st.page_link("pages/3_Outlets.py", label="🏬 Outlets")
    st.page_link("pages/4_SKU_Recommender.py", label="🤖 Recommender")
    st.page_link("pages/6_Routes.py", label="🗺️ Routes")
    st.page_link("pages/5_Settings.py", label="⚙️ Settings")
    st.page_link("pages/7_Merchandisers.py", label="🧑‍🤝‍🧑 Merchandisers")
    st.page_link("pages/chatbot_page.py", label="💬 Chatbot")
    st.page_link("pages/predict_page.py", label="📈 Predict Item Performance")
    st.page_link("pages/sku_detection.py", label="👁️ Detector")
    logout_button()

app_header("Admin Dashboard")

# Tabs
tab_overview, tab_china = st.tabs(["United States", "China"])

with tab_overview:
   
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
        if st.button("🗺️ Routes"):
            st.switch_page("pages/6_Routes.py")

    # KPIs
    df = fetch_sales_superstore()
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_tile("Total Sales", f"${df['sales'].sum():,.2f}")
    with k2: kpi_tile("Orders", df["order_id"].nunique())
    with k3: kpi_tile("Customers", df["customer_id"].nunique())
    with k4: kpi_tile("SKUs", df["product_id"].nunique())

    st.divider()
    st.subheader("📊 Top Categories (by sales)")
    top_cat = (
        df.groupby("category", dropna=True)["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    st.dataframe(top_cat, use_container_width=True, hide_index=True)

with tab_china:
    st.subheader("China Outlets Dashboard")

    iframe(
        src="https://lookerstudio.google.com/embed/reporting/2f87a2eb-d0c9-4827-ac47-f7e2cdcbedff/page/58qUF",
        width=1200,  
        height=800,
        scrolling=True
    )