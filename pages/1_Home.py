import streamlit as st
from utils.db import fetch_sales_superstore
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
    st.page_link("pages/6_Routes.py", label="🗺️ Routes")
    st.page_link("pages/5_Settings.py", label="⚙️ Settings")
    st.page_link("pages/7_Merchandisers.py", label="🧑‍🤝‍🧑 Merchandisers")
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
    if st.button("🗺️ Routes"):
        st.switch_page("pages/6_Routes.py")


# KPI placeholders …
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
