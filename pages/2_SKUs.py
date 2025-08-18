import streamlit as st
import pandas as pd
from io import StringIO
from utils.auth import require_auth, logout_button
from utils.components import app_header, hide_default_pages_nav
from utils.db import get_sku_catalog

st.set_page_config(page_title="SKUs • Admin", page_icon="📦", layout="wide")
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

# app_header("SKU Management")

# # Initialize in-session table
# if "sku_df" not in st.session_state:
#     st.session_state.sku_df = pd.DataFrame([
#         {"sku_id":"SKU-1001","name":"Mothballs 200g","category":"Home Care","price":2.50,"status":"Active","facing_score":85,"last_updated":"—"},
#         {"sku_id":"SKU-2002","name":"Trash Bags 30L (20s)","category":"Household","price":3.20,"status":"Active","facing_score":78,"last_updated":"—"},
#         {"sku_id":"SKU-3003","name":"Insect Repellent Spray","category":"Home Care","price":4.90,"status":"Inactive","facing_score":55,"last_updated":"—"},
#     ])

# # Controls
# left, right = st.columns([3,1])
# with left:
#     st.write("Filter & Search")
#     name_filter = st.text_input("Name contains", "")
#     status_filter = st.multiselect("Status", ["Active","Inactive"], default=["Active","Inactive"])
# with right:
#     st.write("Import / Export")
#     uploaded = st.file_uploader("Import CSV", type=["csv"], help="Uploads your SKU sheet and converts it into the portal format.")
#     if uploaded is not None:
#         try:
#             raw = pd.read_csv(uploaded)
#             # Normalize basic columns if present
#             mapping = {
#                 "SKU ID":"sku_id","SKU":"sku_id","Id":"sku_id",
#                 "Name":"name","Product Name":"name",
#                 "Category":"category","Price":"price",
#                 "Status":"status","Facing Score":"facing_score"
#             }
#             for col, new in mapping.items():
#                 if col in raw.columns and new not in raw.columns:
#                     raw[new] = raw[col]
#             for col in ["sku_id","name","category","price","status","facing_score"]:
#                 if col not in raw.columns:
#                     raw[col] = None
#             keep = ["sku_id","name","category","price","status","facing_score"]
#             st.session_state.sku_df = raw[keep].copy()
#             st.success("Import successful. Data normalized for the portal.")
#         except Exception as e:
#             st.error(f"Failed to import CSV: {e}")

#     # Export current view
#     csv = st.session_state.sku_df.to_csv(index=False).encode("utf-8")
#     st.download_button("Export CSV", csv, "skus_export.csv", "text/csv")

# # Filtered view
# df = st.session_state.sku_df.copy()
# if name_filter:
#     df = df[df["name"].str.contains(name_filter, case=False, na=False)]
# if status_filter:
#     df = df[df["status"].isin(status_filter)]

# st.write("### Edit SKUs")
# edited = st.data_editor(
#     df,
#     num_rows="dynamic",
#     hide_index=True,
#     use_container_width=True,
#     column_config={
#         "price": st.column_config.NumberColumn("price", step=0.1, format="%.2f"),
#         "facing_score": st.column_config.NumberColumn("facing_score", step=1, min_value=0, max_value=100),
#     },
# )
# if st.button("Save Changes"):
#     st.session_state.sku_df = edited
#     st.success("Changes saved to the current session. Connect DB to persist.")

app_header("SKU Management")

df = get_sku_catalog()
if df.empty:
    st.warning("No SKUs found.")
else:
    # Filters
    left, right = st.columns([3,1])
    with left:
        name_filter = st.text_input("Search name contains", "")
        cat_filter = st.multiselect("Category", sorted(df["category"].dropna().unique().tolist()))
    with right:
        st.caption("Read-only from Supabase")
        st.caption("Source: sales_superstore")

    view = df.copy()
    if name_filter:
        view = view[view["name"].str.contains(name_filter, case=False, na=False)]
    if cat_filter:
        view = view[view["category"].isin(cat_filter)]

    st.write(f"Showing {len(view):,} SKUs")
    st.dataframe(view, use_container_width=True, hide_index=True)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV (filtered)", csv, "skus_from_supabase.csv", "text/csv")
