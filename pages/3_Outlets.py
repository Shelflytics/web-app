import streamlit as st
import pandas as pd
from utils.auth import require_auth, logout_button
from utils.components import app_header, hide_default_pages_nav

st.set_page_config(page_title="Outlets • Admin", page_icon="🏬", layout="wide")
hide_default_pages_nav()
require_auth()
with st.sidebar:
    st.page_link("pages/1_Home.py", label="🏠 Home")
    st.page_link("pages/2_SKUs.py", label="📦 SKUs")
    st.page_link("pages/3_Outlets.py", label="🏬 Outlets")
    st.page_link("pages/4_SKU_Recommender.py", label="🤖 Recommender")
    st.page_link("pages/5_Settings.py", label="⚙️ Settings")
    logout_button()

app_header("Outlets")

if "outlet_df" not in st.session_state:
    st.session_state.outlet_df = pd.DataFrame([
        {"outlet_id":"S1001","name":"Sheng Siong - Clementi","postal_code":"120123","manager":"Ms Lim","phone":"—"},
        {"outlet_id":"N2002","name":"NTUC - Serangoon","postal_code":"550123","manager":"Mr Tan","phone":"—"},
        {"outlet_id":"G3003","name":"Giant - Tampines","postal_code":"520123","manager":"Mr Lee","phone":"—"},
    ])

st.dataframe(st.session_state.outlet_df, use_container_width=True, hide_index=True)