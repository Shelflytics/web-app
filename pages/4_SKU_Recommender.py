import streamlit as st
import pandas as pd
from utils.auth import require_auth, logout_button
from utils.components import app_header, hide_default_pages_nav
from utils.api import get_sku_recommendations, APIError

st.set_page_config(page_title="SKU Recommender • Admin", page_icon="🤖", layout="wide")
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
    st.page_link("pages/chatbot_page.py", label="💬 Chatbot") 
    st.page_link("pages/predict_page.py", label="📈 Predict Item Performance")
    logout_button()

app_header("SKU Recommender (ML)")

st.write("Enter a postal code to get a ranked list of recommended SKUs to display/stock.")

with st.form("reco_form"):
    postal = st.text_input("Postal Code", placeholder="e.g. 42420", max_chars=10)
    top_k = st.slider("How many SKUs?", 1, 50, value=10)
    submit = st.form_submit_button("Get Recommendations")

if submit:
    if not postal.strip():
        st.warning("Please enter a postal code.")
    else:
        with st.spinner("Calling model API..."):
            try:
                data = get_sku_recommendations(postal, top_k)
            except APIError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        # Normalize to a table
        rows = []
        if isinstance(data, dict):
            # Try common keys
            items = data.get("results") or data.get("skus") or data.get("recommendations") or data.get("data") or data
            if isinstance(items, dict):
                items = list(items.items())
            rows = items
        elif isinstance(data, list):
            # If it's a simple list of strings, rank 1..N
            if all(isinstance(x, str) for x in data):
                rows = [{"rank": i+1, "sku": x} for i, x in enumerate(data)]
            elif all(isinstance(x, dict) for x in data):
                # If dicts have ranking/score fields, respect them
                # Fallback to enumerate if not.
                def row(i, d):
                    r = {"rank": d.get("rank", i+1), "sku": d.get("sku") or d.get("id") or d.get("name") or "—",
                         "score": d.get("score") or d.get("prob") or d.get("confidence")}
                    # Include any other keys for visibility
                    for k, v in d.items():
                        if k not in r:
                            r[k] = v
                    return r
                rows = [row(i, d) for i, d in enumerate(data)]
        else:
            # Unknown payload; display raw
            st.code(str(data))
            st.stop()

        df = pd.DataFrame(rows)
        if "rank" in df.columns:
            df = df.sort_values("rank")
        st.success("Recommendations received.")
        st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("API docs (embedded)"):
    st.components.v1.iframe("https://model-deployment-963934256033.asia-southeast1.run.app/docs", height=700)
    st.caption("Note: Only the /api/meow endpoint is used by this app.")