import streamlit as st
import pandas as pd
from urllib.parse import quote_plus
from utils.auth import require_auth, logout_button
from utils.components import app_header, hide_default_pages_nav
from utils.api import get_sku_recommendations, APIError
from utils.db import get_outlets

st.set_page_config(page_title="SKU Recommender • Admin", page_icon="🤖", layout="wide")
hide_default_pages_nav()
require_auth()

with st.sidebar:
    st.image("assets/shelflytics_logo_transparent_white.png")
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
    st.page_link("pages/policy_faq.py", label="❓ Policy FAQ")
    logout_button()

app_header("SKU Recommender")
st.caption("Choose an outlet by **postal code**.")

# ---------- Load postal index from database ----------
@st.cache_data(ttl=600)
def load_postal_index():
    df = get_outlets().copy()
    if "postal_code" not in df.columns:
        df["postal_code"] = None
    df = df.dropna(subset=["postal_code"])
    df["postal_code"] = df["postal_code"].astype(str).str.strip()
    df = df[df["postal_code"] != ""]

    # Optional columns that may or may not exist
    for c in ["city", "state", "region"]:
        if c not in df.columns:
            df[c] = None

    # Deduplicate by postal for a clean dropdown
    opts = df.sort_values(["postal_code"]).drop_duplicates("postal_code", keep="first")

    # Human-friendly label
    def label_for_row(r):
        city = r["city"] or ""
        state = r["state"] or ""
        parts = [r["postal_code"]]
        loc = ", ".join([p for p in [city, state] if p])
        if loc:
            parts.append(loc)
        return " — ".join(parts)

    opts["label"] = opts.apply(label_for_row, axis=1)
    return opts, df  # (options for dropdown, full df for details)

opts_df, full_df = load_postal_index()

# ---------- Search + Dropdown (type to filter) ----------
col_search, col_select = st.columns([1, 2])
with col_search:
    q = st.text_input(
        "Search",
        placeholder="e.g. 10001 or Los Angeles",
    )

with col_select:
    if q:
        m = (
            opts_df["postal_code"].str.contains(q, case=False, na=False)
            | opts_df["city"].astype(str).str.contains(q, case=False, na=False)
            | opts_df["state"].astype(str).str.contains(q, case=False, na=False)
        )
        filtered = opts_df[m].copy()
    else:
        filtered = opts_df.copy()

    filtered = filtered.sort_values("postal_code").head(200)  # keep list manageable
    labels = filtered["label"].tolist()
    pcs = filtered["postal_code"].tolist()
    label_to_pc = {l: p for l, p in zip(labels, pcs)}

    if len(labels) == 0:
        st.warning("No matches. Try a different search.")
        selected_label = None
    else:
        selected_label = st.selectbox(
            "Select outlet (postal code)",
            labels,
            index=0,
            key="postal_select",
        )

# Persist selected postal for the form submit
if selected_label:
    st.session_state.selected_postal = label_to_pc[selected_label]

# ---------- Outlet details preview (before submit) ----------
with st.container(border=True):
    st.subheader("Selected outlet details", anchor=False)
    if "selected_postal" not in st.session_state:
        st.info("Search and select an outlet from the dropdown to preview details.")
    else:
        sel_pc = st.session_state.selected_postal
        st.write(f"**Postal:** `{sel_pc}`")
        cols_show = [c for c in ["city", "state", "region"] if c in full_df.columns]
        details = full_df[full_df["postal_code"] == sel_pc][cols_show].drop_duplicates().head(20)
        if details.empty:
            st.caption("No additional outlet metadata available for this postal.")
        else:
            st.dataframe(details, use_container_width=True, hide_index=True)
            # --- Google Maps embed ---
            # Build a clean address string for better geocoding and to avoid "Singapore" defaulting.
            # We force USA to keep routes domestic.
            # Try to pull 1st city/state for this postal (if present)
            row = full_df[full_df["postal_code"] == sel_pc].head(1)

            city  = ""
            state = ""
            if "city" in full_df.columns and not row.empty and not row["city"].isna().all():
                city = str(row["city"].iat[0])
            if "state" in full_df.columns and not row.empty and not row["state"].isna().all():
                state = str(row["state"].iat[0])

            # Compose address and URL-encode it
            address = " ".join([str(sel_pc), city, state, "USA"]).strip()
            map_url = f"https://www.google.com/maps?q={quote_plus(address)}&output=embed"

            # Render the iframe inside the details box
            st.components.v1.html(
                f'<iframe src="{map_url}" '
                'style="width:100%; height:420px; border:0;" loading="lazy" '
                'referrerpolicy="no-referrer-when-downgrade"></iframe>',
                height=420,
                width=1200
            )

# ---------- Parameters + Submit ----------
with st.form("reco_form", border=True):
    st.write("**Parameters**")
    left, right = st.columns([3, 2])
    with left:
        st.text_input(
            "Selected Postal Code (locked)",
            value=st.session_state.get("selected_postal", ""),
            disabled=True,
        )
    with right:
        top_k = st.slider("How many SKUs?", 1, 50, value=10)

    submit = st.form_submit_button("Get Recommendations")

# ---------- Call API (original logic preserved) ----------
if submit:
    postal = st.session_state.get("selected_postal", "")
    if not postal:
        st.warning("Please select a postal code from the dropdown first.")
        st.stop()

    with st.spinner("Calling model API..."):
        try:
            data = get_sku_recommendations(postal, top_k)
        except APIError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    rows = []
    if isinstance(data, dict):
        items = data.get("results") or data.get("skus") or data.get("recommendations") or data.get("data") or data
        if isinstance(items, dict):
            items = list(items.items())
        rows = items
    elif isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            rows = [{"rank": i + 1, "sku": x} for i, x in enumerate(data)]
        elif all(isinstance(x, dict) for x in data):
            def row(i, d):
                r = {
                    "rank": d.get("rank", i + 1),
                    "sku": d.get("sku") or d.get("id") or d.get("name") or "—",
                    "score": d.get("score") or d.get("prob") or d.get("confidence"),
                }
                for k, v in d.items():
                    if k not in r:
                        r[k] = v
                return r
            rows = [row(i, d) for i, d in enumerate(data)]
    else:
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
