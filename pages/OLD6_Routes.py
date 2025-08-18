import streamlit as st
import pandas as pd
from utils.auth import require_auth, logout_button
from utils.components import app_header, hide_default_pages_nav
from utils.db import get_outlets
from utils.llm import parse_route_instruction
from utils.routing import Stop, greedy_route, apply_priority_updates

st.set_page_config(page_title="Routes • Admin", page_icon="🗺️", layout="wide")
hide_default_pages_nav()
require_auth()

with st.sidebar:
    st.page_link("pages/1_Home.py", label="🏠 Home")
    st.page_link("pages/2_SKUs.py", label="📦 SKUs")
    st.page_link("pages/3_Outlets.py", label="🏬 Outlets")
    st.page_link("pages/4_SKU_Recommender.py", label="🤖 Recommender")
    st.page_link("pages/6_Routes.py", label="🗺️ Routes")
    st.page_link("pages/5_Settings.py", label="⚙️ Settings")
    st.page_link("pages/sku_detection.py", label="👁️ SKU Detection") 
    logout_button()

app_header("GenAI-Powered Route Optimisation")

@st.cache_data(ttl=600)
def sample_merchandiser_routes():
    outs = get_outlets().dropna(subset=["postal_code"]).copy()
    outs = outs.head(50)
    outs["label"] = outs.apply(lambda r: f"{r.get('city','Unknown')} ({r['postal_code']})", axis=1)
    bundles = []; chunk = 10
    for i in range(0, min(len(outs), 30), chunk):
        bundles.append(outs.iloc[i:i+chunk].copy())
    while len(bundles) < 3:
        bundles.append(outs.sample(min(chunk, len(outs)), replace=False).copy())

    return {
        "Alicia (West Team)": {
            "start_postal": str(bundles[0].iloc[0]["postal_code"]),
            "stops": [Stop(str(r["postal_code"]), r["label"], "normal") for _, r in bundles[0].iterrows()],
        },
        "Ben (Central Team)": {
            "start_postal": str(bundles[1].iloc[0]["postal_code"]),
            "stops": [Stop(str(r["postal_code"]), r["label"], "normal") for _, r in bundles[1].iterrows()],
        },
        "Chen (East Team)": {
            "start_postal": str(bundles[2].iloc[0]["postal_code"]),
            "stops": [Stop(str(r["postal_code"]), r["label"], "normal") for _, r in bundles[2].iterrows()],
        },
    }

merch = sample_merchandiser_routes()

left, right = st.columns([2,1])
with left:
    m_name = st.selectbox("Merchandiser", list(merch.keys()))
with right:
    mode = st.selectbox("Travel mode", ["driving", "walking", "transit"], index=0)

if "routing_state" not in st.session_state:
    st.session_state.routing_state = {}
state = st.session_state.routing_state.setdefault(m_name, {
    "stops": merch[m_name]["stops"],
    "start": merch[m_name]["start_postal"],
})

st.markdown("#### Natural language adjustment")
st.caption("Examples:\n- Make 120123 urgent priority due to stock shortage\n- Set 550123 high priority")
instr = st.text_input("Type an instruction and press Apply", key=f"instr_{m_name}")
if st.button("Apply instruction"):
    outlets_ctx = [{"postal_code": s.postal_code, "label": s.label, "priority": s.priority} for s in state["stops"]]
    updates = parse_route_instruction(instr, outlets_ctx).get("set_priority", [])
    if updates:
        state["stops"] = apply_priority_updates(state["stops"], updates)
        st.success(f"Applied {len(updates)} priority update(s).")
    else:
        st.info("No actionable updates parsed from the instruction.")

if st.button("Compute route"):
    with st.spinner("Optimising with Google Distance Matrix..."):
        res = greedy_route(state["start"], state["stops"], mode=mode)

    st.subheader("Recommended order")
    ord_df = pd.DataFrame([{ "order": i+1, "postal_code": s.postal_code, "label": s.label, "priority": s.priority } for i, s in enumerate(res.ordered)])
    st.dataframe(ord_df, use_container_width=True, hide_index=True)

    st.subheader("Leg details")
    legs_df = pd.DataFrame(res.legs)
    st.dataframe(legs_df, use_container_width=True, hide_index=True)

    st.success(f"Total time ≈ {res.total_time_min} min, distance ≈ {res.total_dist_km} km")
    if res.maps_url:
        st.link_button("Open in Google Maps", res.maps_url)

st.caption("Note: Greedy heuristic with simple priority bonuses. Adjust in utils/routing.py.")