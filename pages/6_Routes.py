
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
    st.page_link("pages/policy_faq.py", label="❓ Policy FAQ")
    logout_button()

app_header("GenAI-Powered Route Optimisation")

@st.cache_data(ttl=600)
def sample_merchandiser_routes():
    outs = get_outlets().dropna(subset=["postal_code"]).copy()
    outs = outs.head(120)
    outs["label"] = outs.apply(lambda r: f"{r.get('city','Unknown')} ({r['postal_code']})", axis=1)

    bundles, chunk = [], 10
    for i in range(0, min(len(outs), 60), chunk):
        bundles.append(outs.iloc[i:i+chunk].copy())
    while len(bundles) < 6 and len(outs) >= chunk:
        bundles.append(outs.sample(chunk, replace=False).copy())

    names = ["Alicia (West Team)","Ben (Central Team)","Chen (East Team)","Dana (North Team)","Evan (Night Shift)","Farah (Weekend Crew)"]
    contacts = {
        "Alicia (West Team)": "+1-555-0110",
        "Ben (Central Team)": "+1-555-0111",
        "Chen (East Team)": "+1-555-0112",
        "Dana (North Team)": "+1-555-0113",
        "Evan (Night Shift)": "+1-555-0114",
        "Farah (Weekend Crew)": "+1-555-0115",
    }
    homes = {
        "Alicia (West Team)": ["94618","94708"],
        "Ben (Central Team)": ["10001","10002"],
        "Chen (East Team)": ["98109","98107"],
        "Dana (North Team)": ["60614","60657"],
        "Evan (Night Shift)": ["75201","75001"],
        "Farah (Weekend Crew)": ["30305","30306"],
    }

    merchandisers = {}
    for idx, name in enumerate(names):
        b = bundles[min(idx, len(bundles)-1)]
        stops = [Stop(str(r["postal_code"]), r["label"], "normal") for _, r in b.iterrows()][:10]
        outlet_zips = {s.postal_code for s in stops}
        home_pc = next((pc for pc in homes[name] if pc not in outlet_zips), homes[name][0])
        merchandisers[name] = {
            "home_postal": home_pc,
            "contact": contacts[name],
            "stops": stops,
        }
    return merchandisers

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
    "home": merch[m_name]["home_postal"],
})

st.info(f"**{m_name}** — Home: `{merch[m_name]['home_postal']}` · Contact: `{merch[m_name]['contact']}`")

st.markdown("#### Natural language adjustment")
st.caption("Examples:  • Make 19140 urgent priority due to stock shortage  • Set 78207 high priority")
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
        res = greedy_route(state["home"], state["stops"], mode=mode)

    st.subheader("Recommended order")
    ord_df = pd.DataFrame([{
        "order": i+1,
        "postal_code": s.postal_code,
        "label": s.label,
        "priority": s.priority,
    } for i, s in enumerate(res.ordered)])
    st.dataframe(ord_df, use_container_width=True, hide_index=True)

    st.subheader("Leg details")
    legs_df = pd.DataFrame(res.legs)[["from","to","priority","time_min","dist_km"]]
    st.dataframe(legs_df, use_container_width=True, hide_index=True)

    st.success(f"Total time ≈ {res.total_time_min} min, distance ≈ {res.total_dist_km} km")
    if res.maps_url:
        st.link_button("Open in Google Maps", res.maps_url)

    modes = st.session_state.setdefault("routing_modes", {})
    modes[m_name] = mode
