import streamlit as st
import pandas as pd
import folium
from utils.auth import require_auth, logout_button
from utils.components import app_header, hide_default_pages_nav
from utils.db import get_outlets
from utils.routing import Stop, greedy_route, geocode_pc
from utils.planning import assign_stops_to_merch

st.set_page_config(page_title="Merchandisers • Admin", page_icon="🧑‍🤝‍🧑", layout="wide")
hide_default_pages_nav()
require_auth()

with st.sidebar:
    st.image("assets/shelflytics_logo_transparent_white.png")

    st.page_link("pages/1_Home.py", label="🏠 Home")
    logout_button()

    st.divider()
    st.markdown("**U.S.**")
    st.page_link("pages/2_SKUs.py", label="📦 SKUs")
    st.page_link("pages/3_Outlets.py", label="🏬 Outlets")
    st.page_link("pages/4_SKU_Recommender.py", label="🎁 SKU Recommender")
    st.page_link("pages/6_Routes.py", label="🗺️ Route Optimiser")
    st.page_link("pages/7_Merchandisers.py", label="🧑‍🤝‍🧑 Merchandisers")

    st.divider()
    st.markdown("**China**")
    st.page_link("pages/chatbot_page.py", label="💬 Chatbot")
    st.page_link("pages/predict_page.py", label="📈 Predict Item Performance")

    st.divider()
    st.page_link("pages/sku_detection.py", label="👁️ Detector") 
    st.page_link("pages/policy_faq.py", label="❓ Policy FAQ")
    st.page_link("pages/5_Settings.py", label="⚙️ Settings")


app_header("Merchandiser Profiles & Routes")

@st.cache_data(ttl=1200)
def build_profiles():
    outs = get_outlets().dropna(subset=["postal_code"]).copy()
    outs = outs.head(200)
    outs["postal_code"] = outs["postal_code"].astype(str)
    outs["label"] = outs.apply(lambda r: f"{r.get('city','Unknown')} ({r['postal_code']})", axis=1)

    names = ["Alicia (West Team)","Ben (Central Team)","Chen (East Team)","Dana (North Team)","Evan (Night Shift)","Farah (Weekend Crew)"]
    contacts = {
        "Alicia (West Team)": "+1-555-0110",
        "Ben (Central Team)": "+1-555-0111",
        "Chen (East Team)": "+1-555-0112",
        "Dana (North Team)": "+1-555-0113",
        "Evan (Night Shift)": "+1-555-0114",
        "Farah (Weekend Crew)": "+1-555-0115",
    }
    homes_pref = {
        "Alicia (West Team)": ["94618","94708"],
        "Ben (Central Team)": ["10001","10002"],
        "Chen (East Team)": ["98109","98107"],
        "Dana (North Team)": ["60614","60657"],
        "Evan (Night Shift)": ["75201","75001"],
        "Farah (Weekend Crew)": ["30305","30306"],
    }
    homes = {n: prefs[0] for n, prefs in homes_pref.items()}

    assigned = assign_stops_to_merch(homes, outs, per_merch_max=10)
    profiles = {}
    for name in names:
        profiles[name] = {"home": homes[name], "contact": contacts[name], "stops": assigned.get(name, [])}
    return profiles

profiles = build_profiles()

default_modes = {name: "driving" for name in profiles.keys()}
default_modes.update(st.session_state.get("routing_modes", {}))

@st.cache_data(ttl=None)
def cached_geocode(pc: str):
    return geocode_pc(pc)

@st.cache_data(ttl=None)
def build_map_html(profiles_dict: dict, modes_dict: dict, version: int):
    m = folium.Map(location=[39.50, -98.35], zoom_start=4, width="100%", height=620, control_scale=True, tiles="cartodbpositron")
    colors = ["red","blue","green","purple","orange","darkred"]
    for (name, prof), color in zip(profiles_dict.items(), colors):
        mode = modes_dict.get(name, "driving")
        res = greedy_route(prof["home"], prof["stops"], mode=mode)
        coords = []
        hc = cached_geocode(prof["home"])
        if hc: coords.append((hc[0], hc[1], f"{name} Home"))
        for s in res.ordered:
            c = cached_geocode(s.postal_code)
            if c: coords.append((c[0], c[1], s.label))
        if coords:
            folium.PolyLine([(lat,lng) for lat,lng,_ in coords], tooltip=f"{name} ({mode})", color=color).add_to(m)
            for idx,(lat,lng,label) in enumerate(coords, start=1):
                folium.CircleMarker((lat,lng), radius=4, popup=f"{name} · #{idx}: {label}", color=color, fill=True, fill_opacity=0.7).add_to(m)
    html = m.get_root().render()
    css = "<style>.leaflet-container{width:100%!important;min-height:620px!important}.folium-map{width:100%!important;height:620px!important}figure{width:100%!important;margin:0}</style>"
    return css + html

if "merch_map_version" not in st.session_state:
    st.session_state.merch_map_version = 0
if "merch_map_html" not in st.session_state:
    st.session_state.merch_map_html = None

colA, colB = st.columns([1,2])
with colA:
    if st.button("🔄 Rebuild map with latest routes"):
        st.session_state.merch_map_version += 1
        st.session_state.merch_map_html = build_map_html(profiles, default_modes, st.session_state.merch_map_version)
with colB:
    st.caption("Map persists across refresh/navigation. Click **Rebuild** to update.")

if st.session_state.merch_map_html is None:
    st.session_state.merch_map_html = build_map_html(profiles, default_modes, st.session_state.merch_map_version)

st.components.v1.html(st.session_state.merch_map_html, height=640, width=1200, scrolling=False)

st.divider()
st.subheader("Leg details for a merchandiser")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    sel_name = st.selectbox("Choose merchandiser", list(profiles.keys()))
with col2:
    mode = default_modes.get(sel_name, "driving")
    st.write(f"Mode: **{mode}**")
with col3:
    go = st.button("Load leg details")

st.caption(f"Merchandiser: **{sel_name}**  ·  Home: `{profiles[sel_name]['home']}`  ·  Contact: `{profiles[sel_name]['contact']}`")

if go:
    with st.spinner("Computing leg details..."):
        res = greedy_route(profiles[sel_name]["home"], profiles[sel_name]["stops"], mode=mode)
        legs_df = pd.DataFrame([{
            "from": leg["from"],
            "to": leg["to"],
            "priority": leg["priority"],
            "time_min": leg["time_min"],
            "distance_km": leg["dist_km"],
        } for leg in res.legs])
    st.dataframe(legs_df, use_container_width=True, hide_index=True)