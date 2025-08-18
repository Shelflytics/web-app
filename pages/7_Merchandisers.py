import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from utils.auth import require_auth, logout_button
from utils.components import app_header, hide_default_pages_nav
from utils.db import get_outlets
from utils.routing import Stop, greedy_route, geocode_pc

st.set_page_config(page_title="Merchandisers • Admin", page_icon="🧑‍🤝‍🧑", layout="wide")
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

app_header("Merchandiser Profiles & Original Routes")

@st.cache_data(ttl=600)
def build_profiles():
    outs = get_outlets().dropna(subset=["postal_code"]).copy()
    outs = outs.head(120)
    outs["label"] = outs.apply(lambda r: f"{r.get('city','Unknown')} ({r['postal_code']})", axis=1)
    chunk = 10
    names = ["Alicia (West Team)","Ben (Central Team)","Chen (East Team)","Dana (North Team)","Evan (Night Shift)","Farah (Weekend Crew)"]
    profiles = {}
    for i, name in enumerate(names):
        seg = outs.iloc[i*chunk:(i+1)*chunk]
        if len(seg) == 0:
            seg = outs.sample(min(chunk, len(outs)), replace=False)
        stops = [Stop(str(r["postal_code"]), r["label"], "normal") for _, r in seg.iterrows()][:10]
        start_pc = stops[0].postal_code if stops else str(outs.iloc[0]["postal_code"])
        profiles[name] = {"start": start_pc, "stops": stops}
    return profiles

profiles = build_profiles()

rows = [{"Merchandiser": n, "Start": p["start"], "Stops": len(p["stops"])} for n,p in profiles.items()]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()
st.subheader("Routes on Map")

m = folium.Map(location=[39.50, -98.35], zoom_start=4)  # USA center
colors = ["red","blue","green","purple","orange","darkred"]
for (name, p), color in zip(profiles.items(), colors):
    res = greedy_route(p["start"], p["stops"], mode="driving")
    coords = []
    for s in res.ordered:
        c = geocode_pc(s.postal_code)
        if c: coords.append((c[0], c[1], s.label))
    if coords:
        folium.PolyLine([(lat,lng) for lat,lng,_ in coords], tooltip=name, color=color).add_to(m)
        for idx,(lat,lng,label) in enumerate(coords, start=1):
            folium.CircleMarker((lat,lng), radius=4, popup=f"{name} · #{idx}: {label}", color=color, fill=True, fill_opacity=0.7).add_to(m)

st_folium(m, width=None, height=520)

st.info("Use the 🗺️ Routes (GenAI) page to change priorities via text and recompute on the fly.")