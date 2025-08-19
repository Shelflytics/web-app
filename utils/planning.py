from __future__ import annotations
import math
from typing import Dict, List, Tuple
import streamlit as st
import pandas as pd
from utils.routing import geocode_pc, Stop

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

@st.cache_data(ttl=3600)
def _cached_geocode(pc: str):
    return geocode_pc(pc)

@st.cache_data(ttl=600)
def assign_stops_to_merch(
    homes: Dict[str, str],
    outlets_df: pd.DataFrame,
    per_merch_max: int = 10,
) -> Dict[str, List[Stop]]:
    """Assign each outlet to the nearest merch home (by great-circle km).
    Returns dict[merch_name] -> list[Stop] (<= per_merch_max, closest first).
    """
    home_coords: Dict[str, Tuple[float, float]] = {}
    for name, pc in homes.items():
        c = _cached_geocode(str(pc))
        if c:
            home_coords[name] = (c[0], c[1])

    outlet_coords: Dict[str, Tuple[float, float]] = {}
    df = outlets_df.dropna(subset=["postal_code"]).copy()
    df["postal_code"] = df["postal_code"].astype(str)
    df = df.drop_duplicates(subset=["postal_code"])  # unique postal
    for _, r in df.iterrows():
        pc = str(r["postal_code"])        
        c = _cached_geocode(pc)
        if c:
            outlet_coords[pc] = (c[0], c[1])

    buckets: Dict[str, List[tuple]] = {name: [] for name in homes.keys()}
    for _, r in df.iterrows():
        pc = str(r["postal_code"])        
        if pc not in outlet_coords:
            continue
        lat, lng = outlet_coords[pc]
        label = f"{r.get('city','Unknown')} ({pc})"
        best_name, best_d = None, float("inf")
        for name, (hlat, hlng) in home_coords.items():
            d = _haversine(lat, lng, hlat, hlng)
            if d < best_d:
                best_name, best_d = name, d
        if best_name:
            buckets[best_name].append((pc, best_d, label))

    result: Dict[str, List[Stop]] = {}
    for name, items in buckets.items():
        items.sort(key=lambda x: x[1])
        selected = items[:per_merch_max]
        result[name] = [Stop(pc, label, "normal") for pc, _, label in selected]
    return result