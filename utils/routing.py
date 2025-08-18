from __future__ import annotations
import time, urllib.parse, requests, streamlit as st
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

GMAPS_DM_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
GMAPS_GC_URL = "https://maps.googleapis.com/maps/api/geocode/json"

PRIORITY_BONUS_MIN = {"urgent":30.0, "high":10.0, "normal":0.0, "low":-10.0}

def _country_name() -> str:
    return st.secrets.get("routing", {}).get("country_name", "United States")

def _country_code() -> str:
    return st.secrets.get("routing", {}).get("country_code", "us")

def _gmaps_key() -> str:
    key = st.secrets.get("googlemaps", {}).get("api_key")
    if not key:
        raise RuntimeError("Missing [googlemaps].api_key in .streamlit/secrets.toml")
    return key

def _fmt_addr(pc: str) -> str:
    return f"{pc}, {_country_name()}"

def _duration_matrix_single_origin(origin: str, destinations: List[str], mode: str) -> List[Tuple[float, float, dict]]:
    params = {
        "origins": origin,
        "destinations": "|".join(destinations),
        "mode": mode.lower(),
        "units": "metric",
        "region": _country_code(),
        "departure_time": "now",
        "key": _gmaps_key(),
    }
    r = requests.get(GMAPS_DM_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    row = (data.get("rows") or [{}])[0]
    elements = row.get("elements") or []
    for el in elements:
        if el.get("status") != "OK":
            out.append((float("inf"), float("inf"), el)); continue
        d_val = el["distance"]["value"]
        t_val = el["duration"]["value"]
        mins = (t_val if t_val > 0 else 60) / 60.0
        km   = (d_val if d_val > 0 else 1000) / 1000.0
        out.append((mins, km, el))
    return out

@st.cache_data(ttl=86400, show_spinner=False)
def geocode_pc(pc: str):
    params = {
        "address": pc,
        "components": f"country:{_country_code()}",
        "key": _gmaps_key(),
    }
    try:
        r = requests.get(GMAPS_GC_URL, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        if js.get("status") == "OK" and js.get("results"):
            loc = js["results"][0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"])
    except Exception:
        return None
    return None

from dataclasses import dataclass, field
@dataclass
class Stop:
    postal_code: str
    label: str
    priority: str = "normal"

@dataclass
class RouteResult:
    ordered: List[Stop] = field(default_factory=list)
    legs: List[Dict] = field(default_factory=list)
    total_time_min: float = 0.0
    total_dist_km: float = 0.0
    maps_url: str = ""

def greedy_route(start_pc: str, stops: List[Stop], mode: str = "driving") -> RouteResult:
    remaining = stops[:]
    ordered: List[Stop] = []
    legs: List[Dict] = []
    total_min = 0.0
    total_km = 0.0
    cur = start_pc

    while remaining:
        dest_labels = [_fmt_addr(s.postal_code) for s in remaining]
        results = _duration_matrix_single_origin(_fmt_addr(cur), dest_labels, mode)
        best_idx, best_score = None, float("inf")
        for i, (mins, km, raw) in enumerate(results):
            bonus = PRIORITY_BONUS_MIN.get(remaining[i].priority, 0.0)
            score = mins - bonus
            if score < best_score:
                best_idx, best_score = i, score
        chosen = remaining.pop(best_idx)
        chosen_mins, chosen_km, chosen_raw = results[best_idx]
        ordered.append(chosen)
        legs.append({
            "from": cur,
            "to": chosen.postal_code,
            "time_min": round(chosen_mins, 1),
            "dist_km": round(chosen_km, 2),
            "priority": chosen.priority,
        })
        total_min += chosen_mins; total_km += chosen_km
        cur = chosen.postal_code
        time.sleep(0.08)

    if ordered:
        origin = urllib.parse.quote_plus(_fmt_addr(start_pc))
        destination = urllib.parse.quote_plus(_fmt_addr(ordered[-1].postal_code))
        waypoints = "|".join(urllib.parse.quote_plus(_fmt_addr(s.postal_code)) for s in ordered[:-1])
        url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}&travelmode={mode.lower()}"
        if waypoints:
            url += f"&waypoints={waypoints}"
    else:
        url = ""
    return RouteResult(ordered, legs, round(total_min,1), round(total_km,2), url)

def apply_priority_updates(stops: List[Stop], updates: list[dict]) -> List[Stop]:
    pc_to_stop = {s.postal_code: s for s in stops}
    for u in updates:
        pc = str(u.get("postal_code","")).strip()
        pr = str(u.get("priority","normal")).lower().strip()
        if pc in pc_to_stop and pr in {"urgent","high","normal","low"}:
            pc_to_stop[pc].priority = pr
    return list(pc_to_stop.values())