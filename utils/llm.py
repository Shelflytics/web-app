from __future__ import annotations
import json
import streamlit as st
import google.generativeai as genai  

PRIORITY_MAP = {"urgent":3, "high":2, "normal":1, "low":0}

def _gemini() -> genai.GenerativeModel:
    api_key = st.secrets.get("gemini", {}).get("api_key")
    if not api_key:
        raise RuntimeError("Missing [gemini].api_key in .streamlit/secrets.toml")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")  # free plan friendly

def parse_route_instruction(instruction: str, outlets: list[dict]) -> dict:
    if not instruction.strip():
        return {"set_priority": []}
    model = _gemini()
    sys = (
        "You convert sales manager instructions into STRICT JSON for route priorities. "
        "Only return JSON with this shape and nothing else:\n"
        "{\\n  \\\"set_priority\\\": [\\n    {\\\"postal_code\\\": \\\"string\\\", "
        "\\\"priority\\\": \\\"urgent|high|normal|low\\\", \\\"reason\\\": \\\"string optional\\\"}\\n  ]\\n}\\n\n"
        "Allowed priorities: urgent, high, normal, low. If the text references a city or name, pick the best matching outlet from the provided list. "
        "Outlets you can reference (postal_code and label):\n"
    )
    outlet_lines = [f"- {o.get('label','')} :: {o.get('postal_code','')}" for o in outlets]
    content = sys + "\n".join(outlet_lines) + "\n\nInstruction: " + instruction.strip()

    try:
        resp = model.generate_content(content)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.strip("`\n")
            if text.lower().startswith("json\n"):
                text = text[5:]
        data = json.loads(text)
        items = data.get("set_priority", [])
        norm = []
        for it in items:
            pc = str(it.get("postal_code", "")).strip()
            pr = str(it.get("priority", "")).lower().strip()
            if pr not in PRIORITY_MAP:
                pr = "normal"
            if pc:
                norm.append({"postal_code": pc, "priority": pr, "reason": it.get("reason")})
        return {"set_priority": norm}
    except Exception:
        return {"set_priority": []}