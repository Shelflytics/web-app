from __future__ import annotations
import requests
from typing import Any

BASE_URL = "https://model-deployment-963934256033.asia-southeast1.run.app"
RECOMMEND_PATH = "/api/meow"  # Only endpoint we will call

class APIError(Exception):
    pass

def get_sku_recommendations(postal_code: str, top_k: int = 10) -> list[dict] | list[str] | list[Any]:
    """Call the deployed FastAPI model endpoint and return parsed results.

    The endpoint expects JSON:
        { "postal_code": "string", "top_k": 10 }

    Returns:
        A list of SKUs or a list of dicts with ranking info, best-effort parsed.
    """
    url = BASE_URL + RECOMMEND_PATH
    payload = { "postal_code": postal_code, "top_k": int(top_k) }
    headers = { "Content-Type": "application/json", "Accept": "application/json" }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        raise APIError(f"Network error calling model API: {e}") from e

    if not resp.ok:
        # Try to show FastAPI validation error detail if present
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise APIError(f"API returned {resp.status_code}: {detail}")

    # The API may return a JSON string, list, or object; attempt to parse smartly.
    try:
        data = resp.json()
    except ValueError:
        data = resp.text

    # Normalize to list rows suitable for display
    # If it's a string, try to parse as JSON again; else wrap it
    if isinstance(data, str):
        # Some backends return a JSON string as the root
        try:
            import json
            data = json.loads(data)
        except Exception:
            # Fallback: return single-item list for display
            data = [data]

    return data