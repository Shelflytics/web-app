import streamlit as st
import requests
import base64
import pandas as pd
from io import BytesIO
from datetime import datetime
from utils.auth import require_auth, logout_button
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from utils.auth import require_auth, logout_button
from utils.components import app_header, hide_default_pages_nav
from utils.db import get_outlets
from utils.routing import Stop, greedy_route, geocode_pc

# Configure your SKU detection API endpoint here
SKU_API_URL = "http://localhost:8000/detect/upload?confidence_threshold=0.3"  # <-- replace with real endpoint

st.set_page_config(page_title="Detector • Admin", page_icon="👁️", layout="wide")
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
    st.page_link("pages/sku_detection.py", label="👁️ Detector")
    logout_button()

# Initialize history in session state
if "sku_history" not in st.session_state:
    st.session_state.sku_history = []


st.title("👁️ SKU Detection")

st.write("Upload an image containing SKUs. The image will be sent to the SKU detection API and results will be shown below.")

with st.form("sku_form"):
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    submit = st.form_submit_button("Detect")

if submit:
    if uploaded_file is None:
        st.error("Please upload an image file first.")
    else:
        try:
            # Try multiple form field names to match the API's expected multipart field
            file_tuple = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            field_names = ["file", "image"]
            attempts = []
            resp = None
            data = None

            with st.spinner("Sending image to SKU detection API..."):
                for field in field_names:
                    attempt = {"field": field}
                    files = {field: file_tuple}
                    try:
                        r = requests.post(SKU_API_URL, files=files, timeout=60)
                    except Exception as e:
                        attempt["error"] = str(e)
                        attempts.append(attempt)
                        continue

                    attempt["status_code"] = r.status_code
                    # capture response body (json if possible)
                    try:
                        attempt["body"] = r.json()
                    except Exception:
                        attempt["body"] = r.text

                    attempts.append(attempt)

                    if r.ok:
                        resp = r
                        break

            # If no successful response, show diagnostics
            if resp is None:
                st.error("Request did not succeed with any tested form field names.")
                st.write("Attempts:")
                st.json(attempts)
                data = None
            else:
                # Successful response
                try:
                    data = resp.json()
                except Exception:
                    st.error(f"Response is not valid JSON (status {resp.status_code}).")
                    st.write(resp.text)
                    data = None

        except Exception as e:
            st.error(f"Request failed: {e}")
            data = None

        if data:
            # Expected keys: detections, num_detections, image_base64
            num = data.get("num_detections", None)
            detections = data.get("detections", [])
            image_b64 = data.get("image_base64", None)

            st.subheader("Results")
            if num is not None:
                st.metric("Number of detections", num)
            if image_b64:
                try:
                    # Handle data URI or raw base64
                    if isinstance(image_b64, str) and image_b64.startswith("data:"):
                        image_b64 = image_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(image_b64)
                    st.image(img_bytes, caption="Detected SKUs", use_column_width=True)
                except Exception:
                    st.info("Returned image could not be decoded and displayed.")

            # Save to history (newest first)
            entry = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "filename": uploaded_file.name,
                "num_detections": int(num) if num is not None else 0,
            }
            st.session_state.sku_history.insert(0, entry)

# Show history table and CSV export
st.write("### Detection History")
if len(st.session_state.sku_history) == 0:
    st.info("No detections yet.")
else:
    df = pd.DataFrame(st.session_state.sku_history)
    st.dataframe(df)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export history to CSV",
        data=csv_bytes,
        file_name="sku_detections_history.csv",
        mime="text/csv",
    )
