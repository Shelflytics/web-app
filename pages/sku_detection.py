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
from PIL import Image

# Configure your SKU detection API endpoint here
SKU_API_URL = "http://localhost:5000/detect/upload"  # <-- replace with real endpoint

st.set_page_config(page_title="Detector • Admin", page_icon="👁️", layout="wide")
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


# Initialize history in session state
if "sku_history" not in st.session_state:
    st.session_state.sku_history = []


app_header("SKU Detector (RetinaNet)")

st.write("Upload an image containing SKUs. The image will be sent to the SKU detection API and results will be shown below.")

with st.form("sku_form"):
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    confidence = st.slider(
        "Select confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%0.2f",
    )
    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("More detections")
    with col_r:
        # add a narrow rightmost subcolumn so the caption sits flush to the far right
        spacer, rightmost = st.columns([2.5, 1])
        with rightmost:
            st.caption("Fewer detections")
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
                    attempt = {"field": field, "params": {"confidence_threshold": confidence}}
                    files = {field: file_tuple}
                    try:
                        # include confidence threshold as a query param
                        r = requests.post(
                            SKU_API_URL,
                            params={"confidence_threshold": confidence},
                            files=files,
                            timeout=60,
                        )
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
                    st.image(img_bytes, caption="Detected SKUs", use_container_width=True)

                    # Provide a download button that converts the returned image to JPG
                    try:
                        img_buf = BytesIO(img_bytes)
                        img = Image.open(img_buf).convert("RGB")
                        out_buf = BytesIO()
                        img.save(out_buf, format="JPEG", quality=95)
                        jpg_bytes = out_buf.getvalue()

                        base_name = (uploaded_file.name.rsplit(".", 1)[0]
                                     if (uploaded_file and "." in uploaded_file.name)
                                     else "detected_image")
                        file_name = f"{base_name}_detected.jpg"

                        st.download_button(
                            label="Download image as JPG",
                            data=jpg_bytes,
                            file_name=file_name,
                            mime="image/jpeg",
                        )
                    except Exception:
                        # Fallback: offer raw bytes if conversion fails
                        try:
                            raw_name = (uploaded_file.name
                                        if uploaded_file and uploaded_file.name
                                        else "detected_image.jpg")
                            st.download_button(
                                label="Download raw image",
                                data=img_bytes,
                                file_name=raw_name,
                                mime="application/octet-stream",
                            )
                        except Exception:
                            pass
                except Exception:
                    st.info("Returned image could not be decoded and displayed.")

            # Save to history (newest first)
            entry = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "filename": uploaded_file.name,
                "num_detections": int(num) if num is not None else 0,
                "confidence_threshold": float(confidence),
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
