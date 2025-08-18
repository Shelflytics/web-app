import streamlit as st
import httpx
import datetime
from utils.auth import logout_button
from utils.components import hide_default_pages_nav

hide_default_pages_nav()

# Modal dialog for prediction
@st.dialog("Item Performance Classification")
def _prediction_dialog():
    """
    Modal that runs the prediction (shows a spinner), then displays ONLY the prediction.
    Clicking OK closes the modal and reloads the page.
    """
    if st.session_state.get("prediction_result") is None:
        with st.spinner("Classifying… Please wait"):
            try:
                resp = httpx.post(
                    "http://localhost:8000/predict_item_sales",  # Updated port to 8000
                    json=st.session_state["pred_payload"],
                    timeout=120.0,
                )
                resp.raise_for_status()
                st.session_state["prediction_result"] = resp.json()
                st.rerun()
            except httpx.ConnectError:
                st.error("Could not connect to the classification service.")
            except httpx.TimeoutException:
                st.error("The classification request timed out.")
            except httpx.HTTPStatusError as e:
                st.error(f"Classification API error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

            if st.button("Close"):
                st.session_state["show_dialog"] = False
                st.session_state["prediction_result"] = None
                st.session_state.pop("pred_payload", None)
                st.rerun()
        return

    pr = st.session_state["prediction_result"]
    st.subheader("Result")
    st.markdown(f"**Prediction:** {pr.get('prediction_label', '—')}")

    if st.button("OK"):
        st.session_state["show_dialog"] = False
        st.session_state["prediction_result"] = None
        st.session_state.pop("pred_payload", None)
        st.session_state.selected_item_id = "<New Item>"
        st.session_state.selected_outlet_id = "<New Outlet>"
        st.session_state.is_existing_item_selected = False
        st.session_state.is_existing_outlet_selected = False
        st.session_state.input_item_state = {
            "item_identifier_value": "",
            "item_weight_value": 15.0,
            "item_fat_content_value": "Low Fat",
            "item_visibility_value": 0.06,
            "item_type_value": "Dairy",
            "item_mrp_value": 150.0,
            "item_avg_sales_value": 1000.0,
        }
        current_year = datetime.datetime.now().year
        st.session_state.input_outlet_state = {
            "outlet_identifier_value": "",
            "outlet_establishment_year_value": current_year - 10,
            "outlet_size_value": "Medium",
            "outlet_location_type_value": "Tier 1",
            "outlet_type_value": "Supermarket Type1",
            "outlet_age_value": 10,
        }
        st.rerun()

def fetch_identifiers(endpoint):
    """Fetches list of identifiers from unified server"""
    try:
        response = httpx.get(f"http://localhost:8000/{endpoint}", timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Failed to fetch {endpoint} from backend: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching {endpoint}: {e}")
        return []

def fetch_entity_info(endpoint, identifier):
    """Fetches detailed info for a specific item or outlet."""
    if not identifier:
        return None
    try:
        response = httpx.get(f"http://localhost:8000/{endpoint}/{identifier}", timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            st.warning(f"Identifier '{identifier}' not found in database. You can proceed to add it.")
            return {}
        else:
            st.error(f"Failed to fetch {endpoint} info for {identifier}: {e.response.text}")
            return None
    except httpx.RequestError as e:
        st.error(f"Could not connect to backend to fetch {endpoint} info for {identifier}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching {endpoint} info for {identifier}: {e}")
        return None

def render_prediction_page():
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        st.warning("Please log in to access the prediction page.")
        st.switch_page("app.py") 
        return

    with st.sidebar:
        st.image("assets\shelflytics_logo_transparent_white.png")
        st.page_link("pages/1_Home.py", label="🏠 Home")
        st.page_link("pages/2_SKUs.py", label="📦 SKUs")
        st.page_link("pages/3_Outlets.py", label="🏬 Outlets")
        st.page_link("pages/4_SKU_Recommender.py", label="🤖 Recommender")
        st.page_link("pages/6_Routes.py", label="🗺️ Routes")
        st.page_link("pages/5_Settings.py", label="⚙️ Settings")
        st.page_link("pages/7_Merchandisers.py", label="🧑‍🤝‍🧑 Merchandisers")
        st.page_link("pages/chatbot_page.py", label="💬 Chatbot") 
        st.page_link("pages/predict_page.py", label="📈 Predict Item Performance")
        st.page_link("pages/sku_detection.py", label="👁️ Detector") 
        logout_button()
    
    st.title("📈 Item Performance Classification")
    st.write("Enter the details below to classify the sales performance of an item.")

    current_year = datetime.datetime.now().year

    # Ensure dialog session keys exist
    st.session_state.setdefault("show_dialog", False)
    st.session_state.setdefault("prediction_result", None)

    if st.session_state.get("show_dialog"):
        _prediction_dialog()

    # Fetch lists using unified server endpoints
    item_identifiers = ["<New Item>"] + fetch_identifiers("items_list")
    outlet_identifiers = ["<New Outlet>"] + fetch_identifiers("outlets_list")

    # Initialize session state
    if 'selected_item_id' not in st.session_state:
        st.session_state.selected_item_id = "<New Item>"
    if 'selected_outlet_id' not in st.session_state:
        st.session_state.selected_outlet_id = "<New Outlet>"

    default_item_state = {
        "item_identifier_value": "",
        "item_weight_value": 15.0,
        "item_fat_content_value": "Low Fat",
        "item_visibility_value": 0.06,
        "item_type_value": "Dairy",
        "item_mrp_value": 150.0,
        "item_avg_sales_value": 1000.0
    }
    default_outlet_state = {
        "outlet_identifier_value": "",
        "outlet_establishment_year_value": current_year - 10,
        "outlet_size_value": "Medium",
        "outlet_location_type_value": "Tier 1",
        "outlet_type_value": "Supermarket Type1",
        "outlet_age_value": 10
    }

    if 'input_item_state' not in st.session_state:
        st.session_state.input_item_state = default_item_state.copy()
    if 'input_outlet_state' not in st.session_state:
        st.session_state.input_outlet_state = default_outlet_state.copy()
    if 'is_existing_item_selected' not in st.session_state:
        st.session_state.is_existing_item_selected = False
    if 'is_existing_outlet_selected' not in st.session_state:
        st.session_state.is_existing_outlet_selected = False

    # Callbacks for selectbox changes
    def on_item_select_callback():
        st.session_state.selected_item_id = st.session_state.item_selectbox_key
        if st.session_state.selected_item_id != "<New Item>":
            item_info = fetch_entity_info("item_info", st.session_state.selected_item_id)
            if item_info:
                st.session_state.is_existing_item_selected = True
                st.session_state.input_item_state.update({
                    "item_identifier_value": item_info.get("item_identifier", ""),
                    "item_weight_value": float(item_info.get("item_weight", 15.0)),
                    "item_fat_content_value": item_info.get("item_fat_content", "Low Fat"),
                    "item_visibility_value": float(item_info.get("item_visibility", 0.06)),
                    "item_type_value": item_info.get("item_type", "Dairy"),
                    "item_mrp_value": float(item_info.get("item_mrp", 150.0)),
                    "item_avg_sales_value": float(item_info.get("item_avg_sales", 1000.0))
                })
            else:
                st.session_state.is_existing_item_selected = False
                st.session_state.input_item_state.update(default_item_state.copy())
                st.session_state.input_item_state["item_identifier_value"] = st.session_state.selected_item_id
        else:
            st.session_state.is_existing_item_selected = False
            st.session_state.input_item_state.update(default_item_state.copy())
        st.rerun()

    def on_outlet_select_callback():
        st.session_state.selected_outlet_id = st.session_state.outlet_selectbox_key
        if st.session_state.selected_outlet_id != "<New Outlet>":
            outlet_info = fetch_entity_info("outlet_performance", st.session_state.selected_outlet_id)
            if outlet_info:
                st.session_state.is_existing_outlet_selected = True
                fetched_outlet_age = int(outlet_info.get("outlet_age", 10))
                fetched_establishment_year = int(outlet_info.get("outlet_establishment_year", current_year - fetched_outlet_age))
                st.session_state.input_outlet_state.update({
                    "outlet_identifier_value": outlet_info.get("outlet_identifier", ""),
                    "outlet_establishment_year_value": fetched_establishment_year,
                    "outlet_size_value": outlet_info.get("outlet_size", "Medium"),
                    "outlet_location_type_value": outlet_info.get("outlet_location_type", "Tier 1"),
                    "outlet_type_value": outlet_info.get("outlet_type", "Supermarket Type1"),
                    "outlet_age_value": fetched_outlet_age
                })
            else:
                st.session_state.is_existing_outlet_selected = False
                st.session_state.input_outlet_state.update(default_outlet_state.copy())
                st.session_state.input_outlet_state["outlet_identifier_value"] = st.session_state.selected_outlet_id
        else:
            st.session_state.is_existing_outlet_selected = False
            st.session_state.input_outlet_state.update(default_outlet_state.copy())
        st.rerun()

    # Item section
    st.subheader("Item Details")
    col_item_id, col_item_type_selector = st.columns([0.6, 0.4])
    with col_item_type_selector:
        st.selectbox(
            "Select Existing Item",
            options=item_identifiers,
            index=item_identifiers.index(st.session_state.selected_item_id) if st.session_state.selected_item_id in item_identifiers else 0,
            key="item_selectbox_key",
            on_change=on_item_select_callback,
            help="Choose an existing item or select '<New Item>' to enter new details."
        )
    with col_item_id:
        item_identifier_value = st.text_input(
            "Item Identifier",
            value=st.session_state.input_item_state["item_identifier_value"],
            placeholder="e.g., FDX07", max_chars=10,
            help="Unique ID of the item.",
            key="item_identifier_input_key",
            disabled=st.session_state.is_existing_item_selected,
        )

    item_weight_value = st.number_input(
        "Item Weight (kg)",
        min_value=1.0, max_value=30.0, value=st.session_state.input_item_state["item_weight_value"], step=0.1,
        format="%.2f", help="Weight of the item in kg",
        key="pred_item_weight_key",
        disabled=st.session_state.is_existing_item_selected,
        on_change=lambda: st.session_state.input_item_state.update({"item_weight_value": st.session_state.pred_item_weight_key})
    )

    item_fat_content_value = st.selectbox(
        "Item Fat Content",
        options=["Low Fat", "Regular", "LF"],
        index=["Low Fat", "Regular", "LF"].index(st.session_state.input_item_state["item_fat_content_value"]),
        help="Fat content category of the item",
        key="pred_item_fat_content_key",
        disabled=st.session_state.is_existing_item_selected,
        on_change=lambda: st.session_state.input_item_state.update({"item_fat_content_value": st.session_state.pred_item_fat_content_key})
    )

    item_visibility_value = st.number_input(
        "Item Visibility",
        min_value=0.0, max_value=1.0, value=st.session_state.input_item_state["item_visibility_value"], step=0.001,
        format="%.3f", help="Proportion of total display area allocated to the item",
        key="pred_item_visibility_key",
        on_change=lambda: st.session_state.input_item_state.update({"item_visibility_value": st.session_state.pred_item_visibility_key})
    )

    item_type_value = st.selectbox(
        "Item Type",
        options=[
            "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
            "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
            "Health and Hygiene", "Hard Drinks", "Canned", "Breads",
            "Starchy Foods", "Others", "Seafood"
        ],
        index=[
            "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
            "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
            "Health and Hygiene", "Hard Drinks", "Canned", "Breads",
            "Starchy Foods", "Others", "Seafood"
        ].index(st.session_state.input_item_state["item_type_value"]),
        help="Category of the item",
        key="pred_item_type_key",
        disabled=st.session_state.is_existing_item_selected,
        on_change=lambda: st.session_state.input_item_state.update({"item_type_value": st.session_state.pred_item_type_key})
    )

    item_mrp_value = st.number_input(
        "Item MRP ($)",
        min_value=50.0, max_value=300.0, value=st.session_state.input_item_state["item_mrp_value"], step=0.01,
        format="%.2f", help="Maximum Retail Price of the item",
        key="pred_item_mrp_key",
        on_change=lambda: st.session_state.input_item_state.update({"item_mrp_value": st.session_state.pred_item_mrp_key})
    )

    item_outlet_sales_value = st.number_input(
        "Item Outlet Sales ($)",
        min_value=0.0, max_value=20000.0, value=0.0, step=10.0, format="%.2f",
        help="Sales of THIS item at THIS outlet for this observation.",
        key="pred_item_outlet_sales_key"
    )

    # Outlet section
    st.subheader("Outlet Details")
    col_outlet_id, col_outlet_type_selector = st.columns([0.6, 0.4])
    with col_outlet_type_selector:
        st.selectbox(
            "Select Existing Outlet",
            options=outlet_identifiers,
            index=outlet_identifiers.index(st.session_state.selected_outlet_id) if st.session_state.selected_outlet_id in outlet_identifiers else 0,
            key="outlet_selectbox_key",
            on_change=on_outlet_select_callback,
            help="Choose an existing outlet or select '<New Outlet>' to enter new details."
        )
    with col_outlet_id:
        outlet_identifier_value = st.text_input(
            "Outlet Identifier",
            value=st.session_state.input_outlet_state["outlet_identifier_value"],
            placeholder="e.g., OUT027", max_chars=10,
            help="Unique ID of the outlet.",
            key="outlet_identifier_input_key",
            disabled=st.session_state.is_existing_outlet_selected,
        )

    if st.session_state.is_existing_outlet_selected:
        st.write(f"**Outlet Age:** {st.session_state.input_outlet_state['outlet_age_value']} years")
        st.info(f"Outlet established in **{st.session_state.input_outlet_state['outlet_establishment_year_value']}**")
        outlet_establishment_year_for_payload = st.session_state.input_outlet_state["outlet_establishment_year_value"]
    else:
        outlet_establishment_year_for_payload = st.number_input(
            "Outlet Establishment Year",
            min_value=1900, max_value=current_year, value=st.session_state.input_outlet_state["outlet_establishment_year_value"], step=1,
            help="Year the outlet was established.",
            key="pred_outlet_establishment_year_key",
            on_change=lambda: st.session_state.input_outlet_state.update({"outlet_establishment_year_value": st.session_state.pred_outlet_establishment_year_key})
        )
        calculated_age = current_year - outlet_establishment_year_for_payload
        st.info(f"Calculated Outlet Age: **{calculated_age} years**")

    outlet_size_value = st.selectbox(
        "Outlet Size",
        options=["Medium", "Small", "High"],
        index=["Medium", "Small", "High"].index(st.session_state.input_outlet_state["outlet_size_value"]),
        help="Size of the outlet",
        key="pred_outlet_size_key",
        disabled=st.session_state.is_existing_outlet_selected,
        on_change=lambda: st.session_state.input_outlet_state.update({"outlet_size_value": st.session_state.pred_outlet_size_key})
    )
    outlet_location_type_value = st.selectbox(
        "Outlet Location Type",
        options=["Tier 1", "Tier 2", "Tier 3"],
        index=["Tier 1", "Tier 2", "Tier 3"].index(st.session_state.input_outlet_state["outlet_location_type_value"]),
        help="Type of location where the outlet is situated",
        key="pred_outlet_location_type_key",
        disabled=st.session_state.is_existing_outlet_selected,
        on_change=lambda: st.session_state.input_outlet_state.update({"outlet_location_type_value": st.session_state.pred_outlet_location_type_key})
    )
    outlet_type_value = st.selectbox(
        "Outlet Type",
        options=["Supermarket Type1", "Supermarket Type2", "Grocery Store", "Supermarket Type3"],
        index=["Supermarket Type1", "Supermarket Type2", "Grocery Store", "Supermarket Type3"].index(st.session_state.input_outlet_state["outlet_type_value"]),
        help="Type of the outlet",
        key="pred_outlet_type_key",
        disabled=st.session_state.is_existing_outlet_selected,
        on_change=lambda: st.session_state.input_outlet_state.update({"outlet_type_value": st.session_state.pred_outlet_type_key})
    )

    # Submit button
    submit_button = st.button("Get Prediction")

    if submit_button:
        payload = {
            "Item_Identifier": st.session_state.input_item_state["item_identifier_value"],
            "Item_Weight": float(st.session_state.input_item_state["item_weight_value"]),
            "Item_Fat_Content": st.session_state.input_item_state["item_fat_content_value"],
            "Item_Visibility": float(st.session_state.input_item_state["item_visibility_value"]),
            "Item_Type": st.session_state.input_item_state["item_type_value"],
            "Item_MRP": float(st.session_state.input_item_state["item_mrp_value"]),

            "Outlet_Identifier": st.session_state.input_outlet_state["outlet_identifier_value"],
            "Outlet_Establishment_Year": int(outlet_establishment_year_for_payload),
            "Outlet_Size": st.session_state.input_outlet_state["outlet_size_value"],
            "Outlet_Location_Type": st.session_state.input_outlet_state["outlet_location_type_value"],
            "Outlet_Type": st.session_state.input_outlet_state["outlet_type_value"],

            "item_outlet_sales": float(item_outlet_sales_value),
            "always_add_new_rows": True
        }

        if not payload["Item_Identifier"] or not payload["Outlet_Identifier"]:
            st.error("Item Identifier and Outlet Identifier cannot be empty.")
        else:
            st.session_state["pred_payload"] = payload
            st.session_state["prediction_result"] = None
            st.session_state["show_dialog"] = True
            st.rerun()

if __name__ == "__main__":
    render_prediction_page()