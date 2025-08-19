import streamlit as st
import multiprocessing
import sys
import os
import time # Import time for potential small startup delay
# from utils.server import run_server

from utils.auth import login_ui, setup_2fa_ui, is_authenticated
from utils.components import hide_default_pages_nav, hide_entire_sidebar

# Function to start the FastAPI server in a background process
def start_backend_server():
    """Function to start the FastAPI server in a background process."""
    try:
        # Check if the process is already running to avoid restarting
        if 'backend_process' in st.session_state and st.session_state.backend_process.is_alive():
            print("DEBUG: Backend server process already running.")
            return

        print("DEBUG: Attempting to start backend server process...")
        # target should be the function that starts the Uvicorn server
        process = multiprocessing.Process(target=run_server)
        process.daemon = True  # Ensures the process dies when the main Streamlit process dies
        process.start()
        st.session_state.backend_process = process
        st.session_state.servers_started = True
        print(f"DEBUG: Backend server process started with PID: {process.pid}")

        # Add a small delay to give the server time to start
        time.sleep(3) # Wait 3 seconds for server to initialize
        print("DEBUG: Backend server startup delay complete.")

    except Exception as e:
        st.error(f"Failed to start backend server: {e}")
        print(f"CRITICAL ERROR: Failed to start backend server process: {e}")

# This block ensures that the servers are started only once when the app runs
if __name__ == "__main__":
    # Ensure servers are started when the app launches for the first time
    if 'servers_started' not in st.session_state or not st.session_state.get('backend_process', False) or not st.session_state.backend_process.is_alive():
        start_backend_server()

    st.set_page_config(page_title="SKU Admin Portal", page_icon="📦", layout="wide", initial_sidebar_state="collapsed")
    hide_default_pages_nav()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not is_authenticated():
        hide_entire_sidebar()
        tab1, tab2 = st.tabs(["Login", "Setup 2FA"])
        with tab1: login_ui()
        with tab2: setup_2fa_ui()
    else:
        st.switch_page("pages/1_Home.py")

