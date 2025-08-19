import streamlit as st
import httpx

from utils.auth import logout_button
from utils.components import hide_default_pages_nav

hide_default_pages_nav()

def render_chatbot_page():
    with st.sidebar:
        st.image("assets/shelflytics_logo_transparent_white.png")
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
        st.page_link("pages/policy_faq.py", label="❓ Policy FAQ")
        logout_button()

    # Authentication Check
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        st.warning("Please log in to access the chatbot.")
        st.switch_page("app.py") 
        return
            
    st.title("🤖 Outlet Performance Chatbot")
    st.write("Ask questions about the outlet performance data.")

    # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box
    if prompt := st.chat_input("Ask the chatbot about your data..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Now pointing to the unified server on port 8000
            response = httpx.post("http://localhost:8000/chat", json={"message": prompt}, timeout=60.0) 
            response.raise_for_status()

            chatbot_response = response.json().get("response", "Sorry, I couldn't process that query.")
            
            if "error" in chatbot_response:
                chatbot_response = f"Error: {chatbot_response['error']}"

            with st.chat_message("assistant"):
                st.markdown(chatbot_response)
            st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

        except httpx.ConnectError:
            st.error("Could not connect to the unified server. Please make sure it is running on port 8000.")
        except httpx.TimeoutException: 
            st.error("The request to the chatbot timed out. Please try again or rephrase your query.")
        except httpx.HTTPStatusError as e:
            st.error(f"Error from unified server: {e.response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    render_chatbot_page()