import streamlit as st
import httpx # Required for API calls from UI

# Chatbot UI Logic
def render_chatbot_page():
    # Authentication Check
    # Redirect to the main app (login page) if not authenticated
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        st.warning("Please log in to access the chatbot.")
        # Use st.switch_page to redirect to the main app.py (login page)
        st.switch_page("app.py") 
        return # Stop execution of this function if not authenticated

    # Sidebar with logout
    with st.sidebar:
        st.write(f"Welcome! You are logged in.")

        # Navigation links using st.page_link
        st.page_link("app.py", label="📊 Dashboard") 
        st.page_link("pages/chatbot_page.py", label="💬 Chatbot") 
        st.page_link("pages/predict_page.py", label="📈 Predict Item Performance")

        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
            
    st.title("🤖 Outlet Performance Chatbot")
    st.write("Ask questions about the outlet performance data.")

    # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    # This area will automatically scroll if content overflows the page
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box always stays at the bottom of the page
    if prompt := st.chat_input("Ask the chatbot about your data..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Point to port 8000 where main.py (Gemini chatbot) is running
            response = httpx.post("http://localhost:8000/chat", json={"message": prompt}, timeout=60.0) 
            response.raise_for_status()

            chatbot_response = response.json().get("response", "Sorry, I couldn't process that query.")
            
            if "error" in chatbot_response:
                chatbot_response = f"Error: {chatbot_response['error']}"

            with st.chat_message("assistant"):
                st.markdown(chatbot_response)
            st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

        except httpx.ConnectError:
            st.error("Could not connect to the FastAPI backend (main.py). Please make sure it is running on port 8000.")
        except httpx.TimeoutException: 
            st.error("The request to the chatbot timed out. Please try again or rephrase your query.")
        except httpx.HTTPStatusError as e:
            st.error(f"Error from FastAPI backend (main.py): {e.response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    render_chatbot_page()