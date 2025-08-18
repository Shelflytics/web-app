import streamlit as st
import pyotp
import qrcode
from io import BytesIO
import base64

# Configuration
MASTER_PASSWORD = "ILOVEAAP"
SECRET_KEY = "JBSWY3DPEHPK3PXP"  # Base32 secret for 2FA (you can generate a new one)

def generate_qr_code():
    """Generate QR code for 2FA setup"""
    totp_uri = pyotp.totp.TOTP(SECRET_KEY).provisioning_uri(
        name="user@yourapp.com",
        issuer_name="Your Streamlit App"
    )
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode()

def verify_2fa(token):
    """Verify 2FA token"""
    totp = pyotp.TOTP(SECRET_KEY)
    return totp.verify(token)

def login_page():
    """Display login page"""
    st.title("🔐 Login")
    
    with st.form("login_form"):
        password = st.text_input("Master Password", type="password")
        totp_token = st.text_input("2FA Code (6 digits)")
        submit = st.form_submit_button("Login")
        
        if submit:
            if password == MASTER_PASSWORD and verify_2fa(totp_token):
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials or 2FA code")

def setup_2fa_page():
    """Display 2FA setup page"""
    st.title("📱 Setup 2FA")
    st.write("Scan this QR code with your authenticator app (Google Authenticator, Authy, etc.)")
    
    qr_code = generate_qr_code()
    st.image(f"data:image/png;base64,{qr_code}")
    
    st.write(f"**Manual entry key:** `{SECRET_KEY}`")
    st.write("After setting up, use the login page to authenticate.")
    
    if st.button("Go to Login"):
        st.session_state.show_setup = False
        st.rerun()

def main_app():
    """Main application after authentication"""
    st.title("📊 Dashboard")
    
    # Sidebar with logout
    with st.sidebar:
        st.write(f"Welcome! You are logged in.")

        # Navigation links using st.page_link
        # 'pages/chatbot_page.py' refers to the new chatbot page file.
        st.page_link("app.py", label="📊 Dashboard") 
        st.page_link("pages/chatbot_page.py", label="💬 Chatbot") 
        st.page_link("pages/predict_page.py", label="📈 Predict Item Performance")

        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    looker_studio_html = """
    <style>
        /* Ensure the iframe scales responsively */
        .iframe-container {
            position: relative;
            width: 100%;
            padding-bottom: 75%; /* Aspect Ratio 5:4 (height/width * 100). Adjust as needed. */
            height: 0;
            overflow: hidden;
            border-radius: 0.5rem;
        }
        .iframe-container iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
            border-radius: 0.5rem;
        }
    </style>
    <div class="iframe-container">
        <iframe
            src="https://lookerstudio.google.com/embed/reporting/2f87a2eb-d0c9-4827-ac47-f7e2cdcbedff/page/58qUF"
            frameborder="0"
            allowfullscreen
            referrerpolicy="strict-origin-when-cross-origin"
            aria-label="Looker Studio Embedded Report"
        ></iframe>
    </div>
    """
    st.markdown(looker_studio_html, unsafe_allow_html=True)

def main():
    """Main application logic"""
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'show_setup' not in st.session_state:
        st.session_state.show_setup = False
    
    # Navigation
    if not st.session_state.authenticated:
        # Show setup or login page
        tab1, tab2 = st.tabs(["Login", "Setup 2FA"])
        
        with tab1:
            login_page()
        
        with tab2:
            setup_2fa_page()
    else:
        main_app()

if __name__ == "__main__":
    main()