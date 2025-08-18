# import streamlit as st
# import pyotp
# import qrcode
# from io import BytesIO
# import base64

# # Configuration
# MASTER_PASSWORD = "ILOVEAAP"
# SECRET_KEY = "JBSWY3DPEHPK3PXP"  # Base32 secret for 2FA (you can generate a new one)

# def generate_qr_code():
#     """Generate QR code for 2FA setup"""
#     totp_uri = pyotp.totp.TOTP(SECRET_KEY).provisioning_uri(
#         name="user@yourapp.com",
#         issuer_name="Your Streamlit App"
#     )
    
#     qr = qrcode.QRCode(version=1, box_size=10, border=5)
#     qr.add_data(totp_uri)
#     qr.make(fit=True)
    
#     img = qr.make_image(fill_color="black", back_color="white")
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     buf.seek(0)
    
#     return base64.b64encode(buf.getvalue()).decode()

# def verify_2fa(token):
#     """Verify 2FA token"""
#     totp = pyotp.TOTP(SECRET_KEY)
#     return totp.verify(token)

# def login_page():
#     """Display login page"""
#     st.title("🔐 Login")
    
#     with st.form("login_form"):
#         password = st.text_input("Master Password", type="password")
#         totp_token = st.text_input("2FA Code (6 digits)")
#         submit = st.form_submit_button("Login")
        
#         if submit:
#             if password == MASTER_PASSWORD and verify_2fa(totp_token):
#                 st.session_state.authenticated = True
#                 st.success("Login successful!")
#                 st.rerun()
#             else:
#                 st.error("Invalid credentials or 2FA code")

# def setup_2fa_page():
#     """Display 2FA setup page"""
#     st.title("📱 Setup 2FA")
#     st.write("Scan this QR code with your authenticator app (Google Authenticator, Authy, etc.)")
    
#     qr_code = generate_qr_code()
#     st.image(f"data:image/png;base64,{qr_code}")
    
#     st.write(f"**Manual entry key:** `{SECRET_KEY}`")
#     st.write("After setting up, use the login page to authenticate.")
    
#     if st.button("Go to Login"):
#         st.session_state.show_setup = False
#         st.rerun()

# def main_app():
#     """Main application after authentication"""
#     st.title("🚀 Your Streamlit App")
    
#     # Sidebar with logout
#     with st.sidebar:
#         st.write(f"Welcome! You are logged in.")
#         if st.button("Logout"):
#             st.session_state.authenticated = False
#             st.rerun()
    
#     # Main content
#     st.write("## Dashboard")
#     st.write("This is where your main application content will go.")
    
#     # Placeholder sections for future features
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.write("### FastAPI ML Models")
#         st.info("Connect to your Google Cloud deployed models here")
#         if st.button("Test ML Model Connection"):
#             st.write("🔄 This will connect to your FastAPI endpoints")
    
#     with col2:
#         st.write("### Database")
#         st.info("MySQL database connection will go here")
#         if st.button("Test Database Connection"):
#             st.write("🔄 This will test your MySQL connection")
    
#     # Sample data display
#     st.write("### Sample Data")
#     import pandas as pd
#     sample_data = pd.DataFrame({
#         'Name': ['Alice', 'Bob', 'Charlie'],
#         'Value': [10, 20, 30],
#         'Status': ['Active', 'Inactive', 'Active']
#     })
#     st.dataframe(sample_data)

# def main():
#     """Main application logic"""
#     # Initialize session state
#     if 'authenticated' not in st.session_state:
#         st.session_state.authenticated = False
#     if 'show_setup' not in st.session_state:
#         st.session_state.show_setup = False
    
#     # Navigation
#     if not st.session_state.authenticated:
#         # Show setup or login page
#         tab1, tab2 = st.tabs(["Login", "Setup 2FA"])
        
#         with tab1:
#             login_page()
        
#         with tab2:
#             setup_2fa_page()
#     else:
#         main_app()

# if __name__ == "__main__":
#     main()