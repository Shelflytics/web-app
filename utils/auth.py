import streamlit as st
import pyotp, qrcode
from io import BytesIO
import base64
from extra_streamlit_components import CookieManager   

# Defaults; can be overridden via .streamlit/secrets.toml
MASTER_PASSWORD = st.secrets.get("MASTER_PASSWORD", "ILOVEAAP")
SECRET_KEY = st.secrets.get("SECRET_KEY", "JBSWY3DPEHPK3PXP")

# --- Cookie settings ---
cookie_manager = CookieManager()              
COOKIE_KEY = "sku_admin_auth"                 
COOKIE_MAX_AGE = 60 * 60 * 12                 # 12 hours 

def _qr_png_b64(secret: str) -> str:
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name="user@sku-admin-portal", issuer_name="SKU Admin Portal"
    )
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def is_authenticated() -> bool:                
    """Check session flag or auth cookie."""
    if st.session_state.get("authenticated"):
        return True
    token = cookie_manager.get(COOKIE_KEY)
    if token == "true":
        st.session_state.authenticated = True
        return True
    return False

def login_ui():
    st.title("🔐 Admin Login")
    with st.form("login_form"):
        password = st.text_input("Master Password", type="password")
        token = st.text_input("2FA Code (6 digits)")
        submit = st.form_submit_button("Login")
        if submit:
            totp = pyotp.TOTP(SECRET_KEY)
            if password == MASTER_PASSWORD and totp.verify(token):
                st.session_state.authenticated = True
                # Persist for 12h (adjust as needed)
                cookie_manager.set(COOKIE_KEY, "true", max_age=COOKIE_MAX_AGE, key="auth_set")
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials or 2FA code")

def setup_2fa_ui():
    st.title("📱 Setup 2FA")
    st.write("Scan this QR code with Google Authenticator, Authy, etc.")
    b64 = _qr_png_b64(SECRET_KEY)
    st.image(f"data:image/png;base64,{b64}")
    st.write(f"**Manual key:** `{SECRET_KEY}`")
    st.caption("After setting up, go back to Login and authenticate.")

def require_auth():
    # Hide default nav + sidebar if not logged in
    from utils.components import hide_entire_sidebar, hide_default_pages_nav
    hide_default_pages_nav()
    if not is_authenticated():
        hide_entire_sidebar()
        st.info("Please login first.")
        st.stop()

def logout_button(sidebar=True):
    area = st.sidebar if sidebar else st
    if area.button("Logout"):
        cookie_manager.delete(COOKIE_KEY, key="auth_del")
        st.session_state.authenticated = False
        st.rerun()
