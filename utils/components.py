import streamlit as st

def app_header(title: str = "SKU Admin Portal"):
    # Simple header with placeholder for a logo
    left, right = st.columns([1, 10])
    with left:
        st.image("assets\shelflytics_logo_no_word_transparent_white.png")
    with right:
        st.title(title)

def kpi_tile(label: str, value, help_text: str | None = None):
    st.metric(label, value, help=help_text)

def pill(text: str):
    st.markdown(
        f'<span style="background:#24262b;padding:6px 10px;border-radius:999px;font-size:12px;color:#E2E2E6;">{text}</span>',
        unsafe_allow_html=True
    )
    
def hide_default_pages_nav():
    # Hides the built-in “Pages” nav so only our custom sidebar shows
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] { display: none !important; }
        div[data-testid="stSidebarNav"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)

def hide_entire_sidebar():
    # Used pre-login to hide the whole sidebar
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)
