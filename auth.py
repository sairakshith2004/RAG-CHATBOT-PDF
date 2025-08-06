import streamlit as st
import json
import os
import hashlib
from datetime import datetime

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def apply_auth_styles():
    """Apply custom CSS styles for the authentication page"""
    st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin-top: 3rem;
    }
    
    /* Title styling */
    .auth-title {
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .auth-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Form styling - transparent input boxes with clear text */
    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 16px;
        color: white !important;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.6) !important;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 15px rgba(255,255,255,0.3);
        background-color: rgba(255,255,255,0.15) !important;
        border: 2px solid rgba(255,255,255,0.5) !important;
        color: white !important;
        outline: none;
    }
    
    /* Input labels styling */
    .stTextInput > label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        margin-bottom: 8px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255,107,107,0.3);
    }
    
    # Tab styling - make text more visible, bold and bright
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 700 !important;
        font-size: 18px !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.3);
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        font-weight: 800 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
        background-color: rgba(255,255,255,0.15);
        transform: translateY(-1px);
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    /* Welcome message styling */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin: 2rem auto;
        max-width: 500px;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .welcome-message {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
    
    .logout-btn {
        background: rgba(255,255,255,0.2);
        color: white;
        border: 2px solid white;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .logout-btn:hover {
        background: white;
        color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

def show_welcome_header():
    """Display compact welcome header for authenticated users"""
    st.markdown("""
    <style>
    .welcome-header {
        position: fixed;
        top: 10px;
        right: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 10px 20px;
        color: white;
        z-index: 999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 15px;
        font-size: 14px;
    }
    
    .welcome-text {
        margin: 0;
        font-weight: 500;
    }
    
    .logout-btn-header {
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .logout-btn-header:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create the header HTML
    st.markdown(f"""
    <div class="welcome-header">
        <p class="welcome-text">👋 Hello, {st.session_state['user']}!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button in sidebar or bottom
    with st.sidebar:
        st.markdown("### User Account")
        st.write(f"Logged in as: **{st.session_state['user']}**")
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def validate_input(username, password, is_signup=False):
    """Validate user input"""
    errors = []
    
    if not username.strip():
        errors.append("Username is required")
    elif len(username.strip()) < 3:
        errors.append("Username must be at least 3 characters long")
    elif not username.replace('_', '').replace('-', '').isalnum():
        errors.append("Username can only contain letters, numbers, hyphens, and underscores")
    
    if not password:
        errors.append("Password is required")
    elif len(password) < 6:
        errors.append("Password must be at least 6 characters long")
    elif is_signup:
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
    
    return errors

def login():
    """Display professional login/signup interface"""
    st.session_state['authenticated'] = False
    
    # Apply custom styles
    apply_auth_styles()
    
    # Header
    st.markdown("""
    <div class="auth-container">
        <h1 class="auth-title">📄 Welcome PDF AI</h1>
        <p class="auth-subtitle">Please sign in to your account or create a new one</p>
    </div>
    """, unsafe_allow_html=True)
    
    users = load_users()
    
    # Create tabs
    login_tab, signup_tab = st.tabs(["🔑 Sign In", "📝 Create Account"])
    
    with login_tab:
        st.markdown("### Sign In to Your Account")
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                login_submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if login_submitted:
                errors = validate_input(username, password)
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                elif username in users and users[username]["password"] == hash_password(password):
                    st.success("✅ Login successful! Redirecting...")
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = username
                    st.session_state['login_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
    
    with signup_tab:
        st.markdown("### Create Your Account")
        
        with st.form("signup_form", clear_on_submit=True):
            new_user = st.text_input("👤 Choose Username", placeholder="Enter a unique username")
            new_pass = st.text_input("🔒 Choose Password", type="password", placeholder="Create a strong password")
            confirm_pass = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
            
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                signup_submitted = st.form_submit_button("Create Account", use_container_width=True)
            
            if signup_submitted:
                errors = validate_input(new_user, new_pass, is_signup=True)
                
                if new_pass != confirm_pass:
                    errors.append("Passwords do not match")
                
                if new_user in users:
                    errors.append("Username already exists")
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    users[new_user] = {
                        "password": hash_password(new_pass),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_users(users)
                    st.success("✅ Account created successfully! Please sign in.")
                    st.balloons()

def require_auth():
    """Check authentication and redirect if needed"""
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        login()
        st.stop()
    else:
        show_welcome_header()  # Show compact header instead of full page

# Example usage
if __name__ == "__main__":
    st.set_page_config(
        page_title="Welcome PDF AI - Login",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    require_auth()
    
    # Your main app content goes here after authentication
    st.markdown("## 🎉 Main Application Content")
    st.write("This is where your main application would be displayed after successful authentication.")