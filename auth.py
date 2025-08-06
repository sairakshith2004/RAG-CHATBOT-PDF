import streamlit as st
import json
import os
import hashlib

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login():
    st.session_state['authenticated'] = False
    users = load_users()
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        st.header("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if username in users and users[username]["password"] == hash_password(password):
                st.success("Logged in successfully!")
                st.session_state['authenticated'] = True
                st.session_state['user'] = username
                st.rerun()
            else:
                st.error("Invalid username or password")

    with signup_tab:
        st.header("Sign Up")
        new_user = st.text_input("Choose a username", key="signup_user")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            if new_user in users:
                st.warning("Username already exists")
            elif len(new_user) < 3 or len(new_pass) < 3:
                st.warning("Username and password must be at least 3 characters")
            else:
                users[new_user] = {"password": hash_password(new_pass)}
                save_users(users)
                st.success("Account created! Please login.")

def require_auth():
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        login()
        st.stop()