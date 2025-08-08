# AI-Powered PDF Chat Assistant

This project is an AI-powered chatbot that allows you to upload PDF documents and ask questions about their content. It leverages Google Gemini (via LangChain) for question answering and FAISS for efficient document retrieval, all within a user-friendly Streamlit interface.

## Features

- Upload one or more PDF files and extract their content
- Ask questions about the uploaded PDFs and get AI-generated answers
- Handles large documents with adaptive chunking and progress tracking
- Caches vector stores for faster subsequent queries
- Download your conversation history as CSV or JSON
- Secure login system (customizable)
- Modern, responsive UI with custom HTML/CSS support

## Setup

1. **Create and activate a virtual environment:**
    ```sh
    python -m venv myenv
    myenv\Scripts\activate  # On Windows
    # or
    source myenv/bin/activate  # On Linux/Mac
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the app:**
    ```sh
    streamlit run app.py
    ```

## Usage

1. Log in using your credentials (or sign up if enabled).
2. Enter your [Google AI API Key](https://ai.google.dev/) in the sidebar.
3. Upload one or more PDF files.
4. Ask questions about the content of your PDFs.
5. Download your conversation history if needed.

## Project Structure


## Project Structure

- `app.py` — Main Streamlit application
- `auth/` — Authentication utilities and login/signup logic
- `auth.py` — Handles user authentication, session management, and access control for the app
- `login_signup.py` — Streamlit login and signup forms
- `auth_utils.py` — Helper functions for authentication
- `faiss_index/` — Stores FAISS vector index and cache files
- `images/` — UI images and screenshots
- `login.html`, `style.css` — Custom login page and styles
- `users.json` — Stores registered user credentials and metadata
- `requirements.txt
## Credits

Developed by [Sai Rakshith Talluru](https://www.linkedin.com/in/sairakshith-talluru-a69272265/)

---

## Quick Start

To run this project in your terminal, use:
```sh
streamlit run app.py
```
