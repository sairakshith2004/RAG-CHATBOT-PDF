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
# 🤖 MyRAG Chatbot AI

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A smart and interactive **RAG (Retrieval-Augmented Generation) Chatbot**, built using **LangChain**, **OpenAI**, and **FAISS**, designed to answer domain-specific questions by combining LLM power with custom knowledge bases.

---

## 🚀 Features

- 🔍 Retrieval-based Question Answering
- 🧠 Powered by OpenAI / Local LLMs
- 🗃️ Vector Store (FAISS) for efficient knowledge retrieval
- 🌐 Hosted with Gradio / Streamlit / Render
- 💬 Clean chatbot UI with real-time responses

---

## 🔧 Tech Stack

- Python 🐍  
- LangChain  
- OpenAI / Hugging Face Transformers  
- FAISS (Vector DB)  
- Gradio / Streamlit (Frontend UI)  
- Hugging Face Spaces / Render / Streamlit Cloud (Hosting)

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/myrag-chatbot.git
cd myrag-chatbot
pip install -r requirements.txt
python app.py


# AI-Powered Document Chat Assistant

This project is an AI-powered chatbot that allows you to upload **PDF, PPTX, and DOCX documents** and ask questions about their content. It leverages **Google Gemini** for question answering and FAISS for efficient document retrieval, all within a user-friendly Streamlit interface.

## Features

- Upload one or more PDF, PowerPoint, or Word files and extract their content
- Automatic AI-generated summary/explanation of uploaded documents
- Ask questions about the uploaded documents and get detailed AI answers
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
    # Also install document libraries if not included:
    pip install python-pptx python-docx
    ```

3. **Run the app:**
    ```sh
    streamlit run RAG-CHATBOT-PDF-main/app.py
    ```

## Usage

1. Log in using your credentials (or sign up if enabled).
2. Enter your [Google AI API Key](https://ai.google.dev/) in the sidebar.
3. Upload one or more PDF, PPTX, or DOCX files.
4. View the automatic AI-generated summary of your documents.
5. Ask questions about the content of your documents.
6. Download your conversation history if needed.

## Project Structure

- `RAG-CHATBOT-PDF-main/app.py` — Main Streamlit application
- `auth/` — Authentication utilities and login/signup logic
- `auth.py` — Handles user authentication, session management, and access control for the app
- `login_signup.py` — Streamlit login and signup forms
- `auth_utils.py` — Helper functions for authentication
- `faiss_index/` — Stores FAISS vector index and cache files
- `images/` — UI images and screenshots
- `login.html`, `style.css` — Custom login page and styles
- `users.json` — Stores registered user credentials and metadata
- `requirements.txt` — Python dependencies

## Credits

Developed by [Sai Rakshith Talluru](https://www.linkedin.com/in/sairakshith-talluru-a69272265/)

---

## Quick Start

To run this project in your terminal, use:
```sh
streamlit run RAG-CHATBOT-PDF-main/app.py
```

---

# 🤖 MyRAG Chatbot AI

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A smart and interactive **RAG (Retrieval-Augmented Generation) Chatbot**, built using **LangChain**, **Google Gemini**, and **FAISS**, designed to answer domain-specific questions by combining LLM power with custom knowledge bases.

---

## 🚀 Features

- 🔍 Retrieval-based Question Answering
- 🧠 Powered by Google Gemini / Local LLMs
- 🗃️ Vector Store (FAISS) for efficient knowledge retrieval
- 🌐 Hosted with Streamlit
- 💬 Clean chatbot UI with real-time responses

---

## 🔧 Tech Stack

- Python 🐍  
- LangChain  
- Google Gemini / Hugging Face Transformers  
- FAISS (Vector DB)  
- Streamlit (Frontend UI)  
- Hugging Face Spaces / Render / Streamlit Cloud (Hosting)

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/myrag-chatbot.git
cd myrag-chatbot
pip install -r requirements.txt
python app.py
```

