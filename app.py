import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple
import tempfile
import json
import asyncio
import nest_asyncio

# Apply nest_asyncio to fix event loop issues
nest_asyncio.apply()

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFChatApp:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="AI-Powered PDF Chat Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'processed_pdfs' not in st.session_state:
            st.session_state.processed_pdfs = []
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = "ready"
        if 'pdf_hash' not in st.session_state:
            st.session_state.pdf_hash = None
        if 'embeddings_cache' not in st.session_state:
            st.session_state.embeddings_cache = None
            
    def get_pdf_hash(self, pdf_docs: List) -> str:
        """Generate a hash for the uploaded PDFs to enable caching"""
        import hashlib
        hasher = hashlib.md5()
        for pdf in pdf_docs:
            # Read file content for hashing
            pdf.seek(0)
            hasher.update(pdf.read())
            pdf.seek(0)  # Reset file pointer
        return hasher.hexdigest()
    
    def is_vector_store_cached(self, pdf_docs: List) -> bool:
        """Check if vector store is already cached for these PDFs"""
        current_hash = self.get_pdf_hash(pdf_docs)
        return (st.session_state.pdf_hash == current_hash and 
                st.session_state.vector_store is not None and
                os.path.exists("faiss_index"))
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format"""
        if not api_key:
            return False
        return len(api_key) > 20 and api_key.startswith(('AIza', 'sk-'))
    
    def extract_pdf_text(self, pdf_docs: List) -> str:
        """Extract text from uploaded PDF files with error handling and progress tracking"""
        text = ""
        successful_pdfs = []
        failed_pdfs = []
        
        # Calculate total pages for progress tracking
        total_pages = 0
        pdf_page_counts = {}
        
        # First pass: count pages
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                page_count = len(pdf_reader.pages)
                pdf_page_counts[pdf.name] = page_count
                total_pages += page_count
            except Exception as e:
                failed_pdfs.append(pdf.name)
                logger.error(f"Error reading {pdf.name}: {str(e)}")
        
        if total_pages > 500:
            st.warning(f"📄 Large document detected: {total_pages} pages. This may take a few minutes to process.")
        
        # Second pass: extract text with progress
        processed_pages = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for pdf in pdf_docs:
            if pdf.name in failed_pdfs:
                continue
                
            try:
                pdf_reader = PdfReader(pdf)
                pdf_text = ""
                
                status_text.text(f"Processing {pdf.name}...")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            pdf_text += page_text + "\n"
                        
                        processed_pages += 1
                        progress_bar.progress(processed_pages / total_pages)
                        
                        # Update status every 50 pages for large documents
                        if page_num % 50 == 0 and total_pages > 100:
                            status_text.text(f"Processing {pdf.name}: Page {page_num + 1}/{len(pdf_reader.pages)}")
                            
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {pdf.name}: {str(e)}")
                        continue
                
                if pdf_text.strip():
                    text += f"\n\n--- Content from {pdf.name} ---\n\n{pdf_text}"
                    successful_pdfs.append(pdf.name)
                else:
                    failed_pdfs.append(pdf.name)
                    
            except Exception as e:
                logger.error(f"Error processing {pdf.name}: {str(e)}")
                failed_pdfs.append(pdf.name)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if failed_pdfs:
            st.warning(f"⚠️ Could not process: {', '.join(failed_pdfs)}")
        if successful_pdfs:
            st.success(f"✅ Successfully processed: {', '.join(successful_pdfs)} ({total_pages} pages)")
            
        return text
    
    def create_text_chunks(self, text: str, chunk_size: int = 8000, chunk_overlap: int = 800) -> List[str]:
        """Split text into chunks for processing with adaptive sizing for large documents"""
        if not text.strip():
            return []
        
        # Adaptive chunk sizing based on document length
        text_length = len(text)
        if text_length > 1000000:  # Very large document (>1MB text)
            chunk_size = 12000  # Larger chunks for efficiency
            chunk_overlap = 1200
            st.info(f"📊 Large document detected ({text_length:,} characters). Using optimized chunking.")
        elif text_length > 500000:  # Medium-large document
            chunk_size = 10000
            chunk_overlap = 1000
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        st.info(f"📝 Created {len(chunks)} text chunks (avg. {len(text)//len(chunks):,} chars each)")
        
        return chunks
    
    def create_vector_store(self, text_chunks: List[str], api_key: str) -> Optional[FAISS]:
        """Create and save vector store with event loop handling and batch processing"""
        if not text_chunks:
            st.error("No text chunks to process")
            return None
            
        try:
            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            
            # For large documents, process in batches to avoid memory issues
            batch_size = 50 if len(text_chunks) > 200 else len(text_chunks)
            
            with st.spinner(f"🔄 Creating vector embeddings for {len(text_chunks)} chunks..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if len(text_chunks) <= batch_size:
                    # Small document - process all at once
                    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                else:
                    # Large document - process in batches
                    status_text.text("Processing large document in batches...")
                    
                    # Process first batch
                    first_batch = text_chunks[:batch_size]
                    vector_store = FAISS.from_texts(first_batch, embedding=embeddings)
                    progress_bar.progress(batch_size / len(text_chunks))
                    
                    # Process remaining batches
                    for i in range(batch_size, len(text_chunks), batch_size):
                        batch = text_chunks[i:i + batch_size]
                        status_text.text(f"Processing batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")
                        
                        # Create temporary vector store for this batch
                        batch_vector_store = FAISS.from_texts(batch, embedding=embeddings)
                        
                        # Merge with main vector store
                        vector_store.merge_from(batch_vector_store)
                        
                        progress_bar.progress(min(i + batch_size, len(text_chunks)) / len(text_chunks))
                
                # Save the vector store
                vector_store.save_local("faiss_index")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            st.success(f"✅ Vector store created successfully with {len(text_chunks)} embeddings!")
            return vector_store
            
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            logger.error(f"Vector store creation failed: {str(e)}")
            return None
    
    def create_qa_chain(self, api_key: str):
        """Create the question-answering chain"""
        prompt_template = """
        You are an AI assistant specialized in analyzing PDF documents. Please follow these guidelines:
        
        1. Answer questions based ONLY on the provided context from the PDF documents
        2. Provide detailed, comprehensive answers when the information is available
        3. If the answer is not in the context, clearly state: "The answer is not available in the provided documents"
        4. Include relevant details, examples, and explanations when possible
        5. Maintain a professional and helpful tone
        
        Context from PDF documents:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=api_key,
                max_output_tokens=2048
            )
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = load_qa_chain(
                model,
                chain_type="stuff",
                prompt=prompt,
                verbose=True
            )
            
            return chain
            
        except Exception as e:
            st.error(f"❌ Error creating QA chain: {str(e)}")
            logger.error(f"QA chain creation failed: {str(e)}")
            return None
    
    def process_user_question(self, user_question: str, api_key: str, pdf_docs: List) -> None:
        """Process user question and generate response with event loop handling and caching"""
        if not user_question.strip():
            st.warning("Please enter a question")
            return
            
        if not api_key or not self.validate_api_key(api_key):
            st.error("❌ Please provide a valid API key")
            return
            
        if not pdf_docs:
            st.error("❌ Please upload PDF files first")
            return
        
        try:
            # Check if we can use cached vector store
            if self.is_vector_store_cached(pdf_docs):
                st.info("🚀 Using cached vector store for faster processing!")
                vector_store = st.session_state.vector_store
            else:
                # Process PDFs and create new vector store
                with st.spinner("📄 Extracting text from PDFs..."):
                    text = self.extract_pdf_text(pdf_docs)
                    
                if not text.strip():
                    st.error("❌ No text could be extracted from the uploaded PDFs")
                    return
                
                # Create text chunks
                with st.spinner("✂️ Processing text chunks..."):
                    text_chunks = self.create_text_chunks(text)
                    
                if not text_chunks:
                    st.error("❌ Could not create text chunks")
                    return
                
                # Create vector store
                vector_store = self.create_vector_store(text_chunks, api_key)
                if not vector_store:
                    return
                
                # Cache the vector store and PDF hash
                st.session_state.vector_store = vector_store
                st.session_state.pdf_hash = self.get_pdf_hash(pdf_docs)
                
            # Load existing vector store and perform similarity search
            with st.spinner("🔍 Searching for relevant information..."):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=api_key
                    )
                    
                    # Use cached vector store if available
                    if vector_store:
                        db = vector_store
                    else:
                        db = FAISS.load_local(
                            "faiss_index",
                            embeddings,
                            allow_dangerous_deserialization=True
                        )
                    
                    # Get relevant documents with higher k for large documents
                    k_docs = min(8, len(db.docstore._dict)) if hasattr(db, 'docstore') else 6
                    docs = db.similarity_search(user_question, k=k_docs)
                    
                    if not docs:
                        st.warning("⚠️ No relevant content found for your question.")
                        return
                    
                except Exception as e:
                    st.error(f"❌ Error loading vector store: {str(e)}")
                    return
                
            # Create QA chain and get response
            with st.spinner("🤔 Generating response..."):
                chain = self.create_qa_chain(api_key)
                if not chain:
                    return
                    
                try:
                    response = chain(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    return
                
            # Store conversation
            pdf_names = [pdf.name for pdf in pdf_docs]
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            st.session_state.conversation_history.append({
                "question": user_question,
                "answer": response['output_text'],
                "timestamp": timestamp,
                "pdf_names": pdf_names,
                "model": "Google AI - Gemini 1.5 Flash"
            })
            
            # Display the conversation
            self.display_conversation(user_question, response['output_text'], pdf_names)
            
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            logger.error(f"Question processing failed: {str(e)}")
            
            # Additional debugging information
            if "event loop" in str(e).lower():
                st.error("🔄 Event loop issue detected. Please try refreshing the page and trying again.")
                st.info("💡 Tip: If this persists, restart the Streamlit application.")
    
    def display_conversation(self, question: str, answer: str, pdf_names: List[str]):
        """Display conversation in a chat-like interface"""
        st.markdown("### 💬 Conversation")
        
        # Current conversation
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                        YOU
                    </div>
                </div>
                <div style="margin-left: 15px; font-size: 16px;">
                    {question}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background-color: #2196F3; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                        AI ASSISTANT
                    </div>
                </div>
                <div style="margin-left: 15px; font-size: 16px; line-height: 1.6;">
                    {answer}
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #666;">
                    📄 Sources: {', '.join(pdf_names)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display conversation history
        if len(st.session_state.conversation_history) > 1:
            st.markdown("### 📝 Previous Conversations")
            for i, conv in enumerate(reversed(st.session_state.conversation_history[:-1])):
                with st.expander(f"Conversation {len(st.session_state.conversation_history) - i - 1}: {conv['question'][:50]}..."):
                    st.write(f"**Question:** {conv['question']}")
                    st.write(f"**Answer:** {conv['answer']}")
                    st.write(f"**Time:** {conv['timestamp']}")
                    st.write(f"**Sources:** {', '.join(conv['pdf_names'])}")
    
    def export_conversation_history(self):
        """Export conversation history to various formats"""
        if not st.session_state.conversation_history:
            return
            
        # Prepare data for export
        export_data = []
        for conv in st.session_state.conversation_history:
            export_data.append({
                "Question": conv["question"],
                "Answer": conv["answer"],
                "Timestamp": conv["timestamp"],
                "PDF Sources": ", ".join(conv["pdf_names"]),
                "Model": conv["model"]
            })
        
        df = pd.DataFrame(export_data)
        
        # CSV Export
        csv = df.to_csv(index=False)
        csv_b64 = base64.b64encode(csv.encode()).decode()
        
        st.sidebar.markdown("### 📥 Export Options")
        st.sidebar.markdown(
            f'<a href="data:file/csv;base64,{csv_b64}" download="pdf_chat_history.csv">'
            f'<button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">📊 Download CSV</button></a>',
            unsafe_allow_html=True
        )
        
        # JSON Export
        json_data = json.dumps(export_data, indent=2)
        json_b64 = base64.b64encode(json_data.encode()).decode()
        
        st.sidebar.markdown(
            f'<a href="data:file/json;base64,{json_b64}" download="pdf_chat_history.json">'
            f'<button style="background-color: #2196F3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">📋 Download JSON</button></a>',
            unsafe_allow_html=True
        )
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.title("🤖 AI PDF Chat Assistant")
        st.sidebar.markdown("---")
        
        # Profile links
        st.sidebar.markdown("### 👨‍💻 Developer")
        linkedin_link = "https://www.linkedin.com/in/snsupratim/"
        kaggle_link = "https://www.kaggle.com/snsupratim/"
        github_link = "https://github.com/snsupratim/"
        
        st.sidebar.markdown(
            f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_link})"
        )
        st.sidebar.markdown(
            f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_link})"
        )
        st.sidebar.markdown(
            f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_link})"
        )
        
        st.sidebar.markdown("---")
        
        # API Key input
        st.sidebar.markdown("### 🔑 API Configuration")
        api_key = st.sidebar.text_input(
            "Enter your Google AI API Key:",
            type="password",
            help="Get your API key from https://ai.google.dev/"
        )
        
        if api_key and not self.validate_api_key(api_key):
            st.sidebar.error("❌ Invalid API key format")
        elif api_key:
            st.sidebar.success("✅ API key validated")
        
        st.sidebar.markdown("[Get API Key](https://ai.google.dev/)")
        st.sidebar.markdown("---")
        
        # File upload
        st.sidebar.markdown("### 📁 Upload PDFs")
        pdf_docs = st.sidebar.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF files to chat with"
        )
        
        if pdf_docs:
            st.sidebar.success(f"✅ {len(pdf_docs)} PDF(s) uploaded")
            for pdf in pdf_docs:
                st.sidebar.write(f"📄 {pdf.name}")
        
        # Control buttons
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Controls")
        
        col1, col2 = st.sidebar.columns(2)
        
        if col1.button("🔄 Reset All", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.vector_store = None
            st.session_state.processed_pdfs = []
            st.session_state.pdf_hash = None
            if os.path.exists("faiss_index"):
                import shutil
                shutil.rmtree("faiss_index", ignore_errors=True)
            st.rerun()
        
        if col2.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Performance tips for large documents
        if pdf_docs:
            total_size = sum(len(pdf.getvalue()) for pdf in pdf_docs)
            if total_size > 10 * 1024 * 1024:  # > 10MB
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 💡 Performance Tips")
                st.sidebar.info(
                    "📊 Large document detected!\n\n"
                    "• First processing will take time\n"
                    "• Subsequent questions will be faster\n"
                    "• Vector store is cached automatically\n"
                    "• Use specific questions for better results"
                )
        
        # Export options
        if st.session_state.conversation_history:
            st.sidebar.markdown("---")
            self.export_conversation_history()
        
        return api_key, pdf_docs
    
    def render_main_content(self, api_key: str, pdf_docs: List):
        """Render the main content area"""
        st.title("🤖 AI-Powered PDF Chat Assistant")
        st.markdown("### Upload PDFs and ask questions about their content!")
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if api_key and self.validate_api_key(api_key):
                st.success("🔑 API Key: Ready")
            else:
                st.error("🔑 API Key: Missing")
        
        with col2:
            if pdf_docs:
                st.success(f"📄 PDFs: {len(pdf_docs)} uploaded")
            else:
                st.error("📄 PDFs: None uploaded")
        
        with col3:
            if st.session_state.conversation_history:
                st.info(f"💬 Conversations: {len(st.session_state.conversation_history)}")
            else:
                st.info("💬 Conversations: None")
        
        st.markdown("---")
        
        # Question input
        user_question = st.text_input(
            "Ask a question about your PDF documents:",
            placeholder="e.g., What are the main topics discussed in the document?",
            key="user_input"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            ask_button = st.button("🚀 Ask Question", use_container_width=True)
        
        # Process question
        if ask_button or user_question:
            if user_question:
                self.process_user_question(user_question, api_key, pdf_docs)
            else:
                st.warning("Please enter a question")
        
        # Display instructions if no conversation yet
        if not st.session_state.conversation_history:
            st.markdown("### 📋 How to use:")
            st.markdown("""
            1. **Enter your Google AI API Key** in the sidebar
            2. **Upload one or more PDF files** using the file uploader
            3. **Ask questions** about the content of your PDFs
            4. **Download** your conversation history when done
            """)
            
            st.markdown("### 🎯 Example Questions:")
            st.markdown("""
            - "What are the main topics discussed in this document?"
            - "Can you summarize the key findings?"
            - "What recommendations are mentioned?"
            - "Who are the main authors or contributors?"
            """)
    
    def run(self):
        """Main application runner"""
        try:
            # Render sidebar and get inputs
            api_key, pdf_docs = self.render_sidebar()
            
            # Render main content
            self.render_main_content(api_key, pdf_docs)
            
        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")
            logger.error(f"Application error: {str(e)}")

def main():
    """Main function to run the application"""
    app = PDFChatApp()
    app.run()

if __name__ == "__main__":
    main()