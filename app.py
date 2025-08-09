import os
import tempfile
import logging
from typing import List, Optional
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------
# Configuration Constants
# ----------------
INDEX_NAME = "rag-index-1536"  # Updated to avoid dimension conflicts

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small output size

# Alternative configurations (uncomment to use):
# EMBEDDING_MODEL = "text-embedding-3-large"
# EMBEDDING_DIMENSION = 3072  # text-embedding-3-large output size

# EMBEDDING_MODEL = "text-embedding-ada-002" 
# EMBEDDING_DIMENSION = 1536  # text-embedding-ada-002 output size

# Document processing configuration - Optimized for better accuracy
DEFAULT_CHUNK_SIZE = 1000  # Increased for more context per chunk
DEFAULT_CHUNK_OVERLAP = 200  # Increased overlap for better continuity
DEFAULT_RETRIEVAL_K = 8  # Retrieve more relevant chunks

class RAGAssistant:
    """
    A Retrieval-Augmented Generation assistant that uses Pinecone for vector storage
    and OpenAI for embeddings and chat completion.
    """
    
    def __init__(self):
        self.embedding_model = None
        self.db = None
        self.chat_model = None
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = ""
        if "pinecone_api_key" not in st.session_state:
            st.session_state.pinecone_api_key = ""
        if "documents_processed" not in st.session_state:
            st.session_state.documents_processed = 0
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
    
    def validate_api_key(self, key: str, key_type: str) -> bool:
        """Validate API key format."""
        if not key or not isinstance(key, str):
            return False
        
        if key_type == "openai":
            return key.startswith("sk-") and len(key) > 20
        elif key_type == "pinecone":
            return len(key) > 20  # Basic length check for Pinecone keys
        
        return False
    
    def get_openai_key(self) -> str:
        """Get and validate OpenAI API key."""
        key = st.session_state.get("openai_api_key", "").strip()
        if not self.validate_api_key(key, "openai"):
            st.error("Please enter a valid OpenAI API key in the sidebar.")
            st.stop()
        return key

    def get_pinecone_key(self) -> str:
        """Get and validate Pinecone API key."""
        key = st.session_state.get("pinecone_api_key", "").strip()
        if not self.validate_api_key(key, "pinecone"):
            st.error("Please enter a valid Pinecone API key in the sidebar.")
            st.stop()
        return key

    @st.cache_resource
    def get_embedding_model(_self):
        """Get cached embedding model."""
        try:
            return OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=_self.get_openai_key()
            )
        except Exception as e:
            logger.error(f"Error creating embedding model: {e}")
            st.error(f"Failed to create embedding model ({EMBEDDING_MODEL}): {e}")
            st.stop()

    def get_pinecone_client(self) -> Pinecone:
        """Get Pinecone client."""
        try:
            # Ensure API key is available to SDK
            api_key = self.get_pinecone_key()
            os.environ["PINECONE_API_KEY"] = api_key
            return Pinecone(api_key=api_key)
        except Exception as e:
            logger.error(f"Error creating Pinecone client: {e}")
            st.error(f"Failed to connect to Pinecone: {e}")
            st.stop()

    def get_db(self) -> PineconeVectorStore:
        """Get or create Pinecone vector database."""
        if self.db is not None:
            return self.db
        
        try:
            # Make sure Pinecone SDK can see the key
            os.environ["PINECONE_API_KEY"] = self.get_pinecone_key()
            
            pc = self.get_pinecone_client()

            # Check existing indexes and handle dimension conflicts
            existing_indexes = [index["name"] for index in pc.list_indexes().get("indexes", [])]
            
            if INDEX_NAME in existing_indexes:
                # Check if existing index has correct dimensions
                index_stats = pc.describe_index(INDEX_NAME)
                existing_dimension = index_stats.dimension
                
                if existing_dimension != EMBEDDING_DIMENSION:
                    logger.warning(f"Dimension mismatch: existing index has {existing_dimension}, expected {EMBEDDING_DIMENSION}")
                    
                    # Ask user what to do
                    st.error(f"⚠️ **Dimension Mismatch Detected**")
                    st.warning(f"Your existing Pinecone index '{INDEX_NAME}' has dimension {existing_dimension}, but the current configuration expects {EMBEDDING_DIMENSION}.")
                    
                    st.markdown("### 🔧 **Solutions:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🗑️ Force Delete & Recreate", type="primary"):
                            try:
                                with st.spinner("Forcefully deleting and recreating index..."):
                                    # Force delete the index
                                    pc.delete_index(INDEX_NAME)
                                    
                                    # Wait longer for deletion to complete
                                    import time
                                    st.write("⏳ Waiting for deletion to complete...")
                                    time.sleep(10)  # Increased wait time
                                    
                                    # Check if deletion is complete
                                    max_retries = 6
                                    for i in range(max_retries):
                                        try:
                                            existing_indexes = [index["name"] for index in pc.list_indexes().get("indexes", [])]
                                            if INDEX_NAME not in existing_indexes:
                                                break
                                        except:
                                            pass
                                        st.write(f"⏳ Checking deletion status... ({i+1}/{max_retries})")
                                        time.sleep(5)
                                    
                                    # Create new index
                                    pc.create_index(
                                        name=INDEX_NAME,
                                        dimension=EMBEDDING_DIMENSION,
                                        metric="cosine",
                                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                                    )
                                    
                                    logger.info(f"Successfully recreated Pinecone index with correct dimensions: {INDEX_NAME}")
                                    st.success("✅ Index recreated successfully!")
                                    
                                    # Clear cached database connection
                                    if hasattr(self, 'db'):
                                        self.db = None
                                    
                                    st.rerun()
                                    
                            except Exception as e:
                                st.error(f"Failed to recreate index: {e}")
                                logger.error(f"Error recreating index: {e}")
                    
                    with col2:
                        if st.button("📝 Use New Index Name"):
                            st.code(f"""
# Change this in app.py line 22:
INDEX_NAME = "rag-index-new"  # or any other name
                            """)
                            st.info("This will create a new index with the correct dimensions.")
                    
                    with col3:
                        if st.button("⚙️ Manual Pinecone Fix"):
                            st.markdown("### Manual Steps:")
                            st.markdown("""
                            1. Go to [Pinecone Console](https://app.pinecone.io/)
                            2. Delete the `rag-index` manually
                            3. Come back and refresh this page
                            4. The app will create a new index automatically
                            """)
                    
                    st.stop()
            else:
                # Create new index
                with st.spinner("Creating Pinecone index..."):
                    pc.create_index(
                        name=INDEX_NAME,
                        dimension=EMBEDDING_DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    logger.info(f"Created new Pinecone index: {INDEX_NAME}")

            embedding_model = self.get_embedding_model()

            self.db = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=embedding_model
            )
            
            return self.db
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            
            # Check if it's a dimension mismatch error
            if "dimension" in str(e).lower() and "does not match" in str(e).lower():
                st.error("🔧 **Vector Dimension Mismatch**")
                st.warning("The embedding model produces vectors of a different dimension than your Pinecone index expects.")
                st.info("💡 **Solutions:**")
                st.markdown("""
                1. **Recreate the index** - Delete your current index and let the app create a new one
                2. **Use a different index name** - Change INDEX_NAME in the code
                3. **Check your embedding model** - Ensure you're using the correct model
                """)
            else:
                st.error(f"Failed to setup vector database: {e}")
            
            st.stop()

    def add_documents_to_db(self, uploaded_files: List) -> bool:
        """Add uploaded PDF documents to the vector database."""
        if not uploaded_files:
            st.error("No files uploaded!")
            return False

        try:
            db = self.get_db()
            total_chunks = 0
            
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"Processing: {uploaded_file.name}")
                
                # Create a safe cross-platform temp file path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_file_path = tmp.name

                try:
                    loader = PyPDFLoader(temp_file_path)
                    data = loader.load()

                    if not data:
                        st.warning(f"No content found in {uploaded_file.name}")
                        continue

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        chunk_overlap=DEFAULT_CHUNK_OVERLAP
                    )
                    chunks = splitter.split_documents(data)
                    
                    if chunks:
                        db.add_documents(chunks)
                        total_chunks += len(chunks)
                        logger.info(f"Added {len(chunks)} chunks from {uploaded_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {uploaded_file.name}: {e}")
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.session_state.documents_processed += len(uploaded_files)
            st.success(f"✅ Successfully processed {len(uploaded_files)} documents ({total_chunks} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to database: {e}")
            st.error(f"Failed to add documents: {e}")
            return False

    def format_docs(self, docs) -> str:
        """Format retrieved documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    @st.cache_resource
    def get_chat_model(_self, temperature: float = 0) -> ChatOpenAI:
        """Get cached chat model."""
        try:
            return ChatOpenAI(
                model="gpt-4o-mini",  # Fast and cost-effective model
                api_key=_self.get_openai_key(),
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Error creating chat model: {e}")
            st.error(f"Failed to create chat model: {e}")
            st.stop()

    def run_rag_chain(self, query: str) -> str:
        """Run the RAG chain to answer a query."""
        if not query or not query.strip():
            return "Please provide a valid question."
        
        try:
            db = self.get_db()
            # Use MMR (Maximum Marginal Relevance) for better diversity in results
            retriever = db.as_retriever(
                search_type="mmr",  # Changed from similarity to MMR
                search_kwargs={
                    'k': DEFAULT_RETRIEVAL_K,
                    'fetch_k': DEFAULT_RETRIEVAL_K * 2,  # Fetch more candidates
                    'lambda_mult': 0.7  # Balance between relevance and diversity
                }
            )

            PROMPT_TEMPLATE = """
            You are an expert research assistant specializing in analyzing academic and technical documents.
            
            INSTRUCTIONS:
            1. Analyze the provided context thoroughly to answer the question
            2. Provide detailed, comprehensive answers with specific examples when available
            3. If the context mentions tables, figures, or specific data, reference them explicitly
            4. Synthesize information from multiple parts of the context if relevant
            5. If the context is insufficient, explain what specific information is missing
            6. Structure your response clearly with bullet points or numbered lists when appropriate
            
            CONTEXT:
            {context}
            
            QUESTION: {question}
            
            DETAILED ANSWER:
            """
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

            chat_model = self.get_chat_model()

            rag_chain = {
                "context": retriever | self.format_docs,
                "question": RunnablePassthrough()
            } | prompt_template | chat_model | StrOutputParser()

            return rag_chain.invoke(query)
            
        except Exception as e:
            logger.error(f"Error running RAG chain: {e}")
            return f"An error occurred while processing your question: {e}"

def create_sidebar(rag_assistant: RAGAssistant) -> List:
    """Create and manage the sidebar UI."""
    with st.sidebar:
        st.title("🔑 API Keys")
        
        # API Key inputs with validation indicators
        openai_key = st.text_input(
            "Enter your OpenAI API key:",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
            help="Your OpenAI API key (starts with 'sk-')"
        )
        st.session_state.openai_api_key = openai_key
        
        if openai_key:
            if rag_assistant.validate_api_key(openai_key, "openai"):
                st.success("✅ Valid OpenAI key")
            else:
                st.error("❌ Invalid OpenAI key format")
        
        pinecone_key = st.text_input(
            "Enter your Pinecone API key:",
            type="password",
            value=st.session_state.get("pinecone_api_key", ""),
            help="Your Pinecone API key"
        )
        st.session_state.pinecone_api_key = pinecone_key
        
        if pinecone_key:
            if rag_assistant.validate_api_key(pinecone_key, "pinecone"):
                st.success("✅ Valid Pinecone key")
            else:
                st.error("❌ Invalid Pinecone key format")

        st.markdown("---")
        st.title("📄 Document Upload")
        
        # Display stats
        if st.session_state.documents_processed > 0:
            st.info(f"📊 Documents processed: {st.session_state.documents_processed}")
        
        pdf_docs = st.file_uploader(
            "Upload PDF documents:",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to add to your knowledge base"
        )
        
        process_button = st.button(
            "🚀 Submit & Process",
            disabled=not pdf_docs or not openai_key or not pinecone_key
        )
        
        if process_button:
            if not pdf_docs:
                st.warning("Please upload at least one PDF file")
            elif not rag_assistant.validate_api_key(openai_key, "openai"):
                st.error("Please enter a valid OpenAI API key")
            elif not rag_assistant.validate_api_key(pinecone_key, "pinecone"):
                st.error("Please enter a valid Pinecone API key")
            else:
                with st.spinner("Processing documents..."):
                    success = rag_assistant.add_documents_to_db(pdf_docs)
                    if success:
                        st.balloons()
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        st.info(f"""
        **Embedding Model:** {EMBEDDING_MODEL}  
        **Vector Dimension:** {EMBEDDING_DIMENSION}  
        **Index Name:** {INDEX_NAME}
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Upload multiple PDFs at once
        - Ask specific questions about your documents
        - Use clear, detailed questions for better results
        - If you get dimension errors, try recreating the index
        """)
        
        return pdf_docs

def create_main_interface(rag_assistant: RAGAssistant):
    """Create the main query interface."""
    st.markdown("### 💬 Ask Questions About Your Documents")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What is the main topic discussed in the documents? What are the key findings? etc.",
        help="Ask specific questions about the content in your uploaded documents"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        submit_query = st.button(
            "🔍 Ask Question",
            type="primary",
            disabled=not query.strip() or not st.session_state.get("openai_api_key") or not st.session_state.get("pinecone_api_key")
        )
    
    if submit_query:
        if not query.strip():
            st.warning("Please enter a question")
        elif not st.session_state.get("documents_processed", 0):
            st.warning("Please upload and process some documents first")
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = rag_assistant.run_rag_chain(query.strip())
                    
                    st.markdown("### 📝 Answer")
                    st.markdown(result)
                    
                    # Add query to history
                    if "query_history" not in st.session_state:
                        st.session_state.query_history = []
                    st.session_state.query_history.append({
                        "question": query.strip(),
                        "answer": result
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error(f"Error in main interface: {e}")

def display_query_history():
    """Display previous queries and answers."""
    if st.session_state.get("query_history"):
        with st.expander("📜 Query History", expanded=False):
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
                st.markdown(f"**Q{len(st.session_state.query_history)-i}:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}{'...' if len(item['answer']) > 200 else ''}")
                st.markdown("---")

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Researcher RAG Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("📚 Researcher RAG Assistant")
    st.markdown("Upload your documents and ask intelligent questions powered by AI")
    
    # Initialize RAG assistant
    rag_assistant = RAGAssistant()
    rag_assistant.initialize_session_state()
    
    # Create UI components
    pdf_docs = create_sidebar(rag_assistant)
    
    # Main content area
    create_main_interface(rag_assistant)
    
    # Query history
    display_query_history()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, LangChain, OpenAI, and Pinecone | "
        "[GitHub](https://github.com/amany4864/ResearcherRAG)"
    )

if __name__ == "__main__":
    main()
