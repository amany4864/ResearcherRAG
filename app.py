import os
import streamlit as st
# from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load env vars
# load_dotenv()

# Get API key
def get_api_key():
    key = st.session_state.get("openai_api_key")
    if not key:
        st.error("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()
    return key


# Initialize embedding model
def get_embedding_model():
    return OpenAIEmbeddings(api_key=get_api_key())

# Initialize database
def get_db():
    return Chroma(
        collection_name="knowledge_base",
        embedding_function=get_embedding_model(),
        persist_directory='./vector_db'
    )

# Format docs into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Add documents to DB
def add_to_db(uploaded_files):
    db = get_db()
    if not uploaded_files:
        st.error("No files uploaded!")
        return

    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        doc_metadata = [d.metadata for d in data]
        doc_content = [d.page_content for d in data]

        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50    
        )
        

        chunks = splitter.create_documents(doc_content, doc_metadata)

        db.add_documents(chunks)
        os.remove(temp_file_path)

# Run RAG chain
def run_rag_chain(query):
    db = get_db()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    PROMPT_TEMPLATE = """
    You are a highly knowledgeable and concise assistant. 
    Use the provided context to answer the user's question as accurately as possible.
    Only use information from the context and do not make up facts.

    Context:
    {context}

    Question: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chat_model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=get_api_key(),
        temperature=0
    )

    rag_chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt_template | chat_model | StrOutputParser()

    return rag_chain.invoke(query)

# Streamlit UI
def main():
    st.set_page_config(page_title="Researcher RAG Assistant", page_icon=":books:")
    st.header("Researcher RAG Assistant")

    with st.sidebar:
        st.title("API Keys")
        openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if st.button("Save API Key"):
            if openai_api_key:
                st.session_state.openai_api_key = openai_api_key
                st.success("API key saved!")
            else:
                st.warning("Please enter your API key.")

        st.markdown("---")
        pdf_docs = st.file_uploader("Upload PDFs (Optional)", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload a file")
            else:
                with st.spinner("Processing documents..."):
                    add_to_db(pdf_docs)
                    st.success("Documents added to the knowledge base!")

    query = st.text_area(":bulb: Ask your question about any topic:")
    if st.button("Submit Query"):
        if not query:
            st.warning("Please enter a question")
        elif not get_api_key():
            st.error("Please enter your API key in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                result = run_rag_chain(query)
                st.write(result)

if __name__ == "__main__":
    main()
