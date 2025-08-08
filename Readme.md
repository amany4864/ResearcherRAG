📚 RAG-based Research Assistant
A domain-independent Retrieval-Augmented Generation (RAG) application built with OpenAI and Streamlit that allows users to upload documents, store them in a vector database, and ask context-aware questions.
The tool supports any industry/domain — business, legal, education, research, etc. — by dynamically retrieving relevant document chunks before generating responses.

🚀 Features
Domain Agnostic → Works with any type of text documents (PDF, TXT, DOCX, etc.).

Dynamic RAG Pipeline → Uses embeddings + vector database for semantic search.

OpenAI LLM Integration → GPT models for context-aware responses.

User-Provided API Key → Users can securely use their own OpenAI key.

Multi-File Upload Support → Process multiple documents at once.

Memoryless & Secure → No storage of your API key or private data in backend.

Simple Deployment → Easily deployable on Streamlit Cloud, Render, or HuggingFace Spaces.

📂 Project Structure
bash
Copy
Edit
.
├── app.py               # Main Streamlit app
├── requirements.txt     # Dependencies
├── .gitignore           # Ignored files
├── vector_db/           # Local vector database (ignored in Git)
├── utils/               # Helper functions for RAG pipeline
└── README.md            # Documentation
🛠️ Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/amany4864/ResearcherRAG.git
cd rag-assistant
Create and activate a virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Install dependencies


pip install -r requirements.txt
🔑 API Key Setup
This app allows users to paste their own OpenAI API key in the UI.

OPENAI_API_KEY=your_default_key_here
▶️ Running Locally
bash
Copy
Edit
streamlit run app.py
Then open your browser at:
http://localhost:8501