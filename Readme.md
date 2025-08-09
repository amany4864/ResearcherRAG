# 📚 ResearcherRAG - Intelligent Document Assistant

A robust, domain-independent Retrieval-Augmented Generation (RAG) application built with OpenAI, Pinecone, and Streamlit. Upload your documents and ask intelligent questions powered by AI.
DEPLOYED LINK:- https://amanrag.streamlit.app/
## ✨ Features

- **🎯 Domain Agnostic** - Works with any type of PDF documents across industries
- **🔍 Smart Retrieval** - Uses semantic search with Pinecone vector database
- **🤖 AI-Powered** - GPT-4 integration for accurate, context-aware responses
- **🔐 Secure** - User-provided API keys, no data stored on servers
- **📁 Multi-Document** - Process multiple PDFs simultaneously
- **📊 Progress Tracking** - Real-time processing feedback and statistics
- **🎨 Modern UI** - Clean, responsive interface with query history
- **⚡ Performance** - Optimized with caching and error handling

## 🏗️ Architecture

```
User Interface (Streamlit)
    ↓
Document Processing (LangChain)
    ↓
Vector Storage (Pinecone)
    ↓
Retrieval & Generation (OpenAI GPT-4)
```

## 📂 Project Structure

```
ResearcherRAG/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Readme.md          # Documentation
├── vector_db/         # Local vector storage (auto-generated)
└── temp/              # Temporary file processing
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/amany4864/ResearcherRAG.git
cd ResearcherRAG
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Getting Started

1. **Enter API Keys** - Add your OpenAI and Pinecone API keys in the sidebar
2. **Upload Documents** - Upload one or more PDF files
3. **Process Documents** - Click "Submit & Process" to add documents to the vector database
4. **Ask Questions** - Enter questions about your documents and get AI-powered answers

### API Keys

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Pinecone API Key**: Get from [Pinecone Console](https://app.pinecone.io/)

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file for default API keys:
```
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

### Customization

The application supports various configuration options in `app.py`:

- `DEFAULT_CHUNK_SIZE`: Document chunk size (default: 500)
- `DEFAULT_CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `DEFAULT_RETRIEVAL_K`: Number of chunks to retrieve (default: 5)

## 🔒 Security & Privacy

- API keys are stored only in session state
- No data is permanently stored on servers
- Temporary files are automatically cleaned up
- All communication is encrypted

## 🐛 Troubleshooting

### Common Issues

1. **Invalid API Key Error**
   - Ensure your API keys are correct and have sufficient credits
   - Check key format (OpenAI keys start with 'sk-')

2. **Document Processing Fails**
   - Ensure PDFs are not password-protected
   - Check file size (large files may take longer)

3. **Connection Errors**
   - Verify internet connection
   - Check Pinecone service status

### Logging

The application includes comprehensive logging. Check the console output for detailed error messages.

## 🚀 Deployment

### Streamlit Cloud

1. Fork this repository
2. Connect your GitHub account to Streamlit Cloud
3. Deploy directly from the repository
4. Set environment variables in Streamlit Cloud settings

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [LangChain](https://langchain.com/) for RAG pipeline
- [OpenAI](https://openai.com/) for embeddings and language models
- [Pinecone](https://pinecone.io/) for vector database

## 📧 Support

For support, please open an issue on GitHub or contact [amany4864@gmail.com](mailto:amany4864@gmail.com).

---

Built with ❤️ by [Aman Yadav](https://github.com/amany4864)
