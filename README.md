# EduLLM Backend

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A powerful AI-driven educational backend that provides Retrieval-Augmented Generation (RAG) capabilities, file management, and intelligent resource recommendations for enhanced learning experiences.

## 🎯 Features

- **🤖 RAG-Powered Q&A**: Intelligent question-answering system using user-uploaded documents
- **📁 File Management**: Upload, store, and manage educational documents with embeddings
- **🔍 Smart Search**: Web-based resource discovery and topic extraction
- **🔒 User Session Management**: Secure user-specific data handling
- **🌐 Multi-language Support**: French-localized search results for Moroccan users
- **⚡ Streaming Responses**: Real-time AI response streaming for better UX
- **🗃️ Vector Database**: Efficient document embeddings storage and retrieval

## 🏗️ Architecture

The backend is built with a modular architecture:

```
EduLLM-backend/
├── server.py              # FastAPI application and API endpoints
├── rag_engine.py          # Core RAG functionality and embeddings
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker orchestration
├── Dockerfile            # Container configuration
├── embeddings/           # User-specific vector embeddings storage
├── data/                 # Document storage directory
└── Files/                # Uploaded files directory
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or conda
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mohamedmdi/EduLLM-backend.git
   cd EduLLM-backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create a .env file with the following variables:
   SERPAPI_KEY=your_serpapi_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the server**
   ```bash
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run with Docker
docker build -t edullm-backend .
docker run -p 8000:8000 edullm-backend
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 🏠 Health Check
```http
GET /
```
**Response:**
```json
{
  "message": "Hello from the RAG server!"
}
```

#### 🤖 Ask Questions
```http
POST /ask
```
**Parameters:**
- `query` (string): The question to ask
- `file` (file[], optional): Documents to upload and analyze
- `user_id` (string): Unique user identifier

**Response:** Streaming text response

**Example:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -F "query=Explain machine learning concepts" \
  -F "user_id=user123" \
  -F "file=@document.pdf"
```

#### 📁 List User Files
```http
GET /list_files?user_id={user_id}
```
**Response:**
```json
{
  "success": true,
  "files": [
    {
      "file": "document.pdf",
      "hash": "abc123..."
    }
  ],
  "message": "Files retrieved successfully"
}
```

#### 🗑️ Delete File
```http
DELETE /delete_file?user_id={user_id}&file_hash={file_hash}
```
**Response:**
```json
{
  "success": true,
  "message": "File and embeddings deleted successfully"
}
```

#### 🔍 Search Resources
```http
GET /search?user_id={user_id}
```
**Response:**
```json
{
  "topics": ["machine learning", "neural networks"],
  "search_results": {
    "machine learning": [
      {
        "title": "Machine Learning Course",
        "link": "https://example.com",
        "snippet": "Learn ML fundamentals..."
      }
    ]
  }
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SERPAPI_KEY` | API key for web search functionality | Yes |
| `GROQ_API_KEY` | GROQ API key for LLM operations | Yes |
| `HOST` | Server host (default: 0.0.0.0) | No |
| `PORT` | Server port (default: 8000) | No |

### CORS Settings

The server is configured to accept requests from any origin. For production, update the CORS settings in `server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🧠 RAG Engine Features

### Document Processing
- **Supported Formats**: PDF, DOCX, TXT, and more
- **Text Extraction**: Intelligent content parsing
- **Chunking**: Smart document segmentation for optimal embeddings
- **Metadata Storage**: File information and user association

### Vector Embeddings
- **OpenAI Embeddings**: High-quality text representations
- **FAISS Integration**: Efficient similarity search
- **User Isolation**: Separate embedding spaces per user
- **Incremental Updates**: Add new documents without rebuilding

### Query Processing
- **Context Retrieval**: Relevant document chunks identification
- **LLM Integration**: GPT-powered response generation
- **Streaming Output**: Real-time response delivery
- **Source Attribution**: Document-aware answers

## 🔍 Search Integration

### Web Search Features
- **SerpAPI Integration**: Google search results
- **Localized Results**: Morocco-specific, French-language results
- **Topic Extraction**: Automatic learning topic identification
- **Resource Aggregation**: Educational content discovery

### Search Configuration
```python
# Customizable search parameters
{
    "q": "des resources et vidéos pour: " + topic,
    "num": 5,                    # Number of results
    "location": "Morocco",       # Geographic focus
    "gl": "ma",                 # Country code
    "hl": "fr"                  # Language preference
}
```

## 📊 Data Storage

### File Storage Structure
```
embeddings/
├── user_{user_id}/
│   ├── chunks.json          # Document chunks
│   ├── embeddings.npy       # Vector embeddings
│   └── index.index          # FAISS index
```

### Database Schema
- **User Sessions**: Individual embedding spaces
- **Document Chunks**: Segmented text with metadata
- **File Mappings**: Hash-based file identification
- **Vector Indices**: Optimized similarity search structures

## 🛠️ Development

### Project Structure
```python
# Core modules
server.py           # FastAPI application
rag_engine.py       # RAG implementation
requirements.txt    # Dependencies

# Supporting files
Dockerfile         # Container setup
docker-compose.yml # Multi-service orchestration
.env.example       # Environment template
```

### Adding New Features

1. **New Endpoints**: Add to `server.py`
2. **RAG Enhancements**: Modify `rag_engine.py`
3. **Dependencies**: Update `requirements.txt`
4. **Testing**: Create test files in `tests/`

### Code Style
- **Python**: Follow PEP 8 guidelines
- **FastAPI**: Use async/await patterns
- **Type Hints**: Include for better code clarity
- **Documentation**: Add docstrings for functions

## 🚨 Error Handling

The API includes comprehensive error handling:

```python
# Example error responses
{
  "detail": "File not found",
  "status_code": 404
}

{
  "detail": "Invalid user_id format",
  "status_code": 400
}
```

## 📈 Performance Optimization

### Recommendations
- **Async Operations**: All I/O operations are asynchronous
- **Connection Pooling**: Reuse HTTP connections
- **Caching**: Implement Redis for frequent queries
- **Load Balancing**: Use multiple server instances
- **Resource Monitoring**: Track memory and CPU usage

### Scaling Considerations
- **Horizontal Scaling**: Deploy multiple backend instances
- **Database Optimization**: Use dedicated vector databases
- **CDN Integration**: Cache static file responses
- **Background Tasks**: Queue heavy processing operations

## 🔒 Security

### Best Practices
- **Input Validation**: Sanitize all user inputs
- **File Type Checking**: Validate uploaded file formats
- **Rate Limiting**: Implement request throttling
- **API Keys**: Secure environment variable storage
- **HTTPS**: Use TLS encryption in production

### User Data Protection
- **Isolation**: Separate user embeddings
- **Cleanup**: Automatic temporary file removal
- **Access Control**: User-specific data access
- **Audit Logging**: Track API usage patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- **Issues**: [GitHub Issues](https://github.com/mohamedmdi/EduLLM-backend/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mohamedmdi/EduLLM-backend/discussions)
- **Email**: Contact the development team

## 🎓 About EduLLM

EduLLM is an AI-powered educational platform designed to enhance learning through intelligent document analysis, personalized Q&A, and smart resource discovery. The backend serves as the core engine powering the educational experience.

---

**Built with ❤️ for education and powered by AI**
backend/
│
├── data/           # 📥 Place your source PDF/DOCX files here
├── embeddings/     # 🧬 Stores generated embeddings and indexes
```

---

## 🏁 Getting Started

### 1️⃣ Clone the Repository

Clone the repository and navigate to the backend folder:

```bash
git clone <your-repo-url>
cd backend
```

### 2️⃣ Create Required Folders

Create the folders for your data and embeddings (these are not tracked in git):

```bash
mkdir data embeddings Files
```

- `data/`: Place your source PDF/DOCX files here for processing.
- `embeddings/`: This folder will store generated embeddings and vector indexes.
- `Files/`: (Optional) For any extra files you want to process or keep.

### 3️⃣ Build & Run with Docker

Build and start the backend server using Docker Compose:

```bash
docker-compose up --build
```

- The server will be available at `http://localhost:8000` by default.
- All dependencies are handled inside the container.

### 4️⃣ Run Locally (Without Docker)

If you prefer to run the backend directly on your machine:

1. Install Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Start the backend server:

    ```bash
    python server.py
    ```

---

## 📦 Main Files

- `server.py` — 🌐 Main API server
- `db_builder.py` — 🏗️ Build the vector database
- `rag_engine.py` — 🤖 Retrieval-Augmented Generation engine
- `run_query.py` — 🔎 Query the vector database
- `Dockerfile` & `docker-compose.yml` — 🐳 Docker setup

---

## 📝 Usage

- 📂 **Add Documents:** Place your PDF or DOCX files in the `data/` folder. These will be used to build the vector database.
- 🏗️ **Build the Database:** Use `db_builder.py` to process documents and generate embeddings. Example:

    ```bash
    python db_builder.py
    ```

- 🔎 **Query the Database:** Use `run_query.py` to test queries against your vector database. Example:

    ```bash
    python run_query.py --query "What is AI?"
    ```

- 🌐 **API Access:** Once the server is running, you can interact with the REST API at `http://localhost:8000` (or as configured in your Docker setup).

---

## 📜 License

This project is licensed under the MIT License.

---
