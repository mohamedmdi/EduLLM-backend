# 🤖 AI APP Backend

Welcome to the backend of the **AI APP** project! This repository contains all the code and Docker setup you need to run a powerful document-based vector search API. 

---

## 🚀 Features

- ⚡ Fast REST API for document search
- 🧠 Embedding & vector database tools
- 🐳 Easy Docker deployment
- 📄 Supports PDF & DOCX files

---

## 🛠️ Prerequisites

- [🐳 Docker](https://www.docker.com/get-started) & [Docker Compose](https://docs.docker.com/compose/)
- [🐍 Python 3.11+](https://www.python.org/) (if running locally)
- (Optional) `pip` for Python package management

---

## 📁 Folder Structure

> **⚠️ Important:**  
> This repository only includes the backend code and Docker setup.  
> **You must manually create the following folders before running the application:**

```
backend/
│
├── data/           # 📥 Place your source PDF/DOCX files here
├── embeddings/     # 🧬 Stores generated embeddings and indexes
├── Files/          # 🗂️ (Optional) Additional files for processing
```

---

## 🏁 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd backend
```

### 2️⃣ Create Required Folders

```bash
mkdir data embeddings Files
```

### 3️⃣ Build & Run with Docker

```bash
docker-compose up --build
```

_This will build the Docker image and start the backend server._

### 4️⃣ Run Locally (Without Docker)

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Start the server:

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

- 📂 Place your documents in the `data/` folder.
- 🏗️ Use the provided scripts to build the database and run queries.
- 🌐 The API will be available at `http://localhost:8000` (or as configured).

---

## 📜 License

This project is licensed under the MIT License.

---

> Made with ❤️ by the AI APP Team
