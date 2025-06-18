# EduLLM Backend

<div align="center">

![EduLLM Logo](https://img.shields.io/badge/EduLLM-AI%20Learning%20Companion-emerald?style=for-the-badge&logo=graduation-cap)

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?style=flat-square&logo=docker)](https://www.docker.com/)
[![REST API](https://img.shields.io/badge/API-REST-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-VectorDB-orange?style=flat-square&logo=vectorworks)](https://en.wikipedia.org/wiki/Vector_database)

**Your AI-Powered Document Search Engine Backend**

A robust, containerized backend for EduLLM, enabling fast, intelligent document search and retrieval with modern vector database technology.

[🚀 Features](#-features) • [🛠️ Getting-Started](#-getting-started) • [📦 Main-Files](#-main-files) • [📝 Usage](#-usage)

</div>

---

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
