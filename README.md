# 🎓 RAG-Based Intelligent Academic Advisor System for University Student Support

An enterprise-grade Retrieval-Augmented Generation (RAG) system built with **Python**, **LangChain**, **HuggingFace Embeddings (`all-MiniLM-L6-v2`)**, **FAISS Vector Store**, and **Streamlit**.

Designed specifically for higher education institutions to assist university students with real-time, grounded answers regarding:
- Course prerequisites and corequisites
- Degree requirements and credit allocations
- Academic probation, warnings, suspensions, and Dean's List requirements
- Add/Drop dates, withdrawal deadlines, and grade forgiveness/retake policies

---

## 📂 Project Structure

```text
academic_advisor_rag/
│
├── catalog_cs.md           # University academic catalog & student policy document
├── ingest_and_index.py     # Document chunking, HuggingFace embeddings & FAISS indexing
├── app.py                  # Streamlit web interface with interactive RAG advising chat
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment variables template
└── vectorstore_db/         # Persisted local FAISS vector database (created upon indexing)
    ├── index.faiss
    └── index.pkl
```

---

## ⚡ Quickstart Guide

### 1. Clone / Navigate to the Directory
```bash
cd academic_advisor_rag
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ingest and Build the FAISS Vector Database
Run the ingestion script to parse `catalog_cs.md`, segment it using `RecursiveCharacterTextSplitter`, embed it with HuggingFace's `all-MiniLM-L6-v2`, and save the index into `vectorstore_db`:

```bash
python ingest_and_index.py
```
*(Optional flags: `--file path/to/catalog.md`, `--chunk_size 600`, `--chunk_overlap 100`)*

### 5. Launch the Streamlit Advisor Web App
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 🧠 System Architecture & Capabilities

1. **Deterministic Grounding**: The system prompt strictly prohibits hallucinations and instructs the advisor to answer strictly from retrieved policy excerpts.
2. **Transparent Citations**: Every response provides an expandable source tray showing exact retrieved catalog passages and distance metrics.
3. **Flexible LLM Backends**:
   - **OpenAI** (`gpt-4o-mini`, `gpt-4o`)
   - **Retrieval-Only Mode**: Instant similarity lookup without needing an API key.
