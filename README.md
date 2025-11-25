# **Flat RAG System v2.0 (Improved Stable Release)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

A modern, modular Retrieval-Augmented Generation (RAG) engine built with **FastAPI**, **Qdrant**, and **Ollama**.

This release includes:
- A reworked, clean architecture  
- Dependency injection & service layer  
- Robust chunking pipeline  
- Stable clients for LLM / embeddings / vector DB  
- Full debug Web UI  
- PCA vector visualization  

---

## 🚀 Key Improvements in v2.0

### 🧱 Architecture (Clean, Modular)
- Layered structure  
- DI via FastAPI `app.state`  
- Strong error handling  
- Extensible client factory  
- Cleaner separation of concerns  

### 🧩 SmartTextSplitter
- Semantic chunking (if supported)
- Token-aware fallback  
- Window overlap  
- Adaptive chunk sizing (from `.env`)  

### 🧠 RAG Pipeline
- Better context builder  
- Stable prompt generator  
- Reduced hallucinations  
- Predictable answers  

### 🔍 Retrieval Layer
- Unified vector client  
- More robust Qdrant integration  
- Safe collection handling  
- Configurable thresholds  

### 💬 Clients (LLM / Embeddings / Vector)
Supports formats:
- Ollama (`{"response": "..."}`)
- `generated_text`
- `text`
- OpenAI-style (`choices[].message.content`)  

Handles:
- API errors  
- timeouts  
- invalid JSON  
- connection failures  

### 📊 PCA Visualization
- `/api/plot`
- PCA → PNG export  
- Visual debugging of embedding space  

### 💻 Full Web UI
- RAG UI  
- Search UI  
- Embedding UI  
- PCA Viewer  
- Health Dashboard  

---

## ✨ Features
- Full local RAG pipeline  
- Modular structure  
- Smart chunking  
- Semantic retrieval  
- Local LLM  
- PCA visualization  
- Docker-ready  
- Strong logging  

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI |
| Vector DB | Qdrant |
| LLM Runtime | Ollama |
| Embeddings | mxbai-embed-large / Ollama |
| Config | Pydantic Settings |
| HTTP | httpx |
| UI | Jinja2 |
| Visualization | PCA (sklearn) + matplotlib |
| Containers | Docker + Compose |

---

## 📁 Project Structure

backend/
├── app/
│ ├── clients/
│ ├── services/
│ ├── dependencies/
│ ├── routes/
│ ├── templates/
│ ├── core/
│ └── main.py
├── Dockerfile
├── compose.yml
└── .env.example


---

## 🧰 Getting Started

### 1) Clone
```bash
git clone https://github.com/baiazetsh/flat-rag-system.git
cd flat-rag-system

2) Configure
cp .env.example .env

3) Run
docker compose up --build

🖥️ Web Interface
Page	URL
RAG UI	/ui
Semantic Search	/ui/search
Embedding Form	/api/embed/form
Health UI	/api/health/ui
PCA Viewer	/api/plot
📡 API Endpoints
Health

GET /api/health

RAG

POST /api/search_with_llm

Search

POST /api/search

Embedding

POST /api/embed

LLM Direct

POST /api/llm/ask

Upload

POST /api/upload

PCA

GET /api/plot

🧭 Architecture Diagrams
ASCII Diagram
                 ┌─────────────────────────────┐
                 │         Web Browser          │
                 │   (RAG UI / Search UI /      │
                 │    Embed UI / PCA Viewer)    │
                 └───────────────┬─────────────┘
                                 │
                                 ▼
                     ┌──────────────────────┐
                     │      FastAPI App      │
                     │     (main.py)         │
                     └─────────┬─────────────┘
                               │
        ┌──────────────────────┼──────────────────────────┐
        │                      │                          │
        ▼                      ▼                          ▼
┌──────────────────┐   ┌──────────────────┐     ┌─────────────────────┐
│     Routes        │   │  Dependencies    │     │   Templates (UI)    │
│  /api/* /ui/*     │   │  DI Factories    │     │   Jinja2 HTML       │
└─────────┬────────┘   └─────────┬────────┘     └───────────┬─────────┘
          │                      │                            │
          ▼                      ▼                            ▼
┌──────────────────┐   ┌──────────────────┐     ┌─────────────────────────┐
│    Services       │   │  SmartTextSplitter│     │    PCA Visualization    │
│ (RAG / Vector /   │   │  Context Builder │     │  (sklearn + matplotlib) │
│  LLM / Embeddings)│   │  Prompter        │     └─────────────────────────┘
└─────────┬────────┘   └─────────┬────────┘
          │                      │
          ▼                      ▼
 ┌──────────────────┐   ┌─────────────────────────┐
 │   Clients         │   │        Config + Logger  │
 │ - LLM Client      │   │   (Pydantic Settings)   │
 │ - Embedding Client│   │                         │
 │ - Vector Client   │   └─────────────────────────┘
 └───────┬──────────┘
         │
         ▼
  ┌───────────────────────────────────────────────────┐
  │                   External Services                │
  │   ┌───────────────┐   ┌─────────────────┐         │
  │   │    Ollama      │   │     Qdrant      │         │
  │   │ (LLM + Embeds) │   │ (Vector Search) │         │
  │   └───────────────┘   └─────────────────┘         │
  └───────────────────────────────────────────────────┘

Mermaid Architecture
flowchart TD

A[Web Browser<br>(UI: RAG / Search / Embed / PCA)] --> B(FastAPI App<br>main.py)

B --> C[Routes<br>/api/*  /ui/*]
B --> D[Dependencies<br>DI Factories]
B --> E[Templates<br>Jinja2 UI]

C --> F[Services<br>RAG / Search / Embeddings / LLM / Vector]
D --> F

F --> G[SmartTextSplitter<br>Context Builder<br>Prompter]
F --> H[PCA Visualization<br>sklearn + matplotlib]

F --> I[Clients<br>LLM / Embedding / Vector]

I --> J[Ollama<br>(LLM + Embeddings)]
I --> K[Qdrant<br>(Vector DB)]

B --> L[Config + Logger<br>Pydantic Settings]

RAG Flow (Mermaid)
flowchart TD

A[User Query] --> B[SmartTextSplitter<br>(optional preprocessing)]
B --> C[Embedding Client<br>Ollama Embeddings]
C --> D[Vector DB Search<br>Qdrant]
D --> E[Top-K Retrieved Chunks]
E --> F[Context Builder<br>Merge / Rank / Filter]
F --> G[Prompter<br>Construct Final Prompt]
G --> H[LLM Client<br>Ollama Generate]
H --> I[Final Answer]
I --> J[Return to UI / API]

📄 License

MIT License.

👤 Author

@baiazetsh
Flat RAG System — Local AI Retrieval Engine (2025)
v2.0.0