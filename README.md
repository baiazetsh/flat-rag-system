# 🧠 Migration RAG Assistant

[![Build Status](https://img.shields.io/github/actions/workflow/status/baiazetsh/flat-rag-system/tests.yml?branch=main&label=Build&color=brightgreen)](https://github.com/baiazetsh/flat-rag-system/actions)
[![Test Coverage](https://img.shields.io/badge/Coverage-90%25-blue.svg)](test_reports/coverage_all/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)


A lightweight yet powerful **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **Ollama**, and **Qdrant** — including a simple web interface for asking questions and viewing results.

---

## 🚀 Overview

This project connects a local language model with a vector database to build a working RAG pipeline.  
You can upload your text files, index them into Qdrant, and then ask natural-language questions.  
The system finds the most relevant chunks, builds a context, and lets the LLM generate the final answer.

---

## ✨ Features

- 🔍 **Semantic Search** powered by Qdrant  
- 💬 **Context-aware answers** using local LLMs (via Ollama)  
- 🧩 **Full RAG pipeline** — embeddings, retrieval, generation  
- 🧱 **FastAPI backend + Jinja2 templates (HTML UI)**  
- 📦 **Dockerized setup** — easy to run anywhere  
- 🧾 **File uploader** with automatic text chunking and vectorization  
- 🧪 **Built-in tests** (Pytest + Coverage + GitHub Actions ready)

---

## ⚙️ Tech Stack

| Layer | Tool |
|-------|------|
| Backend | FastAPI |
| Vector DB | Qdrant |
| LLM Engine | Ollama |
| Data Models | Pydantic |
| Async HTTP | httpx |
| Frontend | Jinja2 Templates |
| Containerization | Docker Compose |
| Testing | Pytest, pytest-cov |

---

## 📁 Project Structure

flat-rag-system/
├── backend/ # FastAPI app, routes, models, services
│ ├── app/
│ ├── tests/ # Unit & integration tests
│ ├── Dockerfile
│ └── requirements-test.txt
├── scripts/ # Utility scripts (init, run_tests, etc.)
├── screenshots/ # UI and result images
├── .github/ # Workflows (CI/CD)
├── compose.yml # Multi-container setup
├── .env.example # Environment template
├── README.md # You are here
├── LICENSE # MIT License
└── analyze_project.py # Analyzer script

---

## 🧰 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/baiazetsh/flat-rag-system.git
cd flat-rag-system


### 2. Configure environment
cp .env.example .env

Update values in .env (especially Ollama and Qdrant settings if needed).

3. Run via Docker Compose
docker compose up --build
Backend will be available at:
👉 http://localhost:8000/

4. Open the web UI

Upload .txt files, then ask questions via the form — the system retrieves and generates answers instantly.


🧪 Run Tests
docker exec -it rag_backend ./run_tests.sh
cd backend
pytest -v --cov=app
Coverage report will be available at:
backend/test_reports/coverage_all/index.html


## 🖼️ Screenshots

### Main Interface
![Home UI](screenshots/ui_home.png)

### Example Answer
![Result UI](screenshots/ui_result.png)

![Tests](https://github.com/baiazetsh/flat-rag-system/actions/workflows/tests.yml/badge.svg)


🚧 Roadmap
✅ Done

 FastAPI backend with endpoints (/ask, /search, /embed)

 Qdrant integration for semantic search

 Ollama embeddings + generation

 Unit & API tests (pytest)

 Docker Compose environment

🧠 In Progress

 Web UI (Jinja2 templates + upload form)

 Dynamic Qdrant collection creation

 Improved context builder with adaptive chunking

 Logging + timing metrics (embedding, search, LLM)

🔮 Planned

 Deploy demo to Render / HuggingFace Spaces

 Multiple LLMs (Gemma, Qwen, Mistral)

 Real-time streamed responses via WebSocket

 Admin dashboard for document management

 Prometheus + Grafana metrics dashboard


Author: @baiazetsh

Project: Flat RAG System — lightweight local AI retrieval assistant
v 1.0.0