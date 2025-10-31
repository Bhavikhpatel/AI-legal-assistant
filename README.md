This project was deployed in huggingface spaces - https://huggingface.co/spaces/bhavikhpatelhf/legal-assistant

# ⚖️ AI-Powered Legal Assistant

An intelligent Flask-based legal reasoning system that interprets user queries, identifies relevant legal offenses, retrieves contextual data using GraphRAG, and generates expert legal interpretations powered by large language models.

---

## 🧠 Overview

This system serves as an **AI-powered legal assistant** that understands natural language queries and provides structured legal analysis. It combines **graph-based retrieval**, **semantic understanding**, and **LLM reasoning** to generate contextually accurate responses.

The architecture integrates:
- **GraphQuery** for hybrid graph + full-text retrieval  
- **LegalInference** for LLM-driven legal reasoning  
- **Confidence scoring** and **multi-strategy offense matching**  
- **Event-streamed responses** for real-time insights

---

## ⚙️ Features

- Flask REST API with CORS support  
- Graph-enhanced retrieval (GraphRAG architecture)  
- Query understanding via LegalInference LLM  
- Hybrid semantic and keyword search  
- Real-time streaming (`/api/analyze-stream`)  
- Confidence scoring and reranking  
- Extensive structured logging and exception tracing  
- Health monitoring endpoint (`/api/health`)

---

## 🧩 Tech Stack

**Framework:** Flask, Flask-CORS  
**AI Components:** GraphQuery, LegalInference (LLM-based)  
**Model Backend:** BAAI/bge-base-en-v1.5 (embeddings)  
**Core Libraries:** NumPy, dotenv, logging, JSON, datetime  
**Data Structure:** Graph database with full-text index
