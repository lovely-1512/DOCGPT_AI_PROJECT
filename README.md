---
title: DocGPT AI
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---

# 📚 DocGPT AI — Evidence-First Document Intelligence

Upload PDF / PPTX / DOCX / TXT / CSV / XLSX → hybrid retrieval (FAISS + BM25 with
Reciprocal Rank Fusion) → Cross-Encoder reranking → Groq LLM grounded answer
with citations.

## Features
- Chat with citations and confidence scoring
- Compare two documents
- Study mode (MCQ, flashcards, viva, revision notes)
- Quiz mode with scoring
- Voice input (Whisper via Groq)

## Setup
Set the `GROQ_API_KEY` secret in **Space Settings → Repository secrets**.
