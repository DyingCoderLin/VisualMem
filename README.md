<p align="center">
   <a href="README.md">English</a> | <a href="README-zh_CN.md">简体中文</a>
</p>

<p align="center">
   <h1 align="center">VisualMem</h1>
   <p align="center">Personal visual memory assistant powered by 24/7 desktop history</p>
   <p align="center">open source | 100% local capable | dev friendly | 24/7 screen recording & smart retrieval</p>
</p>

<p align="center" style="font-family: monospace;">
   <code>[ recording reality, one pixel at a time ]</code>
</p>

<p align="center">
    <a href="./Quickstart.md">
        <img src="https://img.shields.io/badge/Get%20Started-Quickstart-blue?style=for-the-badge" alt="Get Started">
    </a>
</p>

---

# How it works?

- **24/7 Recording**: Automatically capture your screen 24/7. Supports 100% local deployment.
- **Smart Indexing**: Deeply understand and index every frame using CLIP vision models and OCR.
- **Natural Language Retrieval**: Find any memory using natural language via API or Pro GUI.

# Why VisualMem?

- **Context is Everything**: AI is only as good as its context, and the most valuable context is on your screen.
- **Privacy & Flexibility**: Your data belongs to you. Supports 100% local processing for maximum privacy, while remaining flexible to connect with remote VLM APIs.
- **Professional Experience**: Modern Electron-based UI with timeline browsing and real-time Q&A.

## ✨ Powerful Features & Intelligent Workflow

<p align="center">
  <img src="./docs/demo.png" alt="VisualMem GUI Screenshot" width="800">
</p>

VisualMem is more than just a screen recorder; it's your **second brain**:
- **Recent Task Assistance**: Forgot what you just changed in your code? Or want to find a document you saw 30 minutes ago? VisualMem helps you locate it instantly.
- **Computer Memory Retrieval**: Use natural language to retrieve any visual information you've seen on your computer over the past days or weeks.

### 🧠 Intelligent RAG Workflow
To achieve precise retrieval, we've built a sophisticated visual RAG pipeline:
<p align="center">
  <img src="./docs/visualmem_workflow.png" alt="VisualMem Workflow" width="800">
</p>

1. **Multidimensional Indexing**: The system uses **CLIP** for real-time image vector encoding and **OCR** for text extraction.
2. **Intent Understanding**: During search, LLMs perform **Query Expansion** and **Time Range Extraction** to capture your search intent accurately.
3. **Two-Stage Retrieval**:
   - **Stage 1 (Coarse Filtering)**: Dense (vector) and Sparse (keyword) search work together to quickly recall candidate frames from massive data.
   - **Stage 2 (Reranking)**: Candidate frames are passed to a **Reranker** model for deep sorting, ensuring the most relevant results come first.
4. **VLM Summarization**: Finally, the selected keyframes are fed into a **VLM model** to generate accurate, evidence-based natural language answers.

## 📺 Demo Video

<p align="center">
  <video src="https://github.com/user-attachments/assets/1150db7c-4c93-4ed4-9a3c-a2fcf86f29fe" width="800" controls></video>
</p>

## 🚀 Get Started

Ready to set up VisualMem? Follow our step-by-step guide:

👉 **[Quickstart Guide](./Quickstart.md)**

### Hardware Requirements
- **OS**: macOS (Apple Silicon recommended) or Linux (Ubuntu 22.04+).
- **GPU**: Minimum **4GB VRAM** for local inference (CLIP + OCR). At least **8GB VRAM** required to enable Reranker.
- **Storage**: Default capture frequency (3s) consumes ~15GB/month.

## ✨ Core Features

- **Smart Capture & Filtering**: Records only when content changes significantly.
- **Multimodal Retrieval**: Supports hybrid multimodal retrieval mode with semantic search (CLIP) and OCR full-text search.
- **Real-time Tracing**: Analyze current screen content for instant Q&A.
- **Timeline View**: Browse your history intuitively like a social media feed.

## 🏗️ Technical Architecture

- **Frontend**: Electron + React + TypeScript + Vite
- **Backend**: Python + FastAPI + SQLite + LanceDB
- **AI Models**: CLIP + Qwen3-VL as a reranker + Qwen3-VL / GPT-5 / any VLM model you like

---
*VisualMem - Never let your visual memory fade.*

## 📂 Documentation

- [Quickstart Guide](./Quickstart.md) - Get up and running in 5 minutes.
- [Architecture Design](./docs/ENCODER_RETRIEVER_ARCHITECTURE.md) - Deep dive into the system design.

## 🛠️ Development

Contributions are welcome! Please feel free to submit a Pull Request.

---
*VisualMem - Recording reality, one pixel at a time.*

