# Quickstart Guide

This guide will walk you through the installation and setup of VisualMem.

## 📋 Prerequisites

- **Python**: 3.12
- **Node.js**: v20.x or higher (Tested with v23.9.0)
- **npm**: 10.x or higher (Tested with 10.9.2)
- **VLM Service**: An OpenAI-compatible multimodal LLM service (e.g., vLLM, Ollama, or OpenAI API).
- **Hardware**: 
    - **GPU**: Minimum **4GB VRAM** for local inference (CLIP/OCR). 8GB+ recommended for reranker enabled.
    - **OS**: macOS (Apple Silicon) or Linux.

## 🛠️ Step 1: Backend Configuration

### 1. Clone the Repository
```bash
git clone <repository-url>
cd VLM-research
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or you can create by conda if you like
```

### 3. Install Python Dependencies
Choose the requirement file based on your OS:
- **macOS**:
  ```bash
  pip install -r requirements_macos.txt
  ```
- **Linux (CUDA)**:
  ```bash
  pip install -r requirements_linux_cuda.txt
  ```

### 4. Configure Environment Variables
Copy the example environment file and edit it:
```bash
cp env.example .env
```
Edit the `.env` file to ensure the following keys are correctly configured:
```ini
# VLM API Configuration (for understanding screenshots)
VLM_API_URI=http://localhost:8081/v1  # VLM service address (Avoid port 8080, used by backend)
VLM_API_MODEL=Qwen3-VL-8B-Instruct  # Your VLM Model name
VLM_API_KEY=your_api_key_if_needed # if deployed by your self, just None will work

# Storage Path (Optional, defaults to ./visualmem_storage, no need to change if your SSD has enough space)
# STORAGE_BASE_PATH=./visualmem_storage
```

## 🚀 Step 2: Launch VisualMem

The frontend handles screen capture and UI, while automatically managing the backend service for indexing and retrieval.

### 1. Navigate to Frontend Directory
```bash
cd pro_gui
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Start the Application
For development (with hot-reload):
```bash
npm run dev
```
*Note: Use `npm run dev:no-devtools` to hide the Chrome DevTools.*

The VisualMem Pro GUI will launch, and the backend service will start automatically in the background.

## 📖 Usage Guide

1.  **Start Recording**: Click the "Start Recording" button at the top of the GUI. The system will automatically capture frames based on screen changes.
2.  **Browse Timeline**: On the Home (Timeline) page, scroll horizontally to view screenshots from each day.
3.  **Smart Search**: Enter any query in the search bar (e.g., "The React documentation I was reading earlier"). The system combines vector search and VLM to provide answers and relevant screenshots.
4.  **Real-time Tracing**: Switch to the "Real-time Tracing" page to ask questions about your current screen content instantly.

## ❓ Troubleshooting

- **Images not displaying**: Ensure the backend service is running and the storage path in `.env` is correct.
- **VLM connection failed**: Verify that `VLM_API_URI` is accessible from your machine.

---
Back to [README.md](./README.md)

