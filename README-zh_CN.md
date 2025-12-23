<p align="center">
   <a href="README.md">English</a> | <a href="README-zh_CN.md">简体中文</a>
</p>

<p align="center">
   <h1 align="center">[ VisualMem ]</h1>
   <p align="center">由 24/7 桌面历史记录驱动的个人视觉记忆助手</p>
   <p align="center">开源 | 支持 100% 本地运行 | 开发者友好 | 24/7 屏幕记录与智能检索</p>
</p>

<p align="center" style="font-family: monospace;">
   <code>[ 记录现实，一次一个像素 ]</code>
</p>

<p align="center">
    <a href="./Quickstart.md">
        <img src="https://img.shields.io/badge/开始使用-快速启动-blue?style=for-the-badge" alt="开始使用">
    </a>
</p>

---

# 它是如何工作的？

- **全天候记录**：24/7 自动捕捉屏幕，支持 100% 本地部署并运行。
- **智能索引**：利用 CLIP 视觉模型和 OCR 技术对每一帧进行深度理解并建立索引。
- **自然语言检索**：通过 API 或 Pro GUI，使用自然语言找回任何记忆。

# 为什么选择 VisualMem？

- **上下文即一切**：AI 的能力取决于它拥有的上下文，而最宝贵的上下文就在你的屏幕上。
- **隐私与灵活性**：你的数据属于你。支持 100% 本地处理以保护隐私，同时也支持灵活连接远程 VLM API。
- **专业级体验**：提供现代化 UI，支持时间轴浏览和实时问答。

## ✨ 强大功能与智能工作流

<p align="center">
  <img src="./demo.png" alt="VisualMem GUI 运行截图" width="800">
</p>

VisualMem 不仅仅是一个录屏工具，它是你的**第二大脑**：
- **近期任务协助**：忘记了刚才在代码里改了什么？或者想找回半小时前看到的文档？VisualMem 帮你瞬间定位。
- **电脑记忆检索**：通过自然语言描述，检索你过去几天、几周在电脑上看到的任何视觉信息。

### 🧠 智能 RAG 工作流
为了实现精准检索，我们构建了一套成熟的视觉 RAG 流程：
<p align="center">
  <img src="./visualmem_workflow.png" alt="VisualMem Workflow" width="800">
</p>

1. **多维索引**：系统实时使用 **CLIP** 对图像进行向量编码，并同步通过 **OCR** 提取文字内容。
2. **意图理解**：搜索时，利用 LLM 进行 **Query 扩写**与**时间范围解析**，精准捕捉你的搜索意图。
3. **两阶段检索**：
   - **第一阶段（粗筛）**：Dense（向量）与 Sparse（关键词）检索协同作用，从海量数据中快速召回候选帧。
   - **第二阶段（精排）**：将候选帧交给 **Reranker** 模型进行深度排序，确保最相关的结果排在最前面。
4. **VLM 总结**：最后将精选的关键帧喂给 **VLM 模型**，为你生成准确、带证据的自然语言回答。

## � 演示视频


## �🚀 快速开始

准备好设置 VisualMem 了吗？请按照我们的分步指南进行操作：

👉 **[快速启动指南 (Quickstart)](./Quickstart.md)**

### 硬件要求
- **操作系统**: macOS (推荐 Apple Silicon) 或 Linux (Ubuntu 22.04+)。
- **显卡 (GPU)**: 本地推理（CLIP + OCR）至少需要 **4GB 显存**。若要启用 Reranker 则至少需要 **8GB 显存**。
- **存储**: 默认截屏频率（3s）约消耗磁盘15G/月。

## ✨ 核心特性

- **智能捕捉与过滤**：基于帧差算法，仅在内容变化时记录。
- **多模态检索**：支持语义搜索（CLIP）和 OCR 全文检索的混合多模态检索模式。
- **实时追踪**：实时分析当前屏幕，支持即时问答。
- **时间轴视图**：直观浏览历史记录。

## 🏗️ 技术架构

- **前端**：Electron + React + TypeScript + Vite
- **后端**：Python + FastAPI + SQLite + LanceDB
- **AI 模型**：CLIP + Qwen2.5-VL / GPT-4o

---
*VisualMem - 让你的视觉记忆永不褪色。*

## 📂 文档指南

- [快速启动指南](./Quickstart.md) - 5 分钟内完成安装并运行。
- [架构设计](./docs/ENCODER_RETRIEVER_ARCHITECTURE.md) - 深入了解系统设计。

## 🛠️ 开发与贡献

欢迎提交 Issue 或 Pull Request 来完善这个项目。

---
*VisualMem - 记录现实，一次一个像素。*
