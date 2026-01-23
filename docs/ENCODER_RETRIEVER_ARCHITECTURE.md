# Encoder & Retriever 架构文档

## 📋 概述

本文档描述了 VisualMem 项目中 **编码器（Encoder）** 和 **检索器（Retriever）** 的统一架构设计。

通过抽象基类，我们为文本和图像编码/检索提供了一致的接口，方便扩展和维护。

---

## 🏗️ 架构设计

### 1. VLM RAG 工作流 (VLM RAG Workflow)

VisualMem 实现了一个成熟且完整的，基于视觉 RAG 的个人智能助手工作流，确保从海量屏幕截图中精准找回信息：

1.  **Query Rewrite (查询扩写)**: 
    - 利用 LLM 对原始查询进行语义扩写（Expand）。
    - 自动解析查询中的时间范围（如“昨天下午”、“上周五”）。
2.  **Hybrid Search (混合检索)**:
    - **Dense Search (向量检索)**: 使用 CLIP 模型将查询转化为向量，在 LanceDB 中进行语义匹配。
    - **Sparse Search (全文检索)**: 利用 SQLite FTS5 对 OCR 提取的文本进行关键词匹配。
3.  **Two-stage Search (两阶段检索)**:
    - 第一阶段：分别从向量库和 OCR 库中检索 Top-K 候选帧。
    - 第二阶段：对两路结果进行合并与去重（Dedup）。
4.  **Reranking (精排)**:
    - 如果启用了 Rerank 模块，使用交叉熵模型（Cross-Encoder）对候选帧进行二次打分，筛选出最相关的 Top-N 帧。
5.  **VLM Summarization (VLM 总结)**:
    - 将筛选出的关键帧及其时间戳提交给多模态大模型（如 Qwen3-VL 或 GPT-5）。
    - VLM 结合视觉信息和上下文，给出最终的自然语言回答。

---

### 2. 编码器架构（Encoder）

```
BaseEncoder (抽象基类)
    ├── TextEncoderInterface (文本编码接口)
    │   └── TextEncoder (sentence-transformers)
    │
    ├── ImageEncoderInterface (图像编码接口)
    │   └── (待扩展: DINOv2, ResNet 等)
    │
    └── MultiModalEncoderInterface (多模态编码接口)
        └── CLIPEncoder (CLIP 多模态)
```

#### 1.1 抽象基类

**`BaseEncoder`**
- 所有编码器的基类
- 定义通用接口：`encode()`, `encode_batch()`, `get_embedding_dim()`

**`TextEncoderInterface`**
- 纯文本编码接口
- 方法：`encode_text()`, `encode_text_batch()`

**`ImageEncoderInterface`**
- 纯图像编码接口
- 方法：`encode_image()`, `encode_image_batch()`

**`MultiModalEncoderInterface`**
- 多模态编码接口（继承 Text + Image）
- 智能 `encode()` 方法：根据输入类型自动选择

#### 1.2 具体实现

**`TextEncoder`** (实现 `TextEncoderInterface`)
- **模型**: sentence-transformers
- **默认模型**: `google/siglip-large-patch16-384` (1024维)
- **特点**: 纯文本语义编码，快速轻量
- **用途**: OCR 文本的语义检索

**`CLIPEncoder`** (实现 `MultiModalEncoderInterface`)
- **模型**: OpenAI CLIP
- **默认模型**: `google/siglip-large-patch16-384` (1024维)
- **特点**: 文本和图像共享 embedding 空间
- **用途**: 多模态检索（文本→图像，图像→图像）

---

### 2. 检索器架构（Retriever）

```
BaseRetriever (抽象基类)
    ├── TextRetrieverInterface (文本检索接口)
    │   └── TextRetriever (LanceDB + FTS)
    │       • 数据库: visualmem_textdb
    │       • 表名: ocr_texts
    │       • 支持: Dense, Sparse, Hybrid
    │
    ├── ImageRetrieverInterface (图像检索接口)
    │   └── (待扩展)
    │
    └── MultiModalRetrieverInterface (多模态检索接口)
        └── ImageRetriever (CLIP + LanceDB)
            • 数据库: visualmem_db/screen_analyses.lance
            • 表名: screen_analyses
            • 支持: Dense only
```

#### 2.1 抽象基类

**`BaseRetriever`**
- 所有检索器的基类
- 统一接口：`retrieve()`, `retrieve_dense()`, `retrieve_sparse()`, `retrieve_hybrid()`
- 统一配置：`DEFAULT_DB_PATH`, `DEFAULT_TABLE_NAME`

**`TextRetrieverInterface`**
- 纯文本检索接口
- 默认数据库：`./visualmem_textdb`
- 默认表名：`ocr_texts`

**`ImageRetrieverInterface`**
- 图像检索接口
- 默认数据库：`./visualmem_db`
- 默认表名：`screen_analyses`

**`MultiModalRetrieverInterface`**
- 多模态检索接口
- 额外方法：`retrieve_by_text()`, `retrieve_by_image()`, `retrieve_by_image_path()`

#### 2.2 具体实现

**`TextRetriever`** (实现 `TextRetrieverInterface`)

| 特性 | 说明 |
|------|------|
| **编码器** | `CLIPEncoder` (CLIP 文本编码) |
| **数据库** | LanceDB (`./visualmem_textdb`) |
| **表名** | `ocr_texts` |
| **数据来源** | SQLite OCR 数据库 |
| **检索模式** | ✅ Dense, ✅ Sparse (FTS/BM25), ✅ Hybrid |
| **Reranker** | Linear, RRF, Cross-Encoder |
| **特点** | 🌟 与图像检索使用相同 embedding 空间 |

**检索模式详解**:

1. **Dense Search** (纯语义)
   - 使用 sentence-transformers 生成 embedding
   - 向量相似度搜索
   - 适合：语义查询（如"机器学习相关代码"）

2. **Sparse Search** (FTS/BM25)
   - 使用 LanceDB 的 FTS 索引
   - 基于 BM25 算法的关键词匹配
   - 适合：精确关键词（如"Error: timeout"）

3. **Hybrid Search** (混合)
   - 同时执行 Dense + Sparse
   - 使用 Reranker 重新排序
   - 最佳实践：结合语义和关键词

**`ImageRetriever`** (实现 `MultiModalRetrieverInterface`)

| 特性 | 说明 |
|------|------|
| **编码器** | `CLIPEncoder` (CLIP) |
| **数据库** | LanceDB (`./visualmem_db`) |
| **表名** | `screen_analyses` |
| **数据来源** | 屏幕截图 + CLIP embedding |
| **检索模式** | ✅ Dense only |
| **查询类型** | 文本→图像，图像→图像 |

---

## 🔄 数据流

### 文本检索流程（使用 CLIP）

```
用户查询 "机器学习代码"
    ↓
CLIPEncoder.encode_text()  ← 使用 CLIP 文本塔
    ↓
query_embedding [512维]  ← 与图像 embedding 同空间！
    ↓
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Dense Search        │ Sparse Search       │ Hybrid Search       │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ 向量相似度          │ BM25 关键词匹配     │ Dense + Sparse      │
│ LanceDB.search()    │ LanceDB.search(fts) │ + Reranker          │
└─────────────────────┴─────────────────────┴─────────────────────┘
    ↓
返回相关 OCR 文本及图片路径

💡 关键优势：
- OCR 文本 embedding 与原始截图 embedding 在同一空间
- 支持跨模态检索：查询 OCR 文本时也能找到原始截图
- 控制变量：文本和图像使用相同的 CLIP 模型
```

### 图像检索流程

```
用户查询 "代码截图" 或 上传图片
    ↓
CLIPEncoder.encode_text() / encode_image()
    ↓
query_embedding [512维]
    ↓
Dense Search (向量相似度)
    ↓
LanceDB.search()
    ↓
返回相似截图及元数据
```

---

## 📂 目录结构

```
visualmem/
├── core/
│   ├── encoder/
│   │   ├── __init__.py              # 模块导出
│   │   ├── base_encoder.py          # 编码器抽象基类
│   │   ├── text_encoder.py          # 文本编码器 (sentence-transformers)
│   │   └── clip_encoder.py          # CLIP 多模态编码器
│   │
│   └── retrieval/
│       ├── __init__.py              # 模块导出
│       ├── base_retriever.py        # 检索器抽象基类
│       ├── text_retriever.py        # 文本检索器 (Dense/Sparse/Hybrid)
│       └── image_retriever.py       # 图像检索器 (CLIP)
│
├── examples/
│   ├── example_text_retrieval.py    # 文本检索示例
│   └── example_clip_retrieval.py    # 图像检索示例
│
└── scripts/
    ├── rebuild_index.py             # 重建图像索引
    └── rebuild_text_index.py        # 重建文本索引
```

---

## 🚀 使用示例

### 1. 文本检索（使用 CLIP）

```python
from core.encoder import CLIPEncoder
from core.retrieval import create_text_retriever
from config import config

# 初始化 - 使用 CLIP 编码器
encoder = CLIPEncoder(model_name=config.EMBEDDING_MODEL)
retriever = create_text_retriever(encoder=encoder)

# Dense 检索（语义）
results = retriever.retrieve_dense("机器学习代码", top_k=10)

# Sparse 检索（关键词）
results = retriever.retrieve_sparse("Error: timeout", top_k=10)

# Hybrid 检索（混合）
results = retriever.retrieve_hybrid(
    "python pandas",
    top_k=10,
    reranker="linear"  # 或 "rrf", "cross-encoder"
)
```

### 2. 图像检索

```python
from core.encoder import CLIPEncoder
from core.storage.lancedb_storage import LanceDBStorage
from core.retrieval import ImageRetriever
from PIL import Image

# 初始化
encoder = CLIPEncoder()
storage = LanceDBStorage(db_path="./visualmem_db")
retriever = ImageRetriever(encoder=encoder, storage=storage)

# 文本→图像检索
results = retriever.retrieve_by_text("代码截图", top_k=5)

# 图像→图像检索
query_image = Image.open("example.jpg")
results = retriever.retrieve_by_image(query_image, top_k=5)
```

## 📝 总结

通过抽象基类设计，我们实现了：

✅ **统一接口**: 文本和图像编码/检索共享一致的 API  
✅ **灵活扩展**: 轻松添加新的编码器和检索器  
✅ **清晰分离**: 数据库路径、表名等配置明确分离  
✅ **多种模式**: Dense、Sparse、Hybrid 三种检索满足不同需求  
✅ **性能优化**: 批量编码、FTS 索引、Reranker 选择  
✅ **跨模态支持**: 🌟 文本和图像使用相同 CLIP 模型，embedding 空间一致  
✅ **控制变量**: 便于对比 Dense/Sparse/Hybrid 的性能差异  

---

## 🔗 相关文档

- [快速开始](../QUICKSTART.md)
- [查询策略](./QUERY_STRATEGY.md)
- [架构总览](./ARCHITECTURE_FINAL.md)

