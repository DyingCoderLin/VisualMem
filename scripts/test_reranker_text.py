import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval.local_reranker import LocalReranker

def test_reranker():
    # 1. 加载测试数据
    json_path = "sentence_pairs.json"
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    if not pairs:
        print("错误: json 文件为空")
        return

    # 提取 Query（假设所有 pairs 的 query 相同，或者只取第一个作为测试）
    query = pairs[0][0]
    
    # 将 documents 包装成 reranker 期望的 frames 格式
    # 由于只有文本，我们存入 ocr_text 字段
    frames = []
    for i, pair in enumerate(pairs):
        frames.append({
            "id": f"doc_{i}",
            "ocr_text": pair[1]
        })

    print(f"--- 准备测试 Reranker ---")
    print(f"Query: {query}")
    print(f"待排序文档数: {len(frames)}")

    # 2. 初始化 Reranker (会加载 Qwen3-VL-Reranker-2B)
    print("\n[Step 1] 正在加载模型...")
    reranker = LocalReranker()

    # 3. 执行重排
    print("\n[Step 2] 正在进行重排打分...")
    results = reranker.rerank(query, frames, top_k=len(frames))

    # 4. 输出结果
    print("\n" + "="*50)
    print(f"{'排名':<5} | {'得分':<8} | {'文档内容摘要'}")
    print("-" * 50)
    for i, res in enumerate(results):
        # 截取前 50 个字符显示
        snippet = res['ocr_text'].replace('\n', ' ')[:60] + "..."
        print(f"{i+1:<5} | {res['rerank_score']:<8.4f} | {snippet}")
    print("="*50)

if __name__ == "__main__":
    test_reranker()
