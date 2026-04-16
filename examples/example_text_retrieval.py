#!/usr/bin/env python3
"""
文本检索示例 - 支持 Dense、Sparse、Hybrid 三种模式

演示如何使用文本编码器和检索器进行：
1. Dense Search: 纯语义搜索
2. Sparse Search: FTS (BM25) 关键词搜索
3. Hybrid Search: 混合搜索 + Reranker

模仿 screenpipe 的 RAG 实现
"""

import sys
import os
from pathlib import Path
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from core.encoder.text_encoder import create_text_encoder
from core.retrieval.text_retriever import create_text_retriever
from config import config
from utils.logger import setup_logger

logger = setup_logger("example_text_retrieval")


def print_results(results: list, title: str, max_display: int = 5):
    """格式化打印检索结果"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    if not results:
        print("警告: 没有找到结果")
        return
    
    print(f"\n找到 {len(results)} 个结果（显示前 {max_display} 个）：\n")
    
    for i, result in enumerate(results[:max_display], 1):
        # 提取主要字段
        frame_id = result.get('frame_id', 'N/A')
        timestamp = result.get('timestamp', 'N/A')
        image_path = result.get('image_path', 'N/A')
        
        # 转换为相对路径（可以 Command+点击打开）
        if image_path != 'N/A':
            try:
                rel_path = Path(image_path).relative_to(Path.cwd())
            except:
                rel_path = image_path
        else:
            rel_path = 'N/A'
        
        # 距离或分数
        distance = result.get('_distance', None)
        score = result.get('_relevance_score', None)
        
        print(f"{i}. Frame ID: {frame_id}")
        print(f"   时间: {timestamp}")
        print(f"   图片: {rel_path}")  # 可以 Command+点击打开
        
        if distance is not None:
            print(f"   距离: {distance:.4f} (越小越相似)")
        if score is not None:
            print(f"   分数: {score:.4f}")
        
        print()


def example_dense_search(retriever, query: str):
    """示例1：Dense 语义搜索"""
    print("\n" + "="*70)
    print("示例 1：Dense Search（纯语义搜索）")
    print("="*70)
    print(f"\n查询: \"{query}\"")
    print("\n工作原理:")
    print("  1. 将查询文本转换为 embedding 向量")
    print("  2. 在向量空间中寻找最相似的文本")
    print("  3. 基于余弦相似度/欧氏距离排序")
    print("\n优点: 理解语义，可以找到意思相近但用词不同的结果")
    print("缺点: 对精确关键词匹配不如 FTS")
    
    start_time = time.time()
    results = retriever.retrieve_dense(query, top_k=10)
    elapsed = (time.time() - start_time) * 1000
    
    print_results(results, f"Dense Search 结果 (耗时: {elapsed:.1f}ms)")


def example_sparse_search(retriever, query: str):
    """示例2：Sparse FTS 搜索"""
    print("\n" + "="*70)
    print("示例 2：Sparse Search（FTS / BM25 关键词搜索）")
    print("="*70)
    print(f"\n查询: \"{query}\"")
    print("\n工作原理:")
    print("  1. 使用 BM25 算法进行全文索引搜索")
    print("  2. 基于词频、文档长度等统计特征")
    print("  3. 精确匹配关键词")
    print("\n优点: 快速，精确匹配关键词")
    print("缺点: 不理解语义，必须包含查询词")
    
    start_time = time.time()
    results = retriever.retrieve_sparse(query, top_k=10)
    elapsed = (time.time() - start_time) * 1000
    
    print_results(results, f"Sparse Search 结果 (耗时: {elapsed:.1f}ms)")


def example_hybrid_search(retriever, query: str, reranker: str = "linear"):
    """示例3：Hybrid 混合搜索"""
    print("\n" + "="*70)
    print(f"示例 3：Hybrid Search（混合搜索 + {reranker.upper()} Reranker）")
    print("="*70)
    print(f"\n查询: \"{query}\"")
    print(f"\n工作原理:")
    print("  1. 同时执行 Dense 和 Sparse 搜索")
    print("  2. 合并两组结果")
    print(f"  3. 使用 {reranker.upper()} Reranker 重新排序")
    print("\n优点: 结合两者优势，又快又准")
    print("缺点: 计算开销稍大")
    
    print(f"\nReranker 说明:")
    if reranker == "linear":
        print("  - LinearCombinationReranker: 简单加权组合（默认 70% 语义 + 30% 关键词）")
    elif reranker == "rrf":
        print("  - RRFReranker: Reciprocal Rank Fusion，基于排名倒数融合")
    elif reranker == "cross-encoder":
        print("  - CrossEncoderReranker: 使用 cross-encoder 模型，最准确但较慢")
    
    start_time = time.time()
    results = retriever.retrieve_hybrid(query, top_k=10, reranker=reranker)
    elapsed = (time.time() - start_time) * 1000
    
    print_results(results, f"Hybrid Search 结果 (耗时: {elapsed:.1f}ms)")


def example_comparison(retriever, query: str):
    """示例4：对比三种检索方式"""
    print("\n" + "="*70)
    print("示例 4：三种检索方式对比")
    print("="*70)
    print(f"\n查询: \"{query}\"")
    
    # Dense
    start = time.time()
    dense_results = retriever.retrieve_dense(query, top_k=5)
    dense_time = (time.time() - start) * 1000
    
    # Sparse
    start = time.time()
    sparse_results = retriever.retrieve_sparse(query, top_k=5)
    sparse_time = (time.time() - start) * 1000
    
    # Hybrid
    start = time.time()
    hybrid_results = retriever.retrieve_hybrid(query, top_k=5, reranker="linear")
    hybrid_time = (time.time() - start) * 1000
    
    print("\n" + "-"*70)
    print("性能对比:")
    print("-"*70)
    print(f"  Dense Search:  {dense_time:>6.1f}ms  |  结果数: {len(dense_results)}")
    print(f"  Sparse Search: {sparse_time:>6.1f}ms  |  结果数: {len(sparse_results)}")
    print(f"  Hybrid Search: {hybrid_time:>6.1f}ms  |  结果数: {len(hybrid_results)}")
    
    # 显示 top 3 结果对比
    print("\n" + "-"*70)
    print("Top 3 结果对比:")
    print("-"*70)
    
    for i in range(min(len(dense_results), len(sparse_results), len(hybrid_results))):
        print(f"\n排名 {i+1}:")
        
        # Dense
        dense_text = dense_results[i].get('text', 'N/A')[:60] if i < len(dense_results) else 'N/A'
        print(f"  Dense:  {dense_text}...")
        
        # Sparse
        sparse_text = sparse_results[i].get('text', 'N/A')[:60] if i < len(sparse_results) else 'N/A'
        print(f"  Sparse: {sparse_text}...")
        
        # Hybrid
        hybrid_text = hybrid_results[i].get('text', 'N/A')[:60] if i < len(hybrid_results) else 'N/A'
        print(f"  Hybrid: {hybrid_text}...")


def example_reranker_comparison(retriever, query: str):
    """示例5：对比不同的 Reranker"""
    print("\n" + "="*70)
    print("示例 5：不同 Reranker 对比")
    print("="*70)
    print(f"\n查询: \"{query}\"")
    
    rerankers = ["linear", "rrf"]  # cross-encoder 太慢，可选启用
    
    for reranker in rerankers:
        print(f"\n--- {reranker.upper()} Reranker ---")
        
        start = time.time()
        results = retriever.retrieve_hybrid(query, top_k=5, reranker=reranker)
        elapsed = (time.time() - start) * 1000
        
        print(f"耗时: {elapsed:.1f}ms | 结果数: {len(results)}")
        
        if results:
            for i, r in enumerate(results, 1):
                image_path = r.get('image_path', 'N/A')
                # 取出它的相对运行目录的相对路径
                image_path = Path(image_path).relative_to(Path.cwd())
                score = r.get('_relevance_score', r.get('_distance', 'N/A'))
                print(f"  {i}. {image_path}... (分数: {score})")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("文本检索示例 - Dense / Sparse / Hybrid")
    print("="*70)
    
    # 1. 初始化编码器和检索器
    print("\n[1/3] 初始化 TextEncoder（基于 CLIP）...")
    encoder = create_text_encoder(model_name=config.EMBEDDING_MODEL)
    print(f"  - 底层 CLIP 模型: {config.EMBEDDING_MODEL}")
    print(f"  - Embedding 维度: {encoder.embedding_dim}")
    print(f"  - 设备: {encoder.device}")
    print(f"  - 说明: 与 CLIPEncoder 共享底层模型（控制变量）")
    
    print("\n[2/3] 初始化文本检索器...")
    retriever = create_text_retriever(
        encoder=encoder,
        db_path=config.TEXT_LANCEDB_PATH,
        table_name="ocr_texts",
        default_reranker="linear"
    )
    print("  - 优势：Embedding 与图像检索在同一空间")
    
    # 3. 检查数据
    print("\n[3/3] 检查数据库状态...")
    stats = retriever.get_stats()
    print("\n数据库信息:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")
    
    if stats.get("total_rows", 0) == 0:
        print("\n" + "="*70)
        print("警告: 数据库为空！")
        print("="*70)
        print("\n请先运行以下命令构建文本索引:")
        print("  python scripts/rebuild_text_index.py")
        print("\n或者确保 main.py 正在运行并开启了 OCR。")
        return
    
    # 4. 运行示例
    print("\n" + "="*70)
    print("开始检索示例")
    print("="*70)
    
    # 示例查询
    queries = [
        "机器学习相关的代码",       # 语义查询
        "Error: connection timeout",  # 精确关键词查询
        "python pandas dataframe",     # 技术术语
    ]
    
    # 如果数据库有数据，使用第一个查询进行演示
    demo_query = queries[0]
    
    try:
        # 示例 1: Dense Search
        example_dense_search(retriever, demo_query)
        
        # 示例 2: Sparse Search
        example_sparse_search(retriever, demo_query)
        
        # 示例 3: Hybrid Search (Linear)
        example_hybrid_search(retriever, demo_query, reranker="linear")
        
        # 示例 4: 三种方式对比
        example_comparison(retriever, demo_query)
        
        # 示例 5: Reranker 对比
        example_reranker_comparison(retriever, demo_query)
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
        print(f"\n错误: {e}")
    
    print("\n" + "="*70)
    print("示例完成")
    print("="*70)
    print("\n提示:")
    print("  - 修改 demo_query 变量尝试不同查询")
    print("  - 调整 top_k 参数控制返回结果数量")
    print("  - 尝试不同的 reranker: linear, rrf, cross-encoder")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
