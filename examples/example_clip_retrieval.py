#!/usr/bin/env python3
"""
多模态 RAG 检索示例

演示如何使用 CLIP + LanceDB 进行文本→图像和图像→图像的检索
"""

import sys
from pathlib import Path
from PIL import Image
import os

# 添加项目路径（examples/ 的父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))

# 确保工作目录是项目根目录（visualmem/）
os.chdir(Path(__file__).parent.parent)

from config import config
from core.encoder import create_encoder
from core.storage.lancedb_storage import LanceDBStorage
from core.retrieval.image_retriever import ImageRetriever
from utils.logger import setup_logger

logger = setup_logger("example")


def example_1_text_to_image():
    """
    示例1：文本查询图像
    使用自然语言描述来搜索相关截图
    """
    print("\n" + "="*60)
    print("示例1：文本查询图像（Text → Image）")
    print("="*60)
    
    # 1. 初始化组件
    logger.info("初始化编码器和 LanceDB...")
    encoder = create_encoder(model_name=config.EMBEDDING_MODEL)
    storage = LanceDBStorage(
        db_path=config.LANCEDB_PATH,
        embedding_dim=encoder.embedding_dim
    )
    retriever = ImageRetriever(encoder=encoder, storage=storage, top_k=5)
    
    # 2. 文本查询
    query = "哪张图片里有人物出现"
    logger.info(f"查询: '{query}'")
    
    results = retriever.retrieve_by_text(query, top_k=5)
    
    # 3. 显示结果
    print(f"\n找到 {len(results)} 个相关结果：")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 距离: {result.get('distance', 0):.3f}")
        print(f"   时间: {result['timestamp']}")
        # 显示相对路径，可以 Command+点击打开
        if 'image_path' in result:
            # 转换为相对路径
            image_path = Path(result['image_path'])
            try:
                rel_path = image_path.relative_to(Path.cwd())
                print(f"   图片: {rel_path}")
            except ValueError:
                print(f"   图片: {result['image_path']}")
        if 'ocr_text' in result and result['ocr_text']:
            print(f"   OCR: {result['ocr_text'][:80]}...")


def example_2_image_to_image():
    """
    示例2：图像查询图像
    找到与给定截图视觉上相似的其他截图
    """
    print("\n" + "="*60)
    print("示例2：图像查询图像（Image → Image）")
    print("="*60)
    
    # 初始化
    encoder = create_encoder(model_name=config.EMBEDDING_MODEL)
    storage = LanceDBStorage(
        db_path=config.LANCEDB_PATH,
        embedding_dim=encoder.embedding_dim
    )
    retriever = ImageRetriever(encoder=encoder, storage=storage, top_k=3)
    
    # 使用图像路径查询
    query_image_path = os.path.join(config.IMAGE_STORAGE_PATH, "20251028/38e34b32e2ba2b4a.jpg")
    logger.info(f"查询图像: {query_image_path}")
    
    results = retriever.retrieve_by_image_path(query_image_path, top_k=3)
    
    print(f"\n找到 {len(results)} 个视觉相似的结果")
    for i, result in enumerate(results, 1):
        distance = result.get('distance', 0)
        print(f"{i}. 距离: {distance:.3f}")
        print(f"   时间: {result['timestamp']}")
        if 'image_path' in result:
            image_path = Path(result['image_path'])
            try:
                rel_path = image_path.relative_to(Path.cwd())
                print(f"   图片: {rel_path}")
            except ValueError:
                print(f"   图片: {result['image_path']}")


def example_3_stats():
    """
    示例3：查看统计信息
    """
    print("\n" + "="*60)
    print("示例3：检索器统计信息")
    print("="*60)
    
    encoder = create_encoder(model_name=config.EMBEDDING_MODEL)
    storage = LanceDBStorage(
        db_path=config.LANCEDB_PATH,
        embedding_dim=encoder.embedding_dim
    )
    retriever = ImageRetriever(encoder=encoder, storage=storage)
    
    stats = retriever.get_stats()
    
    print(f"\n检索器配置：")
    print(f"  • 类型: {stats['retriever_type']}")
    print(f"  • 编码器: {stats['encoder_model']}")
    print(f"  • Embedding 维度: {stats['embedding_dim']}")
    print(f"  • 支持文本查询: {stats['supports_text_query']}")
    print(f"  • 支持图像查询: {stats['supports_image_query']}")
    print(f"  • 默认 top_k: {stats['top_k']}")
    print(f"\n数据库统计：")
    print(f"  • 总帧数: {stats['storage_stats']['total_frames']}")


def main():
    """主函数：运行示例"""
    print("\n" + "="*60)
    print("CLIP 多模态检索系统 - 使用示例")
    print("="*60)
    print("\n特性：")
    print("  ✓ Dense 向量搜索（去掉 hybrid/sparse）")
    print("  ✓ 文本查询图像（CLIP 文本编码器）")
    print("  ✓ 图像查询图像（CLIP 图像编码器）")
    print("  ✓ 图文对齐的共享空间（512维）")
    
    try:
        # 运行示例
        example_1_text_to_image()       # 文本→图像
        # example_2_image_to_image()      # 图像→图像
        # example_3_stats()               # 统计信息
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        print("\n请确保：")
        print("1. 已经运行 main.py 捕获了一些截图")
        print("2. 已经切换到 vector 模式（STORAGE_MODE=vector）")
        print("3. 已经安装了必要的依赖：pip install transformers torch lancedb")


if __name__ == "__main__":
    main()

