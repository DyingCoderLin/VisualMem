#!/usr/bin/env python3
"""
OCR Fallback 查询示例

演示如何使用 SQLite + OCR 文本进行 fallback 查询
（不使用多模态 VLM）
"""

import sys
from pathlib import Path

# 添加项目路径（examples/ 的父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage.sqlite_storage import SQLiteStorage
from PIL import Image
from utils.logger import setup_logger

logger = setup_logger("example_ocr_fallback")


def example_text_search():
    """
    示例1：基于 OCR 文本的全文搜索
    """
    print("\n" + "="*60)
    print("示例1：OCR 文本全文搜索")
    print("="*60)
    
    # 连接到 SQLite 数据库
    sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
    
    # 搜索关键词
    query = "python"
    print(f"\n搜索关键词: '{query}'")
    
    results = sqlite_storage.search_by_text(query, limit=5, min_confidence=0.0)
    
    print(f"\n找到 {len(results)} 个结果：")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Frame ID: {result['frame_id']}")
        print(f"   时间: {result['timestamp']}")
        print(f"   图片: {result['image_path']}")
        print(f"   OCR置信度: {result['ocr_confidence']:.2f}")
        print(f"   文本预览: {result['ocr_text'][:100]}...")


def example_recent_frames():
    """
    示例2：获取最近的帧（带 OCR 文本）
    """
    print("\n" + "="*60)
    print("示例2：最近的帧（带 OCR）")
    print("="*60)
    
    sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
    
    results = sqlite_storage.get_recent_frames(limit=10)
    
    print(f"\n最近的 {len(results)} 帧：")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['timestamp']}")
        print(f"   Frame ID: {result['frame_id']}")
        if result['ocr_text']:
            print(f"   OCR文本: {result['ocr_text'][:80]}...")
        else:
            print(f"   OCR文本: (无)")


def example_stats():
    """
    示例3：查看 OCR fallback 统计信息
    """
    print("\n" + "="*60)
    print("示例3：OCR Fallback 统计")
    print("="*60)
    
    sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
    
    stats = sqlite_storage.get_stats()
    
    print(f"\n数据库统计：")
    print(f"  - 总帧数: {stats['total_frames']}")
    print(f"  - OCR识别结果数: {stats['total_ocr_results']}")
    print(f"  - 总文本长度: {stats['total_text_length']} 字符")
    print(f"  - 数据库路径: {stats['db_path']}")
    print(f"  - 存储模式: {stats['storage_mode']}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("OCR Fallback 查询系统")
    print("="*60)
    print("\n特性：")
    print("  - 基于 OCR 文本的全文搜索（FTS5）")
    print("  - 不依赖多模态 VLM")
    print("  - 支持中英文搜索")
    print("  - SQLite 轻量级存储")
    
    print("\n使用场景：")
    print("  1. VLM 不可用时的 fallback")
    print("  2. 需要精确文本匹配")
    print("  3. 快速关键词搜索")
    
    try:
        # 运行示例
        example_text_search()      # 文本搜索
        # example_recent_frames()    # 最近的帧
        # example_stats()            # 统计信息
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
        print(f"\n错误: {e}")
        print("\n请确保：")
        print("1. 已经运行 main.py 并启用了 OCR（ENABLE_OCR=true）")
        print(f"2. OCR 数据库存在: {config.OCR_DB_PATH}")
        print("3. 已经安装了 pytesseract：pip install pytesseract")


if __name__ == "__main__":
    main()
