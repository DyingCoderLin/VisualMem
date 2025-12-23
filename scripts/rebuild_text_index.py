#!/usr/bin/env python3
"""
重建文本索引脚本

功能：
1. 从 SQLite OCR 数据库中读取所有 OCR 文本
2. 使用 TextEncoder 生成 embedding
3. 存储到 LanceDB 用于语义搜索
4. 创建 FTS 索引用于关键词搜索
5. 支持 Dense、Sparse、Hybrid 三种检索方式

模仿 screenpipe 的 RAG 实现
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from core.encoder.text_encoder import create_text_encoder
from core.storage.sqlite_storage import SQLiteStorage
from utils.logger import setup_logger
import lancedb

logger = setup_logger("rebuild_text_index")


def rebuild_text_index(
    sqlite_db_path: str = None,
    lance_db_path: str = None,
    table_name: str = "ocr_texts",
    clip_model: str = None,
    batch_size: int = 32,
    clear_existing: bool = True,
    confirm: bool = True
):
    """
    重建文本索引（使用 TextEncoder，底层为 CLIP）
    """
    print("\n" + "="*70)
    print("重建文本索引 - Dense + Sparse + Hybrid 检索")
    print("="*70)
    
    sqlite_db_path = sqlite_db_path or config.OCR_DB_PATH
    lance_db_path = lance_db_path or config.TEXT_LANCEDB_PATH
    clip_model = clip_model or config.CLIP_MODEL

    sqlite_path = Path(sqlite_db_path)
    lance_path = Path(lance_db_path)
    
    print("\n配置:")
    print(f"  • SQLite 数据库: {sqlite_path}")
    print(f"  • LanceDB 数据库: {lance_path}")
    print(f"  • 表名: {table_name}")
    print(f"  • CLIP 模型: {clip_model}")
    print(f"  • 批处理大小: {batch_size}")
    print(f"  • 清空现有数据: {clear_existing}")
    
    # 1. 检查 SQLite 数据库
    print("\n[1/6] 检查 SQLite OCR 数据库...")
    if not sqlite_path.exists():
        logger.error(f"SQLite 数据库不存在: {sqlite_path}")
        print(f"\n❌ 错误: SQLite 数据库 {sqlite_path} 不存在")
        print("\n请先运行以下命令之一:")
        print("  • python main.py  (开启 OCR 捕捉)")
        print("  • python scripts/rebuild_sqlite.py  (重建 SQLite 数据库)")
        return
    
    sqlite_storage = SQLiteStorage(db_path=str(sqlite_path))
    stats = sqlite_storage.get_stats()
    
    print(f"✓ SQLite 数据库已连接")
    print(f"  • 总帧数: {stats['total_frames']}")
    print(f"  • OCR 结果数: {stats['total_ocr_results']}")
    
    if stats['total_frames'] == 0:
        logger.warning("SQLite 数据库为空")
        print("\n⚠️ SQLite 数据库为空，无法构建索引")
        return
    
    # 2. 读取所有 OCR 文本
    print("\n[2/6] 从 SQLite 读取 OCR 文本...")
    
    # 使用 search_by_text 获取所有数据（通过一个通配查询）
    # 或者直接查询数据库
    import sqlite3
    conn = sqlite3.connect(sqlite_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            f.frame_id, 
            f.timestamp, 
            f.image_path,
            o.text as ocr_text,
            o.ocr_engine,
            o.confidence
        FROM frames f
        JOIN ocr_text o ON f.frame_id = o.frame_id
        WHERE LENGTH(o.text) > 0
        ORDER BY f.timestamp DESC
    """)
    
    all_data = []
    for row in cursor.fetchall():
        all_data.append({
            "frame_id": row["frame_id"],
            "timestamp": row["timestamp"],
            "image_path": row["image_path"],
            "text": row["ocr_text"],
            "ocr_engine": row["ocr_engine"],
            "confidence": row["confidence"] if row["confidence"] else 0.0
        })
    
    conn.close()
    
    print(f"✓ 读取了 {len(all_data)} 条 OCR 文本记录")
    
    if len(all_data) == 0:
        print("\n⚠️ 没有有效的 OCR 文本，无法构建索引")
        return
    
    # 3. 初始化 TextEncoder（基于 CLIP）
    print("\n[3/6] 初始化 TextEncoder（基于 CLIP）...")
    encoder = create_text_encoder(model_name=clip_model)
    embedding_dim = encoder.get_embedding_dim()
    print(f"✓ TextEncoder 已加载")
    print(f"  • 底层模型: CLIP {clip_model}")
    print(f"  • 维度: {embedding_dim}")
    print(f"  • 设备: {encoder.device}")
    print(f"  • 说明: 与 CLIPEncoder 共享底层CLIP 模型")
    
    # 4. 生成 Embeddings（使用 CLIP 文本编码）
    print("\n[4/6] 生成文本 embeddings（使用 CLIP）...")
    
    texts = [d["text"] for d in all_data]
    
    # 批量生成 embeddings（带进度条）
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="生成 embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = encoder.encode_text_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    print(f"✓ 生成了 {len(all_embeddings)} 个 CLIP 文本 embeddings")
    
    # 5. 准备 LanceDB 数据
    print("\n[5/6] 准备 LanceDB 数据...")
    
    # 添加 embeddings 到数据
    for i, data_item in enumerate(all_data):
        data_item["vector"] = all_embeddings[i]
    
    # 检查是否需要清空现有表
    if clear_existing and lance_path.exists() and confirm:
        response = input(f"\n警告：LanceDB 数据库 {lance_path} 已存在。是否清空并重建？(y/N): ").lower()
        if response != 'y':
            print("操作已取消。")
            return
    
    # 连接 LanceDB
    db = lancedb.connect(str(lance_path))
    
    # 删除旧表（如果存在）
    if clear_existing and table_name in db.table_names():
        db.drop_table(table_name)
        logger.info(f"已删除旧表: {table_name}")
    
    # 6. 创建表并插入数据
    print("\n[6/6] 创建 LanceDB 表并插入数据...")
    
    try:
        # 创建表
        table = db.create_table(
            table_name,
            data=all_data,
            mode="overwrite" if clear_existing else "create"
        )
        
        print(f"✓ 表 '{table_name}' 创建成功")
        print(f"  • 总行数: {table.count_rows()}")
        
        # 创建 FTS 索引（用于 Sparse 和 Hybrid 检索）
        print("\n创建 FTS 全文索引...")
        table.create_fts_index("text", replace=True)
        print("✓ FTS 索引创建成功")
        
    except Exception as e:
        logger.error(f"创建表失败: {e}")
        print(f"\n❌ 错误: {e}")
        return
    
    # 统计信息
    print("\n" + "="*70)
    print("重建完成！")
    print("="*70)
    
    print("\n数据库信息:")
    print(f"  • LanceDB 路径: {lance_path}")
    print(f"  • 表名: {table_name}")
    print(f"  • 总记录数: {len(all_data)}")
    print(f"  • Embedding 维度: {embedding_dim}")
    print(f"  • FTS 索引: ✅ 已创建")
    
    print("\n支持的检索方式:")
    print("  ✅ Dense Search:  纯语义搜索（基于 embedding 相似度）")
    print("  ✅ Sparse Search: FTS 关键词搜索（BM25）")
    print("  ✅ Hybrid Search: 混合搜索（Dense + Sparse + Reranker）")
    
    print("\n" + "="*70)
    print("可以使用以下方式查询:")
    print("  python examples/example_text_retrieval.py")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="重建文本索引，支持 Dense/Sparse/Hybrid 检索"
    )
    parser.add_argument(
        "--sqlite-db",
        type=str,
        default=None,
        help=f"SQLite 数据库路径（默认: {config.OCR_DB_PATH}）"
    )
    parser.add_argument(
        "--lance-db",
        type=str,
        default=None,
        help=f"LanceDB 数据库路径（默认: {config.TEXT_LANCEDB_PATH}）"
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="ocr_texts",
        help="LanceDB 表名"
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help=f"CLIP 模型名称（默认: {config.CLIP_MODEL}）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小"
    )
    parser.add_argument(
        "--no-clear",
        action="store_false",
        dest="clear_existing",
        help="不清空现有数据，追加模式（不推荐）"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过所有确认提示"
    )
    
    args = parser.parse_args()
    
    rebuild_text_index(
        sqlite_db_path=args.sqlite_db,
        lance_db_path=args.lance_db,
        table_name=args.table_name,
        clip_model=args.clip_model,
        batch_size=args.batch_size,
        clear_existing=args.clear_existing,
        confirm=not args.yes
    )


if __name__ == "__main__":
    main()

