#!/usr/bin/env python3
"""
重建索引工具

功能：
1. 清空现有的 LanceDB 数据库
2. 将 IMAGE_STORAGE_PATH 下的所有图片重新编码并存入数据库
"""

import sys
import os
import shutil
from pathlib import Path
from PIL import Image
from datetime import datetime
from tqdm import tqdm

# 添加项目路径（scripts/ 的父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from core.encoder import create_encoder
from core.ocr import create_ocr_engine
from core.storage.lancedb_storage import LanceDBStorage
from utils.logger import setup_logger

logger = setup_logger("rebuild_index")


def clear_database(db_path: str):
    """
    清空 LanceDB 数据库
    
    Args:
        db_path: 数据库路径
    """
    db_path = Path(db_path)
    
    if db_path.exists():
        logger.info(f"正在删除现有数据库: {db_path}")
        try:
            if db_path.is_dir():
                shutil.rmtree(db_path)
            else:
                db_path.unlink()
            logger.info("✓ 数据库已清空")
            return True
        except Exception as e:
            logger.error(f"✗ 删除数据库失败: {e}")
            return False
    else:
        logger.info(f"数据库不存在，无需清空: {db_path}")
        return True


def collect_images(image_dir: str):
    """
    收集所有图片文件
    
    Args:
        image_dir: 图片目录路径
        
    Returns:
        图片文件路径列表
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        logger.error(f"图片目录不存在: {image_dir}")
        return []
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)
    
    logger.info(f"找到 {len(image_files)} 张图片")
    return sorted(image_files)


def extract_metadata_from_path(image_path: Path, image_dir: Path):
    """
    从图片文件名提取时间戳和元数据
    
    文件名格式: YYYYMMDD_HHMMSS_ffffff.jpg
    例如: 20251204_130438_750984.jpg -> 2025年12月4日 13:04:38.750984
    
    Args:
        image_path: 图片完整路径
        image_dir: 图片根目录
        
    Returns:
        (frame_id, timestamp, metadata)
    """
    # 提取相对路径
    rel_path = image_path.relative_to(image_dir)
    
    # 从目录结构提取日期（假设格式为 YYYYMMDD/xxx.jpg）
    parts = rel_path.parts
    date_str = parts[0] if len(parts) > 0 else "unknown"
    
    # 生成 frame_id（使用文件名，去掉扩展名）
    frame_id = image_path.stem
    
    # 从文件名提取时间戳
    try:
        # 文件名格式: YYYYMMDD_HHMMSS_ffffff (22个字符，两个下划线)
        filename = image_path.stem
        if len(filename) == 22 and filename[8] == '_' and filename[15] == '_':
            year = int(filename[0:4])
            month = int(filename[4:6])
            day = int(filename[6:8])
            hour = int(filename[9:11])
            minute = int(filename[11:13])
            second = int(filename[13:15])
            microsecond = int(filename[16:22])
            
            timestamp = datetime(year, month, day, hour, minute, second, microsecond)
        else:
            # 如果文件名格式不对，尝试从目录名解析日期
            if len(date_str) == 8 and date_str.isdigit():
                timestamp = datetime.strptime(date_str, "%Y%m%d")
                # 使用文件的修改时间作为具体时间
                file_mtime = os.path.getmtime(image_path)
                dt = datetime.fromtimestamp(file_mtime)
                timestamp = timestamp.replace(hour=dt.hour, minute=dt.minute, second=dt.second)
            else:
                # 最后回退：使用文件的修改时间
                file_mtime = os.path.getmtime(image_path)
                timestamp = datetime.fromtimestamp(file_mtime)
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse timestamp from filename {image_path.name}: {e}")
        # 回退：使用文件的修改时间
        file_mtime = os.path.getmtime(image_path)
        timestamp = datetime.fromtimestamp(file_mtime)
    
    metadata = {
        "date_folder": date_str,
        "relative_path": str(rel_path),
        "file_size": os.path.getsize(image_path),
    }
    
    return frame_id, timestamp, metadata


def rebuild_index(
    image_dir: str,
    db_path: str,
    model_name: str,
    clear_existing: bool = True,
    batch_size: int = 32,
    cleanup_interval: int = 50,  # 每插入多少个批次后清理一次旧版本
    rebuild_with_ocr: bool = False,
):
    """
    重建索引
    
    Args:
        image_dir: 图片目录
        db_path: 数据库路径
        model_name: CLIP 模型名称
        clear_existing: 是否清空现有数据库
        batch_size: 批处理大小
        cleanup_interval: 清理间隔
        rebuild_with_ocr: 是否进行 OCR 识别
    """
    print("\n" + "="*60)
    print("重建 LanceDB 索引")
    if rebuild_with_ocr:
        print("模式: 图像 Embedding + OCR 识别")
    else:
        print("模式: 仅图像 Embedding")
    print("="*60)
    
    # 步骤1：清空数据库
    if clear_existing:
        print("\n[1/4] 清空现有数据库...")
        if not clear_database(db_path):
            print("❌ 清空数据库失败，终止操作")
            return False
    else:
        print("\n[1/4] 跳过清空数据库（追加模式）")
    
    # 步骤2：收集图片
    print("\n[2/4] 扫描图片目录...")
    image_files = collect_images(image_dir)
    
    if len(image_files) == 0:
        print("❌ 没有找到图片文件")
        return False
    
    print(f"✓ 找到 {len(image_files)} 张图片")
    
    # 步骤3：初始化编码器和存储
    print(f"\n[3/4] 初始化编码器: {model_name}")
    try:
        encoder = create_encoder(model_name=model_name)
        print(f"✓ 编码器加载成功（embedding 维度: {encoder.embedding_dim}）")
    except Exception as e:
        print(f"❌ 编码器加载失败: {e}")
        return False
    
    print(f"\n初始化 LanceDB: {db_path}")
    try:
        storage = LanceDBStorage(
            db_path=db_path,
            embedding_dim=encoder.embedding_dim
        )
        print(f"✓ 数据库初始化成功")
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        return False
    
    # 步骤4：批量编码和存储（支持小 batch，定期清理旧版本）
    ocr_mode_str = " + OCR" if rebuild_with_ocr else ""
    print(f"\n[4/4] 编码并存储图片{ocr_mode_str}（批次大小: {batch_size}，定期清理旧版本）...")
    
    # 初始化 OCR 引擎（如果需要）
    ocr_engine = None
    if rebuild_with_ocr:
        try:
            ocr_engine = create_ocr_engine(config.OCR_ENGINE_TYPE)
            print(f"✓ OCR 引擎初始化成功: {config.OCR_ENGINE_TYPE}")
        except Exception as e:
            print(f"❌ OCR 引擎初始化失败: {e}")
            return False
            
    image_dir_path = Path(image_dir)
    success_count = 0
    error_count = 0
    batch_frames = []  # 累积批量数据
    batch_count = 0
    
    # 使用 tqdm 显示进度条
    for image_path in tqdm(image_files, desc="处理进度", unit="张"):
        try:
            # 加载图片
            image = Image.open(image_path)
            
            # 提取元数据
            frame_id, timestamp, metadata = extract_metadata_from_path(
                image_path, image_dir_path
            )
            
            # 编码图片
            embedding = encoder.encode_image(image)
            
            # 执行 OCR（如果启用）
            ocr_text = ""
            if ocr_engine:
                try:
                    ocr_result = ocr_engine.recognize(image)
                    ocr_text = ocr_result.text
                except Exception as e:
                    logger.warning(f"OCR 识别失败 {image_path}: {e}")
            
            # 累积到批量数据中
            batch_frames.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image": image,
                "embedding": embedding,
                "ocr_text": ocr_text,
                "metadata": metadata
            })
            
            # 当累积到 batch_size 时，批量存储
            if len(batch_frames) >= batch_size:
                success = storage.store_frames_batch(batch_frames)
                if success:
                    success_count += len(batch_frames)
                    batch_count += 1
                else:
                    error_count += len(batch_frames)
                    logger.warning(f"批量存储失败: {len(batch_frames)} 张图片")
                batch_frames = []  # 清空批量数据
                
                # 定期清理旧版本（减少 manifest 文件数量）
                if cleanup_interval > 0 and batch_count % cleanup_interval == 0 and storage.table is not None:
                    try:
                        logger.info(f"清理旧版本（已处理 {batch_count} 个批次）...")
                        stats = storage.cleanup_old_versions(
                            older_than_hours=1.0,  # 清理1小时前的版本
                            delete_unverified=True
                        )
                        if stats:
                            logger.info(f"版本清理完成，释放空间: {stats.get('bytes_freed', 0)} 字节，删除文件: {stats.get('files_removed', 0)}")
                    except Exception as e:
                        logger.warning(f"清理旧版本失败: {e}")
            
        except Exception as e:
            error_count += 1
            logger.error(f"处理图片失败 {image_path}: {e}")
            continue
    
    # 处理剩余的批量数据
    if batch_frames:
        success = storage.store_frames_batch(batch_frames)
        if success:
            success_count += len(batch_frames)
            batch_count += 1
        else:
            error_count += len(batch_frames)
            logger.warning(f"批量存储剩余数据失败: {len(batch_frames)} 张图片")
    
    # 最后清理一次旧版本
    if cleanup_interval > 0 and storage.table is not None:
        try:
            logger.info("最终清理旧版本...")
            stats = storage.cleanup_old_versions(
                older_than_hours=1.0,
                delete_unverified=True
            )
            if stats:
                logger.info(f"最终版本清理完成，释放空间: {stats.get('bytes_freed', 0)} 字节，删除文件: {stats.get('files_removed', 0)}")
        except Exception as e:
            logger.warning(f"最终清理旧版本失败: {e}")
    
    # 完成
    print("\n" + "="*60)
    print("重建完成！")
    print("="*60)
    print(f"✓ 成功处理: {success_count} 张")
    if error_count > 0:
        print(f"✗ 失败: {error_count} 张")
    
    # 显示数据库统计
    stats = storage.get_stats()
    print(f"\n数据库统计：")
    print(f"  • 总帧数: {stats['total_frames']}")
    print(f"  • Embedding 维度: {stats['embedding_dim']}")
    print(f"  • 存储模式: {stats['storage_mode']}")
    
    return success_count > 0


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='重建 LanceDB 索引',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 清空并重建索引
  python rebuild_index.py
  
  # 使用自定义路径
  python rebuild_index.py --image-dir ./my_images --db-path ./my_db.lance
  
  # 追加模式（不清空现有数据）
  python rebuild_index.py --no-clear
  
  # 使用不同的 CLIP 模型
  python rebuild_index.py --model openai/clip-vit-base-patch32
        """
    )
    
    parser.add_argument(
        '--image-dir',
        type=str,
        default=config.IMAGE_STORAGE_PATH,
        help=f'图片目录路径（默认: {config.IMAGE_STORAGE_PATH}）'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default=config.LANCEDB_PATH,
        help=f'LanceDB 数据库路径（默认: {config.LANCEDB_PATH}）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=config.EMBEDDING_MODEL,
        help=f'CLIP 模型名称（默认: {config.EMBEDDING_MODEL}）'
    )
    
    parser.add_argument(
        '--no-clear',
        action='store_true',
        help='不清空现有数据库（追加模式）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批处理大小（默认: 32）'
    )
    
    parser.add_argument(
        '--cleanup-interval',
        type=int,
        default=50,
        help='每插入多少个批次后清理一次旧版本（默认: 50，设置为0禁用自动清理）'
    )
    
    parser.add_argument(
        '--rebuild-with-ocr',
        action='store_true',
        help='在重建索引时同时进行 OCR 识别并存入数据库（默认: False）'
    )
    
    args = parser.parse_args()
    
    # 确认操作
    if not args.no_clear:
        print("\n⚠️  警告：此操作将清空现有数据库！")
        print(f"图片目录: {args.image_dir}")
        print(f"数据库路径: {args.db_path}")
        if args.rebuild_with_ocr:
            print(f"OCR 模式: 已开启")
        response = input("\n确认继续？[y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("操作已取消")
            return
    
    # 执行重建
    success = rebuild_index(
        image_dir=args.image_dir,
        db_path=args.db_path,
        model_name=args.model,
        clear_existing=not args.no_clear,
        batch_size=args.batch_size,
        cleanup_interval=args.cleanup_interval,
        rebuild_with_ocr=args.rebuild_with_ocr,
    )
    
    if success:
        print("\n✅ 索引重建成功！")
        print("\n现在可以使用 query.py 或 example_clip_retrieval.py 进行检索")
    else:
        print("\n❌ 索引重建失败")
        sys.exit(1)


if __name__ == "__main__":
    main()

