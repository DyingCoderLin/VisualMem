#!/usr/bin/env python3
"""
重建 SQLite OCR 数据库

功能：
1. 删除现有的 SQLite 表
2. 扫描配置的 IMAGE_STORAGE_PATH 目录下的所有图片
3. 对每张图片进行 OCR 识别
4. 重新存入 SQLite 数据库

使用方法：
    python scripts/rebuild_sqlite.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import time
import hashlib

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from config import config
from utils.logger import setup_logger
from core.ocr import create_ocr_engine
from core.storage.sqlite_storage import SQLiteStorage

logger = setup_logger("rebuild_sqlite")


def generate_frame_id_from_path(image_path: Path) -> str:
    """
    从图片路径生成 frame_id
    
    新格式文件名: YYYYMMDD_HHMMSS_ffffff.jpg -> 直接使用文件名（去掉后缀）
    """
    filename = image_path.stem  # 去掉 .jpg 后缀
    
    # 检查是否已经是新格式 (YYYYMMDD_HHMMSS_ffffff)
    if len(filename) == 22 and filename[8] == '_' and filename[15] == '_':
        return filename
    
    # 旧格式：使用文件修改时间生成新格式ID
    mtime = datetime.fromtimestamp(image_path.stat().st_mtime)
    return mtime.strftime("%Y%m%d_%H%M%S_") + f"{mtime.microsecond:06d}"


def extract_timestamp_from_path(image_path: Path) -> datetime:
    """
    从图片文件名提取时间戳
    
    文件名格式: YYYYMMDD_HHMMSS_ffffff.jpg
    例如: 20251204_130438_750984.jpg -> 2025年12月4日 13:04:38.750984
    
    如果无法解析，使用文件修改时间
    """
    try:
        # 从文件名（去掉扩展名）提取时间戳
        filename = image_path.stem  # 例如: "20251204_130438_750984"
        
        # 检查格式: YYYYMMDD_HHMMSS_ffffff (22个字符，两个下划线)
        if len(filename) == 22 and filename[8] == '_' and filename[15] == '_':
            year = int(filename[0:4])
            month = int(filename[4:6])
            day = int(filename[6:8])
            hour = int(filename[9:11])
            minute = int(filename[11:13])
            second = int(filename[13:15])
            microsecond = int(filename[16:22])
            
            return datetime(year, month, day, hour, minute, second, microsecond)
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse timestamp from filename {image_path.name}: {e}")
    
    # 回退：尝试从父目录名解析日期，使用文件修改时间作为时分秒
    try:
        date_dir = image_path.parent.name
        if len(date_dir) == 8 and date_dir.isdigit():
            year = int(date_dir[:4])
            month = int(date_dir[4:6])
            day = int(date_dir[6:8])
            
            # 使用文件修改时间作为时分秒
            mtime = image_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            
            return datetime(year, month, day, dt.hour, dt.minute, dt.second)
    except Exception as e:
        logger.debug(f"Failed to parse date from directory: {e}")
    
    # 最后回退：使用文件修改时间
    mtime = image_path.stat().st_mtime
    return datetime.fromtimestamp(mtime)


def scan_images(image_dir: Path) -> List[Path]:
    """
    扫描图片目录
    
    Args:
        image_dir: 图片目录路径
        
    Returns:
        图片路径列表（按时间排序）
    """
    logger.info(f"Scanning images in: {image_dir}")
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return []
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    # 递归查找所有图片
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.rglob(f"*{ext}"))
    
    # 按修改时间排序
    image_paths.sort(key=lambda p: p.stat().st_mtime)
    
    logger.info(f"Found {len(image_paths)} images")
    return image_paths


def clear_database(db_path: str):
    """
    清空数据库（删除所有表）
    
    Args:
        db_path: 数据库文件路径
    """
    logger.info(f"Clearing database: {db_path}")
    
    db_file = Path(db_path)
    if db_file.exists():
        try:
            # 删除数据库文件
            db_file.unlink()
            logger.info(f"Deleted existing database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to delete database: {e}")
            raise
    else:
        logger.info(f"Database file does not exist: {db_path}")


def process_image(
    image_path: Path,
    ocr_engine,
    sqlite_storage: SQLiteStorage
) -> Tuple[bool, str]:
    """
    处理单张图片：OCR 识别 + 存入 SQLite
    
    Args:
        image_path: 图片路径
        ocr_engine: OCR 引擎
        sqlite_storage: SQLite 存储
        
    Returns:
        (成功标志, 错误信息)
    """
    try:
        # 1. 读取图片
        image = Image.open(image_path)
        
        # 2. 生成 frame_id 和时间戳
        frame_id = generate_frame_id_from_path(image_path)
        timestamp = extract_timestamp_from_path(image_path)
        
        # 3. OCR 识别
        ocr_result = ocr_engine.recognize(image)
        
        # 4. 存入 SQLite
        success = sqlite_storage.store_frame_with_ocr(
            frame_id=frame_id,
            timestamp=timestamp,
            image_path=str(image_path.absolute()),
            ocr_text=ocr_result.text,
            ocr_text_json=ocr_result.text_json,
            ocr_engine=ocr_result.engine,
            ocr_confidence=ocr_result.confidence,
            device_name="default",
            metadata={"size": image.size}
        )
        
        if success:
            return True, ""
        else:
            return False, "Storage failed"
        
    except Exception as e:
        return False, str(e)


def rebuild_sqlite(
    image_dir: str = None,
    db_path: str = None,
    ocr_engine_type: str = "pytesseract"
):
    """
    重建 SQLite OCR 数据库
    
    Args:
        image_dir: 图片目录（默认使用 config.IMAGE_STORAGE_PATH）
    db_path: 数据库路径（默认使用 config.OCR_DB_PATH）
        ocr_engine_type: OCR 引擎类型
    """
    print("\n" + "="*60)
    print("重建 SQLite OCR 数据库")
    print("="*60)
    
    # 1. 配置
    image_dir = Path(image_dir or config.IMAGE_STORAGE_PATH)
    db_path = db_path or config.OCR_DB_PATH
    
    print(f"\n配置:")
    print(f"  - 图片目录: {image_dir}")
    print(f"  - 数据库路径: {db_path}")
    print(f"  - OCR 引擎: {ocr_engine_type}")
    
    # 2. 扫描图片
    print(f"\n[1/4] 扫描图片...")
    image_paths = scan_images(image_dir)
    
    if len(image_paths) == 0:
        print(f"\n未找到图片，退出")
        return
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 3. 清空数据库
    print(f"\n[2/4] 清空现有数据库...")
    try:
        clear_database(db_path)
        print(f"数据库已清空")
    except Exception as e:
        print(f"清空数据库失败: {e}")
        return
    
    # 4. 初始化 OCR 引擎和存储
    print(f"\n[3/4] 初始化 OCR 引擎...")
    try:
        ocr_engine = create_ocr_engine(ocr_engine_type, lang="chi_sim+eng")
        sqlite_storage = SQLiteStorage(db_path=db_path)
        print(f"OCR 引擎和存储已初始化")
    except Exception as e:
        print(f"初始化失败: {e}")
        print(f"\n请确保已安装 pytesseract:")
        print(f"  macOS:   brew install tesseract tesseract-lang")
        print(f"  Ubuntu:  sudo apt install tesseract-ocr tesseract-ocr-chi-sim")
        print(f"  Python:  pip install pytesseract")
        return
    
    # 5. 处理图片
    print(f"\n[4/4] 处理图片并进行 OCR...")
    print("="*60)
    
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    for i, image_path in enumerate(image_paths, 1):
        # 相对路径显示
        try:
            rel_path = image_path.relative_to(Path.cwd())
        except ValueError:
            rel_path = image_path
        
        print(f"\n[{i}/{len(image_paths)}] 处理: {rel_path}")
        
        # 处理图片
        success, error_msg = process_image(image_path, ocr_engine, sqlite_storage)
        
        if success:
            success_count += 1
            print(f"  成功")
        else:
            error_count += 1
            print(f"  失败: {error_msg}")
        
        # 每 10 张显示进度
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            eta = avg_time * (len(image_paths) - i)
            print(f"\n进度: {i}/{len(image_paths)} ({i/len(image_paths)*100:.1f}%)")
            print(f"已用时间: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
    
    # 6. 完成统计
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("重建完成！")
    print("="*60)
    
    print(f"\n统计:")
    print(f"  - 总图片数: {len(image_paths)}")
    print(f"  - 成功: {success_count}")
    print(f"  - 失败: {error_count}")
    print(f"  - 总耗时: {elapsed:.1f}s")
    print(f"  - 平均速度: {elapsed/len(image_paths):.2f}s/张")
    
    # 7. 显示数据库统计
    stats = sqlite_storage.get_stats()
    print(f"\n数据库状态:")
    print(f"  - 总帧数: {stats['total_frames']}")
    print(f"  - OCR 结果数: {stats['total_ocr_results']}")
    print(f"  - 总文本长度: {stats['total_text_length']} 字符")
    print(f"  - 数据库路径: {stats['db_path']}")
    
    print("\n" + "="*60)
    print("可以使用以下方式查询:")
    print("  python examples/example_ocr_fallback.py")
    print("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="重建 SQLite OCR 数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/rebuild_sqlite.py
  
  # 指定图片目录
  python scripts/rebuild_sqlite.py --image-dir ./my_images
  
  # 指定数据库路径
  python scripts/rebuild_sqlite.py --db-path ./my_ocr.db
  
  # 仅英文（更快）
  python scripts/rebuild_sqlite.py --ocr-engine pytesseract
        """
    )
    
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help=f"图片目录路径（默认: {config.IMAGE_STORAGE_PATH}）"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help=f"SQLite 数据库路径（默认: {config.OCR_DB_PATH}）"
    )
    
    parser.add_argument(
        "--ocr-engine",
        type=str,
        default="pytesseract",
        choices=["pytesseract", "dummy"],
        help="OCR 引擎类型（默认: pytesseract）"
    )
    
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="跳过确认提示"
    )
    
    args = parser.parse_args()
    
    # 确认提示
    if not args.yes:
        print("\n警告：此操作将删除现有的 SQLite 数据库！")
        print(f"数据库路径: {args.db_path or config.OCR_DB_PATH}")
        response = input("\n是否继续？(yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("已取消")
            return
    
    try:
        rebuild_sqlite(
            image_dir=args.image_dir,
            db_path=args.db_path,
            ocr_engine_type=args.ocr_engine
        )
    except KeyboardInterrupt:
        print("\n\n用户中断，退出")
        sys.exit(1)
    except Exception as e:
        logger.error(f"重建失败: {e}", exc_info=True)
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

