#!/usr/bin/env python3
"""
迁移 frame_id 从哈希格式到时间戳格式

功能：
1. 从 SQLite 读取 frame_id（可能是哈希）和 timestamp
2. 生成新的时间戳格式 frame_id：YYYYMMDD_HHMMSS_ffffff
3. 更新 frames 表的 frame_id（主键）和 image_path
4. 更新 ocr_text 表的 frame_id（外键）
5. 更新文件名（如果存在）
6. 可选：更新 LanceDB（如果使用）

注意：这是一个破坏性操作，建议先备份数据库
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from config import config

logger = setup_logger("migrate_frame_ids")


def generate_timestamp_frame_id(timestamp: datetime) -> str:
    """生成时间戳格式的 frame_id"""
    return timestamp.strftime("%Y%m%d_%H%M%S_") + f"{timestamp.microsecond:06d}"


def is_timestamp_format(frame_id: str) -> bool:
    """检查 frame_id 是否已经是时间戳格式"""
    if len(frame_id) != 22:
        return False
    return frame_id[8] == '_' and frame_id[15] == '_' and frame_id[:8].isdigit()


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    从文件名提取时间戳
    
    支持两种格式：
    1. YYYYMMDD_HHMMSS_ffffff.jpg（时间戳格式）
    2. 1766393803396_xxx.jpg（JavaScript 毫秒时间戳 + 随机字符串）
    """
    try:
        stem = Path(filename).stem  # 去掉 .jpg 后缀
        
        # 格式1: YYYYMMDD_HHMMSS_ffffff (22个字符，两个下划线)
        if len(stem) == 22 and stem[8] == '_' and stem[15] == '_':
            year = int(stem[0:4])
            month = int(stem[4:6])
            day = int(stem[6:8])
            hour = int(stem[9:11])
            minute = int(stem[11:13])
            second = int(stem[13:15])
            microsecond = int(stem[16:22])
            return datetime(year, month, day, hour, minute, second, microsecond)
        
        # 格式2: JavaScript 毫秒时间戳格式（如 1766393803396_xxx）
        # 查找第一个下划线，之前的部分应该是毫秒时间戳
        if '_' in stem:
            parts = stem.split('_', 1)
            millis_str = parts[0]
            # 检查是否是纯数字（13位毫秒时间戳）
            if millis_str.isdigit() and len(millis_str) == 13:
                millis = int(millis_str)
                # 转换为 datetime（毫秒 -> 秒）
                ts = datetime.fromtimestamp(millis / 1000.0)
                # 提取微秒部分（毫秒时间戳的毫秒部分转换为微秒）
                microseconds = (millis % 1000) * 1000
                return ts.replace(microsecond=microseconds)
    except (ValueError, IndexError, OSError) as e:
        logger.debug(f"无法从文件名 {filename} 提取时间戳: {e}")
    return None


def migrate_frame_ids(db_path: str, dry_run: bool = False, frame_id_only: bool = False) -> Dict[str, int]:
    """
    迁移 frame_id 从哈希格式到时间戳格式
    
    Returns:
        统计信息字典
    """
    db_file = Path(db_path)
    if not db_file.exists():
        logger.error(f"数据库文件不存在: {db_path}")
        return {}
    
    logger.info(f"打开数据库: {db_path}")
    
    # 连接数据库
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='frames'")
        if not cursor.fetchone():
            logger.error("表 'frames' 不存在")
            return {}
        
        # 获取所有需要迁移的记录
        cursor.execute("SELECT frame_id, timestamp, image_path FROM frames")
        rows = cursor.fetchall()
        
        logger.info(f"找到 {len(rows)} 条记录")
        
        stats = {
            'total': len(rows),
            'migrated': 0,
            'skipped': 0,
            'errors': 0
        }
        
        # 构建迁移映射：old_frame_id -> (new_frame_id, timestamp, old_image_path, new_image_path)
        migrations: Dict[str, Tuple[str, datetime, str, str]] = {}
        
        # 第一步：收集所有需要迁移的记录
        for row in rows:
            old_frame_id = row["frame_id"]
            timestamp_str = row["timestamp"]
            old_image_path = row["image_path"]
            
            # 如果已经是时间戳格式，跳过
            if is_timestamp_format(old_frame_id):
                stats['skipped'] += 1
                continue
            
            try:
                # 策略1：优先从 image_path 的文件名提取时间戳
                ts = None
                if old_image_path:
                    image_path_obj = Path(old_image_path)
                    if image_path_obj.exists():
                        # 尝试从文件名提取时间戳
                        ts = extract_timestamp_from_filename(image_path_obj.name)
                
                # 策略2：如果无法从文件名提取，从 timestamp 字段解析
                if ts is None:
                    ts = datetime.fromisoformat(timestamp_str)
                else:
                    # 如果从文件名提取了时间戳，验证一下是否与 timestamp 字段接近（可选）
                    try:
                        db_ts = datetime.fromisoformat(timestamp_str)
                        # 允许差异在1小时内（可能是时区或精度问题）
                        if abs((ts - db_ts).total_seconds()) > 3600:
                            logger.warning(f"frame_id={old_frame_id}: 文件名时间戳与数据库时间戳差异较大，使用文件名时间戳")
                    except:
                        pass
                
                # 生成新的 frame_id
                new_frame_id = generate_timestamp_frame_id(ts)
                
                # 生成新的 image_path（从路径中提取目录，更新文件名）
                if frame_id_only:
                    # 只更新 frame_id，保持 image_path 不变
                    new_image_path = old_image_path
                else:
                    # 同时更新 image_path 中的文件名
                    old_path = Path(old_image_path) if old_image_path else Path("")
                    new_filename = f"{new_frame_id}.jpg"
                    if old_image_path:
                        new_image_path = str(old_path.parent / new_filename)
                    else:
                        # 如果 image_path 为空，生成默认路径
                        date_dir = ts.strftime("%Y%m%d")
                        new_image_path = str(Path(config.IMAGE_STORAGE_PATH) / date_dir / new_filename)
                
                # 存储：(new_frame_id, timestamp, old_image_path, new_image_path)
                migrations[old_frame_id] = (new_frame_id, ts, old_image_path, new_image_path)
                
            except Exception as e:
                logger.error(f"处理 frame_id={old_frame_id} 时出错: {e}")
                stats['errors'] += 1
        
        logger.info(f"准备迁移 {len(migrations)} 条记录")
        
        if dry_run:
            logger.info("[DRY RUN] 将要执行的迁移操作：")
            for old_id, (new_id, ts, old_path, new_path) in list(migrations.items())[:10]:  # 只显示前10个
                print(f"  {old_id} -> {new_id}")
                if frame_id_only:
                    print(f"    image_path: 保持不变 ({old_path})")
                else:
                    print(f"    image_path: {old_path} -> {new_path}")
            if len(migrations) > 10:
                print(f"  ... 还有 {len(migrations) - 10} 条记录")
            return stats
        
        # 第二步：执行迁移（需要按顺序处理外键约束）
        # 1. 先更新 ocr_text 表中的 frame_id（外键）
        cursor.execute("SELECT COUNT(*) FROM ocr_text")
        ocr_count = cursor.fetchone()[0]
        logger.info(f"找到 {ocr_count} 条 OCR 记录需要更新")
        
        ocr_updated = 0
        for old_frame_id, (new_frame_id, _, _, _) in migrations.items():
            try:
                # 更新 ocr_text 表中的 frame_id
                cursor.execute(
                    "UPDATE ocr_text SET frame_id = ? WHERE frame_id = ?",
                    (new_frame_id, old_frame_id)
                )
                if cursor.rowcount > 0:
                    ocr_updated += cursor.rowcount
            except Exception as e:
                logger.error(f"更新 ocr_text 表失败 (old_frame_id={old_frame_id}): {e}")
                stats['errors'] += 1
                # 移除这个迁移项，避免后续步骤出错
                if old_frame_id in migrations:
                    del migrations[old_frame_id]
        
        logger.info(f"已更新 {ocr_updated} 条 OCR 记录的 frame_id")
        
        # 2. 更新 frames 表的 frame_id（主键）和 image_path
        # 注意：SQLite 不支持直接更新主键，需要先删除再插入
        frames_updated = 0
        
        for old_frame_id, (new_frame_id, ts, old_image_path, new_image_path) in migrations.items():
            try:
                # 获取旧记录的完整数据
                cursor.execute(
                    "SELECT device_name, metadata, created_at FROM frames WHERE frame_id = ?",
                    (old_frame_id,)
                )
                old_row = cursor.fetchone()
                if not old_row:
                    logger.warning(f"未找到 frame_id={old_frame_id} 的记录")
                    continue
                
                device_name = old_row["device_name"]
                metadata = old_row["metadata"]
                created_at = old_row["created_at"]
                
                # 确定要使用的 image_path
                # 如果 frame_id_only，使用旧的 image_path；否则使用新的 image_path
                final_image_path = old_image_path if frame_id_only else new_image_path
                
                # 删除旧记录
                cursor.execute("DELETE FROM frames WHERE frame_id = ?", (old_frame_id,))
                
                # 插入新记录（使用新的 frame_id）
                cursor.execute(
                    """INSERT INTO frames (frame_id, timestamp, image_path, device_name, metadata, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (new_frame_id, ts.isoformat(), final_image_path, device_name, metadata, created_at)
                )
                
                frames_updated += 1
                
                if frames_updated % 100 == 0:
                    logger.info(f"已迁移 {frames_updated}/{len(migrations)} 条记录...")
                
            except Exception as e:
                logger.error(f"迁移 frame_id={old_frame_id} 失败: {e}")
                stats['errors'] += 1
        
        stats['migrated'] = frames_updated
        
        # 提交更改
        conn.commit()
        logger.info("✓ 数据库迁移完成")
        
        # 3. 重命名文件（仅在非 frame_id_only 模式下）
        if not frame_id_only:
            files_renamed = 0
            for old_frame_id, (new_frame_id, _, old_image_path_str, new_image_path_str) in migrations.items():
                try:
                    old_path = Path(old_image_path_str)
                    new_path = Path(new_image_path_str)
                    
                    # 如果路径不同且旧文件存在且新文件不存在，重命名
                    if old_path != new_path and old_path.exists() and not new_path.exists():
                        old_path.rename(new_path)
                        files_renamed += 1
                        if files_renamed % 100 == 0:
                            logger.info(f"已重命名 {files_renamed} 个文件...")
                    elif not old_path.exists():
                        logger.warning(f"文件不存在: {old_path}")
                    elif new_path.exists():
                        logger.warning(f"目标文件已存在: {new_path}")
                    
                except Exception as e:
                    logger.error(f"重命名文件失败 ({old_frame_id}): {e}")
                    stats['errors'] += 1
            
            logger.info(f"已重命名 {files_renamed} 个文件")
        else:
            logger.info("跳过文件重命名（frame-id-only 模式）")
        
        return stats
        
    except Exception as e:
        logger.error(f"迁移失败: {e}", exc_info=True)
        if not dry_run:
            conn.rollback()
        return stats
    finally:
        conn.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="迁移 frame_id 从哈希格式到时间戳格式"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite 数据库路径（默认: config.OCR_DB_PATH）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要执行的操作，不实际执行"
    )
    parser.add_argument(
        "--frame-id-only",
        action="store_true",
        help="只更新 frame_id，不更新 image_path 和文件名"
    )
    
    args = parser.parse_args()
    
    db_path = args.db_path or config.OCR_DB_PATH
    
    print("=" * 60)
    print("Frame ID 迁移工具")
    print("=" * 60)
    print(f"数据库: {db_path}")
    print(f"旧格式: 哈希值（如 79eebc524dc06b62）")
    print(f"新格式: 时间戳（如 20251201_143025_123456）")
    if args.dry_run:
        print("模式: DRY RUN（不会实际执行）")
    if args.frame_id_only:
        print("模式: 只更新 frame_id（不更新 image_path 和文件名）")
    print("=" * 60)
    
    # 确认
    if not args.dry_run:
        print("\n⚠️  警告：这是一个破坏性操作！")
        print("   建议先备份数据库：")
        print(f"   cp {db_path} {db_path}.backup")
        response = input("\n确认要继续吗？(yes/no): ").strip().lower()
        if response != "yes":
            print("已取消")
            return
    
    stats = migrate_frame_ids(db_path, dry_run=args.dry_run, frame_id_only=args.frame_id_only)
    
    if stats:
        print("\n" + "=" * 60)
        print("迁移完成！")
        print("=" * 60)
        print(f"  总计: {stats['total']}")
        print(f"  已迁移: {stats['migrated']}")
        print(f"  跳过（已是新格式）: {stats['skipped']}")
        print(f"  错误: {stats['errors']}")
        print("=" * 60)
        
        if args.dry_run:
            print("\n[DRY RUN 模式] 以上操作未实际执行")
            print("移除 --dry-run 参数以执行实际迁移")


if __name__ == "__main__":
    main()

