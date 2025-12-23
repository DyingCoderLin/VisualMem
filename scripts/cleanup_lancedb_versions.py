#!/usr/bin/env python3
"""
清理 LanceDB 数据库的旧版本（manifest 文件）

功能：
1. 清理指定时间之前的旧版本，减少 manifest 文件数量
2. 显示清理前后的版本数量和空间使用情况
3. 支持 frames 表和 ocr_texts 表
4. 支持预览模式（dry-run）和实际清理模式
"""

import sys
from pathlib import Path
from datetime import timedelta
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
import lancedb
from utils.logger import setup_logger

logger = setup_logger("cleanup_versions")


def get_version_count(db_path: str, table_name: str) -> int:
    """获取表的版本数量"""
    try:
        db = lancedb.connect(db_path)
        if table_name not in db.table_names():
            return 0
        table = db.open_table(table_name)
        versions = table.list_versions()
        return len(versions)
    except Exception as e:
        logger.error(f"获取版本数量失败: {e}")
        return 0


def get_db_size(db_path: str) -> dict:
    """获取数据库目录大小信息"""
    try:
        import shutil
        db_dir = Path(db_path)
        if not db_dir.exists():
            return {"total": 0, "versions": 0, "data": 0}
        
        total_size = sum(f.stat().st_size for f in db_dir.rglob('*') if f.is_file())
        
        # 计算 _versions 目录大小
        versions_dir = db_dir / "_versions"
        versions_size = 0
        if versions_dir.exists():
            versions_size = sum(f.stat().st_size for f in versions_dir.rglob('*') if f.is_file())
        
        # 计算 data 目录大小
        data_dir = db_dir / "data"
        data_size = 0
        if data_dir.exists():
            data_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
        
        return {
            "total": total_size,
            "versions": versions_size,
            "data": data_size
        }
    except Exception as e:
        logger.error(f"获取数据库大小失败: {e}")
        return {"total": 0, "versions": 0, "data": 0}


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def cleanup_table_versions(
    db_path: str,
    table_name: str,
    older_than_hours: float = 1.0,
    delete_unverified: bool = True,
    dry_run: bool = False,
    do_compact: bool = True
):
    """
    清理指定表的旧版本
    
    Args:
        db_path: LanceDB 数据库路径
        table_name: 表名
        older_than_hours: 清理多少小时前的版本
        delete_unverified: 是否删除未验证的文件
        dry_run: 预览模式，不实际删除
    """
    print(f"\n处理表: {table_name}")
    print("-" * 60)
    
    try:
        # 连接数据库
        db = lancedb.connect(db_path)
        
        if table_name not in db.table_names():
            print(f"⚠️  表 '{table_name}' 不存在，跳过")
            return None
        
        table = db.open_table(table_name)
        
        # 获取清理前的版本信息
        versions_before = table.list_versions()
        version_count_before = len(versions_before)
        
        print(f"清理前版本数量: {version_count_before}")
        if version_count_before > 0:
            print(f"最早版本: {versions_before[0]}")
            print(f"最新版本: {versions_before[-1]}")
        
        if dry_run:
            print(f"\n🔍 预览模式：将会清理 {older_than_hours} 小时前的版本")
            print(f"   delete_unverified: {delete_unverified}")
            return None
        
        # 执行优化（清理旧版本 + 压缩文件）
        cleanup_time = timedelta(hours=older_than_hours)
        if do_compact:
            print(f"\n🧹 开始优化（清理 {older_than_hours} 小时前的版本 + 压缩文件）...")
        else:
            print(f"\n🧹 开始清理 {older_than_hours} 小时前的版本...")
        
        try:
            # 使用 optimize 方法（替代 cleanup_old_versions + compact_files）
            # optimize 会同时执行清理和压缩
            table.optimize(
                cleanup_older_than=cleanup_time,
                delete_unverified=delete_unverified
            )
        except Exception as e:
            logger.error(f"优化失败: {e}")
            print(f"❌ 优化失败: {e}")
            return None
        
        # 获取清理后的版本信息
        versions_after = table.list_versions()
        version_count_after = len(versions_after)
        
        print(f"✓ 优化完成")
        print(f"清理后版本数量: {version_count_after}")
        print(f"删除版本数: {version_count_before - version_count_after}")
        
        # 注意：optimize 方法不返回统计信息，所以无法显示具体的释放空间
        # 如果需要统计信息，可以通过比较清理前后的数据库大小来估算
        
        return {
            "table": table_name,
            "versions_before": version_count_before,
            "versions_after": version_count_after,
            "versions_removed": version_count_before - version_count_after,
        }
        
    except Exception as e:
        logger.error(f"清理表 {table_name} 失败: {e}")
        print(f"❌ 清理失败: {e}")
        return None


def cleanup_lancedb_versions(
    db_path: str = None,
    table_names: list = None,
    older_than_hours: float = 1.0,
    delete_unverified: bool = True,
    dry_run: bool = False,
    do_compact: bool = True
):
    """
    清理 LanceDB 数据库的旧版本
    
    Args:
        db_path: LanceDB 数据库路径
        table_names: 要清理的表名列表，默认清理 frames 和 ocr_texts
        older_than_hours: 清理多少小时前的版本
        delete_unverified: 是否删除未验证的文件（默认True，可清理7天内的文件）
        dry_run: 预览模式，不实际删除
        do_compact: 是否执行文件压缩（默认True）
    """
    db_path = db_path or config.LANCEDB_PATH
    
    if table_names is None:
        table_names = ["frames", "ocr_texts"]
    
    print("\n" + "="*60)
    print("清理 LanceDB 数据库旧版本")
    print("="*60)
    print(f"数据库路径: {db_path}")
    print(f"表名: {', '.join(table_names)}")
    print(f"清理时间阈值: {older_than_hours} 小时前")
    print(f"删除未验证文件: {delete_unverified}")
    print(f"模式: {'预览模式（不会删除）' if dry_run else '实际清理模式'}")
    print("="*60)
    
    # 检查数据库路径
    db_dir = Path(db_path)
    if not db_dir.exists():
        print(f"\n❌ 错误: 数据库路径不存在: {db_path}")
        return
    
    # 获取清理前的数据库大小
    print("\n清理前的数据库大小:")
    size_before = get_db_size(db_path)
    print(f"  总大小: {format_size(size_before['total'])}")
    print(f"  _versions 目录: {format_size(size_before['versions'])}")
    print(f"  data 目录: {format_size(size_before['data'])}")
    
    # 清理每个表
    results = []
    for table_name in table_names:
        result = cleanup_table_versions(
            db_path=db_path,
            table_name=table_name,
            older_than_hours=older_than_hours,
            delete_unverified=delete_unverified,
            dry_run=dry_run,
            do_compact=do_compact
        )
        if result:
            results.append(result)
    
    # 如果实际执行了清理，显示清理后的数据库大小
    if not dry_run and results:
        print("\n清理后的数据库大小:")
        size_after = get_db_size(db_path)
        print(f"  总大小: {format_size(size_after['total'])}")
        print(f"  _versions 目录: {format_size(size_after['versions'])}")
        print(f"  data 目录: {format_size(size_after['data'])}")
        
        space_freed = size_before['total'] - size_after['total']
        versions_freed = size_before['versions'] - size_after['versions']
        
        print(f"\n释放空间:")
        print(f"  总空间: {format_size(space_freed)}")
        print(f"  _versions 目录: {format_size(versions_freed)}")
    
    # 汇总
    print("\n" + "="*60)
    print("清理完成")
    print("="*60)
    if results:
        total_removed = sum(r['versions_removed'] for r in results)
        print(f"总共删除版本数: {total_removed}")
    else:
        if dry_run:
            print("预览模式：未执行实际清理")
        else:
            print("没有找到需要清理的版本")


def main():
    parser = argparse.ArgumentParser(
        description="清理 LanceDB 数据库的旧版本（manifest 文件）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 预览模式（不实际删除）
  python scripts/cleanup_lancedb_versions.py --dry-run
  
  # 清理1小时前的版本
  python scripts/cleanup_lancedb_versions.py --older-than 1
  
  # 清理24小时前的版本
  python scripts/cleanup_lancedb_versions.py --older-than 24
  
  # 清理指定数据库
  python scripts/cleanup_lancedb_versions.py --db-path ./my_db.lance --older-than 1
  
  # 只清理 frames 表
  python scripts/cleanup_lancedb_versions.py --tables frames --older-than 1
        """
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help=f'LanceDB 数据库路径（默认: {config.LANCEDB_PATH}）'
    )
    
    parser.add_argument(
        '--tables',
        nargs='+',
        default=None,
        choices=['frames', 'ocr_texts'],
        help='要清理的表名（默认: frames ocr_texts）'
    )
    
    parser.add_argument(
        '--older-than',
        type=float,
        default=1.0,
        help='清理多少小时前的版本（默认: 1.0）'
    )
    
    parser.add_argument(
        '--no-delete-unverified',
        action='store_true',
        help='不删除未验证的文件（默认会删除7天内的文件）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，只显示统计信息，不实际删除'
    )
    
    parser.add_argument(
        '--no-compact',
        action='store_true',
        help='不执行文件压缩（默认会执行压缩）'
    )
    
    args = parser.parse_args()
    
    cleanup_lancedb_versions(
        db_path=args.db_path,
        table_names=args.tables,
        older_than_hours=args.older_than,
        delete_unverified=not args.no_delete_unverified,
        dry_run=args.dry_run,
        do_compact=not args.no_compact
    )


if __name__ == "__main__":
    main()

