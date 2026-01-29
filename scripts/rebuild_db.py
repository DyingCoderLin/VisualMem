#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量/部分重建 OCR(SQLite) + LanceDB(frames) + TextDB 的工具。

支持普通运行（默认路径）和 benchmark 运行（当设置 BENCHMARK_NAME 或显式传入 --benchmark-name 时）。
默认会重建全部三个库，可用 --targets 选择性重建。
"""

import sys
from pathlib import Path
import argparse
from typing import Optional, List

# 将项目根目录加入路径，便于脚本直接运行
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from utils.logger import setup_logger
from scripts.rebuild_sqlite import rebuild_sqlite as rebuild_sqlite_db
from scripts.rebuild_index import rebuild_index as rebuild_lancedb_index
from scripts.rebuild_text_index import rebuild_text_index

logger = setup_logger("rebuild_db")

TARGET_ALL = ["sqlite", "lancedb", "textdb"]


def _confirm(paths: dict, tasks: List[str], embedding_model: str, clear_existing: bool) -> bool:
    """简单确认提示"""
    print("\n即将重建数据库")
    print(f"  - 图片目录: {paths['image']}")
    print(f"  - LanceDB(frames): {paths['lancedb']}")
    print(f"  - TextDB: {paths['textdb']}")
    print(f"  - OCR SQLite: {paths['sqlite']}")
    print(f"  - CLIP 模型: {embedding_model}")
    print(f"  - 清空现有数据: {clear_existing}")
    print(f"  - 目标任务: {tasks}")
    ans = input("\n确认继续？[y/N]: ").strip().lower()
    return ans in ("y", "yes")


def resolve_paths(
    benchmark_name: Optional[str],
    image_dir: Optional[str],
    lancedb_path: Optional[str],
    textdb_path: Optional[str],
    sqlite_path: Optional[str],
) -> tuple[Path, Path, Path, Path]:
    """
    解析实际使用的路径，优先级：命令行参数 > benchmark_name > 配置默认值
    """
    if benchmark_name:
        base_dir = Path(config.BENCHMARK_DB_ROOT) / benchmark_name
        default_image = Path(config.BENCHMARK_IMAGE_ROOT) / benchmark_name
        default_lancedb = base_dir / "lancedb"
        default_textdb = base_dir / "textdb"
        default_sqlite = base_dir / "ocr.db"
    else:
        base_dir = None
        default_image = Path(config.IMAGE_STORAGE_PATH)
        default_lancedb = Path(config.LANCEDB_PATH)
        default_textdb = Path(config.TEXT_LANCEDB_PATH)
        default_sqlite = Path(config.OCR_DB_PATH)

    resolved_image = Path(image_dir) if image_dir else default_image
    resolved_lancedb = Path(lancedb_path) if lancedb_path else default_lancedb
    resolved_textdb = Path(textdb_path) if textdb_path else default_textdb
    resolved_sqlite = Path(sqlite_path) if sqlite_path else default_sqlite

    return resolved_image, resolved_lancedb, resolved_textdb, resolved_sqlite


def rebuild_db(
    benchmark_name: Optional[str] = None,
    image_dir: Optional[str] = None,
    lancedb_path: Optional[str] = None,
    textdb_path: Optional[str] = None,
    sqlite_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
    image_batch_size: int = 32,
    ocr_batch_size: int = 100,
    min_text_length: int = 10,
    targets: Optional[List[str]] = None,
    clear_existing: bool = True,
    cleanup_interval: int = 50,
):
    embedding_model = embedding_model or config.EMBEDDING_MODEL
    resolved_image, resolved_lancedb, resolved_textdb, resolved_sqlite = resolve_paths(
        benchmark_name, image_dir, lancedb_path, textdb_path, sqlite_path
    )

    tasks = targets or ["all"]
    if "all" in tasks:
        tasks = TARGET_ALL.copy()

    logger.info("开始重建数据库")
    logger.info(f"  - Benchmark: {benchmark_name or '默认'}")
    logger.info(f"  - 图片目录: {resolved_image}")
    logger.info(f"  - LanceDB(frames): {resolved_lancedb}")
    logger.info(f"  - TextDB: {resolved_textdb}")
    logger.info(f"  - OCR SQLite: {resolved_sqlite}")
    logger.info(f"  - 目标任务: {tasks}")

    # 确认提示
    proceed = _confirm(
        {
            "image": resolved_image,
            "lancedb": resolved_lancedb,
            "textdb": resolved_textdb,
            "sqlite": resolved_sqlite,
        },
        tasks,
        embedding_model,
        clear_existing,
    )
    if not proceed:
        print("已取消。")
        return

    if not resolved_image.exists():
        raise FileNotFoundError(f"图片目录不存在: {resolved_image}")

    # 确保输出目录存在
    resolved_lancedb.parent.mkdir(parents=True, exist_ok=True)
    resolved_textdb.parent.mkdir(parents=True, exist_ok=True)
    resolved_sqlite.parent.mkdir(parents=True, exist_ok=True)

    # 1) OCR -> SQLite
    if "sqlite" in tasks:
        rebuild_sqlite_db(
            image_dir=str(resolved_image),
            db_path=str(resolved_sqlite),
            ocr_engine_type="pytesseract",
        )

    # 2) 图像 embedding -> LanceDB
    if "lancedb" in tasks:
        rebuild_lancedb_index(
            image_dir=str(resolved_image),
            db_path=str(resolved_lancedb),
            model_name=embedding_model,
            clear_existing=clear_existing,
            batch_size=image_batch_size,
            cleanup_interval=cleanup_interval,
        )

    # 3) OCR 文本 embedding -> 独立 Text LanceDB
    if "textdb" in tasks:
        rebuild_text_index(
            sqlite_db_path=str(resolved_sqlite),
            lance_db_path=str(resolved_textdb),
            batch_size=ocr_batch_size,
            embedding_model=embedding_model,
            clear_existing=clear_existing,
            confirm=False,
        )

    logger.info("全部任务完成")


def main():
    parser = argparse.ArgumentParser(
        description="重建 OCR(SQLite) + LanceDB(frames/ocr) + TextDB"
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default=None,
        help="指定 benchmark 名称（优先级高于 BENCHMARK_NAME 环境变量）",
    )
    parser.add_argument("--image-dir", type=str, default=None, help="自定义图片目录")
    parser.add_argument("--lancedb-path", type=str, default=None, help="自定义 LanceDB 路径")
    parser.add_argument("--textdb-path", type=str, default=None, help="自定义 TextDB 路径")
    parser.add_argument("--sqlite-path", type=str, default=None, help="自定义 OCR SQLite 路径")
    parser.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help=f"CLIP 模型名称（默认: {config.EMBEDDING_MODEL}）",
    )
    parser.add_argument("--image-batch", type=int, default=32, help="图像 embedding 批次大小")
    parser.add_argument("--ocr-batch", type=int, default=100, help="OCR 文本 embedding 批次大小")
    parser.add_argument("--min-text-length", type=int, default=10, help="OCR 文本最小长度")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["all", "sqlite", "lancedb", "textdb"],
        default=None,
        help="选择性重建的目标，默认 all",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="不清空现有 LanceDB/TextDB，改为追加模式",
    )
    parser.add_argument(
        "--cleanup-interval",
        type=int,
        default=50,
        help="每插入多少个批次后清理一次旧版本（默认: 50，设置为0禁用自动清理）",
    )

    args = parser.parse_args()

    # 仅当显式传入 benchmark 时才使用基准数据集；否则走默认图片目录
    benchmark_name = args.benchmark_name

    rebuild_db(
        benchmark_name=benchmark_name,
        image_dir=args.image_dir,
        lancedb_path=args.lancedb_path,
        textdb_path=args.textdb_path,
        sqlite_path=args.sqlite_path,
        embedding_model=args.embedding_model,
        image_batch_size=args.image_batch,
        ocr_batch_size=args.ocr_batch,
        min_text_length=args.min_text_length,
        targets=args.targets,
        clear_existing=not args.no_clear,
        cleanup_interval=args.cleanup_interval,
    )


if __name__ == "__main__":
    main()
