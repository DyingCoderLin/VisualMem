#!/usr/bin/env python3
"""
Query Rewrite 和 Time Extraction 示例

演示如何使用 LLM 进行查询重写和时间范围提取
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径（examples/ 的父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))

# 确保工作目录是项目根目录（visualmem/）
import os
os.chdir(Path(__file__).parent.parent)

from config import config
from core.retrieval.query_llm_utils import rewrite_and_time
from utils.logger import setup_logger

logger = setup_logger("example_query_rewrite_time")


def print_config():
    """打印当前配置"""
    print("\n" + "="*60)
    print("当前配置")
    print("="*60)
    print(f"  - ENABLE_LLM_REWRITE: {config.ENABLE_LLM_REWRITE}")
    print(f"  - ENABLE_TIME_FILTER: {config.ENABLE_TIME_FILTER}")
    print(f"  - QUERY_REWRITE_NUM: {config.QUERY_REWRITE_NUM}")
    
    # Query Rewrite API 配置
    if config.QUERY_REWRITE_BASE_URL:
        print(f"\n  Query Rewrite API (独立配置):")
        print(f"    - Base URL: {config.QUERY_REWRITE_BASE_URL}")
        print(f"    - Model: {config.QUERY_REWRITE_MODEL or config.VLM_API_MODEL}")
        print(f"    - API Key: {'已设置' if config.QUERY_REWRITE_API_KEY else '未设置'}")
    else:
        print(f"\n  Query Rewrite API (使用 VLM 配置):")
        print(f"    - Base URL: {config.VLM_API_URI}")
        print(f"    - Model: {config.VLM_API_MODEL}")
        print(f"    - API Key: {'已设置' if config.VLM_API_KEY else '未设置'}")
    print("="*60)


def format_time_range(time_range):
    """格式化时间范围显示"""
    if time_range is None:
        return "None"
    start, end = time_range
    return f"{start.strftime('%Y-%m-%d %H:%M:%S')} ~ {end.strftime('%Y-%m-%d %H:%M:%S')}"


def interactive_query():
    """交互式查询"""
    print("\n" + "="*60)
    print("Query Rewrite & Time Extraction 交互式演示")
    print("="*60)
    print("\n提示：")
    print("  - 输入查询文本，系统将进行查询重写和时间提取")
    print("  - 输入 'q' 或 'quit' 退出")
    print("  - 输入 'config' 查看当前配置")
    print("="*60)
    
    while True:
        try:
            query = input("\n请输入查询 (q 退出, config 查看配置): ").strip()
            
            if not query:
                continue
            
            if query.lower() in ('q', 'quit', 'exit'):
                print("退出。")
                break
            
            if query.lower() == 'config':
                print_config()
                continue
            
            print(f"\n处理查询: '{query}'")
            print("-" * 60)
            
            # 调用 rewrite_and_time
            dense_queries, sparse_queries, time_range = rewrite_and_time(
                query=query,
                enable_rewrite=config.ENABLE_LLM_REWRITE,
                enable_time=config.ENABLE_TIME_FILTER,
                expand_n=config.QUERY_REWRITE_NUM,
            )
            
            # 显示结果
            print("\n结果:")
            print(f"\n  Dense Queries ({len(dense_queries)} 条):")
            for i, q in enumerate(dense_queries, 1):
                print(f"    {i}. {q}")
            
            print(f"\n  Sparse Queries ({len(sparse_queries)} 条):")
            for i, q in enumerate(sparse_queries, 1):
                print(f"    {i}. {q}")
            
            print(f"\n  Time Range:")
            print(f"    {format_time_range(time_range)}")
            
            if time_range:
                start, end = time_range
                duration = end - start
                print(f"    持续时间: {duration}")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n用户中断，退出。")
            break
        except Exception as e:
            logger.error(f"处理查询失败: {e}", exc_info=True)
            print(f"\n错误: {e}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Query Rewrite & Time Extraction 示例")
    print("="*60)
    
    # 显示配置
    print_config()
    
    # 检查配置
    if not config.ENABLE_LLM_REWRITE and not config.ENABLE_TIME_FILTER:
        print("\n警告：")
        print("  ENABLE_LLM_REWRITE 和 ENABLE_TIME_FILTER 都为 false")
        print("  请在 .env 文件中至少启用其中一个功能")
        print("\n示例配置：")
        print("  ENABLE_LLM_REWRITE=true")
        print("  ENABLE_TIME_FILTER=true")
        print("  QUERY_REWRITE_NUM=3")
        return
    
    # 检查 API 配置
    use_independent_api = config.QUERY_REWRITE_BASE_URL and config.QUERY_REWRITE_BASE_URL.strip()
    if use_independent_api:
        if not config.QUERY_REWRITE_API_KEY:
            print("\n警告：")
            print("  已配置 QUERY_REWRITE_BASE_URL，但未设置 QUERY_REWRITE_API_KEY")
            print("  某些 API 可能需要 API Key")
    else:
        if not config.VLM_API_URI:
            print("\n警告：")
            print("  未配置 VLM_API_URI，将无法调用 API")
            return
    
    # 启动交互式查询
    interactive_query()


if __name__ == "__main__":
    main()
