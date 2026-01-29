#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
帧差过滤示例

演示查询时如何过滤相似的连续帧
"""

import sys
from pathlib import Path

# 添加项目路径（examples/ 的父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np
from datetime import datetime

def example_frame_diff_filter():
    """示例：帧差过滤功能"""
    print("\n" + "="*60)
    print("示例：帧差过滤功能")
    print("="*60)
    
    # 1. 加载配置
    print("\n[1/4] 加载配置...")
    from config import config
    print(f"  - 存储模式: {config.STORAGE_MODE}")
    print(f"  - 查询时帧差过滤: {config.ENABLE_QUERY_FRAME_DIFF}")
    print(f"  - 帧差阈值: {config.SIMPLE_FILTER_DIFF_THRESHOLD}")
    
    # 2. 导入帧差过滤函数
    print("\n[2/4] 导入帧差过滤函数...")
    from query import _apply_frame_diff_filter
    print("  - 导入成功")
    
    # 3. 创建示例数据
    print("\n[3/4] 创建示例帧（模拟连续截图）...")
    test_frames = []

    # 帧1: 红色（完全不同的内容）
    img1 = Image.new('RGB', (100, 100), color='red')
    test_frames.append({
        'frame_id': 'frame_1_红色',
        'timestamp': datetime.now(),
        'image': img1,
        'ocr_text': ''
    })
    
    # 帧2: 几乎相同的红色（与帧1相似，应被过滤）
    img2 = Image.new('RGB', (100, 100), color=(255, 0, 1))
    test_frames.append({
        'frame_id': 'frame_2_接近红色',
        'timestamp': datetime.now(),
        'image': img2,
        'ocr_text': ''
    })
    
    # 帧3: 蓝色（与帧2差异大，应保留）
    img3 = Image.new('RGB', (100, 100), color='blue')
    test_frames.append({
        'frame_id': 'frame_3_蓝色',
        'timestamp': datetime.now(),
        'image': img3,
        'ocr_text': ''
    })
    
    # 帧4: 绿色（与帧3差异大，应保留）
    img4 = Image.new('RGB', (100, 100), color='green')
    test_frames.append({
        'frame_id': 'frame_4_绿色',
        'timestamp': datetime.now(),
        'image': img4,
        'ocr_text': ''
    })
    
    # 帧5: 几乎相同的绿色（与帧4相似，应被过滤）
    img5 = Image.new('RGB', (100, 100), color=(0, 255, 1))
    test_frames.append({
        'frame_id': 'frame_5_接近绿色',
        'timestamp': datetime.now(),
        'image': img5,
        'ocr_text': ''
    })
    
    print(f"  - 创建了 {len(test_frames)} 个模拟帧")

    # 4. 应用帧差过滤
    print("\n[4/4] 应用帧差过滤...")
    threshold = 0.006
    filtered_frames = _apply_frame_diff_filter(test_frames, threshold=threshold)
    
    print(f"  - 阈值: {threshold}")
    print(f"  - 过滤前: {len(test_frames)} 帧")
    print(f"  - 过滤后: {len(filtered_frames)} 帧")
    print(f"  - 被过滤: {len(test_frames) - len(filtered_frames)} 帧")
    
    # 显示保留的帧
    print("\n保留的帧（会被送入 VLM）:")
    for i, frame in enumerate(filtered_frames, 1):
        print(f"  {i}. {frame['frame_id']}")
    
    # 显示被过滤的帧
    filtered_out = set(f['frame_id'] for f in test_frames) - set(f['frame_id'] for f in filtered_frames)
    if filtered_out:
        print("\n被过滤的帧（避免重复信息）:")
        for frame_id in filtered_out:
            print(f"  - {frame_id}")
    
    # 结果说明
    print("\n" + "="*60)
    print("帧差过滤示例完成")
    print("="*60)
    print("\n工作原理:")
    print("  - 帧差过滤通过比较相邻帧的图像差异")
    print("  - 只保留差异大于阈值的帧")
    print("  - 避免将相似的连续帧重复喂给 VLM")
    print("  - 节省 token 成本，提高理解效率")

def main():
    """主函数"""
    print("\n" + "="*60)
    print("帧差过滤示例")
    print("="*60)
    print("\n背景：")
    print("  在查询时，可能会检索到很多连续的相似截图")
    print("  帧差过滤可以去除这些冗余帧，只保留关键变化")
    print("\n本示例将：")
    print("  1. 创建 5 个模拟截图帧")
    print("  2. 应用帧差过滤（阈值 0.006）")
    print("  3. 显示哪些帧被保留/过滤")
    
    try:
        example_frame_diff_filter()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
