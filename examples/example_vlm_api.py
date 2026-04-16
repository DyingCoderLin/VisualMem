#!/usr/bin/env python3
"""
VLM API 连接示例

演示如何使用 VLM API 进行图像理解
"""

import sys
from pathlib import Path

# 添加项目路径（examples/ 的父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import base64
from PIL import Image
from io import BytesIO
from config import config

def example_vlm_api():
    """示例：使用 VLM API 分析图像"""
    print("\n" + "="*60)
    print("示例：VLM API 图像分析")
    print("="*60)
    print()
    
    # 1. 显示配置
    print("VLM 配置:")
    print(f"  - API 地址: {config.VLM_API_URI}")
    print(f"  - 模型: {config.VLM_API_MODEL}")
    print()
    
    # 2. 创建示例图片
    print("[1/4] 创建示例图片...")
    image = Image.new('RGB', (200, 100), color='red')
    
    # 添加一些文字（如果有PIL的ImageDraw）
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        draw.text((50, 40), "TEST IMAGE", fill='white')
    except:
        pass
    
    # 转换为base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    print(f"  - 图片大小: {len(img_bytes)} bytes")
    print(f"  - Base64长度: {len(img_base64)} chars")
    print()
    
    # 3. 构造请求
    print("\n[2/4] 构造 API 请求...")
    headers = {
        "Content-Type": "application/json"
    }
    
    if config.VLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.VLM_API_KEY}"
    
    payload = {
        "image": f"data:image/png;base64,{img_base64}",
        "text": "请描述这张图片的内容"
    }
    
    print(f"  - 请求方法: POST")
    print(f"  - API 地址: {config.VLM_API_URI}")
    print(f"  - 查询内容: 请描述这张图片的内容")
    
    # 4. 发送请求
    print("\n[3/4] 发送请求到 VLM...")
    try:
        response = requests.post(
            config.VLM_API_URI,
            headers=headers,
            json=payload,
            timeout=60,
            verify=False  # 跳过SSL验证（如果是自签名证书）
        )
        
        print(f"  - HTTP 状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("  - 请求成功！")
            
            # 5. 解析响应
            print("\n[4/4] VLM 响应:")
            print("-" * 60)
            
            try:
                response_json = response.json()
                print(f"JSON响应: {response_json}")
                
                # 尝试提取内容
                content = (
                    response_json.get("response") or 
                    response_json.get("text") or 
                    response_json.get("output") or
                    str(response_json)
                )
                
                print()
                print("提取的内容:")
                print(content)
                
            except:
                print(f"文本响应: {response.text}")
            
            print("-" * 60)
            print("\n" + "="*60)
            print("VLM API 连接成功！")
            print("="*60)
            print("\n接下来可以：")
            print("  1. python main.py   # 开始捕捉屏幕截图")
            print("  2. python query.py  # 查询和理解截图内容")
            
        else:
            print(f"\n请求失败（状态码: {response.status_code}）")
            print(f"响应内容: {response.text[:200]}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"\n连接失败")
        print(f"\n请检查:")
        print(f"  1. VLM 服务是否运行在: {config.VLM_API_URI}")
        print(f"  2. 网络连接是否正常")
        print(f"  3. 端口是否正确")
        print(f"\n详细错误: {e}")
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("\n" + "="*60)
    print("VLM API 连接示例")
    print("="*60)
    print("\n本示例将：")
    print("  1. 创建一张测试图片")
    print("  2. 将图片编码为 base64")
    print("  3. 发送到 VLM API 进行分析")
    print("  4. 显示 VLM 的理解结果")
    
    try:
        example_vlm_api()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n示例运行失败: {e}")

if __name__ == "__main__":
    main()
