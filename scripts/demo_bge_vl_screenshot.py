#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 BAAI/BGE-VL-Screenshot 对指定 benchmark 数据集的图片做 embedding 的示例。

输出：
- embeddings.npy：shape (N, D)
- index.jsonl：每行包含 {"frame_id", "relative_path"}
"""

import sys
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModel

# 项目根路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger("demo_bge_vl_screenshot")


def collect_images(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files: List[Path] = []
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def confirm(image_dir: Path, output_dir: Path, model_name: str, batch_size: int, device: str = "cuda") -> bool:
    print("\n即将生成 BGE-VL-Screenshot 图像向量")
    print(f"  - 图片目录: {image_dir}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 模型: {model_name}")
    print(f"  - 批大小: {batch_size}")
    if device.startswith("cuda") and torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_gb = free_mem / (1024**3)
        print(f"  - GPU 可用内存: {free_gb:.2f} GB")
        if free_gb < 1.0:
            print(f"  警告：GPU 内存不足，建议减小 batch_size 到 1-2")
    ans = input("\n确认继续？[y/N]: ").strip().lower()
    return ans in ("y", "yes")


def embed_benchmark(
    benchmark_name: str,
    image_root: Path,
    output_dir: Path,
    model_name: str = "BAAI/BGE-VL-Screenshot",
    batch_size: int = 16,
    device: str = None,
):
    image_dir = image_root / benchmark_name
    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    index_path = output_dir / "index.jsonl"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("加载模型...")
    attn_impl = None
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    
    # 先检查 flash_attn 是否能导入
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print("检测到 flash_attention_2，将使用加速版本")
        except ImportError:
            logger.warning("flash_attn 模块无法导入，回退到默认注意力实现")
            logger.warning("提示：如果已安装但无法导入，可能是版本不匹配（cxx11abi 或 torch 版本）")
    
    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
        )
    except Exception as e:
        if attn_impl == "flash_attention_2":
            logger.warning(f"使用 flash_attention_2 加载失败: {e}")
            logger.warning("回退到默认注意力实现")
            attn_impl = None
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation=None,
                torch_dtype=torch_dtype,
            )
        else:
            raise
    model.set_processor(model_name)
    model.to(device)
    model.eval()
    
    # 清理初始加载后的 GPU 缓存
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_gb = free_mem / (1024**3)
            print(f"模型已加载，GPU 可用内存: {free_gb:.2f} GB")

    images = collect_images(image_dir)
    if not images:
        print("未找到图片，退出。")
        return

    if not confirm(image_dir, output_dir, model_name, batch_size, device):
        print("已取消。")
        return

    print(f"共 {len(images)} 张，开始编码...")
    all_embeddings: List[torch.Tensor] = []
    with open(index_path, "w", encoding="utf-8") as f_idx:
        for i in tqdm(range(0, len(images), batch_size), desc="encoding"):
            batch_files = images[i : i + batch_size]
            # 使用官方 data_process，仅做图像 candidate 向量
            inputs = model.data_process(
                images=[str(p) for p in batch_files],
                q_or_c="candidate",
            )
            # data_process 返回的 inputs 已经包含 device 信息，直接使用
            with torch.no_grad():
                feats = model(**inputs)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_embeddings.append(feats.cpu())
            
            # 清理 GPU 内存
            del inputs, feats
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            for p in batch_files:
                rel = p.relative_to(image_dir)
                frame_id = p.stem
                f_idx.write(json.dumps({"frame_id": frame_id, "relative_path": str(rel)}) + "\n")

    full = torch.cat(all_embeddings, dim=0).numpy()
    import numpy as np  # 延迟引入

    np.save(embeddings_path, full)
    print(f"完成，向量已保存到: {embeddings_path}")
    print(f"索引信息保存到: {index_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BGE-VL-Screenshot embedding demo (benchmark)")
    parser.add_argument("--benchmark-name", required=True, help="基准数据集名称，对应 BENCHMARK_IMAGE_ROOT/<name>")
    parser.add_argument(
        "--image-root",
        type=str,
        default=config.BENCHMARK_IMAGE_ROOT,
        help=f"图片根目录（默认: {config.BENCHMARK_IMAGE_ROOT}）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: <BENCHMARK_DB_ROOT>/<benchmark>/bge_vl_embeds）",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="批大小（默认4，GPU内存不足时可减小到1-2）")
    parser.add_argument("--device", type=str, default=None, help="cuda 或 cpu，默认自动")
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/BGE-VL-Screenshot",
        help="模型名称（HF Hub）",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(config.BENCHMARK_DB_ROOT) / args.benchmark_name / "bge_vl_embeds"

    embed_benchmark(
        benchmark_name=args.benchmark_name,
        image_root=Path(args.image_root),
        output_dir=out_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
