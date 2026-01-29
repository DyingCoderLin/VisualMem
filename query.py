# query.py - 最终版查询程序
"""
VisualMem - 查询工具

支持两种模式:
1. Simple模式: 加载最近的N张图片 → VLM理解（支持增量更新）
2. Vector模式: CLIP向量检索 → VLM理解

VLM在这里使用，不在捕捉阶段！
"""

import sys
import numpy as np
from PIL import Image, ImageChops
from datetime import datetime
from typing import List, Dict, Optional
from config import config
from utils.logger import setup_logger

logger = setup_logger("query")

# ==================== 加载模块 ====================

logger.info(f"Loading query modules in {config.STORAGE_MODE.upper()} mode...")

# 1. Storage - 根据模式加载
encoder = None

if config.STORAGE_MODE == "simple":
    from core.storage.simple_storage import SimpleStorage
    storage = SimpleStorage(storage_path=config.IMAGE_STORAGE_PATH)
    logger.info("Simple Storage loaded")
    
elif config.STORAGE_MODE == "vector":
    try:
        from core.encoder import create_encoder
        from core.storage.lancedb_storage import LanceDBStorage
        
        encoder = create_encoder(model_name=config.EMBEDDING_MODEL)
        storage = LanceDBStorage(db_path=config.LANCEDB_PATH, embedding_dim=encoder.embedding_dim)
        logger.info(f"Vector mode components loaded (Model: {config.EMBEDDING_MODEL})")
    except ImportError as e:
        logger.error(f"Vector mode requires: pip install transformers torch lancedb")
        sys.exit(1)

# 2. VLM - 用于最终理解
from core.understand.api_vlm import ApiVLM
from core.retrieval.query_llm_utils import rewrite_and_time, filter_by_time
vlm = ApiVLM()

logger.info("All query modules loaded")

# ==================== 增量更新缓存 ====================

class FrameCache:
    """
    帧缓存，支持增量更新
    - 维护最近的50张图片
    - 每次查询时检查新图片
    - 使用帧差过滤(threshold=0.006)
    """
    
    def __init__(self, max_size: int = 50, diff_threshold: float = 0.006):
        self.max_size = max_size
        self.diff_threshold = diff_threshold
        self.cached_frames: List[Dict] = []
        self.last_check_time: Optional[datetime] = None
        logger.info(f"FrameCache initialized (max_size={max_size}, diff_threshold={diff_threshold})")
    
    def _calculate_frame_diff(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        计算两张图片的归一化RMS差异
        返回 0.0 (相同) 到 1.0 (完全不同)
        """
        try:
            # 确保尺寸和模式相同
            if img1.size != img2.size or img1.mode != img2.mode:
                img2 = img2.resize(img1.size).convert(img1.mode)
            
            # 计算差异
            diff_img = ImageChops.difference(img1, img2)
            diff_array = np.array(diff_img)
            rms = np.sqrt(np.mean(np.square(diff_array)))
            normalized_rms = rms / 255.0
            
            return normalized_rms
        except Exception as e:
            logger.warning(f"Frame diff calculation failed: {e}")
            return 1.0  # 出错时认为差异很大
    
    def _should_add_frame(self, new_frame: Dict) -> bool:
        """
        判断是否应该添加新帧
        - 如果缓存为空，直接添加
        - 否则，检查与最后一帧的差异
        """
        if not self.cached_frames:
            return True
        
        # 与最后一帧比较
        last_frame = self.cached_frames[-1]
        diff = self._calculate_frame_diff(last_frame['image'], new_frame['image'])
        # print(f"Frame diff: {diff:.4f}")
        
        if diff > self.diff_threshold:
            logger.debug(f"Frame diff {diff:.4f} > {self.diff_threshold}, adding frame")
            return True
        else:
            logger.debug(f"Frame diff {diff:.4f} <= {self.diff_threshold}, skipping frame")
            return False
    
    def update(self, storage) -> int:
        """
        增量更新缓存
        
        Args:
            storage: SimpleStorage实例
            
        Returns:
            新增的帧数量
        """
        try:
            # 重新加载storage元数据（因为截图可能有新的，所以要进行更新）
            storage.reload_metadata()
            
            # 获取所有帧的元数据（按时间排序）
            all_frames = sorted(
                storage.metadata.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=False  # 从旧到新排序
            )
            
            if not all_frames:
                logger.info("No frames in storage")
                return 0
            
            # 如果缓存为空，初始化（应用帧差过滤）
            if not self.cached_frames:
                logger.info("Initializing cache from storage...")
                # 从最旧的开始遍历，应用帧差过滤
                # 这样可以确保相邻帧之间的帧差 > threshold
                
                for frame_id, meta in all_frames:
                    try:
                        from pathlib import Path
                        image_path = Path(meta["image_path"])
                        if not image_path.exists():
                            continue
                        
                        image = Image.open(image_path)
                        new_frame = {
                            "frame_id": frame_id,
                            "timestamp": datetime.fromisoformat(meta["timestamp"]),
                            "image": image,
                            "ocr_text": meta.get("ocr_text", ""),
                            "metadata": meta.get("metadata", {})
                        }
                        
                        # 检查是否应该添加（帧差过滤）
                        if self._should_add_frame(new_frame):
                            self.cached_frames.append(new_frame)
                            logger.debug(f"Init: Added frame {frame_id}")
                            
                            # 如果已经达到max_size，停止添加旧帧
                            # 但要继续处理，以便更新last_check_time
                            if len(self.cached_frames) >= self.max_size:
                                # 保留最新的max_size张
                                break
                        else:
                            logger.debug(f"Init: Skipped frame {frame_id} (diff <= {self.diff_threshold})")
                    
                    except Exception as e:
                        logger.warning(f"Failed to load frame {frame_id}: {e}")
                
                # 如果初始化后超过max_size，只保留最新的max_size张
                if len(self.cached_frames) > self.max_size:
                    self.cached_frames = self.cached_frames[-self.max_size:]
                
                if self.cached_frames:
                    self.last_check_time = self.cached_frames[-1]['timestamp']
                    logger.info(f"Cache initialized with {len(self.cached_frames)} frames (from {len(all_frames)} total, after frame diff filtering)")
                return len(self.cached_frames)
            
            # 找到上次检查之后的新帧
            new_frames_added = 0
            
            for frame_id, meta in all_frames:
                frame_timestamp = datetime.fromisoformat(meta["timestamp"])
                
                # 只处理上次检查之后的帧
                if self.last_check_time and frame_timestamp <= self.last_check_time:
                    continue
                
                try:
                    from pathlib import Path
                    image_path = Path(meta["image_path"])
                    if not image_path.exists():
                        continue
                    
                    image = Image.open(image_path)
                    new_frame = {
                        "frame_id": frame_id,
                        "timestamp": frame_timestamp,
                        "image": image,
                        "ocr_text": meta.get("ocr_text", ""),
                        "metadata": meta.get("metadata", {})
                    }
                    
                    # 检查是否应该添加（帧差过滤）
                    if self._should_add_frame(new_frame):
                        self.cached_frames.append(new_frame)
                        new_frames_added += 1
                        # logger.info(f"Added new frame: {frame_id} at {frame_timestamp}")
                        
                        # 如果超过最大容量，删除最旧的帧
                        if len(self.cached_frames) > self.max_size:
                            removed_frame = self.cached_frames.pop(0)
                            # logger.info(f"Cache full, removed oldest frame: {removed_frame['frame_id']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_id}: {e}")
            
            # 更新最后检查时间
            if all_frames:
                latest_timestamp = datetime.fromisoformat(all_frames[-1][1]["timestamp"])
                self.last_check_time = latest_timestamp
            
            if new_frames_added > 0:
                logger.info(f"Cache updated: +{new_frames_added} frames, total={len(self.cached_frames)}")
            else:
                logger.debug("No new frames to add")
            
            return new_frames_added
            
        except Exception as e:
            logger.error(f"Cache update failed: {e}", exc_info=True)
            return 0
    
    def get_frames(self) -> List[Dict]:
        """返回缓存中的所有帧（按时间排序，最新在前）"""
        # 返回逆序（最新的在前面）
        return list(reversed(self.cached_frames))
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            "cached_frames": len(self.cached_frames),
            "max_size": self.max_size,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None
        }

# 全局缓存实例（仅用于simple模式）
frame_cache = None
if config.STORAGE_MODE == "simple":
    frame_cache = FrameCache(
        max_size=config.MAX_IMAGES_TO_LOAD,
        diff_threshold=config.SIMPLE_FILTER_DIFF_THRESHOLD
    )

# ==================== 帧差过滤工具函数 ====================

def _apply_frame_diff_filter(frames: List[Dict], threshold: float = 0.006) -> List[Dict]:
    """
    对帧列表应用帧差过滤
    只保留与前一帧差异 > threshold 的图片
    
    Args:
        frames: 帧列表（按时间排序，最新在前）
        threshold: 帧差阈值
        
    Returns:
        过滤后的帧列表
    """
    if not frames:
        return frames
    
    def calculate_frame_diff(img1: Image.Image, img2: Image.Image) -> float:
        """计算两张图片的归一化RMS差异"""
        try:
            if img1.size != img2.size or img1.mode != img2.mode:
                img2 = img2.resize(img1.size).convert(img1.mode)
            
            diff_img = ImageChops.difference(img1, img2)
            diff_array = np.array(diff_img)
            rms = np.sqrt(np.mean(np.square(diff_array)))
            normalized_rms = rms / 255.0
            return normalized_rms
        except Exception as e:
            logger.warning(f"Frame diff calculation failed: {e}")
            return 1.0
    
    filtered = [frames[0]]  # 总是保留第一帧（最新的）
    
    for i in range(1, len(frames)):
        current_frame = frames[i]
        previous_frame = frames[i-1]
        
        # 计算与前一帧的差异
        diff = calculate_frame_diff(current_frame['image'], previous_frame['image'])
        
        if diff > threshold:
            filtered.append(current_frame)
            logger.debug(f"Frame {i}: diff={diff:.4f} > {threshold}, keeping")
        else:
            logger.debug(f"Frame {i}: diff={diff:.4f} <= {threshold}, filtering out")
    
    return filtered

# ==================== 查询Pipeline ====================

def search_and_understand(query: str, top_k: int = None) -> str:
    """
    完整的查询流程
    
    Args:
        query: 用户查询
        top_k: 返回多少个结果（如果None，使用默认值）
        
    Returns:
        VLM的理解和回答
    """
    try:
        logger.info(f"Query: '{query}'")
        
        # 根据模式选择检索方式
        if config.STORAGE_MODE == "simple":
            # Simple模式：使用增量更新缓存
            if top_k is None:
                top_k = config.MAX_IMAGES_TO_LOAD
            
            print(f"\n使用Simple模式（增量更新）...")
            
            # 增量更新缓存
            new_frames_count = frame_cache.update(storage)
            if new_frames_count > 0:
                print(f"发现 {new_frames_count} 张新图片（帧差>0.006）")
            
            # 获取缓存中的帧
            frames = frame_cache.get_frames()
            
            if not frames:
                return "数据库为空，请先运行 main.py 捕捉一些屏幕帧。"
            
            print(f"当前缓存: {len(frames)} 张图片 (最多{frame_cache.max_size}张)")
            
            # 如果需要限制数量
            if len(frames) > top_k:
                frames = frames[:top_k]
        
        else:
            # Vector模式：向量检索
            if top_k is None:
                top_k = 10
            
            print(f"\n使用Vector模式：向量检索 top {top_k}...")
            
            dense_queries = [query]
            time_range = None
            if config.ENABLE_LLM_REWRITE or config.ENABLE_TIME_FILTER:
                print(f"   正在重写查询并提取时间范围...")
                dense_queries, _, time_range = rewrite_and_time(
                    query,
                    enable_rewrite=config.ENABLE_LLM_REWRITE,
                    enable_time=config.ENABLE_TIME_FILTER,
                    expand_n=config.QUERY_REWRITE_NUM,
                )
            
            # 提取时间范围
            start_time = None
            end_time = None
            if time_range:
                start_time, end_time = time_range
                print(f"   时间范围: {start_time} - {end_time}")

            # 执行检索
            frames = []
            for q in dense_queries:
                query_embedding = encoder.encode_text(q)
                res = storage.search(
                    query_embedding, 
                    top_k=top_k,
                    start_time=start_time,
                    end_time=end_time
                )
                frames.extend(res)
            
            # 如果有多个查询，去重并重新排序
            if len(dense_queries) > 1:
                # 按相似度排序并去重
                seen_ids = set()
                unique_frames = []
                for f in sorted(frames, key=lambda x: x.get('similarity', 0), reverse=True):
                    if f['frame_id'] not in seen_ids:
                        unique_frames.append(f)
                        seen_ids.add(f['frame_id'])
                frames = unique_frames[:top_k]
            
            if not frames:
                return "未找到相关的屏幕记录。"
        
        logger.info(f"Found {len(frames)} frames")
        
        # 在Simple模式下，根据配置决定是否进行帧差过滤
        if config.STORAGE_MODE == "simple" and config.ENABLE_QUERY_FRAME_DIFF:
            print(f"\n应用帧差过滤（阈值=0.006）...")
            filtered_frames = _apply_frame_diff_filter(frames)
            removed_count = len(frames) - len(filtered_frames)
            if removed_count > 0:
                print(f"   已过滤掉 {removed_count} 张相似图片")
            frames = filtered_frames
            logger.info(f"After frame diff filtering: {len(frames)} frames")
        
        # 显示检索结果
        print("\n" + "="*60)
        print("检索到的相关帧:")
        print("="*60)
        for i, frame in enumerate(frames, 1):
            print(f"\n{i}. 时间: {frame['timestamp']}")
            if config.STORAGE_MODE == "vector" and 'similarity' in frame:
                print(f"   相似度: {frame.get('similarity', 0):.3f}")
            if frame.get('ocr_text'):
                ocr_preview = frame['ocr_text'][:100].replace('\n', ' ')
                print(f"   OCR文本: {ocr_preview}...")
        
        # 使用VLM深度理解
        print("\n" + "="*60)
        print("正在使用VLM分析...")
        print("="*60)
        
        logger.info("Analyzing with VLM...")
        
        # 构造VLM的prompt
        prompt = f"""You are a helpful visual assistant. Given {len(frames)} screenshots sorted by time, analyze the visual content and answer the user's question.

User Question: {query}

Please analyze these screenshots and focus on:
1. Visual content (UI layout, images, windows, applications, text on screen)
2. What the user was doing at that time
3. Relevant contextual information

IMPORTANT: Please respond in Chinese (中文回答)."""
        
        # 调用VLM - 传递所有图片和时间戳
        # 提取所有图片对象和对应的时间戳
        all_images = []
        all_timestamps = []
        for frame in frames:
            if frame.get('image') is not None:
                all_images.append(frame['image'])
                all_timestamps.append(frame.get('timestamp'))
        
        if not all_images:
            return "错误: 无法加载图片"
        
        logger.info(f"将发送 {len(all_images)} 张图片给VLM进行分析")
        
        # 调用VLM（这里是关键：VLM只在查询时调用！）
        # 传递所有图片和时间戳
        response = vlm._call_vlm(
            prompt, 
            all_images, 
            num_images=len(all_images), 
            image_timestamps=all_timestamps
        )
        
        logger.info("VLM analysis completed")
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return f"查询失败: {str(e)}"

# ==================== 交互界面 ====================

def interactive_mode():
    """交互式查询模式"""
    print("\n" + "="*60)
    print("VisualMem - 查询工具")
    print("="*60)
    
    # 显示统计
    stats = storage.get_stats()
    print(f"\n数据库状态:")
    print(f"  - 模式: {config.STORAGE_MODE.upper()}")
    print(f"  - 截屏的帧数: {stats.get('total_frames', 0)}")
    print(f"  - VLM模型: {config.VLM_API_MODEL}")
    if config.STORAGE_MODE == "simple":
        print(f"  - 每次查询加载: 最近{config.MAX_IMAGES_TO_LOAD}张图片")

        # 初始化缓存并显示状态
        frame_cache.update(storage)
        cache_stats = frame_cache.get_stats()
        print(f"  - 当前缓存: {cache_stats['cached_frames']}张图片")
        if cache_stats['last_check_time']:
            print(f"  - 最后更新: {cache_stats['last_check_time']}")
    print("="*60)
    
    if stats.get('total_frames', 0) == 0:
        print("\n警告: 数据库为空！")
        print("请先运行 python main.py 来捕捉一些屏幕帧。")
        return
    
    print("\n使用说明:")
    print("  - 输入你的问题，系统会找到相关的屏幕截图并用VLM分析")
    if config.STORAGE_MODE == "simple":
        print("  - 每次查询会自动检查新图片（帧差>0.006才会添加到缓存）")
        print("  - 缓存最多保持50张最新的图片")
        if config.ENABLE_QUERY_FRAME_DIFF:
            print("  - 提问VLM时会过滤相似图片（帧差>0.006）")
        else:
            print("  - 提问VLM时不过滤，直接使用所有缓存图片")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'stats' 查看统计信息")
    print("  - 输入 'recent' 查看最近的记录")
    print()
    
    while True:
        try:
            # 获取用户输入
            query = input("\n请输入查询 > ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            
            if query.lower() == 'stats':
                stats = storage.get_stats()
                print(f"\n统计信息:")
                print(f"  - 模式: {config.STORAGE_MODE}")
                print(f"  - 总帧数: {stats.get('total_frames', 0)}")
                if config.STORAGE_MODE == "simple":
                    print(f"  - 存储路径: {stats.get('storage_path', 'N/A')}")
                    # 显示缓存统计
                    cache_stats = frame_cache.get_stats()
                    print(f"  - 缓存帧数: {cache_stats['cached_frames']}/{cache_stats['max_size']}")
                    if cache_stats['last_check_time']:
                        print(f"  - 最后更新: {cache_stats['last_check_time']}")
                    print(f"  - 查询时帧差过滤: {'开启' if config.ENABLE_QUERY_FRAME_DIFF else '关闭'}")
                else:
                    print(f"  - Embedding维度: {stats.get('embedding_dim', 0)}")
                continue
            
            if query.lower() == 'recent':
                print("\n最近的记录:")
                frames = storage.load_recent(limit=10)
                for i, frame in enumerate(frames, 1):
                    print(f"{i}. {frame['timestamp']} - {frame.get('ocr_text', '')[:50]}...")
                continue
            
            # 执行查询
            print()
            answer = search_and_understand(query)
            
            # 显示答案
            print("\n" + "="*60)
            print("VLM的回答:")
            print("="*60)
            print(answer)
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\n错误: {e}")

# ==================== 命令行模式 ====================

def single_query_mode(query: str):
    """单次查询模式（命令行）"""
    answer = search_and_understand(query)
    print("\n" + "="*60)
    print("答案:")
    print("="*60)
    print(answer)

# ==================== 主函数 ====================

def main():
    if len(sys.argv) > 1:
        # 命令行模式
        query = " ".join(sys.argv[1:])
        single_query_mode(query)
    else:
        # 交互模式
        interactive_mode()

if __name__ == "__main__":
    main()
