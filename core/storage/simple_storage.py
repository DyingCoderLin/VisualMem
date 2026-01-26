# core/storage/simple_storage.py
import json
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
from PIL import Image
from utils.logger import setup_logger
from utils.image_utils import resize_image_if_needed
from config import config

logger = setup_logger(__name__)

class SimpleStorage:
    """
    简单的文件系统存储
    - 不需要向量数据库
    - 不需要embedding
    - 直接存储图片 + 元数据到文件夹
    
    适合：Naive实现，快速上手，图片数量不多（几百到几千张）
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or config.IMAGE_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.storage_path / "metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"SimpleStorage initialized at: {self.storage_path}")
        logger.info(f"Current frames: {len(self.metadata)}")
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """保存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def reload_metadata(self) -> bool:
        """重新加载元数据（用于读取其他进程写入的新数据）"""
        try:
            self.metadata = self._load_metadata()
            logger.debug(f"Reloaded metadata: {len(self.metadata)} frames")
            return True
        except Exception as e:
            logger.error(f"Failed to reload metadata: {e}")
            return False
    
    def store_frame(
        self,
        frame_id: str,
        timestamp: datetime,
        image: Image.Image,
        ocr_text: str = "",
        metadata: dict = None
    ) -> bool:
        """
        存储一帧
        
        Args:
            frame_id: 帧ID
            timestamp: 时间戳
            image: PIL图片
            ocr_text: OCR文本
            metadata: 其他元数据
        """
        try:
            # 按日期组织目录
            date_dir = self.storage_path / timestamp.strftime("%Y%m%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # 确保图片是RGB模式（JPEG不支持透明通道）
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # 在保存前按需压缩图片（高清 OCR 和 Embedding 已经在此之前完成）
            image = resize_image_if_needed(image)
            
            # 保存图片为JPEG格式，质量80%
            image_filename = f"{frame_id}.jpg"
            image_path = date_dir / image_filename
            image.save(image_path, "JPEG", quality=config.IMAGE_QUALITY, optimize=True)
            
            # 更新元数据
            self.metadata[frame_id] = {
                "timestamp": timestamp.isoformat(),
                "image_path": str(image_path.resolve()),
                "ocr_text": ocr_text,
                "metadata": metadata or {}
            }
            
            # 保存元数据
            self._save_metadata()
            
            logger.info(f"Stored frame {frame_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store frame: {e}")
            return False
    
    def load_recent(self, limit: int = 50) -> List[Dict]:
        """
        加载最近的N张图片
        
        Args:
            limit: 最多加载多少张
            
        Returns:
            List of dicts: {frame_id, timestamp, image, ocr_text}
        """
        try:
            # 按时间排序，取最近的
            sorted_frames = sorted(
                self.metadata.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True
            )[:limit]
            
            # 加载图片
            results = []
            for frame_id, meta in sorted_frames:
                try:
                    image_path = Path(meta["image_path"])
                    if image_path.exists():
                        image = Image.open(image_path)
                        results.append({
                            "frame_id": frame_id,
                            "timestamp": datetime.fromisoformat(meta["timestamp"]),
                            "image": image,
                            "ocr_text": meta.get("ocr_text", ""),
                            "metadata": meta.get("metadata", {})
                        })
                except Exception as e:
                    logger.warning(f"Failed to load frame {frame_id}: {e}")
                    continue
            
            logger.info(f"Loaded {len(results)} recent frames")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load recent frames: {e}")
            return []
    
    def load_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """
        按时间范围加载图片
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List of dicts
        """
        try:
            # 过滤时间范围内的帧
            frames_in_range = [
                (fid, meta) for fid, meta in self.metadata.items()
                if start_time <= datetime.fromisoformat(meta["timestamp"]) <= end_time
            ]
            
            # 按时间排序
            frames_in_range.sort(key=lambda x: x[1]["timestamp"], reverse=True)
            
            # 加载图片
            results = []
            for frame_id, meta in frames_in_range:
                try:
                    image_path = Path(meta["image_path"])
                    if image_path.exists():
                        image = Image.open(image_path)
                        results.append({
                            "frame_id": frame_id,
                            "timestamp": datetime.fromisoformat(meta["timestamp"]),
                            "image": image,
                            "ocr_text": meta.get("ocr_text", ""),
                            "metadata": meta.get("metadata", {})
                        })
                except Exception as e:
                    logger.warning(f"Failed to load frame {frame_id}: {e}")
                    continue
            
            logger.info(f"Loaded {len(results)} frames in time range")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load by time range: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_frames": len(self.metadata),
            "storage_path": str(self.storage_path),
            "storage_mode": "simple"
        }
    
    def search(self, query_text: str, top_k: int = 50) -> List[Dict]:
        """
        简单的文本搜索（基于OCR）
        注意：这不是语义搜索，只是关键词匹配
        
        Args:
            query_text: 查询文本
            top_k: 返回数量
            
        Returns:
            匹配的帧列表
        """
        try:
            # 简单的关键词匹配
            query_lower = query_text.lower()
            matched_frames = []
            
            for frame_id, meta in self.metadata.items():
                ocr_text = meta.get("ocr_text", "").lower()
                if query_lower in ocr_text:
                    matched_frames.append((frame_id, meta))
            
            # 按时间排序
            matched_frames.sort(key=lambda x: x[1]["timestamp"], reverse=True)
            matched_frames = matched_frames[:top_k]
            
            # 加载图片
            results = []
            for frame_id, meta in matched_frames:
                try:
                    image_path = Path(meta["image_path"])
                    if image_path.exists():
                        image = Image.open(image_path)
                        results.append({
                            "frame_id": frame_id,
                            "timestamp": datetime.fromisoformat(meta["timestamp"]),
                            "image": image,
                            "ocr_text": meta.get("ocr_text", ""),
                            "metadata": meta.get("metadata", {})
                        })
                except Exception as e:
                    logger.warning(f"Failed to load frame {frame_id}: {e}")
                    continue
            
            logger.info(f"Found {len(results)} frames matching query")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

