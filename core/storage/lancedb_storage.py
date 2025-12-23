# core/storage/lancedb_storage.py
"""
LanceDB Storage - 轻量级向量数据库存储（可选，用于Vector模式）

特点：
- 轻量级，无需单独服务器
- 支持向量检索
- 适合大规模图片场景
"""

import os
from typing import List, Optional, Dict
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)

class LanceDBStorage:
    """
    使用 LanceDB 作为向量数据库存储（Vector模式）
    存储: 图片路径 + CLIP embedding + 元数据
    """
    
    def __init__(
        self, 
        db_path: str = config.LANCEDB_PATH,
        embedding_dim: int = 512,  # 默认512（CLIP base），实际使用时应从 encoder.embedding_dim 获取
        image_storage_path: str = config.IMAGE_STORAGE_PATH
    ):
        # 延迟导入（避免模块导入时立即需要）
        import lancedb
        
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.image_storage_path = Path(image_storage_path)
        
        # 创建图片存储目录
        self.image_storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Connecting to LanceDB at {db_path}")
        
        try:
            # 连接到LanceDB
            self.db = lancedb.connect(db_path)
            
            # 创建或加载表
            self._setup_table()
            
            logger.info("LanceDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            raise
    
    def _setup_table(self):
        """创建或加载表（图片向量表和OCR文本向量表）"""
        self.table_name = "frames"
        self.ocr_table_name = "ocr_text"
        
        # 加载图片向量表
        try:
            self.table = self.db.open_table(self.table_name)
            logger.info(f"Loaded existing table: {self.table_name}")
        except Exception:
            self.table = None
            logger.info(f"Frames table will be created on first insert")
        
        # 加载 OCR 文本向量表
        try:
            self.ocr_table = self.db.open_table(self.ocr_table_name)
            logger.info(f"Loaded existing table: {self.ocr_table_name}")
        except Exception:
            self.ocr_table = None
            logger.info(f"OCR table will be created on first insert")
    
    def _save_image(self, image: Image.Image, frame_id: str) -> str:
        """保存图片到本地，返回路径"""
        # 使用日期组织目录
        date_dir = self.image_storage_path / datetime.now(timezone.utc).strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保图片是RGB模式（JPEG不支持透明通道）
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        # 保存图片为JPEG格式，质量80%
        image_filename = f"{frame_id}.jpg"
        image_path = date_dir / image_filename
        image.save(image_path, "JPEG", quality=config.IMAGE_QUALITY, optimize=True)
        
        return str(image_path.resolve())
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """从路径加载图片"""
        try:
            return Image.open(image_path)
        except Exception as e:
            logger.error(f"Failed to load image from {image_path}: {e}")
            return None
    
    def store_frame(
        self, 
        frame_id: str,
        timestamp: datetime,
        image: Image.Image,
        embedding: List[float],
        ocr_text: str = "",
        metadata: dict = None
    ) -> bool:
        """
        存储一帧：图片 + embedding + 元数据
        
        注意：如果 frame_id 已存在，会先删除旧记录再插入新记录（避免重复）
        注意：建议使用 store_frames_batch 进行批量插入以提高性能并减少版本数量
        """
        try:
            # 保存图片
            image_path = self._save_image(image, frame_id)
            
            # 准备数据
            data = [{
                "frame_id": frame_id,
                "timestamp": timestamp.isoformat(),
                "image_path": image_path,
                "vector": embedding,  # LanceDB的embedding列
                "ocr_text": ocr_text or "",
                "metadata": str(metadata or {})
            }]
            
            # 插入到LanceDB
            if self.table is None:
                # 第一次创建表
                self.table = self.db.create_table(self.table_name, data=data)
                logger.info(f"Created table: {self.table_name}")
            else:
                # 检查是否已存在该 frame_id，如果存在则先删除
                try:
                    existing = self.table.search([0.0] * self.embedding_dim).where(f"frame_id = '{frame_id}'").limit(1).to_list()
                    if existing:
                        # 删除旧记录（LanceDB 使用 delete 方法）
                        self.table.delete(f"frame_id = '{frame_id}'")
                        logger.debug(f"Deleted existing frame {frame_id} before inserting new one")
                except Exception as e:
                    # 如果删除失败，记录警告但继续插入
                    logger.warning(f"Failed to check/delete existing frame {frame_id}: {e}")
                
                # 追加数据
                self.table.add(data)
            
            logger.debug(f"Stored frame {frame_id} at {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store frame: {e}")
            return False
    
    def store_frames_batch(
        self,
        frames: List[dict]
    ) -> bool:
        """
        批量存储多帧数据，减少版本数量
        
        Args:
            frames: 帧数据列表，每个元素包含:
                - frame_id: str
                - timestamp: datetime
                - image: Image.Image
                - embedding: List[float]
                - ocr_text: str (可选)
                - metadata: dict (可选)
        
        Returns:
            bool: 是否成功
        """
        if not frames:
            return True
        
        try:
            # 批量保存图片并准备数据
            data_list = []
            frame_ids_to_check = []
            
            for frame in frames:
                frame_id = frame["frame_id"]
                image = frame["image"]
                timestamp = frame["timestamp"]
                embedding = frame["embedding"]
                ocr_text = frame.get("ocr_text", "")
                metadata = frame.get("metadata")
                
                # 如果 frame_data 中已经提供了 image_path，使用它（避免重复保存）
                # 否则才调用 _save_image 保存图片
                if "image_path" in frame and frame["image_path"]:
                    image_path = frame["image_path"]
                else:
                    image_path = self._save_image(image, frame_id)
                
                # 准备数据
                data_list.append({
                    "frame_id": frame_id,
                    "timestamp": timestamp.isoformat(),
                    "image_path": image_path,
                    "vector": embedding,
                    "ocr_text": ocr_text or "",
                    "metadata": str(metadata or {})
                })
                frame_ids_to_check.append(frame_id)
            
            # 批量检查并删除已存在的记录（如果表已存在）
            if self.table is not None:
                try:
                    # 查询所有已存在的 frame_id
                    # 使用一个简单的查询来检查（LanceDB 的 where 子句可以处理多个条件）
                    existing_frame_ids = set()
                    # 由于 LanceDB 的限制，我们需要逐个检查或者使用其他方法
                    # 这里我们使用更高效的方法：先尝试删除所有可能的 frame_id
                    for frame_id in frame_ids_to_check:
                        try:
                            existing = self.table.search([0.0] * self.embedding_dim).where(f"frame_id = '{frame_id}'").limit(1).to_list()
                            if existing:
                                existing_frame_ids.add(frame_id)
                        except Exception:
                            pass
                    
                    # 批量删除已存在的记录
                    if existing_frame_ids:
                        for frame_id in existing_frame_ids:
                            try:
                                self.table.delete(f"frame_id = '{frame_id}'")
                            except Exception as e:
                                logger.warning(f"Failed to delete existing frame {frame_id}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to check existing frames: {e}")
            
            # 批量插入数据
            if self.table is None:
                # 第一次创建表
                self.table = self.db.create_table(self.table_name, data=data_list)
                logger.info(f"Created table: {self.table_name} with {len(data_list)} frames")
            else:
                # 批量追加数据（这样只会创建一个新版本，而不是每个 frame 一个版本）
                self.table.add(data_list)
                logger.debug(f"Batch stored {len(data_list)} frames")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store frames batch: {e}")
            return False
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        向量搜索：根据query embedding找到最相关的图片（支持时间范围过滤）
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            start_time: 开始时间（可选，用于 Pre-filtering）
            end_time: 结束时间（可选，用于 Pre-filtering）
        
        Returns:
            List of dicts containing: frame_id, timestamp, image_path, similarity, ocr_text
        
        Note:
            使用 LanceDB 的 Pre-filtering 功能，在向量搜索时直接过滤时间范围，
            比先搜索再过滤或先过滤再搜索更高效。
        """
        try:
            if self.table is None:
                logger.warning("Table does not exist yet")
                return []
            
            # 构建搜索查询
            search_query = self.table.search(query_embedding)
            
            # 如果有时间范围，使用 Pre-filtering（LanceDB 原生支持）
            if start_time is not None or end_time is not None:
                # 构建 where 条件（timestamp 存储为 ISO 格式字符串）
                conditions = []
                if start_time is not None:
                    start_iso = start_time.isoformat()
                    conditions.append(f"timestamp >= '{start_iso}'")
                if end_time is not None:
                    end_iso = end_time.isoformat()
                    conditions.append(f"timestamp <= '{end_iso}'")
                
                if conditions:
                    where_clause = " AND ".join(conditions)
                    search_query = search_query.where(where_clause)
                    logger.debug(f"Applying time filter: {where_clause}")
            
            # 执行搜索
            results = search_query.limit(top_k).to_list()
            
            # 解析结果
            found_frames = []
            for result in results:
                frame_data = {
                    "frame_id": result["frame_id"],
                    "timestamp": datetime.fromisoformat(result["timestamp"]),
                    "image_path": result["image_path"],
                    "distance": result.get("_distance", 0),  # LanceDB的距离（越小越相似）
                    "ocr_text": result["ocr_text"],
                    "image": self._load_image(result["image_path"]),
                    "metadata": result.get("metadata", {})
                }
                found_frames.append(frame_data)
            
            logger.debug(f"Search returned {len(found_frames)} results")
            return found_frames
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def store_ocr_embedding(
        self,
        frame_id: str,
        timestamp: datetime,
        image_path: str,
        ocr_text: str,
        text_embedding: List[float]
    ) -> bool:
        """
        存储 OCR 文本的 embedding
        
        Args:
            frame_id: 帧ID
            timestamp: 时间戳
            image_path: 图片路径
            ocr_text: OCR 文本内容
            text_embedding: OCR 文本的 CLIP embedding
        """
        try:
            if not ocr_text or not ocr_text.strip():
                logger.debug(f"Skipping empty OCR text for frame {frame_id}")
                return False
            
            data = [{
                "frame_id": frame_id,
                "timestamp": timestamp.isoformat(),
                "image_path": image_path,
                "ocr_text": ocr_text,
                "vector": text_embedding
            }]
            
            if self.ocr_table is None:
                self.ocr_table = self.db.create_table(self.ocr_table_name, data=data)
                logger.info(f"Created OCR table: {self.ocr_table_name}")
            else:
                self.ocr_table.add(data)
            
            logger.debug(f"Stored OCR embedding for frame {frame_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store OCR embedding: {e}")
            return False
    
    def search_ocr(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        搜索 OCR 文本 embedding（支持时间范围过滤）
        
        Args:
            query_embedding: 查询文本的 CLIP embedding
            top_k: 返回数量
            start_time: 开始时间（可选，用于 Pre-filtering）
            end_time: 结束时间（可选，用于 Pre-filtering）
            
        Returns:
            相关帧列表，包含 frame_id, timestamp, image_path, ocr_text, distance
        """
        try:
            if self.ocr_table is None:
                logger.warning("OCR table does not exist yet")
                return []
            
            # 构建搜索查询
            search_query = self.ocr_table.search(query_embedding)
            
            # 如果有时间范围，使用 Pre-filtering
            if start_time is not None or end_time is not None:
                conditions = []
                if start_time is not None:
                    start_iso = start_time.isoformat()
                    conditions.append(f"timestamp >= '{start_iso}'")
                if end_time is not None:
                    end_iso = end_time.isoformat()
                    conditions.append(f"timestamp <= '{end_iso}'")
                
                if conditions:
                    where_clause = " AND ".join(conditions)
                    search_query = search_query.where(where_clause)
                    logger.debug(f"Applying time filter to OCR search: {where_clause}")
            
            results = search_query.limit(top_k).to_list()
            
            found_frames = []
            for result in results:
                frame_data = {
                    "frame_id": result["frame_id"],
                    "timestamp": datetime.fromisoformat(result["timestamp"]),
                    "image_path": result["image_path"],
                    "ocr_text": result["ocr_text"],
                    "distance": result.get("_distance", 0),
                    "image": self._load_image(result["image_path"])
                }
                found_frames.append(frame_data)
            
            logger.info(f"OCR search returned {len(found_frames)} results")
            return found_frames
            
        except Exception as e:
            logger.error(f"OCR search failed: {e}")
            return []
    
    def get_ocr_stats(self) -> Dict:
        """获取 OCR 表的统计信息"""
        try:
            if self.ocr_table is None:
                return {"ocr_frames": 0}
            return {"ocr_frames": self.ocr_table.count_rows()}
        except Exception as e:
            logger.error(f"Failed to get OCR stats: {e}")
            return {"ocr_frames": 0}
    
    def cleanup_old_versions(
        self,
        older_than_hours: float = 1.0,
        delete_unverified: bool = True
    ) -> Optional[Dict]:
        """
        优化数据库（清理旧版本 + 压缩文件），减少 manifest 文件数量
        
        注意：此方法使用新的 optimize API，替代已弃用的 cleanup_old_versions 和 compact_files
        
        Args:
            older_than_hours: 清理多少小时前的版本（默认1小时，设为0只保留最新版本）
            delete_unverified: 是否删除未验证的文件（默认True，可以清理7天内的文件）
        
        Returns:
            优化结果字典（如果成功）或 None
        """
        try:
            if self.table is None:
                logger.warning("Table does not exist, cannot optimize")
                return None
            
            from datetime import timedelta
            
            # 使用新的 optimize 方法（替代 cleanup_old_versions + compact_files）
            cleanup_time = timedelta(hours=older_than_hours)
            self.table.optimize(
                cleanup_older_than=cleanup_time,
                delete_unverified=delete_unverified
            )
            
            logger.info(f"Optimized table (cleanup_older_than={cleanup_time}, delete_unverified={delete_unverified})")
            # optimize 方法不返回统计信息，所以返回一个简单的成功标记
            return {
                "optimized": True,
                "cleanup_older_than_hours": older_than_hours,
            }
        except Exception as e:
            logger.error(f"Failed to optimize table: {e}")
            return None
    
    def list_versions(self) -> List[Dict]:
        """列出所有版本信息"""
        try:
            if self.table is None:
                return []
            return self.table.list_versions()
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        try:
            if self.table is None:
                return {
                    "total_frames": 0,
                    "db_path": self.db_path,
                    "embedding_dim": self.embedding_dim,
                    "storage_mode": "vector"
                }
            
            # LanceDB的count
            num_rows = self.table.count_rows()
            ocr_stats = self.get_ocr_stats()
            
            # 获取版本信息
            versions = self.list_versions()
            
            return {
                "total_frames": num_rows,
                "ocr_frames": ocr_stats.get("ocr_frames", 0),
                "db_path": self.db_path,
                "embedding_dim": self.embedding_dim,
                "storage_mode": "vector",
                "version_count": len(versions)
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_frames": 0,
                "storage_mode": "vector"
            }

