#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询工作线程 - 执行 query.py 的查询逻辑
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple
import base64
import io
import requests
import asyncio

from PySide6.QtCore import QObject, Signal
from PIL import Image as PILImage

from config import config
from utils.logger import setup_logger
from core.capture.screenshot_capturer import ScreenshotCapturer
from core.storage.sqlite_storage import SQLiteStorage
from core.understand.api_vlm import ApiVLM
from core.retrieval.query_llm_utils import (
    rewrite_and_time,
)
from core.retrieval.reranker import Reranker

logger = setup_logger("query_worker")


class QueryWorker(QObject):
    """查询工作线程"""
    
    result_signal = Signal(str)  # 结果信号
    progress_signal = Signal(str)  # 进度信号
    error_signal = Signal(str)  # 错误信号
    
    def __init__(self, storage_mode: str):
        super().__init__()
        # GUI 模式（local / remote）
        self.gui_mode = config.GUI_MODE
        self.backend_url = config.GUI_REMOTE_BACKEND_URL.rstrip("/") if config.GUI_REMOTE_BACKEND_URL else ""

        # 在 remote 模式下，storage_mode 对本地检索无意义，仅用于兼容接口
        self.storage_mode = storage_mode
        
        # 延迟初始化：只在需要时才加载重型组件
        self.encoder = None
        self.retriever = None
        self.storage = None
        self.frame_cache = None
        self._vector_mode_initialized = False
        self._simple_mode_initialized = False
        
        # VLM 也延迟初始化
        self._vlm = None
    
    @property
    def vlm(self):
        """延迟加载 VLM"""
        if self._vlm is None:
            self._vlm = ApiVLM()
        return self._vlm
    
    def _ensure_simple_mode_initialized(self):
        """确保 Simple 模式组件已初始化（延迟加载）"""
        if self.gui_mode == "remote":
            # 远程模式下不在本地初始化存储
            return
        if self._simple_mode_initialized:
            return
        
        self.progress_signal.emit("📂 正在初始化 Simple 模式存储...")
        from core.storage.simple_storage import SimpleStorage
        from query import FrameCache
        
        self.storage = SimpleStorage(storage_path=config.IMAGE_STORAGE_PATH)
        self.encoder = None
        self.retriever = None
        self.frame_cache = FrameCache(
            max_size=config.MAX_IMAGES_TO_LOAD,
            diff_threshold=config.SIMPLE_FILTER_DIFF_THRESHOLD
        )
        self._simple_mode_initialized = True
    
    def _ensure_vector_mode_initialized(self):
        """确保 Vector 模式组件已初始化（延迟加载 CLIP 模型）"""
        if self.gui_mode == "remote":
            # 远程模式下不在本地加载 CLIP / LanceDB
            return
        if self._vector_mode_initialized:
            return
        
        self.progress_signal.emit("🔄 正在加载 CLIP 模型... (首次加载较慢)")
        from core.encoder.clip_encoder import CLIPEncoder
        from core.storage.lancedb_storage import LanceDBStorage
        from core.retrieval.image_retriever import ImageRetriever
        
        self.encoder = CLIPEncoder(model_name=config.CLIP_MODEL)
        self.progress_signal.emit("📦 正在初始化 LanceDB 存储...")
        self.storage = LanceDBStorage(
            db_path=config.LANCEDB_PATH,
            embedding_dim=self.encoder.embedding_dim
        )
        self.retriever = ImageRetriever(
            encoder=self.encoder,
            storage=self.storage,
            top_k=10
        )
        self._vector_mode_initialized = True
        self.progress_signal.emit("✅ CLIP 模型加载完成")
    
    def _ensure_storage_only(self):
        """只初始化存储（不加载 CLIP 模型，用于不需要向量检索的查询）"""
        if self.gui_mode == "remote":
            # 远程模式下不在本地初始化存储
            return
        if self.storage is not None:
            return
        
        self.progress_signal.emit("📦 正在初始化存储...")
        if self.storage_mode == "simple":
            from core.storage.simple_storage import SimpleStorage
            self.storage = SimpleStorage(storage_path=config.IMAGE_STORAGE_PATH)
        else:
            from core.storage.lancedb_storage import LanceDBStorage
            self.storage = LanceDBStorage(
                db_path=config.LANCEDB_PATH,
                embedding_dim=512  # 默认维度，只用于读取
            )
    
    def query_rag(self, query_text: str, top_k: int = None):
        """RAG 快速检索（全库，支持 Hybrid Search）"""
        try:
            self.progress_signal.emit("🔍 正在检索相关图片...")
            logger.info(f"RAG查询: '{query_text}'")
            
            # 延迟初始化：只有在真正需要时才加载 CLIP 模型
            if self.storage_mode == "simple":
                self._ensure_simple_mode_initialized()
                frames = self._rag_simple_mode(query_text, top_k)
            else:
                self._ensure_vector_mode_initialized()
                
                # 获取查询重写结果
                dense_queries = [query_text]
                sparse_queries = [query_text]
                if config.ENABLE_LLM_REWRITE:
                    dense_queries, sparse_queries, _ = rewrite_and_time(
                        query_text,
                        enable_rewrite=config.ENABLE_LLM_REWRITE,
                        enable_time=False,
                        expand_n=config.QUERY_REWRITE_NUM,
                    )
                
                # 定义辅助函数：Dense 搜索
                def _dense_search_task():
                    """Dense 搜索任务（同步函数）"""
                    dense_frames = []
                    for q in dense_queries:
                        frame_list = self.retriever.retrieve_by_text(q, top_k=top_k or config.MAX_IMAGES_TO_LOAD)
                        dense_frames.extend(frame_list)
                    # 加载图片
                    for frame in dense_frames:
                        if 'image' not in frame or frame['image'] is None:
                            self._load_frame_image(frame)
                    return dense_frames
                
                # 定义辅助函数：Sparse 搜索
                def _sparse_search_task():
                    """Sparse 搜索任务（同步函数）"""
                    if not config.ENABLE_HYBRID:
                        return []
                    
                    try:
                        from core.retrieval.text_retriever import create_text_retriever
                        # 不创建 encoder，因为 sparse 搜索（FTS）不需要
                        text_retriever = create_text_retriever(create_encoder=False)
                        
                        sparse_frames = []
                        for q in sparse_queries:
                            sparse_results = text_retriever.retrieve_sparse(
                                query=q,
                                top_k=top_k or config.MAX_IMAGES_TO_LOAD,
                                text_field="text",
                                filter=None  # 全库搜索，不限制时间
                            )
                            
                            for result in sparse_results:
                                score = result.get("_relevance_score") or result.get("_score") or result.get("score", 0.0)
                                if score == 0.0:
                                    continue
                                
                                fid = result.get("frame_id")
                                if not fid:
                                    continue
                                
                                frame = {
                                    "frame_id": fid,
                                    "timestamp": datetime.fromisoformat(result.get("timestamp")) if isinstance(result.get("timestamp"), str) else result.get("timestamp"),
                                    "image_path": result.get("image_path"),
                                    "ocr_text": result.get("text") or result.get("ocr_text", ""),
                                    "distance": 1.0 - score,
                                    "metadata": result.get("metadata", {})
                                }
                                
                                if frame.get("image_path"):
                                    self._load_frame_image(frame)
                                
                                sparse_frames.append(frame)
                        
                        return sparse_frames
                    except Exception as e:
                        logger.error(f"Sparse 检索失败: {e}", exc_info=True)
                        self.progress_signal.emit(f"⚠️ Sparse 检索失败，仅使用 Dense 结果")
                        return []
                
                # 并行执行 Dense 和 Sparse 搜索（使用 asyncio）
                if config.ENABLE_HYBRID:
                    self.progress_signal.emit("🔍 启用 Hybrid Search，并行执行 Dense 和 Sparse 检索...")
                    async def _run_parallel_searches():
                        dense_task = asyncio.to_thread(_dense_search_task)
                        sparse_task = asyncio.to_thread(_sparse_search_task)
                        return await asyncio.gather(dense_task, sparse_task)
                    
                    dense_results, sparse_results = asyncio.run(_run_parallel_searches())
                else:
                    dense_results = _dense_search_task()
                    sparse_results = []
                
                # 合并结果并去重
                frames = []
                seen = set()
                
                for r in dense_results:
                    fid = r.get("frame_id")
                    if fid in seen:
                        continue
                    seen.add(fid)
                    frames.append(r)
                
                for r in sparse_results:
                    fid = r.get("frame_id")
                    if fid in seen:
                        continue
                    seen.add(fid)
                    frames.append(r)
            
            if not frames:
                self.result_signal.emit("未找到相关的屏幕记录。")
                return
            
            logger.info(f"找到 {len(frames)} 个帧")
            
            # 确保所有 frame 都有图片
            frames_with_images = [f for f in frames if f.get("image") is not None]
            if not frames_with_images:
                self.result_signal.emit("检索到的图片无法加载。")
                return
            
            # Rerank 环节（如果启用）
            if config.ENABLE_RERANK:
                self.progress_signal.emit(f"🔄 正在进行 Rerank（返回 top-{config.RERANK_TOP_K}）...")
                reranker = Reranker()
                frames_with_images = reranker.rerank(
                    query=query_text,
                    frames=frames_with_images,
                    top_k=config.RERANK_TOP_K
                )
                
                if not frames_with_images:
                    self.result_signal.emit("Rerank 后没有图片，无法进行 VLM 分析。")
                    return
                
                self.progress_signal.emit(f"✅ Rerank 完成: 返回 top-{len(frames_with_images)} 张图片")
            
            # VLM 分析（只使用通过 rerank 的图片）
            response = self._analyze_with_vlm(query_text, frames_with_images)
            result = self._format_rag_result(response, frames_with_images)
            
            self.result_signal.emit(result)
            
        except Exception as e:
            logger.error(f"RAG查询失败: {e}", exc_info=True)
            self.error_signal.emit(f"查询失败: {str(e)}")
    
    def query_rag_with_time(self, query_text: str, start_time: datetime, end_time: datetime):
        """RAG 查询（带时间范围过滤）

        - GUI_MODE=local  : 使用本地 LanceDB + SQLite + Reranker + VLM
        - GUI_MODE=remote : 通过 HTTP 调用 backend_server，由后端完成检索+RAG+rerank+VLM
        """
        try:
            # Ensure start_time and end_time are UTC for database queries
            if start_time and start_time.tzinfo is None:
                start_time = start_time.astimezone(timezone.utc)
            if end_time and end_time.tzinfo is None:
                end_time = end_time.astimezone(timezone.utc)

            self.progress_signal.emit(f"🔍 RAG语义检索（时间范围: {start_time.strftime('%m/%d %H:%M')} - {end_time.strftime('%m/%d %H:%M')}）...")
            logger.info(f"RAG时间范围查询: '{query_text}' ({start_time} - {end_time})")

            # ---------- Remote GUI 模式 ----------
            if self.gui_mode == "remote":
                if not self.backend_url:
                    err = "GUI_MODE=remote 但 GUI_REMOTE_BACKEND_URL 未配置"
                    logger.error(err)
                    self.error_signal.emit(err)
                    return

                self.progress_signal.emit("🌐 正在通过远程后端执行检索...")
                try:
                    payload = {
                        "query": query_text,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "ocr_mode": False,
                        "enable_hybrid": config.ENABLE_HYBRID,
                        "enable_rerank": config.ENABLE_RERANK,
                    }
                    url = self.backend_url + "/api/query_rag_with_time"
                    resp = requests.post(url, json=payload, timeout=300)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.error(f"远程 RAG 查询失败: {e}", exc_info=True)
                    self.error_signal.emit(f"远程 RAG 查询失败: {e}")
                    return

                answer = data.get("answer", "")
                frames_data = data.get("frames", [])

                # 将返回的 image_base64 转成临时图片文件供缩略图使用（可选）
                image_paths = []
                for idx, f in enumerate(frames_data[:5]):
                    img_b64 = f.get("image_base64")
                    if not img_b64:
                        continue
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        img = PILImage.open(io.BytesIO(img_bytes))
                        tmp_path = Path(config.IMAGE_STORAGE_PATH) / "remote_gui_thumbs"
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        out_path = tmp_path / f"thumb_{idx}.jpg"
                        img.save(out_path, format="JPEG", quality=80)
                        image_paths.append(out_path)
                    except Exception:
                        continue

                # 使用现有 ResultPanel API 展示 VLM 结果 + 缩略图
                self.result_signal.emit(answer)
                # 缩略图通过主窗口的 result_panel.add_thumbnails 调用时再处理，这里仅返回文本
                return

            # ---------- 本地 GUI 模式（原有逻辑） ----------
            
            # 确保向量模式已初始化
            if self.storage_mode == "simple":
                self.result_signal.emit("错误：Simple 模式不支持带时间范围的 RAG 查询，请切换到 Vector 模式。")
                return
            
            self._ensure_vector_mode_initialized()
            
            # 进行语义检索（使用 LanceDB Pre-filtering，一步完成向量搜索和时间过滤）
            self.progress_signal.emit("🔍 正在进行语义检索（LanceDB Pre-filtering）...")
            top_k = config.MAX_IMAGES_TO_LOAD
            
            # 支持查询重写
            dense_queries = [query_text]
            sparse_queries = [query_text]
            if config.ENABLE_LLM_REWRITE:
                dense_queries, sparse_queries, _ = rewrite_and_time(
                    query_text,
                    enable_rewrite=config.ENABLE_LLM_REWRITE,
                    enable_time=False,  # 时间范围已经通过 start_time/end_time 指定
                    expand_n=config.QUERY_REWRITE_NUM,
                )
            
            # 定义辅助函数：Dense 搜索
            def _dense_search_task():
                """Dense 搜索任务（同步函数）"""
                dense_frames = []
                for q in dense_queries:
                    query_embedding = self.encoder.encode_text(q)
                    res = self.storage.search(
                        query_embedding, 
                        top_k=top_k,
                        start_time=start_time,
                        end_time=end_time
                    )
                    dense_frames.extend(res)
                return dense_frames
            
            # 定义辅助函数：Sparse 搜索
            def _sparse_search_task():
                """Sparse 搜索任务（同步函数）"""
                if not config.ENABLE_HYBRID:
                    return []
                
                try:
                    from core.retrieval.text_retriever import create_text_retriever
                    # 不创建 encoder，因为 sparse 搜索（FTS）不需要
                    text_retriever = create_text_retriever(create_encoder=False)
                    
                    # 构建时间过滤条件（SQL 风格）
                    time_filter = None
                    if start_time is not None or end_time is not None:
                        conditions = []
                        if start_time is not None:
                            start_iso = start_time.isoformat()
                            conditions.append(f"timestamp >= '{start_iso}'")
                        if end_time is not None:
                            end_iso = end_time.isoformat()
                            conditions.append(f"timestamp <= '{end_iso}'")
                        if conditions:
                            time_filter = " AND ".join(conditions)
                    
                    sparse_frames = []
                    for q in sparse_queries:
                        sparse_results = text_retriever.retrieve_sparse(
                            query=q,
                            top_k=top_k,
                            text_field="text",
                            filter=time_filter
                        )
                        
                        for result in sparse_results:
                            score = result.get("_relevance_score") or result.get("_score") or result.get("score", 0.0)
                            if score == 0.0:
                                continue
                            
                            fid = result.get("frame_id")
                            if not fid:
                                continue
                            
                            frame = {
                                "frame_id": fid,
                                "timestamp": datetime.fromisoformat(result.get("timestamp")) if isinstance(result.get("timestamp"), str) else result.get("timestamp"),
                                "image_path": result.get("image_path"),
                                "ocr_text": result.get("text") or result.get("ocr_text", ""),
                                "distance": 1.0 - score,
                                "metadata": result.get("metadata", {})
                            }
                            
                            if frame.get("image_path"):
                                self._load_frame_image(frame)
                            
                            sparse_frames.append(frame)
                    
                    return sparse_frames
                except Exception as e:
                    logger.error(f"Sparse 检索失败: {e}", exc_info=True)
                    self.progress_signal.emit(f"⚠️ Sparse 检索失败，仅使用 Dense 结果")
                    return []
            
            # 并行执行 Dense 和 Sparse 搜索（使用 asyncio）
            if config.ENABLE_HYBRID:
                self.progress_signal.emit("🔍 启用 Hybrid Search，并行执行 Dense 和 Sparse 检索...")
                async def _run_parallel_searches():
                    dense_task = asyncio.to_thread(_dense_search_task)
                    sparse_task = asyncio.to_thread(_sparse_search_task)
                    return await asyncio.gather(dense_task, sparse_task)
                
                dense_results, sparse_results = asyncio.run(_run_parallel_searches())
            else:
                dense_results = _dense_search_task()
                sparse_results = []
            
            # 合并结果并去重
            frames = []
            seen = set()
            
            for r in dense_results:
                fid = r.get("frame_id")
                if fid in seen:
                    continue
                seen.add(fid)
                frames.append(r)
            
            for r in sparse_results:
                fid = r.get("frame_id")
                if fid in seen:
                    continue
                seen.add(fid)
                frames.append(r)
            
            if not frames:
                self.result_signal.emit("在指定时间范围内未找到相关的屏幕记录。")
                return
            
            self.progress_signal.emit(f"📊 检索到 {len(frames)} 张相关图片")
            
            # 加载图片
            for frame in frames:
                if 'image' not in frame or frame['image'] is None:
                    self._load_frame_image(frame)
            
            # 确保所有 frame 都有图片
            frames_with_images = [f for f in frames if f.get("image") is not None]
            if not frames_with_images:
                self.result_signal.emit("检索到的图片无法加载。")
                return
            
            # Rerank 环节（如果启用）
            if config.ENABLE_RERANK:
                self.progress_signal.emit(f"🔄 正在进行 Rerank（返回 top-{config.RERANK_TOP_K}）...")
                reranker = Reranker()
                frames_with_images = reranker.rerank(
                    query=query_text,
                    frames=frames_with_images,
                    top_k=config.RERANK_TOP_K
                )
                
                if not frames_with_images:
                    self.result_signal.emit("Rerank 后没有图片，无法进行 VLM 分析。")
                    return
                
                self.progress_signal.emit(f"✅ Rerank 完成: 返回 top-{len(frames_with_images)} 张图片")
            
            # VLM 分析（只使用通过 rerank 的图片）
            response = self._analyze_with_vlm(query_text, frames_with_images)
            result = self._format_rag_result(response, frames_with_images)
            
            self.result_signal.emit(result)
            
        except Exception as e:
            logger.error(f"RAG时间范围查询失败: {e}", exc_info=True)
            self.error_signal.emit(f"查询失败: {str(e)}")
    
    def _rag_simple_mode(self, query_text: str, top_k: int = None) -> List[Dict]:
        """Simple模式的RAG检索"""
        if top_k is None:
            top_k = config.MAX_IMAGES_TO_LOAD
        
        self.progress_signal.emit("📂 Simple模式: 更新缓存...")
        new_frames_count = self.frame_cache.update(self.storage)
        if new_frames_count > 0:
            self.progress_signal.emit(f"✨ 发现 {new_frames_count} 张新图片")
        
        frames = self.frame_cache.get_frames()
        
        if not frames:
            return []
        
        self.progress_signal.emit(f"📊 当前缓存: {len(frames)} 张图片")
        
        if len(frames) > top_k:
            frames = frames[:top_k]
        
        # 帧差过滤
        if config.ENABLE_QUERY_FRAME_DIFF:
            from query import _apply_frame_diff_filter
            self.progress_signal.emit("🔍 应用帧差过滤...")
            filtered_frames = _apply_frame_diff_filter(frames)
            removed_count = len(frames) - len(filtered_frames)
            if removed_count > 0:
                self.progress_signal.emit(f"已过滤 {removed_count} 张相似图片")
            frames = filtered_frames
        
        return frames
    
    def _rag_vector_mode(self, query_text: str, top_k: int = None) -> List[Dict]:
        """Vector模式的RAG检索"""
        if top_k is None:
            top_k = config.MAX_IMAGES_TO_LOAD
        
        self.progress_signal.emit(f"🔍 Vector模式: RAG语义检索 top {top_k}...")
        
        frames = self.retriever.retrieve_by_text(query_text, top_k=top_k)
        
        if not frames:
            return []
        
        self.progress_signal.emit(f"✅ 检索到 {len(frames)} 张相关图片")
        
        # 加载图片
        for frame in frames:
            if 'image' not in frame or frame['image'] is None:
                self._load_frame_image(frame)
        
        return frames
    
    def _load_frame_image(self, frame: Dict):
        """加载帧图片（支持多种路径格式）"""
        try:
            image_path = None
            
            # 1. 首先尝试使用 frame 中的 image_path
            if 'image_path' in frame and frame['image_path']:
                path = Path(frame['image_path'])
                # 如果是绝对路径且存在，直接使用
                if path.is_absolute() and path.exists():
                    image_path = path
                # 如果是相对路径，尝试在当前目录下查找
                elif not path.is_absolute() and path.exists():
                    image_path = path
                # 尝试在项目根目录下查找相对路径
                elif not path.is_absolute():
                    abs_path = Path.cwd() / path
                    if abs_path.exists():
                        image_path = abs_path
            
            # 2. 如果上面都找不到，尝试根据 frame_id 和 timestamp 构建路径
            if image_path is None and 'frame_id' in frame and 'timestamp' in frame:
                date_str = frame['timestamp'].strftime("%Y%m%d")
                # 尝试新格式文件名
                new_path = Path(config.IMAGE_STORAGE_PATH) / date_str / f"{frame['frame_id']}.jpg"
                if new_path.exists():
                    image_path = new_path
            
            # 3. 加载图片
            if image_path and image_path.exists():
                frame['image'] = PILImage.open(image_path)
                logger.debug(f"成功加载图片: {image_path}")
            else:
                original_path = frame.get('image_path', 'unknown')
                logger.warning(f"图片不存在: {original_path}")
                
        except Exception as e:
            logger.error(f"加载图片失败: {e}")
    
    def _analyze_with_vlm(self, query_text: str, frames: List[Dict]) -> str:
        """使用VLM分析帧"""
        self.progress_signal.emit(f"🤖 正在使用VLM分析 {len(frames)} 张图片...")
        
        # System prompt: 定义助手角色
        system_prompt = "You are a helpful visual assistant. You analyze screenshots to answer user questions. Always respond in Chinese (中文回答)."
        
        # User prompt: 直接回答问题，然后用图片内容作为支持
        prompt = f"""User Question: {query_text}

Please directly answer the user's question first, then provide supporting evidence from the {len(frames)} screenshots below. Focus on what the user was doing and how the visual content relates to their question."""
        
        all_images = [frame['image'] for frame in frames if frame.get('image') is not None]
        
        if not all_images:
            raise ValueError("无法加载图片")
        
        # 提取时间戳
        timestamps = [frame.get('timestamp') for frame in frames if frame.get('image') is not None]
        
        return self.vlm._call_vlm(
            prompt, 
            all_images, 
            num_images=len(all_images),
            image_timestamps=timestamps if timestamps else None,
            system_prompt=system_prompt
        )
    
    def _format_rag_result(self, response: str, frames: List[Dict]) -> str:
        """格式化RAG结果"""
        result = f"📝 VLM 回答:\n\n{response}\n\n"
        result += "="*60 + "\n"
        result += f"检索信息:\n"
        result += f"• 模式: {self.storage_mode.upper()}\n"
        result += f"• 检索方法: {'缓存加载' if self.storage_mode == 'simple' else 'RAG语义检索'}\n"
        result += f"• 图片数量: {len(frames)}\n"
        
        if frames:
            oldest = frames[-1]['timestamp']
            newest = frames[0]['timestamp']
            result += f"• 时间范围: {oldest} 到 {newest}\n"
            
            if self.storage_mode == "vector" and frames and '_distance' in frames[0]:
                result += f"• 最高相似度: {1.0 - frames[0].get('_distance', 0):.3f}\n"
        
        return result
    
    def query_time_summary(self, start_time: datetime, end_time: datetime):
        """模式2: 时间段总结"""
        try:
            # Ensure start_time and end_time are UTC for database queries
            if start_time and start_time.tzinfo is None:
                start_time = start_time.astimezone(timezone.utc)
            if end_time and end_time.tzinfo is None:
                end_time = end_time.astimezone(timezone.utc)

            self.progress_signal.emit(f"📅 正在加载 {start_time} 到 {end_time} 的截图...")
            logger.info(f"时间段总结: {start_time} - {end_time}")
            
            # 从 SQLite 获取时间范围内的帧
            frames_with_images = self._load_frames_in_timerange(start_time, end_time)
            
            if not frames_with_images:
                self.result_signal.emit("该时间段内没有截图记录或无法加载图片。")
                return
            
            # 限制数量
            if len(frames_with_images) > 20:
                self.progress_signal.emit(f"图片数量较多,采样到20张...")
                step = len(frames_with_images) // 20
                frames_with_images = frames_with_images[::step][:20]
            
            # VLM 总结
            response = self._summarize_with_vlm(start_time, end_time, frames_with_images)
            result = self._format_summary_result(response, start_time, end_time, frames_with_images)
            
            self.result_signal.emit(result)
            
        except Exception as e:
            logger.error(f"时间段总结失败: {e}", exc_info=True)
            self.error_signal.emit(f"总结失败: {str(e)}")
    
    def _load_frames_in_timerange(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """加载时间范围内的帧"""
        sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
        frames_meta = sqlite_storage.get_recent_frames(limit=1000)
        
        # 过滤时间范围
        filtered_frames = [
            f for f in frames_meta
            if start_time <= f['timestamp'] <= end_time
        ]
        
        if not filtered_frames:
            return []
        
        self.progress_signal.emit(f"找到 {len(filtered_frames)} 张截图")
        
        # 加载图片
        frames_with_images = []
        for frame_meta in filtered_frames:
            try:
                image_path = Path(frame_meta['image_path'])
                if image_path.exists():
                    image = PILImage.open(image_path)
                    frames_with_images.append({
                        'image': image,
                        'timestamp': frame_meta['timestamp'],
                        'ocr_text': frame_meta.get('ocr_text', '')
                    })
            except Exception as e:
                logger.warning(f"加载图片失败: {e}")
        
        return frames_with_images
    
    def _summarize_with_vlm(self, start_time: datetime, end_time: datetime, 
                           frames: List[Dict]) -> str:
        """使用VLM总结时间段"""
        self.progress_signal.emit(f"🤖 正在使用VLM总结 {len(frames)} 张截图...")
        
        # System prompt: 定义助手角色
        system_prompt = "You are a helpful visual assistant. You analyze screenshots to provide summaries. Always respond in Chinese (中文回答)."
        
        # User prompt: 直接总结，然后用图片内容作为支持
        prompt = f"""Please provide a comprehensive summary of what the user was doing from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}.

Please directly provide the summary first, then use the {len(frames)} screenshots below as supporting evidence. Focus on:
1. Main activities and tasks
2. Applications and software used
3. Important work or browsing content
4. Overall time allocation"""
        
        all_images = [frame['image'] for frame in frames]
        
        # 提取时间戳
        timestamps = [frame.get('timestamp') for frame in frames]
        
        return self.vlm._call_vlm(
            prompt, 
            all_images, 
            num_images=len(all_images),
            image_timestamps=timestamps if timestamps else None,
            system_prompt=system_prompt
        )
    
    def _format_summary_result(self, response: str, start_time: datetime, 
                               end_time: datetime, frames: List[Dict]) -> str:
        """格式化总结结果"""
        result = f"📝 时间段总结:\n\n{response}\n\n"
        result += "="*60 + "\n"
        result += f"时间范围: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"分析了 {len(frames)} 张截图\n"
        return result
    
    def query_realtime(self, question: str):
        """模式3: 实时问答"""
        try:
            self.progress_signal.emit("📸 正在获取当前屏幕和历史截图...")
            logger.info(f"实时问答: '{question}'")
            
            # 实时问答不需要初始化 storage 或 CLIP 模型
            # 直接使用 SQLite 获取历史截图
            
            # 获取当前和历史图片
            images, timestamps = self._get_realtime_images()
            
            if not images:
                self.error_signal.emit("无法获取屏幕截图")
                return
            
            # VLM分析
            response = self._analyze_realtime_with_vlm(question, images, timestamps)
            result = self._format_realtime_result(response, images)
            
            self.result_signal.emit(result)
            
        except Exception as e:
            logger.error(f"实时问答失败: {e}", exc_info=True)
            self.error_signal.emit(f"问答失败: {str(e)}")
    
    def _get_realtime_images(self) -> tuple[List[PILImage.Image], List[datetime]]:
        """获取当前和历史图片及其时间戳"""
        # 1. 当前屏幕截图
        capturer = ScreenshotCapturer()
        current_frame = capturer.capture()
        
        if not current_frame:
            return [], []
        
        images = [current_frame.image]
        timestamps = [current_frame.timestamp]
        
        # 2. 从 SQLite 获取最近5张截图（不依赖 storage 的 load_recent 方法）
        self.progress_signal.emit("📂 加载最近5张截图...")
        
        try:
            sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
            recent_frames = sqlite_storage.get_recent_frames(limit=5)
            
            for frame_meta in recent_frames:
                try:
                    image_path = Path(frame_meta['image_path'])
                    if image_path.exists():
                        image = PILImage.open(image_path)
                        images.append(image)
                        timestamps.append(frame_meta['timestamp'])
                except Exception as e:
                    logger.warning(f"加载历史截图失败: {e}")
        except Exception as e:
            logger.warning(f"从SQLite获取历史截图失败: {e}")
        
        return images, timestamps
    
    def _analyze_realtime_with_vlm(self, question: str, images: List[PILImage.Image], timestamps: List[datetime]) -> str:
        """使用VLM分析实时问题"""
        self.progress_signal.emit(f"🤖 正在使用VLM分析 {len(images)} 张图片...")
        
        # System prompt: 定义助手角色
        system_prompt = "You are a helpful visual assistant. You analyze screenshots to answer user questions. Always respond in Chinese (中文回答)."
        
        # User prompt: 直接回答问题
        prompt = f"""User Question: {question}

The first image is the current screen, and the remaining images are recent history. Please directly answer the user's question first, then provide supporting evidence from the screenshots."""
        
        return self.vlm._call_vlm(
            prompt, 
            images, 
            num_images=len(images),
            image_timestamps=timestamps,
            system_prompt=system_prompt
        )
    
    def _format_realtime_result(self, response: str, images: List[PILImage.Image]) -> str:
        """格式化实时问答结果"""
        result = f"📝 实时问答结果:\n\n{response}\n\n"
        result += "="*60 + "\n"
        result += f"分析了 {len(images)} 张图片 (1张当前 + {len(images)-1}张历史)\n"
        return result
    
    # ============ OCR 模式查询 ============
    
    def query_ocr_rag(self, query_text: str, start_time: datetime, end_time: datetime):
        """OCR 模式 RAG 查询 - 基于 OCR 文本 embedding 检索（使用 LanceDB Pre-filtering）"""
        try:
            # Ensure start_time and end_time are UTC for database queries
            if start_time and start_time.tzinfo is None:
                start_time = start_time.astimezone(timezone.utc)
            if end_time and end_time.tzinfo is None:
                end_time = end_time.astimezone(timezone.utc)

            self.progress_signal.emit("📝 OCR模式: 正在检索相关文本...")
            logger.info(f"OCR RAG查询: '{query_text}' ({start_time} - {end_time})")
            
            # 使用 CLIP 对查询文本进行 embedding
            self._ensure_vector_mode_initialized()
            
            # 获取查询文本的 embedding
            query_embedding = self.encoder.encode_text(query_text)
            
            # 在 OCR 表中搜索（使用 LanceDB Pre-filtering，直接传递时间范围）
            frames = self.storage.search_ocr(
                query_embedding, 
                top_k=20,
                start_time=start_time,
                end_time=end_time
            )
            
            if not frames:
                # 如果 OCR 表为空，回退到 SQLite 全文搜索
                self.progress_signal.emit("📂 OCR向量表为空，使用全文搜索...")
                frames = self._search_ocr_fulltext(query_text, start_time, end_time)
            
            if not frames:
                self.result_signal.emit("在指定时间范围内未找到相关的OCR文本记录。")
                return
            
            self.progress_signal.emit(f"📊 找到 {len(frames)} 条相关OCR记录")
            
            # 使用纯文本 VLM 分析
            response = self._analyze_ocr_with_vlm(query_text, frames)
            result = self._format_ocr_result(response, frames)
            
            self.result_signal.emit(result)
            
        except Exception as e:
            logger.error(f"OCR RAG查询失败: {e}", exc_info=True)
            self.error_signal.emit(f"OCR查询失败: {str(e)}")
    
    def _search_ocr_fulltext(self, query_text: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """使用 SQLite 全文搜索 OCR 文本"""
        try:
            sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
            
            # 使用 SQLite 的文本搜索
            results = sqlite_storage.search_text(query_text, limit=20)
            
            # 过滤时间范围并构建结果
            frames = []
            for r in results:
                ts = r.get('timestamp')
                if ts and start_time <= ts <= end_time:
                    frames.append({
                        'frame_id': r['frame_id'],
                        'timestamp': ts,
                        'image_path': r.get('image_path', ''),
                        'ocr_text': r.get('ocr_text', ''),
                        'image': None  # OCR 模式不需要加载图片
                    })
            
            return frames
            
        except Exception as e:
            logger.error(f"全文搜索失败: {e}")
            return []
    
    def _analyze_ocr_with_vlm(self, query_text: str, frames: List[Dict]) -> str:
        """使用 VLM 分析 OCR 文本（纯文本模式）"""
        self.progress_signal.emit(f"🤖 正在分析 {len(frames)} 条OCR文本...")
        
        # 构建 OCR 文本上下文
        ocr_context = []
        for i, frame in enumerate(frames[:15], 1):  # 最多15条
            ts = frame['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            text = frame.get('ocr_text', '')[:500]  # 截断过长的文本
            ocr_context.append(f"[{i}] 时间: {ts}\n内容: {text}")
        
        ocr_text_block = "\n\n".join(ocr_context)
        
        # System prompt: 定义助手角色
        system_prompt = "You are a helpful assistant. You analyze OCR text to answer user questions. Always respond in Chinese (中文回答)."
        
        # User prompt: 直接回答问题
        prompt = f"""User Question: {query_text}

Please directly answer the user's question first, then provide supporting evidence from the OCR text below.

OCR Text Records (sorted by relevance):

{ocr_text_block}

If relevant content is found, please explain in detail:
1. At which time point the relevant content was seen
2. What the specific text content is
3. Possible context (what the user was doing at that time)"""
        
        return self.vlm._call_vlm_text_only(prompt, system_prompt=system_prompt)
    
    def _format_ocr_result(self, response: str, frames: List[Dict]) -> str:
        """格式化 OCR 查询结果"""
        result = f"📝 OCR文本检索结果:\n\n{response}\n\n"
        result += "="*60 + "\n"
        result += f"检索信息:\n"
        result += f"• 模式: OCR文本检索\n"
        result += f"• 匹配记录: {len(frames)} 条\n"
        
        if frames:
            result += f"\n相关时间点:\n"
            for i, frame in enumerate(frames[:5], 1):
                ts = frame['timestamp'].strftime('%H:%M:%S')
                text_preview = (frame.get('ocr_text', '')[:50] + "...") if len(frame.get('ocr_text', '')) > 50 else frame.get('ocr_text', '')
                result += f"  {i}. [{ts}] {text_preview}\n"
        
        return result
    
    def query_realtime_ocr(self, question: str):
        """OCR 模式实时问答 - 基于当前和历史 OCR 文本"""
        try:
            self.progress_signal.emit("📝 OCR模式: 获取当前和历史OCR文本...")
            logger.info(f"OCR实时问答: '{question}'")
            
            # 获取当前屏幕的 OCR
            from core.capture.capturer import ScreenshotCapturer
            from core.ocr.paddleocr_worker import PaddleOCRWorker
            
            capturer = ScreenshotCapturer()
            current_frame = capturer.capture()
            
            ocr_texts = []
            
            if current_frame:
                # 对当前屏幕进行 OCR
                self.progress_signal.emit("📸 正在识别当前屏幕文字...")
                try:
                    ocr_worker = PaddleOCRWorker()
                    current_ocr = ocr_worker.process(current_frame.image)
                    if current_ocr:
                        ocr_texts.append({
                            'time': '当前',
                            'text': current_ocr
                        })
                except Exception as e:
                    logger.warning(f"当前屏幕OCR失败: {e}")
            
            # 获取最近的历史 OCR 文本
            self.progress_signal.emit("📂 加载最近的OCR记录...")
            try:
                sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
                recent_frames = sqlite_storage.get_recent_frames(limit=10)
                
                for frame in recent_frames:
                    ocr_text = frame.get('ocr_text', '')
                    if ocr_text:
                        # 将存储的 UTC 时间转换为本地时间
                        ts_obj = frame['timestamp']
                        if ts_obj.tzinfo is None:
                            ts_obj = ts_obj.replace(tzinfo=timezone.utc)
                        ts = ts_obj.astimezone().strftime('%H:%M:%S')
                        
                        ocr_texts.append({
                            'time': ts,
                            'text': ocr_text[:500]  # 截断
                        })
            except Exception as e:
                logger.warning(f"获取历史OCR失败: {e}")
            
            if not ocr_texts:
                self.error_signal.emit("无法获取OCR文本")
                return
            
            # 使用 VLM 分析
            response = self._analyze_realtime_ocr_with_vlm(question, ocr_texts)
            result = self._format_realtime_ocr_result(response, ocr_texts)
            
            self.result_signal.emit(result)
            
        except Exception as e:
            logger.error(f"OCR实时问答失败: {e}", exc_info=True)
            self.error_signal.emit(f"OCR问答失败: {str(e)}")
    
    def _analyze_realtime_ocr_with_vlm(self, question: str, ocr_texts: List[Dict]) -> str:
        """使用 VLM 分析实时 OCR 文本"""
        self.progress_signal.emit(f"🤖 正在分析 {len(ocr_texts)} 条OCR文本...")
        
        context_parts = []
        for item in ocr_texts:
            context_parts.append(f"[{item['time']}]\n{item['text']}")
        
        ocr_context = "\n\n---\n\n".join(context_parts)
        
        # System prompt: 定义助手角色
        system_prompt = "You are a helpful assistant. You analyze OCR text to answer user questions. Always respond in Chinese (中文回答)."
        
        # User prompt: 直接回答问题
        prompt = f"""User Question: {question}

Please directly answer the user's question first, then provide supporting evidence from the OCR text below.

OCR Text Extracted from Screenshots (first is current screen, others are historical records):

{ocr_context}"""
        
        return self.vlm._call_vlm_text_only(prompt, system_prompt=system_prompt)
    
    def _format_realtime_ocr_result(self, response: str, ocr_texts: List[Dict]) -> str:
        """格式化实时 OCR 问答结果"""
        result = f"📝 OCR实时问答结果:\n\n{response}\n\n"
        result += "="*60 + "\n"
        result += f"分析了 {len(ocr_texts)} 条OCR文本记录\n"
        return result



