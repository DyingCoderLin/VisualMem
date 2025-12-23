import json
import re
import requests
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Set, Dict

from config import config
from core.understand.api_vlm import ApiVLM
from core.storage.sqlite_storage import SQLiteStorage
from utils.constants import COMBINED_EXAMPLES, QUERY_REWRITE_EXAMPLES, TIME_RANGE_EXAMPLES
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _call_query_rewrite_api(messages: list, temperature: float = 0) -> str:
    """
    调用 query rewrite API（支持独立的 OpenAI API 配置或使用 VLM 配置）
    参考 OpenAI SDK 格式，处理思考模型的特殊响应
    
    Args:
        messages: OpenAI 格式的消息列表
        temperature: 温度参数（默认 0，用于确定性输出）
    
    Returns:
        API 返回的文本内容（已处理思考模型的特殊格式）
    """
    # 检查是否配置了独立的 query rewrite API
    use_independent_api = (
        config.QUERY_REWRITE_BASE_URL and 
        config.QUERY_REWRITE_BASE_URL.strip()
    )
    
    if use_independent_api:
        # 使用独立的 OpenAI API 配置
        api_key = config.QUERY_REWRITE_API_KEY or ""
        base_url = config.QUERY_REWRITE_BASE_URL.rstrip('/')
        # 如果未配置模型，回退到 VLM 模型
        model = config.QUERY_REWRITE_MODEL or config.VLM_API_MODEL
        endpoint = f"{base_url}/v1/chat/completions"
        
        logger.info(f"Using Query Rewrite API: {base_url}, model: {model}")
    else:
        # 使用 VLM 配置（默认行为）
        api_key = config.VLM_API_KEY
        base_url = config.VLM_API_URI.rstrip('/')
        model = config.VLM_API_MODEL
        endpoint = f"{base_url}/v1/chat/completions"
        
        logger.info(f"使用 VLM API 进行 Query Rewrite: {base_url}, 模型: {model}")
    
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {api_key}"
    
    # 构建 payload，不设置 max_tokens，让模型自己决定
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=120, verify=False)
        resp.raise_for_status()
        data = resp.json()
        
        # 提取响应内容（参考 OpenAI SDK 格式）
        if not data or "choices" not in data or not data["choices"]:
            raise ValueError("API 响应格式错误：缺少 choices")
        
        choice = data["choices"][0]
        # print(f"choice: {choice}")
        if not choice or "message" not in choice:
            raise ValueError("API 响应格式错误：缺少 message")
        
        message = choice["message"]
        # print(f"message: {message}")
        content = message.get("content", "").strip() if message.get("content") else ""
        # print(f"content: {content}")
        
        # 处理思考模型的特殊格式：移除 <think> 标签
        # 参考 OpenAI SDK 格式处理思考模型响应
        if content.startswith('<think>') and '</think>' in content:
            think_end = content.find('</think>')
            if think_end != -1:
                content = content[think_end + len('</think>'):].strip()
        
        if not content:
            logger.warning(f"API 返回空内容，响应数据: {data}")
            raise ValueError("API 返回空内容")
        
        return content
    except Exception as e:
        logger.error(f"Query Rewrite API 调用失败: {e}")
        raise


def rewrite_and_time(
    query: str,
    enable_rewrite: bool,
    enable_time: bool,
    expand_n: int,
    api_client: Optional[ApiVLM] = None,
) -> Tuple[List[str], List[str], Optional[Tuple[datetime, datetime]]]:
    """
    调用 LLM 生成扩写查询和时间范围（可选）。
    返回 (dense_queries, sparse_queries, time_range)；time_range 为 (start_dt, end_dt) 或 None。
    
    如果配置了 QUERY_REWRITE_BASE_URL，将使用独立的 OpenAI API；
    否则使用 VLM 的配置（通过 api_client 或默认 ApiVLM）。
    """
    dense_queries = [query]
    sparse_queries = [query]
    time_range = None

    if not (enable_rewrite or enable_time):
        return dense_queries, sparse_queries, time_range

    prompt = _build_combined_prompt(expand_n, enable_rewrite, enable_time)
    # 添加 /no_think 后缀以禁用思考过程（如果模型支持）
    user_content = f"{query}, current time for reference: {datetime.now(timezone.utc).isoformat()} /no_think"
    print(f"user_content: {user_content}")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        # 检查是否使用独立的 API
        use_independent_api = (
            config.QUERY_REWRITE_BASE_URL and 
            config.QUERY_REWRITE_BASE_URL.strip()
        )
        
        if use_independent_api:
            # 使用独立的 OpenAI API（temperature=0 用于确定性输出）
            resp = _call_query_rewrite_api(messages, temperature=0)
        else:
            # 使用 VLM API（默认行为，也不设置 max_tokens）
            api = api_client or ApiVLM()
            # 直接调用 API，不设置 max_tokens，temperature=0
            resp = api.chat_text(messages, max_tokens=None, temperature=0)
    except Exception as e:
        logger.warning(f"Query rewrite 调用失败，使用原始查询: {e}")
        return dense_queries, sparse_queries, time_range

    # 解析响应（参考 OpenAI SDK 格式的错误处理）
    try:
        if resp and resp.strip():
            content = resp.strip()
            
            # 尝试解析 JSON
            try:
                parsed = json.loads(content)
                print(f"parsed: {parsed}")
                
                # 提取 dense queries
                if enable_rewrite and "dense_queries" in parsed and isinstance(parsed["dense_queries"], list):
                    dense_queries = parsed["dense_queries"] or [query]
                
                # 提取 sparse queries
                if enable_rewrite and "sparse_queries" in parsed and isinstance(parsed["sparse_queries"], list):
                    sparse_queries = parsed["sparse_queries"] or [query]
                
                # 提取时间范围
                if enable_time:
                    tr = parsed.get("time_range") or parsed.get("time_range_str")
                    if tr:
                        # 如果 time_range 是字典对象，直接提取 start 和 end
                        if isinstance(tr, dict):
                            start_str = tr.get("start")
                            end_str = tr.get("end")
                            if start_str and end_str:
                                try:
                                    start = datetime.fromisoformat(start_str.replace(" ", "T") if "T" not in start_str else start_str)
                                    end = datetime.fromisoformat(end_str.replace(" ", "T") if "T" not in end_str else end_str)
                                    if start > end:
                                        start, end = end, start
                                    time_range = (start, end)
                                except Exception as e:
                                    logger.warning(f"解析时间范围字典失败: {e}, 尝试字符串提取")
                                    time_range = extract_time_range(str(tr))
                            else:
                                time_range = None
                        else:
                            # 如果是字符串，使用正则提取
                            time_range = extract_time_range(str(tr))
                        
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}, 尝试从文本中提取时间范围")
                # 如果 JSON 解析失败，尝试从文本中提取时间范围
                if enable_time:
                    time_range = extract_time_range(content)
    except (AttributeError, KeyError) as e:
        logger.warning(f"Error processing response: {e}, fallback to original query")
        # 如果处理失败，尝试从原始响应中提取时间范围
        if enable_time:
            time_range = extract_time_range(resp)

    if not dense_queries:
        dense_queries = [query]
    if not sparse_queries:
        sparse_queries = [query]
    print(f"dense_queries: {dense_queries}, sparse_queries: {sparse_queries}, time_range: {time_range}")

    return dense_queries, sparse_queries, time_range


def _build_combined_prompt(expand_n: int, rewrite: bool, need_time: bool) -> str:
    """
    构造统一的 system prompt，包含 JSON 结构要求与示例。
    """
    examples = []
    if rewrite and not need_time:
        examples.append(QUERY_REWRITE_EXAMPLES)
    if need_time and not rewrite:
        examples.append(TIME_RANGE_EXAMPLES)
    if rewrite and need_time:
        examples.append(COMBINED_EXAMPLES)

    body = f"""
You are part of an academic information system that processes researchers' queries about computer systems.
For each query, return JSON only. If you cannot infer, fall back to the original query and time_range null.

Fields:
- dense_queries: {expand_n} queries that are similar in meaning and semantically related to the original query
- sparse_queries: Important keywords and key phrases extracted from the query
- time_range: object with "start"/"end" in ISO "YYYY-MM-DD HH:MM:SS", or "null" if you cannot infer the time range

Your Task: 
Generate both dense and sparse query expansions for the following query. Return only valid JSON in the specified format. No extra text.

{chr(10).join(examples)}
"""
    return body.strip()


def extract_time_range(text: str) -> Optional[Tuple[datetime, datetime]]:
    """
    用正则从文本中抽取时间范围，格式 YYYY-MM-DD HH:MM(:SS)
    """
    # 处理"null"的情况    
    if text == "null":
        return None
    pattern = r"(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}(?::\\d{2})?)"
    matches = re.findall(pattern, text)
    if len(matches) >= 2:
        try:
            start = datetime.fromisoformat(matches[0])
            end = datetime.fromisoformat(matches[1])
            print(f"extracted time range: {start} - {end}")
            if start > end:
                start, end = end, start
            return start, end
        except Exception:
            return None
    return None


def get_frame_ids_in_timerange(
    start_time: datetime,
    end_time: datetime,
    sqlite_db_path: Optional[str] = None,
    limit: int = 10000
) -> Set[str]:
    """
    从 SQLite 获取时间范围内的 frame_id 集合（用于在向量检索前过滤候选集）
    复用 GUI 的时间过滤逻辑，直接在 SQL 中按时间范围查询，更高效
    
    Args:
        start_time: 开始时间
        end_time: 结束时间
        sqlite_db_path: SQLite 数据库路径，默认使用 config.OCR_DB_PATH
        limit: 最大返回数量（防止内存溢出）
    
    Returns:
        frame_id 集合
    """
    try:
        sqlite_storage = SQLiteStorage(db_path=sqlite_db_path or config.OCR_DB_PATH)
        # 直接在 SQL 中按时间范围查询，复用 GUI 的逻辑
        frames_meta = sqlite_storage.get_frames_in_timerange(start_time, end_time, limit=limit)
        
        # 提取 frame_id
        frame_ids = {f['frame_id'] for f in frames_meta}
        
        return frame_ids
    except Exception as e:
        from utils.logger import setup_logger
        logger = setup_logger(__name__)
        logger.warning(f"获取时间范围内的 frame_id 失败: {e}")
        return set()


def filter_by_time(frames, time_range: Optional[Tuple[datetime, datetime]]):
    """
    根据时间范围过滤帧列表（在检索结果后过滤）
    """
    if not time_range:
        return frames
    start, end = time_range
    
    # 确保 start/end 与待比较的 ts 时区属性一致
    is_start_aware = start.tzinfo is not None
    
    filtered = []
    for f in frames:
        ts = f.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except Exception:
                continue
        
        if not isinstance(ts, datetime):
            continue
            
        # 处理 offset-naive 和 offset-aware 的比较问题
        if is_start_aware and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        elif not is_start_aware and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
            
        if start <= ts <= end:
            filtered.append(f)
    return filtered


def filter_by_frame_ids(frames: List[Dict], frame_ids: Set[str]) -> List[Dict]:
    """
    根据 frame_id 集合过滤帧列表（用于向量检索后只保留时间范围内的结果）
    
    如果 frame_ids 为空集合，返回空列表（表示时间范围内没有候选帧）
    """
    if not frame_ids:
        # 如果时间范围内没有候选帧，返回空列表
        return []
    return [f for f in frames if f.get("frame_id") in frame_ids]

