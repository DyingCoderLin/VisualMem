"""
VLM-based activity label generation and fuzzy matching.

Extracted from scripts/backfill_activity_clusters.py for reuse in the
real-time clustering pipeline.
"""

import base64
import io
import json
import re
import time

import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from config import config
from utils.logger import setup_logger

logger = setup_logger("activity.vlm_labeler")
_CLUSTER_LABEL_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "cluster_labeling_events.jsonl"


def is_cluster_vlm_endpoint_configured() -> bool:
    """True if CLUSTER_VLM_URL is set (e.g. https://host/api → .../api/v1/chat/completions)."""
    return bool((config.CLUSTER_VLM_URL or "").strip())


def resolve_cluster_chat_completions_url(vlm_url_override: str = "") -> str:
    """OpenAI-compatible POST URL: {base}/v1/chat/completions."""
    base = (vlm_url_override or config.CLUSTER_VLM_URL or "").strip().rstrip("/")
    if not base:
        return ""
    return f"{base}/v1/chat/completions"


def _cluster_label_model_id(use_vision: bool) -> str:
    """Model id is passed through as configured (e.g. litellm provider/model)."""
    default = "Qwen/Qwen3.5-9B"
    if use_vision:
        return (config.CLUSTER_VLM_MODEL or default) or default
    if config.CLUSTER_VLM_TEXT_MODEL:
        return config.CLUSTER_VLM_TEXT_MODEL
    return (config.CLUSTER_VLM_MODEL or default) or default


# ---------------------------------------------------------------------------
# Label cleaning / validation
# ---------------------------------------------------------------------------

_GARBAGE_PATTERNS = re.compile(
    r'(xxx|用户|截图|显示|正在|需要|提供|可以|应该|让我|候选|分析|或者|这里截断'
    r'|如果截图|标签要求|中文词|描述具体|请分析|分析完后|输出标签|回复格式'
    r'|属于以上|第一行|看起来像)'
)


def _clean_label(text: str) -> str:
    """Clean a candidate label string."""
    text = text.strip().strip('"').strip("'").strip('*').strip('`').strip()
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'[。.，,！!？?；;：:]+$', '', text).strip()
    text = re.sub(r'^[\d]+[.、)\]]\s*', '', text).strip()
    text = re.sub(r'^[-•·]\s*', '', text).strip()
    text = re.sub(r'^或者\s*[：:]?\s*', '', text).strip()
    if '、' in text:
        text = text.split('、')[0].strip()
    for sep in ['，', ',']:
        if sep in text:
            text = text.split(sep)[0].strip()
    text = text.strip('"').strip("'").strip('"').strip('"').strip('(').strip(')').strip('（').strip('）').strip()
    return text


def _is_valid_label(text: str) -> bool:
    """Check if a cleaned string is a valid label."""
    if not text or len(text) < 2 or len(text) > 30:
        return False
    if _GARBAGE_PATTERNS.search(text):
        return False
    return True


def parse_label(raw: str) -> str:
    """Extract a clean short label from VLM response."""
    matches = re.findall(r'标签[：:]\s*(.+)', raw)
    for match in reversed(matches):
        label = _clean_label(match)
        if _is_valid_label(label):
            return label

    m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
    if m:
        label = _clean_label(m.group(1))
        if _is_valid_label(label):
            return label

    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        cleaned = _clean_label(line)
        if 3 <= len(cleaned) <= 20 and _is_valid_label(cleaned):
            return cleaned

    return ""


def fuzzy_match_label(candidate: str, existing_labels: List[str]) -> Optional[str]:
    """Check if candidate label matches any existing label.

    Returns the matched existing label (canonical form), or None if no match.
    """
    if not existing_labels:
        return None

    def _normalize(s: str) -> str:
        return re.sub(r'[\s，。、！？""\'\'·\-]', '', s).lower()

    c_norm = _normalize(candidate)

    for existing in existing_labels:
        e_norm = _normalize(existing)
        if candidate == existing or c_norm == e_norm:
            return existing
        if len(c_norm) >= 3 and len(e_norm) >= 3:
            if c_norm in e_norm or e_norm in c_norm:
                return existing
        c_set = set(c_norm)
        e_set = set(e_norm)
        if c_set and e_set:
            jaccard = len(c_set & e_set) / len(c_set | e_set)
            if jaccard > 0.6:
                return existing

    return None


def _image_to_base64(img) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def log_cluster_labeling_event(event: Dict[str, object]) -> None:
    """Append a cluster labeling event into logs/cluster_labeling_events.jsonl."""
    try:
        _CLUSTER_LABEL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with _CLUSTER_LABEL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to append cluster labeling event log: {e}")


def build_region_layout_text(regions: List[dict]) -> str:
    """Build compact region layout context from OCR regions."""
    if not regions:
        return ""

    sorted_regions = sorted(
        [r for r in regions if r.get("text") and str(r.get("text")).strip()],
        key=lambda r: (
            int((r.get("bbox") or [0, 0, 0, 0])[1]),
            int((r.get("bbox") or [0, 0, 0, 0])[0]),
        ),
    )
    if not sorted_regions:
        return ""

    image_width = int(sorted_regions[0].get("image_width", 0) or 0)
    image_height = int(sorted_regions[0].get("image_height", 0) or 0)
    canvas_text = ""
    if image_width > 0 and image_height > 0:
        canvas_text = f"画布尺寸：{image_width}x{image_height} 像素\n"

    lines = []
    for region in sorted_regions[:50]:
        bbox = region.get("bbox") or [0, 0, 0, 0]
        text = str(region.get("text", "")).strip().replace("\n", " ")
        if not text:
            continue
        text = text[:120]
        lines.append(f"bbox={bbox} | text={text}")

    if not lines:
        return ""
    return canvas_text + "区域布局（按从上到下、从左到右）：\n" + "\n".join(lines)


def call_vlm(
    vlm_url: str,
    app_name: str,
    image,
    existing_labels: List[str],
    ocr_text: str = "",
    layout_text: str = "",
    window_name: str = "",
    force_use_vision: Optional[bool] = None,
    return_meta: bool = False,
) -> Union[str, Tuple[str, Dict[str, object]]]:
    """Call VLM to generate a semantic label for a cluster.

    Args:
        vlm_url: VLM server base URL (e.g. http://localhost:8088)
        app_name: Application name
        image: PIL Image (representative frame)
        existing_labels: Already-known labels for this app (for reuse hints)
        ocr_text: OCR text from the frame

    Returns:
        Parsed label string, or "" on failure.
    """
    use_vision = config.CLUSTER_VLM_USE_VISION if force_use_vision is None else force_use_vision
    ocr_snippet = ocr_text.strip()[: config.CLUSTER_VLM_TEXT_OCR_MAX_CHARS] if ocr_text else ""
    layout_snippet = layout_text.strip()[:2000] if layout_text else ""

    def _empty_meta(mode: str) -> Dict[str, object]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "elapsed_ms": 0,
            "mode": mode,
        }

    existing_labels_text = ""
    if existing_labels:
        existing_labels_text = (
            f"该应用已有标签供参考：{', '.join(existing_labels)}\n"
            f"如果截图内容与某个已有标签相同，可以复用。但不要强行匹配，内容不同就创建新标签。\n\n"
        )

    user_text = f"应用：{app_name}\n"
    if window_name:
        user_text += f"窗口标题：{window_name}\n"
    user_text += existing_labels_text
    if ocr_snippet:
        user_text += f"屏幕文字：{ocr_snippet}\n"
    if layout_snippet and config.CLUSTER_VLM_INCLUDE_REGION_LAYOUT:
        user_text += f"{layout_snippet}\n"
    if use_vision:
        user_text += (
            "请简要描述这张桌面截图中用户在做什么操作，然后在最后一行输出：\n"
            "标签：（你的标签）\n"
            "标签要求3-8个中文词，描述具体操作。如：和朋友聊天、阅读arXiv论文、编写Python代码、查看邮件通知"
        )
    else:
        user_text += (
            "请根据以上屏幕文字和布局线索推断用户在做什么操作，然后在最后一行输出：\n"
            "标签：（你的标签）\n"
            "标签要求3-8个中文词，描述具体操作。如：和朋友聊天、阅读arXiv论文、编写Python代码、查看邮件通知"
        )

    # Text-only: never upgrade to vision when OCR/layout are empty (user may use a text-only API).
    if not use_vision and not ocr_snippet and not layout_snippet:
        logger.debug(
            "Skip cluster VLM: text-only mode with no OCR/layout context (not sending image)"
        )
        if return_meta:
            return "", _empty_meta("text")
        return ""

    if use_vision:
        if image is None:
            if return_meta:
                return "", _empty_meta("vision")
            return ""
        b64 = _image_to_base64(image)
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]

    model_id = _cluster_label_model_id(use_vision)
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.1,
    }

    url = resolve_cluster_chat_completions_url(vlm_url)
    if not url:
        logger.error("Cluster VLM URL not configured: set CLUSTER_VLM_URL in .env")
        if return_meta:
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_ms": 0, "mode": "vision" if use_vision else "text"}
        return ""
    start = time.perf_counter()
    try:
        headers = {"Content-Type": "application/json"}
        api_key = config.CLUSTER_VLM_API_KEY or config.VLM_API_KEY
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        raw = result["choices"][0]["message"]["content"].strip()
        parsed_label = parse_label(raw)
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        mode = "vision" if use_vision else "text"
        logger.info(
            f"Cluster labeling call | app={app_name} window={window_name or '-'} mode={mode} "
            f"in={prompt_tokens} out={completion_tokens} total={total_tokens} "
            f"time_ms={elapsed_ms} label={parsed_label or '-'}"
        )
        meta: Dict[str, object] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "elapsed_ms": elapsed_ms,
            "mode": mode,
        }
        if return_meta:
            return parsed_label, meta
        return parsed_label
    except requests.HTTPError as e:
        body = ""
        code = ""
        if e.response is not None:
            code = str(e.response.status_code)
            body = (e.response.text or "")[:2000]
        logger.error(
            "Cluster VLM HTTP error status=%s model=%s mode=%s err=%s body=%r",
            code or "?",
            model_id,
            "vision" if use_vision else "text",
            str(e),
            body,
        )
        if return_meta:
            return "", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
                "mode": "vision" if use_vision else "text",
                "http_status": code,
                "error_body": body,
            }
        return ""
    except Exception as e:
        logger.error(f"VLM call failed: {e}")
        if return_meta:
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_ms": int((time.perf_counter() - start) * 1000), "mode": "vision" if use_vision else "text"}
        return ""
