"""
VLM-based activity label generation and fuzzy matching.

Extracted from scripts/backfill_activity_clusters.py for reuse in the
real-time clustering pipeline.
"""

import base64
import io
import re
from typing import List, Optional

from utils.logger import setup_logger

logger = setup_logger("activity.vlm_labeler")


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


def call_vlm(
    vlm_url: str,
    app_name: str,
    image,
    existing_labels: List[str],
    ocr_text: str = "",
) -> str:
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
    import requests

    b64 = _image_to_base64(image)
    ocr_snippet = ocr_text.strip()[:500] if ocr_text else ""

    existing_labels_text = ""
    if existing_labels:
        existing_labels_text = (
            f"该应用已有标签供参考：{', '.join(existing_labels)}\n"
            f"如果截图内容与某个已有标签相同，可以复用。但不要强行匹配，内容不同就创建新标签。\n\n"
        )

    user_text = f"应用：{app_name}\n{existing_labels_text}"
    if ocr_snippet:
        user_text += f"屏幕文字：{ocr_snippet}\n"
    user_text += (
        "请简要描述这张桌面截图中用户在做什么操作，然后在最后一行输出：\n"
        "标签：（你的标签）\n"
        "标签要求3-8个中文词，描述具体操作。如：和朋友聊天、阅读arXiv论文、编写Python代码、查看邮件通知"
    )

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": user_text},
        ]},
    ]

    payload = {
        "model": "Qwen/Qwen3.5-9B",
        "messages": messages,
        "temperature": 0.1,
    }

    url = vlm_url.rstrip("/") + "/v1/chat/completions"
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        raw = result["choices"][0]["message"]["content"].strip()
        return parse_label(raw)
    except Exception as e:
        logger.error(f"VLM call failed: {e}")
        return ""
