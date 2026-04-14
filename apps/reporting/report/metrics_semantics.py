"""Human-readable explanation of data-platform metrics for the daily report."""

# Shown at the top of CLI JSON output so users and the Reduce model share the same semantics.

METRICS_SEMANTICS_ZH = """\
【指标含义（与 API 对齐）】
- 「各应用前台时长 / focused_minutes」来自主库 frames：在每个采样时刻，系统记录的**当前前台（键盘焦点）应用**为该应用时，计入 1 帧；时长 = 帧数 × 捕获间隔。不是眼动/注意力时长；**应用在后台时不会累计**。
- 因此：若支付成功页、聊天窗口等**一直留在前台**（即使人未操作），也会累计前台时长；若已切到其他应用，则不会算在该应用上。
- 「活动标签 breakdown 中的 focused_minutes」同样是**该标签对应子帧与前台应用一致**时的采样累计，标签来自聚类/OCR 语义，**不要把某条 OCR 文本（如「支付成功」）直接等同于「盯着这行字看了 N 分钟」**；N 分钟是前台窗口维度的粗粒度代理。
- 「各应用 purpose_keywords」由 Reduce 阶段 **LLM** 根据前台时长 + Map 摘要 + 下方证据提示（会话标签/窗口标题）**综合生成**，表示**当日用该应用大致在做什么**；不是 OCR 词频。
"""

METRICS_SEMANTICS_EN = """\
[Metric semantics — aligned with HTTP APIs]
- Per-app focused_minutes counts capture intervals where frames.focused_app_name == that app (foreground at sample time) × CAPTURE_INTERVAL. Not eye-tracking; background apps do not accrue.
- A static foreground window (e.g. payment receipt left open) still accrues time even if idle.
- Activity-label breakdown minutes are still foreground-based; labels describe cluster/OCR context — do not equate a phrase in OCR with “stared at this text for N minutes.”
- Per-app `purpose_keywords` are **LLM-synthesized** from Map summaries and evidence hints — not raw OCR term frequency.
"""

# Single blob for JSON `metrics_semantics` field (CLI / file output).
METRICS_SEMANTICS_FOR_JSON = METRICS_SEMANTICS_ZH.strip() + "\n\n" + METRICS_SEMANTICS_EN.strip()
