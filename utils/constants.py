QUERY_REWRITE_EXAMPLES = """Examples:
1. Query: "climate change effects"
Response: {{
    "dense_queries": ["impact of climate change", "consequences of global warming", "effects of environmental changes"],
    "sparse_queries": ["climate change", "effects", "environmental impact", "global warming"]
}}

2. Query: "machine learning algorithms for image recognition"
Response: {{
    "dense_queries": ["deep learning for computer vision", "neural networks in image processing", "AI algorithms for visual recognition"],
    "sparse_queries": ["machine learning", "algorithms", "image recognition", "computer vision", "neural networks"]
}}

3. Query: "Please show me some paper about NVM filesystem"
Response: {{
    "dense_queries": ["non-volatile memory file systems research", "NVM storage systems papers", "persistent memory filesystem studies"],
    "sparse_queries": ["NVM filesystem", "NVM", "filesystem", "paper", "non-volatile memory"]
}}
"""

TIME_RANGE_EXAMPLES = """Examples:
1. Query: "show me screenshots from yesterday afternoon, current time is 2025-12-13 12:00:00"
Response: {
  "time_range": {"start": "2025-12-12 12:00:00", "end": "2025-12-12 18:00:00"}
}

2. Query: "I want to know what I did last week, current time is 2025-12-13 8:00:00"
Response: {
  "time_range": {"start": "2025-12-06 00:00:00", "end": "2025-12-12 23:59:59"}
}

3. Query: "I forget something, I only remember it was mentioned from 12-06 to 12-12, current time is 2025-12-13 8:00:00"
Response: {
  "time_range": {"start": "2025-12-06 00:00:00", "end": "2025-12-12 23:59:59"}
}
"""

COMBINED_EXAMPLES = """Examples:
1. Query: "Please show me some paper about NVM filesystem last week, current time is 2025-12-13 8:00:00"
Response: {{
  "dense_queries": ["non-volatile memory file systems research", "NVM storage systems papers", "persistent memory filesystem studies"],
  "sparse_queries": ["NVM filesystem", "non-volatile memory", "paper", "storage"],
  "time_range": {{"start": "2025-12-06 00:00:00", "end": "2025-12-12 23:59:59"}}
}}

2. Query: "machine learning algorithms for image recognition, current time is 2025-12-13 8:00:00"
Response: {{
    "dense_queries": ["deep learning for computer vision", "neural networks in image processing", "AI algorithms for visual recognition"],
    "sparse_queries": ["machine learning", "algorithms", "image recognition", "computer vision", "neural networks"]
    "time_range": "null"
}}

3. Query: "我看过点什么视频, current time is 2025-12-13 8:00:00"
Response: {{
  "dense_queries": ["我看过哪些视频", "查看我看过的视频记录", "最近我看过的视频有哪些"],
  "sparse_queries": ["观看记录", "视频", "看过", "观看历史","播放记录"],
  "time_range": "null"
}}
"""