#!/usr/bin/env python3
"""
查找指定时间点（主屏）对应的帧在哪个 MP4 的哪一帧。
用法: python scripts/lookup_frame_by_time.py "2026-03-11 10:04:19"
"""
import os
import sqlite3
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import config as config_module
cfg = getattr(config_module, "config", None) or config_module.Config()

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/lookup_frame_by_time.py \"2026-03-11 10:04:19\"")
        sys.exit(1)
    time_str = sys.argv[1].strip()
    # 解析为 ISO 格式
    if " " in time_str:
        date_part, time_part = time_str.split(" ", 1)
        if len(date_part) == 8 and "-" not in date_part:  # 20260311
            time_str_iso = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}T{time_part}"
        else:
            time_str_iso = f"{date_part}T{time_part}"
    else:
        time_str_iso = time_str
    if "T" not in time_str_iso:
        time_str_iso = time_str_iso.replace(" ", "T", 1)

    db_path = cfg.OCR_DB_PATH
    if not Path(db_path).exists():
        print(f"数据库不存在: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 主屏帧: app_name IS NULL OR app_name = ''
    ts = time_str_iso[:19]  # 2026-03-11T10:04:19
    if len(ts) >= 19 and ts[17:19].isdigit():
        sec = int(ts[17:19]) + 1
        ts_next = ts[:17] + str(sec).zfill(2)
    else:
        ts_next = ts + "Z"
    cur.execute("""
        SELECT f.frame_id, f.timestamp, f.image_path, f.video_chunk_id, f.offset_index
        FROM frames f
        WHERE (f.app_name IS NULL OR f.app_name = '')
        AND f.timestamp >= ? AND f.timestamp < ?
        ORDER BY f.timestamp ASC
        LIMIT 10
    """, (ts, ts_next))
    rows = cur.fetchall()
    if not rows:
        # 扩大范围：该分钟内的帧
        cur.execute("""
            SELECT f.frame_id, f.timestamp, f.image_path, f.video_chunk_id, f.offset_index
            FROM frames f
            WHERE (f.app_name IS NULL OR f.app_name = '')
            AND f.timestamp LIKE ?
            ORDER BY f.timestamp ASC
            LIMIT 20
        """, (ts[:16] + "%",))  # 2026-03-11T10:04%
        rows = cur.fetchall()
    if not rows:
        # 数据库存的是 UTC；用户可能输入的是本地时间（如 UTC+8 的 10:04 = UTC 02:04）
        cur.execute("""
            SELECT f.frame_id, f.timestamp, f.image_path, f.video_chunk_id, f.offset_index
            FROM frames f
            WHERE (f.app_name IS NULL OR f.app_name = '')
            AND date(f.timestamp) = date(?)
            AND (f.timestamp LIKE '%' || strftime('%H:%M', ?) || '%'
                 OR f.timestamp LIKE '%' || strftime('%H:%M', datetime(?, '-8 hours')) || '%')
            ORDER BY f.timestamp ASC
            LIMIT 20
        """, (time_str_iso, time_str_iso, time_str_iso))
        rows = cur.fetchall()

    if not rows:
        print(f"未找到 {time_str} 附近的主屏帧。")
        conn.close()
        sys.exit(1)

    print(f"找到 {len(rows)} 条主屏帧（时间接近 {time_str}）：\n")
    for row in rows:
        print(f"  frame_id:      {row['frame_id']}")
        print(f"  timestamp:     {row['timestamp']}")
        print(f"  image_path:    {row['image_path']}")
        print(f"  video_chunk_id: {row['video_chunk_id']}")
        print(f"  offset_index:  {row['offset_index']}")

        if row["video_chunk_id"] is not None and row["offset_index"] is not None:
            cur.execute("SELECT id, file_path, fps FROM video_chunks WHERE id = ?", (row["video_chunk_id"],))
            vc = cur.fetchone()
            if vc:
                abs_path = vc["file_path"]
                if not os.path.isabs(abs_path):
                    abs_path = str(PROJECT_ROOT / abs_path)
                print(f"  -> MP4 文件:   {abs_path}")
                print(f"  -> 帧索引:     {row['offset_index']} (0-based，即第 {int(row['offset_index'])+1} 帧)")
                print(f"  -> FPS:        {vc['fps']}")
        print()

    conn.close()

if __name__ == "__main__":
    main()
