#!/usr/bin/env python3
"""
Test script for Data Platform API endpoints.

Usage:
    1. Start backend:  python gui_backend_server.py
    2. Run this:       python scripts/test_data_platform_api.py
"""

import json
import sys
from datetime import datetime, timedelta

import requests

BASE = "http://localhost:18080/api"

# Fixed test date: 2026-03-24 (large dataset)
test_date = "2026-03-24"
start_ts = f"{test_date}T00:00:00"
end_ts = f"{test_date}T23:59:59"

# Shorter range for OCR (morning block)
ocr_start = f"{test_date}T09:00:00"
ocr_end = f"{test_date}T11:00:00"


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def call(method, url, **kwargs):
    try:
        if method == "GET":
            r = requests.get(url, params=kwargs.get("params"), timeout=30)
        else:
            r = requests.post(url, json=kwargs.get("json"), timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        print("ERROR: Cannot connect to backend. Start it first:")
        print("  python gui_backend_server.py")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response: {e.response.text[:500]}")
        return None


# ------------------------------------------------------------------
# 1. Timeline / Apps
# ------------------------------------------------------------------
sep(f"1. GET /api/timeline/apps  ({test_date})")
data = call("GET", f"{BASE}/timeline/apps", params={"start": start_ts, "end": end_ts})
if data:
    print(f"Query range: {data['query_start']}  →  {data['query_end']}")
    print(f"Apps found: {len(data['apps'])}")
    for app in data["apps"][:5]:
        total_segs = len(app["segments"])
        print(f"  [{app['app_name']}] {total_segs} segments")
        for seg in app["segments"][:3]:
            dur = seg['duration_seconds']
            dur_str = f"{dur//60}m{dur%60}s" if dur >= 60 else f"{dur}s"
            print(f"    {seg['start'][:19]} ~ {seg['end'][:19]}  {dur_str}  label={seg['activity_label']}")
        if total_segs > 3:
            print(f"    ... and {total_segs - 3} more")
    if len(data["apps"]) > 5:
        print(f"  ... and {len(data['apps']) - 5} more apps")


# ------------------------------------------------------------------
# 2. Timeline / Activities
# ------------------------------------------------------------------
sep(f"2. GET /api/timeline/activities  ({test_date})")
data = call("GET", f"{BASE}/timeline/activities", params={"start": start_ts, "end": end_ts})
if data:
    acts = data["activities"]
    print(f"Total activities: {len(acts)}")
    for a in acts[:8]:
        dur = a['duration_seconds']
        dur_str = f"{dur//3600}h{dur%3600//60}m" if dur >= 3600 else f"{dur//60}m{dur%60}s" if dur >= 60 else f"{dur}s"
        print(f"  {a['start'][:19]} ~ {a['end'][:19]}  {dur_str:>8s}  [{a['app']}] {a['label']}")
    if len(acts) > 8:
        print(f"  ... and {len(acts) - 8} more")


# ------------------------------------------------------------------
# 3. OCR Texts
# ------------------------------------------------------------------
sep("3. GET /api/ocr/texts  (09:00-11:00)")
data = call("GET", f"{BASE}/ocr/texts", params={
    "start": ocr_start, "end": ocr_end, "limit": 5,
})
if data:
    print(f"Total OCR records: {data['total_count']}")
    for item in data["items"]:
        text_preview = (item["text"] or "")[:80].replace("\n", "↵")
        print(f"  {item['timestamp'][:19]}  [{item['app_name']}] {item['window_name']}")
        print(f"    confidence={item['confidence']:.2f}  len={item['text_length']}  label={item.get('activity_label', '-')}")
        print(f"    text: {text_preview}...")


# ------------------------------------------------------------------
# 4. OCR Texts with keyword search
# ------------------------------------------------------------------
sep(f"4. GET /api/ocr/texts  (FTS5 keyword='python', {test_date})")
data = call("GET", f"{BASE}/ocr/texts", params={
    "start": start_ts, "end": end_ts, "keyword": "python", "limit": 3,
})
if data:
    print(f"Total matches: {data['total_count']}")
    for item in data["items"]:
        text_preview = (item["text"] or "")[:80].replace("\n", "↵")
        print(f"  {item['timestamp'][:19]}  [{item['app_name']}]  len={item['text_length']}")
        print(f"    text: {text_preview}...")


# ------------------------------------------------------------------
# 5. OCR Regions
# ------------------------------------------------------------------
sep("5. GET /api/ocr/regions  (first sub_frame from test 3)")
ocr_data = call("GET", f"{BASE}/ocr/texts", params={
    "start": start_ts, "end": end_ts, "limit": 1,
})
if ocr_data and ocr_data["items"]:
    sfid = ocr_data["items"][0]["sub_frame_id"]
    print(f"Using sub_frame_id: {sfid}")
    data = call("GET", f"{BASE}/ocr/regions", params={"sub_frame_id": sfid})
    if data:
        print(f"Image size: {data['image_width']}x{data['image_height']}")
        print(f"Regions: {len(data['regions'])}")
        for r in data["regions"][:5]:
            bbox = r["bbox"]
            text_preview = (r["text"] or "")[:60].replace("\n", "↵")
            print(f"  [{r['region_index']}] ({bbox['x1']},{bbox['y1']})-({bbox['x2']},{bbox['y2']})  conf={r['confidence']:.2f}")
            print(f"       {text_preview}")
else:
    print("No OCR data available to test regions")


# ------------------------------------------------------------------
# 6. Focus Score
# ------------------------------------------------------------------
sep(f"6. GET /api/focus/score  ({test_date})")
data = call("GET", f"{BASE}/focus/score", params={"start": start_ts, "end": end_ts})
if data:
    m = data["metrics"]
    print(f"Focus score: {data['focus_score']}/100")
    print(f"Longest streak: {m['longest_focus_streak_minutes']} min")
    print(f"Avg streak: {m['avg_focus_streak_minutes']} min")
    print(f"Switches/hour: {m['app_switches_per_hour']}")
    print(f"Top distractions: {m['top_distraction_apps']}")
    print(f"Deep work blocks: {len(m['deep_work_blocks'])}")
    for b in m["deep_work_blocks"][:3]:
        print(f"  {b['start'][:19]} ~ {b['end'][:19]}  [{b['app']}] {b.get('minutes', '?')} min")
    print(f"Fragmented periods: {len(m['fragmented_periods'])}")
    for p in m["fragmented_periods"][:3]:
        print(f"  {p['start'][:19]} ~ {p['end'][:19]}  {p['switches']} switches")


# ------------------------------------------------------------------
# 7. Context Keywords
# ------------------------------------------------------------------
sep(f"7. GET /api/context/keywords  ({test_date})")
data = call("GET", f"{BASE}/context/keywords", params={
    "start": start_ts, "end": end_ts, "top_k": 15,
})
if data:
    print(f"Keywords ({len(data['keywords'])}):")
    for kw in data["keywords"]:
        apps_str = ", ".join(kw["apps"][:3])
        if len(kw["apps"]) > 3:
            apps_str += f" +{len(kw['apps'])-3}"
        print(f"  {kw['word']:30s}  count={kw['count']:4d}  apps=[{apps_str}]")


# ------------------------------------------------------------------
# 7b. Context Keywords by App (per-app OCR term frequencies)
# ------------------------------------------------------------------
sep(f"7b. GET /api/context/keywords-by-app  ({test_date})")
data = call("GET", f"{BASE}/context/keywords-by-app", params={
    "start": start_ts, "end": end_ts, "per_app": 5,
})
if data:
    apps_kw = data.get("apps") or []
    print(f"Apps with OCR keywords: {len(apps_kw)}")
    for entry in apps_kw[:12]:
        app = entry.get("app", "?")
        kws = entry.get("keywords") or []
        kw_str = ", ".join(f"{k['word']}({k['count']})" for k in kws[:5])
        print(f"  {app:30s}  {kw_str}")


# ------------------------------------------------------------------
# 8. Apps Focused Time
# ------------------------------------------------------------------
sep(f"8. GET /api/apps/focused-time  ({test_date})")
data = call("GET", f"{BASE}/apps/focused-time", params={"start": start_ts, "end": end_ts})
if data:
    print(f"Total focused: {data['total_focused_minutes']} min")
    for u in data["apps"][:10]:
        bar = "█" * int(u["percentage"] / 3)
        print(f"  {u['app']:25s}  {u['focused_minutes']:6.1f}min  {u['percentage']:5.1f}%  {bar}")


# ------------------------------------------------------------------
# 9. Recording Span
# ------------------------------------------------------------------
sep(f"9. GET /api/recording/span  ({test_date})")
data = call("GET", f"{BASE}/recording/span", params={"start": start_ts, "end": end_ts})
if data:
    first = data['first_active'][:19] if data['first_active'] else '-'
    last = data['last_active'][:19] if data['last_active'] else '-'
    print(f"First active: {first}")
    print(f"Last active:  {last}")


# ------------------------------------------------------------------
# 10. Activities Breakdown
# ------------------------------------------------------------------
sep(f"10. GET /api/activities/breakdown  ({test_date})")
data = call("GET", f"{BASE}/activities/breakdown", params={"start": start_ts, "end": end_ts})
if data:
    print(f"Activities ({len(data['breakdown'])}):")
    for a in data["breakdown"][:10]:
        print(f"  {a['label']:40s}  {a['focused_minutes']:6.1f}min (focused)")
    if len(data["breakdown"]) > 10:
        print(f"  ... and {len(data['breakdown']) - 10} more")


# ------------------------------------------------------------------
# 11. Window Titles
# ------------------------------------------------------------------
sep(f"11. GET /api/windows/titles  ({test_date})")
data = call("GET", f"{BASE}/windows/titles", params={"start": start_ts, "end": end_ts, "limit": 15})
if data:
    print(f"Window titles ({len(data['window_titles'])}):")
    for w in data["window_titles"][:10]:
        print(f"  [{w['app']}] {w['window'][:60]}  (x{w['count']})")


print(f"\n{'='*60}")
print("  All tests completed.")
print(f"{'='*60}")
