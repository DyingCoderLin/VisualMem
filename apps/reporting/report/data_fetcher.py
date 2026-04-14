"""
Async HTTP client for the VisualMem data platform API.

Fully decoupled — communicates only via HTTP.  Can be pointed at a remote
server by changing *base_url*.
"""

from __future__ import annotations

import asyncio
import aiohttp
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from utils.logger import setup_logger

logger = setup_logger("report.data_fetcher")


class DataFetcher:
    """Thin async wrapper around the data platform REST endpoints."""

    def __init__(self, base_url: str = "http://localhost:18080"):
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Fast fail when backend is down (connect refused); avoid 7×60s wall time.
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30, connect=5, sock_connect=5)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        session = await self._ensure_session()
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode({k: v for k, v in params.items() if v is not None})}"
        logger.debug("HTTP GET %s params=%s", path, params)
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(f"GET {path} returned {resp.status}: {body[:200]}")
                    return {}
                data = await resp.json()
                logger.debug(
                    "HTTP GET %s -> 200; top-level keys=%s",
                    path,
                    list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                )
                return data
        except asyncio.TimeoutError as e:
            logger.warning("GET %s timeout after connect/read limit: %r url=%s", path, e, url)
            return {}
        except aiohttp.ClientError as e:
            # Connection refused, DNS, TLS, etc. — str(e) is often empty; use %r
            logger.warning("GET %s client error: %r url=%s", path, e, url)
            return {}
        except Exception as e:
            logger.warning("GET %s failed: %r url=%s", path, e, url)
            return {}

    # ------------------------------------------------------------------
    # Data platform endpoints
    # ------------------------------------------------------------------

    async def get_activities(self, start: str, end: str) -> Dict:
        return await self._get("/api/timeline/activities", {"start": start, "end": end})

    async def get_ocr_texts(
        self,
        start: str,
        end: str,
        app_name: Optional[str] = None,
        limit: int = 200,
    ) -> Dict:
        return await self._get("/api/ocr/texts", {
            "start": start,
            "end": end,
            "app_name": app_name,
            "limit": str(limit),
        })

    async def get_focus_score(self, start: str, end: str) -> Dict:
        return await self._get("/api/focus/score", {"start": start, "end": end})

    async def get_keywords(self, start: str, end: str, top_k: int = 30) -> Dict:
        return await self._get("/api/context/keywords", {
            "start": start, "end": end, "top_k": str(top_k),
        })

    async def get_app_focused_time(self, start: str, end: str) -> Dict:
        return await self._get("/api/apps/focused-time", {"start": start, "end": end})

    async def get_window_titles(self, start: str, end: str, limit: int = 100) -> Dict:
        return await self._get("/api/windows/titles", {
            "start": start, "end": end, "limit": str(limit),
        })

    async def get_activities_breakdown(self, start: str, end: str) -> Dict:
        return await self._get("/api/activities/breakdown", {"start": start, "end": end})

    async def get_recording_span(self, start: str, end: str) -> Dict:
        return await self._get("/api/recording/span", {"start": start, "end": end})

    async def get_context_snapshot(self, timestamp: str) -> Dict:
        return await self._get("/api/context/snapshot", {"timestamp": timestamp})

    # ------------------------------------------------------------------
    # Batch: fetch all day data in parallel
    # ------------------------------------------------------------------

    async def _probe_backend(self) -> bool:
        """Single fast check so we do not wait for 7×HTTP timeouts when nothing is listening."""
        session = await self._ensure_session()
        url = f"{self.base_url}/health"
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=8, connect=5, sock_connect=5),
            ) as resp:
                if resp.status == 200:
                    return True
                logger.warning("health probe: %s returned HTTP %s", url, resp.status)
                return False
        except Exception as e:
            logger.warning("health probe failed: %r url=%s", e, url)
            return False

    async def fetch_day_data(self, start: str, end: str) -> Dict[str, Any]:
        """Fetch all data needed for daily report generation in parallel."""
        keys = [
            "activities", "focus_score", "keywords",
            "app_focused_time", "window_titles",
            "activities_breakdown", "recording_span",
        ]
        if not await self._probe_backend():
            logger.error(
                "fetch_day_data: data platform unreachable at %s (GET /health failed). "
                "Start gui_backend_server (or Electron) on that host/port, or set REPORT_DATA_API_BASE in .env.",
                self.base_url,
            )
            return {k: {} for k in keys}

        logger.debug(
            "fetch_day_data: parallel GET 7 endpoints for range [%s, %s] — "
            "activities, focus_score, keywords, app_focused_time, window_titles, "
            "activities_breakdown, recording_span",
            start,
            end,
        )
        results = await asyncio.gather(
            self.get_activities(start, end),
            self.get_focus_score(start, end),
            self.get_keywords(start, end),
            self.get_app_focused_time(start, end),
            self.get_window_titles(start, end),
            self.get_activities_breakdown(start, end),
            self.get_recording_span(start, end),
            return_exceptions=True,
        )

        data = {}
        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                logger.warning("fetch_day_data: %s failed: %r", key, result)
                data[key] = {}
            else:
                data[key] = result

        # Timeline failure vs "no sessions": success shape is {"activities": [...]}; total failure is {}.
        if data.get("activities") == {}:
            logger.error(
                "fetch_day_data: /api/timeline/activities returned no JSON payload ({}). "
                "See GET warnings above (non-200, timeout, or connection error).",
            )

        act = data.get("activities") or {}
        activities = act.get("activities") if isinstance(act, dict) else []
        if isinstance(activities, list):
            apps = sorted({(a.get("app") or "?") for a in activities})
            logger.debug(
                "fetch_day_data summary: %d activity sessions; apps=%s",
                len(activities),
                apps,
            )
            for i, a in enumerate(activities[:50]):
                logger.debug(
                    "  session[%d] %s | %s | %s",
                    i,
                    a.get("app"),
                    (a.get("label") or "")[:80],
                    a.get("start", "")[:19],
                )
            if len(activities) > 50:
                logger.debug("  ... %d more sessions omitted", len(activities) - 50)
        return data
