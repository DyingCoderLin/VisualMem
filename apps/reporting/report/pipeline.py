"""
Daily report generation pipeline — deterministic Map-Reduce.

Flow: fetch data → chunk → parallel Map → pre-enrich → single Reduce → format.
No Agent loops, no tool calling. Token consumption is fully predictable.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config import config
from apps.reporting.report.app_usage import (
    build_app_usage_minutes_only,
    format_app_usage_hints_for_reduce,
    merge_app_usage_with_llm_purpose,
)
from apps.reporting.report.chunker import build_chunks, estimate_tokens
from apps.reporting.report.data_fetcher import DataFetcher
from apps.reporting.report.llm_caller import LLMCaller
from apps.reporting.report.metrics_semantics import METRICS_SEMANTICS_FOR_JSON
from apps.reporting.report.models import (
    Chunk,
    DailyReport,
    LifeModule,
    LLMResult,
    MapChunkResult,
    MapTask,
    PipelineStats,
    WorkModule,
)
from apps.reporting.report.history_loader import load_prior_reports_for_reduce
from apps.reporting.report.prompts import (
    MAP_SYSTEM_PROMPT,
    MAP_USER_TEMPLATE,
    REDUCE_SYSTEM_PROMPT,
    REDUCE_USER_TEMPLATE,
    reduce_language_addon,
)
from utils.logger import setup_logger

logger = setup_logger("report.pipeline")


class ReportPipeline:
    """Orchestrates the full daily report generation pipeline."""

    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        llm_caller: Optional[LLMCaller] = None,
    ):
        self.fetcher = data_fetcher or DataFetcher(config.REPORT_DATA_API_BASE)
        self.llm = llm_caller or LLMCaller(
            api_key=config.REPORT_API_KEY,
            api_base=config.REPORT_API_BASE,
        )
        self.stats = PipelineStats()

    async def generate(
        self,
        date: str,
        daily_goal: str = "",
        language: str = "auto",
    ) -> Dict[str, Any]:
        """Run the full pipeline and return report + stats.

        Args:
            date: ISO date string, e.g. "2026-04-11"
            daily_goal: optional user goal for the day
            language: "zh", "en", or "auto"

        Returns:
            {"report": DailyReport, "pipeline_stats": PipelineStats}
        """
        pipeline_t0 = time.monotonic()

        start = f"{date}T00:00:00"
        end = f"{date}T23:59:59"

        logger.debug(
            "Pipeline start date=%s range=[%s, %s] — purpose: pull day data from "
            "data platform (HTTP), chunk by activity sessions, Map LLM per chunk, "
            "optional pre-enrich weak chunks, Reduce LLM once.",
            date,
            start,
            end,
        )

        # Stage 1: Data Collection
        t0 = time.monotonic()
        day_data = await self.fetcher.fetch_day_data(start, end)
        self.stats.data_fetch_ms = int((time.monotonic() - t0) * 1000)
        logger.info("Stage 1 (data fetch): %dms", self.stats.data_fetch_ms)

        activities_list = day_data.get("activities", {}).get("activities", [])
        if not activities_list:
            logger.warning(
                "No activities for %s — timeline/activities returned empty "
                "(no committed activity_sessions for this local day, or clustering disabled).",
                date,
            )
            return self._empty_result(
                date,
                pipeline_t0,
                reason="no_activity_sessions",
                message=(
                    "该日期没有活动会话数据：activity_sessions 为空或未覆盖此日。"
                    "请换一天试（例如用 GET /api/date-range 看有记录的区间），"
                    "或确认已开启活动聚类并产生会话。"
                ),
            )

        logger.debug(
            "Stage 1 result: %d activity_sessions — next Stage 2 groups them into "
            "chunks (token_limit=%s gap_min=%s min)",
            len(activities_list),
            config.REPORT_CHUNK_TOKEN_LIMIT,
            config.REPORT_GAP_MINUTES,
        )

        # Stage 2: Dynamic Chunking
        t0 = time.monotonic()
        chunks = build_chunks(
            activities_list,
            token_limit=config.REPORT_CHUNK_TOKEN_LIMIT,
            gap_minutes_threshold=config.REPORT_GAP_MINUTES,
        )
        # Enrich chunks with OCR texts and window titles
        await self._enrich_chunks_with_ocr(chunks)
        self.stats.chunking_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Stage 2 (chunking): %d chunks in %dms",
            len(chunks),
            self.stats.chunking_ms,
        )
        logger.debug(
            "Stage 2: per chunk, GET /api/ocr/texts + /api/windows/titles for "
            "[chunk.start .. chunk.end] to build Map prompt context",
        )

        # Stage 3: Parallel Map
        logger.info(
            "Stage 3 (Map): calling remote LLM once per chunk — latency dominated by model + network"
        )
        t0 = time.monotonic()
        map_results = await self._run_map_phase(chunks)
        self.stats.map_latency_ms = int((time.monotonic() - t0) * 1000)
        self.stats.map_calls = len(map_results)
        self.stats.map_model = config.REPORT_MAP_MODEL
        logger.info(
            "Stage 3 (map): %d calls, in=%d out=%d, %dms",
            len(map_results),
            self.stats.map_total_input_tokens,
            self.stats.map_total_output_tokens,
            self.stats.map_latency_ms,
        )

        # Stage 4: Pre-Enrichment
        t0 = time.monotonic()
        enriched_count = await self._pre_enrich(chunks, map_results)
        self.stats.pre_enrichment_ms = int((time.monotonic() - t0) * 1000)
        self.stats.pre_enrichment_chunks_enriched = enriched_count
        logger.info(
            "Stage 4 (pre-enrich): %d chunks enriched in %dms",
            enriched_count,
            self.stats.pre_enrichment_ms,
        )
        if enriched_count == 0:
            logger.info(
                "Stage 4 skipped extra OCR: all Map chunks had confidence=high and "
                "empty ambiguous_points"
            )

        # Stage 5: Single-Pass Reduce
        logger.debug(
            "Stage 5 (Reduce): single LLM call merging Map summaries + focus_score + "
            "activities_breakdown + keywords into work_module / life_module / today_summary JSON"
        )
        t0 = time.monotonic()
        prior_reports_text = load_prior_reports_for_reduce(
            date,
            config.REPORT_LOG_DIR,
            config.REPORT_HISTORY_DAYS,
        )
        logger.debug(
            "Reduce prior_reports: %d chars from dir=%s (max_days=%s)",
            len(prior_reports_text),
            config.REPORT_LOG_DIR,
            config.REPORT_HISTORY_DAYS,
        )
        if not prior_reports_text.strip():
            prior_reports_text = (
                "(No prior daily_report JSON files found in REPORT_LOG_DIR for dates before "
                f"{date} — you may still write today_summary from today alone; use [] only if "
                "there is nothing worth saying.)"
            )
        report = await self._run_reduce_phase(
            map_results, chunks, day_data, daily_goal, prior_reports_text,
            language,
        )
        base_rows = build_app_usage_minutes_only(day_data)
        app_usage_summary = merge_app_usage_with_llm_purpose(
            base_rows, report.app_purpose,
        )
        self.stats.reduce_latency_ms = int((time.monotonic() - t0) * 1000)
        self.stats.reduce_model = config.REPORT_REDUCE_MODEL
        logger.info(
            "Stage 5 (reduce): in=%d out=%d, %dms",
            self.stats.reduce_input_tokens,
            self.stats.reduce_output_tokens,
            self.stats.reduce_latency_ms,
        )

        # Finalize stats
        self.stats.total_pipeline_ms = int((time.monotonic() - pipeline_t0) * 1000)
        self.stats.total_input_tokens = (
            self.stats.map_total_input_tokens + self.stats.reduce_input_tokens
        )
        self.stats.total_output_tokens = (
            self.stats.map_total_output_tokens + self.stats.reduce_output_tokens
        )

        logger.info(
            "daily_report LLM tokens total: input=%d output=%d "
            "(map: %d+%d, reduce: %d+%d) | pipeline_ms=%d",
            self.stats.total_input_tokens,
            self.stats.total_output_tokens,
            self.stats.map_total_input_tokens,
            self.stats.map_total_output_tokens,
            self.stats.reduce_input_tokens,
            self.stats.reduce_output_tokens,
            self.stats.total_pipeline_ms,
        )

        report.date = date
        # Never log app_usage_summary / report body at INFO — only token/stage lines above.
        return {
            "app_usage_summary": app_usage_summary,
            "metrics_semantics": METRICS_SEMANTICS_FOR_JSON,
            "report": _report_to_dict(report),
            "pipeline_stats": _stats_to_dict(self.stats),
        }

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    async def _enrich_chunks_with_ocr(self, chunks: List[Chunk]):
        """Fetch OCR texts and window titles for each chunk's time range."""
        tasks = []
        for chunk in chunks:
            tasks.append(self._fetch_chunk_context(chunk))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_chunk_context(self, chunk: Chunk):
        """Populate a chunk's ocr_texts and window_titles."""
        try:
            ocr_resp, win_resp = await asyncio.gather(
                self.fetcher.get_ocr_texts(chunk.start, chunk.end, limit=50),
                self.fetcher.get_window_titles(chunk.start, chunk.end, limit=30),
            )
            chunk.ocr_texts = ocr_resp.get("items", [])
            chunk.window_titles = win_resp.get("window_titles", [])

            text_total = sum(len(item.get("text", "")) for item in chunk.ocr_texts)
            chunk.estimated_tokens = estimate_tokens("x" * text_total)
            logger.debug(
                "chunk context: ocr_items=%d window_title_rows=%d est_tokens~%d",
                len(chunk.ocr_texts),
                len(chunk.window_titles),
                chunk.estimated_tokens,
            )
        except Exception as e:
            logger.warning(f"Failed to fetch context for chunk {chunk.start}-{chunk.end}: {e}")

    async def _run_map_phase(self, chunks: List[Chunk]) -> List[MapChunkResult]:
        """Run Map LLM calls in parallel for all chunks."""
        tasks = [self._map_single(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        map_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Map chunk {i} failed: {result}")
                map_results.append(MapChunkResult(
                    time_range=f"{chunks[i].start} - {chunks[i].end}",
                    confidence="low",
                    ambiguous_points=[f"Map call failed: {result}"],
                ))
            else:
                map_results.append(result)
        return map_results

    async def _map_single(self, chunk: Chunk) -> MapChunkResult:
        """Process a single chunk through the Map LLM."""
        activities_text = self._format_activities(chunk.sessions)
        ocr_text = self._format_ocr_texts(chunk.ocr_texts)
        window_text = self._format_window_titles(chunk.window_titles)

        user_prompt = MAP_USER_TEMPLATE.format(
            start=chunk.start,
            end=chunk.end,
            activities=activities_text or "(no activities)",
            ocr_texts=ocr_text or "(no OCR text available)",
            window_titles=window_text or "(no window titles)",
        )

        logger.debug(
            "Map chunk [%s .. %s]: purpose=fact extraction JSON; user_prompt_chars=%d",
            chunk.start[:19] if chunk.start else "",
            chunk.end[:19] if chunk.end else "",
            len(user_prompt),
        )

        result = await self.llm.call(
            model=config.REPORT_MAP_MODEL,
            system_prompt=MAP_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        self.stats.map_total_input_tokens += result.input_tokens
        self.stats.map_total_output_tokens += result.output_tokens

        parsed = self._parse_map_result(result.content, chunk)
        logger.debug(
            "Map chunk parsed: confidence=%s tasks=%d ambiguous=%s",
            parsed.confidence,
            len(parsed.tasks),
            bool(parsed.ambiguous_points),
        )
        return parsed

    async def _pre_enrich(
        self, chunks: List[Chunk], map_results: List[MapChunkResult],
    ) -> int:
        """Scan Map outputs for low-confidence chunks and fetch supplementary data."""
        enriched = 0
        for i, mr in enumerate(map_results):
            need = mr.confidence == "low" or bool(mr.ambiguous_points)
            logger.debug(
                "pre_enrich check chunk[%d]: confidence=%s ambiguous_points=%s -> %s",
                i,
                mr.confidence,
                mr.ambiguous_points,
                "fetch extra OCR" if need else "skip",
            )
            if mr.confidence == "low" or mr.ambiguous_points:
                chunk = chunks[i]
                try:
                    logger.debug(
                        "pre_enrich: GET /api/ocr/texts limit=100 for [%s .. %s]",
                        chunk.start,
                        chunk.end,
                    )
                    ocr_resp = await self.fetcher.get_ocr_texts(
                        chunk.start, chunk.end, limit=100,
                    )
                    extra_items = ocr_resp.get("items", [])
                    extra_text = "\n".join(
                        f"[{item.get('app_name', '?')} {item.get('timestamp', '')}] "
                        f"{item.get('text', '')[:300]}"
                        for item in extra_items
                    )
                    mr.supplementary_context = extra_text
                    enriched += 1
                    logger.info(
                        "Pre-enriched chunk %d (%s-%s): +%d OCR records",
                        i,
                        chunk.start,
                        chunk.end,
                        len(extra_items),
                    )
                except Exception as e:
                    logger.warning(f"Pre-enrichment failed for chunk {i}: {e}")
        return enriched

    async def _run_reduce_phase(
        self,
        map_results: List[MapChunkResult],
        chunks: List[Chunk],
        day_data: Dict[str, Any],
        daily_goal: str,
        prior_reports_text: str,
        language: str,
    ) -> DailyReport:
        """Single-pass Reduce: all summaries → one LLM call → structured report."""
        app_hints = format_app_usage_hints_for_reduce(day_data)
        map_summaries = self._format_map_summaries(map_results)
        focus_data = day_data.get("focus_score", {})
        focus_text = json.dumps(focus_data, indent=2, ensure_ascii=False)

        breakdown = day_data.get("activities_breakdown", {})
        breakdown_text = json.dumps(
            breakdown.get("breakdown", []), indent=2, ensure_ascii=False,
        )

        keywords_data = day_data.get("keywords", {})
        keywords_list = keywords_data.get("keywords", [])
        keywords_text = ", ".join(
            f"{kw['word']}({kw['count']})" for kw in keywords_list[:20]
        )

        user_prompt = REDUCE_USER_TEMPLATE.format(
            metrics_semantics=METRICS_SEMANTICS_FOR_JSON,
            app_usage_hints=app_hints,
            daily_goal=daily_goal or "(none provided)",
            map_summaries=map_summaries,
            focus_metrics=focus_text,
            activity_breakdown=breakdown_text,
            keywords=keywords_text or "(no keywords)",
            prior_reports=prior_reports_text,
        )

        system_prompt = REDUCE_SYSTEM_PROMPT + "\n\n" + reduce_language_addon(language)

        result = await self.llm.call(
            model=config.REPORT_REDUCE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        self.stats.reduce_input_tokens = result.input_tokens
        self.stats.reduce_output_tokens = result.output_tokens

        return self._parse_reduce_result(result.content)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_activities(self, sessions: List[Dict]) -> str:
        lines = []
        for s in sessions:
            dur = s.get("duration_seconds", 0)
            dur_min = round(dur / 60, 1) if dur else "?"
            lines.append(
                f"- [{s.get('start', '?')} ~ {s.get('end', '?')}] "
                f"{s.get('app', '?')}: {s.get('label', '?')} ({dur_min} min)"
            )
        return "\n".join(lines)

    def _format_ocr_texts(self, ocr_items: List[Dict], max_chars: int = 4000) -> str:
        seen = set()
        lines = []
        total = 0
        for item in ocr_items:
            text = (item.get("text") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            app = item.get("app_name", "?")
            snippet = text[:200]
            line = f"[{app}] {snippet}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    def _format_window_titles(self, titles: List[Dict]) -> str:
        lines = []
        for t in titles:
            lines.append(f"- {t.get('app', '?')}: {t.get('window', '?')} (x{t.get('count', 0)})")
        return "\n".join(lines)

    def _format_map_summaries(self, map_results: List[MapChunkResult]) -> str:
        blocks = []
        for i, mr in enumerate(map_results):
            block = f"### Block {i+1}: {mr.time_range}\n"
            block += f"Apps: {', '.join(mr.primary_apps)}\n"
            for task in mr.tasks:
                block += f"- {task.description} ({task.app})"
                if task.evidence:
                    block += f" [evidence: {task.evidence}]"
                block += "\n"
            if mr.key_artifacts:
                block += f"Artifacts: {', '.join(mr.key_artifacts)}\n"
            if mr.ambiguous_points:
                block += f"⚠ Unclear: {'; '.join(mr.ambiguous_points)}\n"
            if mr.supplementary_context:
                block += f"Supplementary OCR:\n{mr.supplementary_context[:1000]}\n"
            blocks.append(block)
        return "\n".join(blocks)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_map_result(self, content: str, chunk: Chunk) -> MapChunkResult:
        """Parse Map LLM JSON output into MapChunkResult."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Map output is not valid JSON, treating as low-confidence")
            return MapChunkResult(
                time_range=f"{chunk.start} - {chunk.end}",
                confidence="low",
                ambiguous_points=["LLM output was not valid JSON"],
            )

        tasks = []
        for t in data.get("tasks", []):
            tasks.append(MapTask(
                description=t.get("description", ""),
                app=t.get("app", ""),
                evidence=t.get("evidence", ""),
            ))

        return MapChunkResult(
            time_range=data.get("time_range", f"{chunk.start} - {chunk.end}"),
            primary_apps=data.get("primary_apps", []),
            tasks=tasks,
            key_artifacts=data.get("key_artifacts", []),
            confidence=data.get("confidence", "high"),
            ambiguous_points=data.get("ambiguous_points", []),
        )

    def _parse_reduce_result(self, content: str) -> DailyReport:
        """Parse Reduce LLM JSON output into DailyReport."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Reduce output is not valid JSON, returning raw markdown")
            return DailyReport(date="", raw_markdown=content)

        wm = data.get("work_module", {})
        lm = data.get("life_module", {})

        work = WorkModule(
            core_accomplishments=wm.get("core_accomplishments", []),
            supporting_research=wm.get("supporting_research", []),
            blockers_and_unfinished=wm.get("blockers_and_unfinished", []),
            tomorrow_suggestions=wm.get("tomorrow_suggestions", []),
        )

        life = LifeModule(
            focus_score=lm.get("focus_score", 0),
            focus_interpretation=lm.get("focus_interpretation", ""),
            deep_work_blocks=lm.get("deep_work_blocks", []),
            fragmentation_diagnosis=lm.get("fragmentation_diagnosis", ""),
            distraction_patterns=lm.get("distraction_patterns", ""),
            intervention_suggestions=lm.get("intervention_suggestions", []),
        )

        raw_ts = data.get("today_summary")
        if raw_ts is None:
            raw_ts = data.get("cross_day_reflection", [])
        if not isinstance(raw_ts, list):
            raw_ts = []
        today_summary = [str(x).strip() for x in raw_ts if str(x).strip()]

        ap = data.get("app_purpose", [])
        if not isinstance(ap, list):
            ap = []

        return DailyReport(
            date="",
            work_module=work,
            life_module=life,
            today_summary=today_summary,
            app_purpose=ap,
        )

    def _empty_result(
        self,
        date: str,
        pipeline_t0: float,
        reason: str = "no_activity_sessions",
        message: str = "",
    ) -> Dict[str, Any]:
        self.stats.total_pipeline_ms = int((time.monotonic() - pipeline_t0) * 1000)
        return {
            "status": reason,
            "message": message,
            "app_usage_summary": [],
            "metrics_semantics": METRICS_SEMANTICS_FOR_JSON,
            "report": _report_to_dict(DailyReport(date=date)),
            "pipeline_stats": _stats_to_dict(self.stats),
        }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _report_to_dict(report: DailyReport) -> Dict[str, Any]:
    return {
        "date": report.date,
        "work_module": {
            "core_accomplishments": report.work_module.core_accomplishments,
            "supporting_research": report.work_module.supporting_research,
            "blockers_and_unfinished": report.work_module.blockers_and_unfinished,
            "tomorrow_suggestions": report.work_module.tomorrow_suggestions,
        },
        "life_module": {
            "focus_score": report.life_module.focus_score,
            "focus_interpretation": report.life_module.focus_interpretation,
            "deep_work_blocks": report.life_module.deep_work_blocks,
            "fragmentation_diagnosis": report.life_module.fragmentation_diagnosis,
            "distraction_patterns": report.life_module.distraction_patterns,
            "intervention_suggestions": report.life_module.intervention_suggestions,
        },
        "today_summary": report.today_summary,
        "raw_markdown": report.raw_markdown,
    }


def _stats_to_dict(stats: PipelineStats) -> Dict[str, Any]:
    return {
        "data_fetch_ms": stats.data_fetch_ms,
        "chunking_ms": stats.chunking_ms,
        "map_calls": stats.map_calls,
        "map_total_input_tokens": stats.map_total_input_tokens,
        "map_total_output_tokens": stats.map_total_output_tokens,
        "map_model": stats.map_model,
        "map_latency_ms": stats.map_latency_ms,
        "pre_enrichment_ms": stats.pre_enrichment_ms,
        "pre_enrichment_chunks_enriched": stats.pre_enrichment_chunks_enriched,
        "reduce_input_tokens": stats.reduce_input_tokens,
        "reduce_output_tokens": stats.reduce_output_tokens,
        "reduce_model": stats.reduce_model,
        "reduce_latency_ms": stats.reduce_latency_ms,
        "total_pipeline_ms": stats.total_pipeline_ms,
        "total_input_tokens": stats.total_input_tokens,
        "total_output_tokens": stats.total_output_tokens,
        "llm_tokens": {
            "input_total": stats.total_input_tokens,
            "output_total": stats.total_output_tokens,
            "map_input": stats.map_total_input_tokens,
            "map_output": stats.map_total_output_tokens,
            "reduce_input": stats.reduce_input_tokens,
            "reduce_output": stats.reduce_output_tokens,
        },
    }
