"""Pydantic schemas for the report generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# LLM call tracking
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    """Result of a single LLM API call with usage tracking."""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Map output
# ---------------------------------------------------------------------------

@dataclass
class MapTask:
    """A single extracted task from a Map chunk."""
    description: str
    app: str = ""
    evidence: str = ""


@dataclass
class MapChunkResult:
    """Structured output from a single Map LLM call."""
    time_range: str = ""
    primary_apps: List[str] = field(default_factory=list)
    tasks: List[MapTask] = field(default_factory=list)
    key_artifacts: List[str] = field(default_factory=list)
    confidence: str = "high"
    ambiguous_points: List[str] = field(default_factory=list)
    supplementary_context: str = ""


# ---------------------------------------------------------------------------
# Chunk definition
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A group of activity sessions to be processed by one Map call."""
    sessions: List[Dict[str, Any]]
    start: str = ""
    end: str = ""
    estimated_tokens: int = 0
    ocr_texts: List[Dict[str, Any]] = field(default_factory=list)
    window_titles: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------

@dataclass
class WorkModule:
    core_accomplishments: List[str] = field(default_factory=list)
    supporting_research: List[str] = field(default_factory=list)
    blockers_and_unfinished: List[str] = field(default_factory=list)
    tomorrow_suggestions: List[str] = field(default_factory=list)


@dataclass
class LifeModule:
    focus_score: int = 0
    focus_interpretation: str = ""
    deep_work_blocks: List[Dict[str, Any]] = field(default_factory=list)
    fragmentation_diagnosis: str = ""
    distraction_patterns: str = ""
    intervention_suggestions: List[str] = field(default_factory=list)


@dataclass
class DailyReport:
    date: str
    work_module: WorkModule = field(default_factory=WorkModule)
    life_module: LifeModule = field(default_factory=LifeModule)
    # Closing notes for the day: unexpected angles + self-examination (may use prior JSON in logs/)
    today_summary: List[str] = field(default_factory=list)
    # [{ "app": str, "purpose_keywords": [str, ...] }] from Reduce LLM
    app_purpose: List[Dict[str, Any]] = field(default_factory=list)
    raw_markdown: str = ""


# ---------------------------------------------------------------------------
# Pipeline stats
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    data_fetch_ms: int = 0
    chunking_ms: int = 0
    map_calls: int = 0
    map_total_input_tokens: int = 0
    map_total_output_tokens: int = 0
    map_model: str = ""
    map_latency_ms: int = 0
    pre_enrichment_ms: int = 0
    pre_enrichment_chunks_enriched: int = 0
    reduce_input_tokens: int = 0
    reduce_output_tokens: int = 0
    reduce_model: str = ""
    reduce_latency_ms: int = 0
    total_pipeline_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
