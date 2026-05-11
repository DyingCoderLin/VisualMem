"""Long-lived worker threads for offloading heavy per-frame work from the
HTTP request path (e.g. embedding, window OCR, cluster assignment).

Workers here must follow the project's worker lifecycle discipline:
start once at server startup and live until shutdown. Do NOT spawn
per-request worker threads.
"""

from core.worker.frame_enrichment_worker import (
    EnrichmentJob,
    FrameEnrichmentWorker,
)

__all__ = ["EnrichmentJob", "FrameEnrichmentWorker"]
