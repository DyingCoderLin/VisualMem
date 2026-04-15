"""
Daily report — top-level application (外挂 agent), not part of the FastAPI backend.

Run via ``python scripts/daily_report.py`` (see script docstring).  The pipeline
uses :class:`apps.reporting.report.data_fetcher.DataFetcher` to call the data platform
(``config.REPORT_DATA_API_BASE``: local backend, or ``GUI_REMOTE_BACKEND_URL`` when
``GUI_MODE=remote``, optional ``REPORT_DATA_API_BASE`` override) over HTTP only.
"""
