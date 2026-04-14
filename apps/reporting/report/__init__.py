"""
Daily report — top-level application (外挂 agent), not part of the FastAPI backend.

Run via ``python scripts/daily_report.py`` (see script docstring).  The pipeline
uses :class:`apps.reporting.report.data_fetcher.DataFetcher` to call the data platform
(``REPORT_DATA_API_BASE``, default ``http://localhost:18080``) over HTTP only.
"""
