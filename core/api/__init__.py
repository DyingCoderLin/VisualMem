"""
Data Platform API — the highest-level abstraction layer for VisualMem.

Provides structured data query endpoints (timeline, OCR, focus, reports)
that sit on top of the raw storage layer.  Mounted as a FastAPI router
inside gui_backend_server.py.
"""
