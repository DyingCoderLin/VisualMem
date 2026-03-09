#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VisualMem GUI package."""

__all__ = ["VisualMemMainWindow"]


def __getattr__(name):
    # Lazy import to avoid importing GUI dependencies in CLI-only environments.
    if name == "VisualMemMainWindow":
        from .main_window import VisualMemMainWindow
        return VisualMemMainWindow
    raise AttributeError(f"module 'gui' has no attribute '{name}'")

