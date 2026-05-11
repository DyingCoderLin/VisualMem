"""Activity clustering and semantic labeling."""

__all__ = [
    "ClusterManager",
    "PendingEntry",
    "build_region_layout_text",
    "call_vlm",
    "fuzzy_match_label",
    "is_cluster_vlm_endpoint_configured",
    "log_cluster_labeling_event",
    "parse_label",
    "resolve_cluster_chat_completions_url",
]


def __getattr__(name: str):
    if name in {"ClusterManager", "PendingEntry"}:
        from .cluster_manager import ClusterManager, PendingEntry

        return {"ClusterManager": ClusterManager, "PendingEntry": PendingEntry}[name]
    if name in {
        "build_region_layout_text",
        "call_vlm",
        "fuzzy_match_label",
        "is_cluster_vlm_endpoint_configured",
        "log_cluster_labeling_event",
        "parse_label",
        "resolve_cluster_chat_completions_url",
    }:
        from . import vlm_labeler

        return getattr(vlm_labeler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
