"""VLM understanding backends."""

__all__ = ["AbstractVLM", "ApiVLM", "LocalVLM"]


def __getattr__(name: str):
    if name == "AbstractVLM":
        from .base_vlm import AbstractVLM

        return AbstractVLM
    if name == "ApiVLM":
        from .api_vlm import ApiVLM

        return ApiVLM
    if name == "LocalVLM":
        from .local_vlm import LocalVLM

        return LocalVLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

