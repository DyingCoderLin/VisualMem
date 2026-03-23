import json
import os
from typing import Dict, List, Set

from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)


class AppNameManager:
    """
    Manages persistence of application names and their window names seen during capture.
    Stored in a JSON file.
    """

    def __init__(self, storage_path: str | None = None):
        if storage_path is None:
            # Use STORAGE_ROOT from config if available
            storage_root = getattr(config, "STORAGE_ROOT", "./visualmem_storage")
            storage_path = os.path.join(storage_root, "app_names.json")

        self.storage_path = storage_path

        # In‑memory structures
        self.app_names: Set[str] = set()
        # Mapping: app_name -> set(window_name)
        self.app_windows: Dict[str, Set[str]] = {}

        self._load_from_disk()
        logger.info(
            f"AppNameManager initialized with {len(self.app_names)} apps from {storage_path}"
        )

    def _load_from_disk(self) -> None:
        """
        Load app and window names from JSON file.

        Backward compatibility:
        - Old format: ["WeChat", "Chrome", ...]
        - New format: { "WeChat": ["聊天窗口", "朋友圈"], "Chrome": ["窗口1"] }
        """
        if not os.path.exists(self.storage_path):
            self.app_names = set()
            self.app_windows = {}
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Old format: simple list of app names
            if isinstance(data, list):
                self.app_names = set(data)
                self.app_windows = {}
                return

            # New format: dict[app_name] -> list[window_name]
            if isinstance(data, dict):
                apps: Set[str] = set()
                app_windows: Dict[str, Set[str]] = {}

                for app, windows in data.items():
                    if not app:
                        continue
                    apps.add(app)
                    if isinstance(windows, list):
                        # Deduplicate while preserving as set in memory
                        app_windows[app] = {w for w in windows if w}
                    else:
                        # If value is not a list, treat as "no windows recorded yet"
                        app_windows[app] = set()

                self.app_names = apps
                self.app_windows = app_windows
                return

        except Exception as e:
            logger.error(f"Failed to load app names from {self.storage_path}: {e}")

        # Fallback to empty on error
        self.app_names = set()
        self.app_windows = {}

    def _save_to_disk(self) -> None:
        """
        Save apps and window names to JSON file in the new structure:

        {
          "微信": ["聊天窗口", "朋友圈", "..."],
          "WeChat": ["Chat with Alice", "Search"],
          ...
        }
        """
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

            # Ensure every app has an entry, even if it has no windows yet
            data: Dict[str, List[str]] = {}
            for app in sorted(self.app_names):
                windows = sorted(self.app_windows.get(app, set()))
                data[app] = windows

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save app names to {self.storage_path}: {e}")

    def add_apps(self, apps: List[str]) -> None:
        """Add new apps and persist if any new ones are found"""
        new_apps = [app for app in apps if app and app not in self.app_names]
        if not new_apps:
            return

        for app in new_apps:
            self.app_names.add(app)
            # Ensure an empty window set exists for the app
            self.app_windows.setdefault(app, set())

        self._save_to_disk()
        logger.info(f"Added new apps: {new_apps}")

    def add_windows_for_app(self, app_name: str, window_names: List[str]) -> None:
        """
        Add window names for a single app and persist if new ones are found.
        """
        if not app_name or not window_names:
            return

        # Ensure app is tracked
        if app_name not in self.app_names:
            self.app_names.add(app_name)

        window_set = self.app_windows.setdefault(app_name, set())
        new_windows = [w for w in window_names if w and w not in window_set]
        if not new_windows:
            return

        window_set.update(new_windows)
        self._save_to_disk()
        logger.info(f"Added new windows for app '{app_name}': {new_windows}")

    def add_window_pairs(self, app_window_pairs: List[tuple[str, str]]) -> None:
        """
        Batch add (app_name, window_name) pairs.
        This is more efficient than calling add_windows_for_app repeatedly.
        """
        changed = False
        for app_name, window_name in app_window_pairs:
            if not app_name or not window_name:
                continue

            if app_name not in self.app_names:
                self.app_names.add(app_name)

            window_set = self.app_windows.setdefault(app_name, set())
            if window_name not in window_set:
                window_set.add(window_name)
                changed = True

        if changed:
            self._save_to_disk()

    def get_app_list(self) -> List[str]:
        """Return sorted list of app names"""
        return sorted(self.app_names)

    def get_app_window_map(self) -> Dict[str, List[str]]:
        """
        Return mapping: app_name -> sorted list of window_names.
        """
        return {app: sorted(self.app_windows.get(app, set())) for app in self.app_names}


# Global instance
app_name_manager = AppNameManager()
