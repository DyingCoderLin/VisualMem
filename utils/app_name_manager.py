import json
import os
from typing import List, Set
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)

class AppNameManager:
    """
    Manages persistence of application names seen during capture.
    Stored in a JSON file.
    """
    def __init__(self, storage_path: str = None):
        if storage_path is None:
            # Use STORAGE_ROOT from config if available
            storage_root = getattr(config, 'STORAGE_ROOT', './visualmem_storage')
            storage_path = os.path.join(storage_root, "app_names.json")
        
        self.storage_path = storage_path
        self.app_names: Set[str] = self._load_apps()
        logger.info(f"AppNameManager initialized with {len(self.app_names)} apps from {storage_path}")

    def _load_apps(self) -> Set[str]:
        """Load app names from JSON file"""
        if not os.path.exists(self.storage_path):
            return set()
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return set(data)
        except Exception as e:
            logger.error(f"Failed to load app names from {self.storage_path}: {e}")
        return set()

    def add_apps(self, apps: List[str]):
        """Add new apps and persist if any new ones are found"""
        new_apps = [app for app in apps if app and app not in self.app_names]
        if new_apps:
            self.app_names.update(new_apps)
            self._save_apps()
            logger.info(f"Added new apps: {new_apps}")

    def _save_apps(self):
        """Save app names to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(sorted(list(self.app_names)), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save app names to {self.storage_path}: {e}")

    def get_app_list(self) -> List[str]:
        """Return sorted list of app names"""
        return sorted(list(self.app_names))

# Global instance
app_name_manager = AppNameManager()
