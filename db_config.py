import json
import os
from pathlib import Path

CONFIG_FILE = "db_config.json"

DEFAULT_CONFIG = {
    "databases": {
        "Default": "./lancedb"
    },
    "active_db": "Default"
}

def load_config():
    """Load configuration from JSON file."""
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to JSON file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def add_database(name: str, path: str):
    """Add a new database path to config."""
    config = load_config()
    config["databases"][name] = str(path)
    # If it's the first one (other than default default), make it active?
    save_config(config)

def remove_database(name: str):
    """Remove a database from config."""
    config = load_config()
    if name in config["databases"]:
        del config["databases"][name]
        # If active was removed, reset active
        if config.get("active_db") == name:
            config["active_db"] = next(iter(config["databases"]), "Default")
        save_config(config)

def get_config():
    """Get the full config object."""
    return load_config()
