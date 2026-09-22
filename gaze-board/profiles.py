"""Local per-user profile storage - the app's "backend".

There's no server or network involved: Gaze Board runs on one device
in front of one person at a time, so "backend" here just means a
proper data layer instead of ad-hoc files. Each signed-in user gets
their own JSON file under profiles/ holding their settings,
calibration baseline, favorite phrases, and a history of past
sessions' adaptive-timing stats.
"""

import json
import os
import re

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")

DEFAULT_SETTINGS = {
    "layout": "optimized", "adaptive": True, "dwell_time": 700, "manual_dwell": 700,
    "prediction_on": True, "highlight_strength": 1.0, "key_scale": 1.0,
    "show_gaze_dot": True, "typing_mode": True, "blink_mode": False,
}


def _slug(display_name):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", display_name.strip().lower()).strip("_")
    return slug or "user"


def _path_for(display_name):
    return os.path.join(PROFILES_DIR, _slug(display_name) + ".json")


def list_profiles():
    """Display names of every saved profile, alphabetically."""
    if not os.path.isdir(PROFILES_DIR):
        return []
    names = []
    for fname in sorted(os.listdir(PROFILES_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(PROFILES_DIR, fname)) as f:
                data = json.load(f)
            names.append(data.get("display_name", fname[:-5]))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(names, key=str.lower)


def profile_exists(display_name):
    return os.path.exists(_path_for(display_name))


def new_profile_data(display_name):
    return {
        "display_name": display_name,
        "settings": DEFAULT_SETTINGS.copy(),
        "calibration": None,
        "favorites": [],
        "adaptive_history": [],
    }


def create_profile(display_name):
    data = new_profile_data(display_name)
    save_profile(display_name, data)
    return data


def load_profile(display_name):
    """Load a profile, creating it if it doesn't exist yet. Missing
    fields (from an older save, or a corrupt file) fall back to defaults
    rather than crashing the app."""
    path = _path_for(display_name)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return create_profile(display_name)

    merged_settings = DEFAULT_SETTINGS.copy()
    merged_settings.update(data.get("settings") or {})
    data["settings"] = merged_settings
    data.setdefault("display_name", display_name)
    data.setdefault("calibration", None)
    data.setdefault("favorites", [])
    data.setdefault("adaptive_history", [])
    return data


def save_profile(display_name, data):
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(_path_for(display_name), "w") as f:
        json.dump(data, f, indent=2)
