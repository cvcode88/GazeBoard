"""Gaze Board - a mouse-simulated eye-gaze AAC keyboard.

Python/Pygame port of the browser prototype. The mouse stands in for
gaze until real eye-tracking is wired in - see the note above
update_interaction() for exactly where that would plug in.

Run:  python gaze_board.py
Needs: pygame, english_words, wordfreq  (pyttsx3 optional, for Speak)
"""

import math
import os
import queue
import re
import threading
import time

import pygame

import profiles
from predictor import predict_next_letters, get_word_suggestions

# ---------------------------------------------------------------- window
WINDOW_W, WINDOW_H = 1280, 880
FPS = 60

# ---------------------------------------------------------------- colors
BG = (242, 244, 246)
PANEL = (255, 255, 255)
PANEL_2 = (233, 237, 240)
TEXT = (24, 32, 41)
MUTED = (91, 103, 115)
BORDER = (215, 222, 227)
ACCENT = (37, 109, 130)
ACCENT_STRONG = (24, 77, 92)
FOCUS = (217, 142, 43)
KEY_BG = (255, 255, 255)
KEY_BORDER = (215, 222, 227)

# ---------------------------------------------------------------- layout geometry
TOPBAR_H = 74
OUTPUT_H = 106
SIDEBAR_W = 190
SIDEBAR_PAD = 16

GRID_COLS, GRID_ROWS = 6, 5
CELL_W, CELL_H = 166, 104
CELL_GAP = 8
BASE_KEY_W, BASE_KEY_H, BASE_KEY_FONT = 123, 75, 26

GRID_TOTAL_W = GRID_COLS * CELL_W + (GRID_COLS - 1) * CELL_GAP
GRID_TOTAL_H = GRID_ROWS * CELL_H + (GRID_ROWS - 1) * CELL_GAP
BOARD_AREA_X = SIDEBAR_W
BOARD_AREA_W = WINDOW_W - SIDEBAR_W
GRID_ORIGIN = (BOARD_AREA_X + (BOARD_AREA_W - GRID_TOTAL_W) // 2, TOPBAR_H + OUTPUT_H + 26)

DOCK_GAP = 8
DOCK_H = 76
DOCK_ORIGIN = (GRID_ORIGIN[0], GRID_ORIGIN[1] + GRID_TOTAL_H + 18)
DOCK_TOTAL_W = GRID_TOTAL_W

SETTINGS_PANEL_W = 340
CALIB_TARGETS = [
    (0.5, 0.5, "center"), (0.15, 0.22, "top-left"), (0.85, 0.22, "top-right"),
    (0.15, 0.82, "bottom-left"), (0.85, 0.82, "bottom-right"),
]
CALIB_WORDS = ["cat", "wave", "help"]
CALIB_DWELL = 0.8
CALIB_TARGET_R = 40

# ---------------------------------------------------------------- persistence
# Everything - settings, calibration, favorites, adaptive-timing history -
# lives in the signed-in user's profile (see profiles.py), one JSON file
# per person. `settings` and `profile_data` are both mutated in place
# (never reassigned) so every function below that already does
# settings["x"] keeps working unchanged across a sign-in/sign-out.
DEFAULT_SETTINGS = profiles.DEFAULT_SETTINGS

current_profile = None
profile_data = {"display_name": None, "calibration": None, "favorites": [], "adaptive_history": []}


def save_settings():
    """Persist the whole current profile (settings + calibration +
    favorites + history) - named save_settings for the many call sites
    that only touch settings, but it saves everything together since
    it's all one file per user."""
    if not current_profile:
        return
    profile_data["settings"] = settings
    profile_data["display_name"] = current_profile
    try:
        profiles.save_profile(current_profile, profile_data)
    except OSError:
        pass


def sign_in(display_name):
    global current_profile, profile_data, buffer, has_calibration
    profile_data = profiles.load_profile(display_name)
    current_profile = display_name
    settings.clear()
    settings.update(profile_data["settings"])
    has_calibration = bool(profile_data.get("calibration"))
    buffer = ""
    build_keyboard()
    build_dock()
    build_suggestions()


def record_session_metrics():
    if len(key_log) < 3:
        return  # nothing meaningful typed this session
    profile_data.setdefault("adaptive_history", []).append({
        "ended_at": time.strftime("%Y-%m-%d %H:%M"),
        "final_dwell_ms": settings["dwell_time"],
        "wpm": round(current_wpm(), 1),
        "backspace_rate": round(backspace_rate(), 3),
        "chars_typed": len(buffer),
    })
    profile_data["adaptive_history"] = profile_data["adaptive_history"][-50:]


def sign_out():
    global current_profile, buffer
    if current_profile:
        record_session_metrics()
        save_settings()
    current_profile = None
    buffer = ""


def add_favorite(phrase):
    phrase = phrase.strip()
    favorites = profile_data.setdefault("favorites", [])
    if phrase and phrase not in favorites:
        favorites.append(phrase)
        save_settings()
        build_suggestions()


settings = DEFAULT_SETTINGS.copy()
has_calibration = False

# ---------------------------------------------------------------- text-to-speech
_speech_queue = queue.Queue()


def _speech_worker():
    engine = None
    try:
        import pyttsx3
        engine = pyttsx3.init()
    except Exception as exc:  # pragma: no cover - environment dependent
        print("Text-to-speech unavailable:", exc)
    while True:
        text = _speech_queue.get()
        if engine is None:
            continue
        try:
            engine.stop()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:  # pragma: no cover
            print("Speech error:", exc)


threading.Thread(target=_speech_worker, daemon=True).start()


def speak(text):
    if text.strip():
        _speech_queue.put(text.strip())


# ---------------------------------------------------------------- keyboard layout
LETTER_FREQ = [
    ("E", 12.70), ("T", 9.06), ("A", 8.17), ("O", 7.51), ("I", 6.97), ("N", 6.75), ("S", 6.33),
    ("H", 6.09), ("R", 5.99), ("D", 4.25), ("L", 4.03), ("C", 2.78), ("U", 2.76), ("M", 2.41),
    ("W", 2.36), ("F", 2.23), ("G", 2.02), ("Y", 1.97), ("P", 1.93), ("B", 1.29), ("V", 0.98),
    ("K", 0.77), ("J", 0.15), ("X", 0.15), ("Q", 0.10), ("Z", 0.07),
]
PUNCT_FREQ = [(".", 0.06), (",", 0.05), ("?", 0.04), ("!", 0.03)]
ALL_ITEMS = LETTER_FREQ + PUNCT_FREQ


def _build_cells():
    center_r, center_c = (GRID_ROWS - 1) / 2, (GRID_COLS - 1) / 2
    cells = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cells.append({"r": r, "c": c, "dist": math.hypot(r - center_r, c - center_c)})
    return cells


def _lerp(a, b, t):
    return a + (b - a) * t


def _make_optimized_layout():
    cells = sorted(_build_cells(), key=lambda cell: cell["dist"])
    max_dist = cells[-1]["dist"] or 1
    items = sorted(ALL_ITEMS, key=lambda item: item[1], reverse=True)
    layout = []
    for cell, (ch, _freq) in zip(cells, items):
        t = cell["dist"] / max_dist
        layout.append({"r": cell["r"], "c": cell["c"], "ch": ch, "scale": _lerp(1.22, 0.85, t)})
    return layout


def _make_alphabetical_layout():
    items = sorted(ALL_ITEMS, key=lambda item: (not item[0].isalpha(), item[0]))
    cells = _build_cells()
    return [{"r": cell["r"], "c": cell["c"], "ch": ch, "scale": 1.0} for cell, (ch, _f) in zip(cells, items)]


LAYOUTS = {"optimized": _make_optimized_layout(), "alphabetical": _make_alphabetical_layout()}

# ---------------------------------------------------------------- app state
buffer = ""
key_log = []  # [{"type": "char"|"space"|"backspace", "t": float}]
session_start = time.time()
hovered = None
dragging_slider = None
settings_open = False
calib = None
prev_typing_mode = True

entering_name = False
name_input = ""
signin_profiles = []

key_selectables = []
dock_selectables = []
suggestion_selectables = []


class Selectable:
    """One dwell/blink-selectable button: the keyboard keys, the dock
    buttons (space/backspace/clear/speak) and the suggestion words are
    all just instances of this, driven by the same interaction code."""

    def __init__(self, rect, label, on_select, font_size=22, align="center"):
        self.rect = rect
        self.label = label
        self.on_select = on_select
        self.font_size = font_size
        self.align = align
        self.dwell_start = None
        self.flash_until = 0.0
        self.highlight_alpha = 0.0


def current_word():
    parts = re.split(r"[^a-zA-Z]", buffer)
    return parts[-1] if parts else ""


def current_wpm():
    minutes = (time.time() - session_start) / 60
    return (len(buffer) / 5) / minutes if minutes > 0 else 0.0


def backspace_rate():
    recent = key_log[-20:]
    total = len(recent)
    if not total:
        return 0.0
    backspaces = sum(1 for k in recent if k["type"] == "backspace")
    return backspaces / total


def handle_char_commit(ch):
    global buffer
    if calib and calib["phase"] == "words":
        calib_type_char(ch)
        return
    buffer += ch
    key_log.append({"type": "space" if ch == " " else "char", "t": time.time()})
    after_edit()


def handle_backspace():
    global buffer
    if calib and calib["phase"] == "words":
        calib_backspace()
        return
    if not buffer:
        return
    buffer = buffer[:-1]
    key_log.append({"type": "backspace", "t": time.time()})
    after_edit()


def handle_clear():
    global buffer
    buffer = ""
    after_edit()


def handle_speak():
    speak(buffer)


def handle_save_favorite():
    add_favorite(buffer)


def select_suggestion(word):
    global buffer
    word_len = len(current_word())
    buffer = buffer[: len(buffer) - word_len] + word + " "
    key_log.append({"type": "char", "t": time.time()})
    after_edit()


def select_favorite(phrase):
    global buffer
    buffer += phrase + " "
    key_log.append({"type": "char", "t": time.time()})
    after_edit()


def after_edit():
    update_predictions()
    build_suggestions()
    adaptive_adjust()


def update_predictions():
    for sel in key_selectables:
        sel.highlight_alpha = 0.0
    if not settings["prediction_on"]:
        return
    preds = predict_next_letters(current_word())
    if not preds or preds[0][1] <= 0:
        return
    max_weight = preds[0][1]
    for letter, weight in preds:
        confidence = min(1.0, weight / max_weight)
        alpha = min(0.85, max(0.0, 0.12 + 0.62 * confidence * settings["highlight_strength"]))
        for sel in key_selectables:
            if sel.label.upper() == letter.upper():
                sel.highlight_alpha = alpha


def adaptive_adjust():
    if not settings["adaptive"]:
        return
    recent = key_log[-20:]
    total = len(recent)
    if total < 6:
        return
    rate = sum(1 for k in recent if k["type"] == "backspace") / total
    if rate > 0.15:
        settings["dwell_time"] = min(1400, settings["dwell_time"] + 50)
        save_settings()
    elif rate < 0.05:
        settings["dwell_time"] = max(350, settings["dwell_time"] - 30)
        save_settings()


def reset_stats():
    global key_log, session_start
    key_log = []
    session_start = time.time()


# ---------------------------------------------------------------- building selectables
def build_keyboard():
    global key_selectables
    key_selectables = []
    for k in LAYOUTS[settings["layout"]]:
        w = round(BASE_KEY_W * k["scale"] * settings["key_scale"])
        h = round(BASE_KEY_H * k["scale"] * settings["key_scale"])
        font_size = round(BASE_KEY_FONT * k["scale"] * settings["key_scale"])
        cx = GRID_ORIGIN[0] + k["c"] * (CELL_W + CELL_GAP) + CELL_W / 2
        cy = GRID_ORIGIN[1] + k["r"] * (CELL_H + CELL_GAP) + CELL_H / 2
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (round(cx), round(cy))
        ch = k["ch"]
        key_selectables.append(Selectable(rect, ch, lambda ch=ch: handle_char_commit(ch), font_size))


def build_dock():
    global dock_selectables
    weights = [2.0, 1.0, 1.0, 1.0, 1.0]
    actions = [
        ("space", lambda: handle_char_commit(" ")),
        ("back", handle_backspace),
        ("clear", handle_clear),
        ("speak", handle_speak),
        ("favorite", handle_save_favorite),
    ]
    avail = DOCK_TOTAL_W - DOCK_GAP * (len(actions) - 1)
    wsum = sum(weights)
    dock_selectables = []
    x = DOCK_ORIGIN[0]
    for (label, fn), weight in zip(actions, weights):
        w = avail * weight / wsum
        rect = pygame.Rect(round(x), DOCK_ORIGIN[1], round(w), DOCK_H)
        dock_selectables.append(Selectable(rect, label, fn, font_size=22))
        x += w + DOCK_GAP


def build_suggestions():
    global suggestion_selectables
    word = current_word()
    if word:
        items = [(w, select_suggestion) for w in get_word_suggestions(word)]
    else:
        items = [(p, select_favorite) for p in profile_data.get("favorites", [])[:6]]
    suggestion_selectables = []
    x = SIDEBAR_PAD
    w = SIDEBAR_W - SIDEBAR_PAD * 2
    y = TOPBAR_H + 56
    for label, handler in items:
        rect = pygame.Rect(x, y, w, 46)
        suggestion_selectables.append(Selectable(rect, label, lambda label=label, handler=handler: handler(label), 20, "left"))
        y += 46 + 8


build_keyboard()
build_dock()
build_suggestions()

# ---------------------------------------------------------------- topbar widget rects
TITLE_POS = (20, 16)
TYPING_SWITCH_RECT = pygame.Rect(240, 26, 42, 22)
BLINK_SWITCH_RECT = pygame.Rect(430, 26, 42, 22)
GAZE_DOT_BTN_RECT = pygame.Rect(600, 22, 130, 30)
CALIBRATE_BTN_RECT = pygame.Rect(742, 22, 110, 30)
SETTINGS_BTN_RECT = pygame.Rect(WINDOW_W - 56, 20, 36, 36)
SIGNOUT_BTN_RECT = pygame.Rect(WINDOW_W - 176, 22, 100, 30)


def toggle_typing_mode():
    settings["typing_mode"] = not settings["typing_mode"]
    if settings["typing_mode"]:
        settings["blink_mode"] = False
    save_settings()


def toggle_blink_mode():
    settings["blink_mode"] = not settings["blink_mode"]
    if settings["blink_mode"]:
        settings["typing_mode"] = False
    save_settings()


def toggle_gaze_dot():
    settings["show_gaze_dot"] = not settings["show_gaze_dot"]
    save_settings()


def handle_topbar_click(pos):
    if TYPING_SWITCH_RECT.collidepoint(pos):
        toggle_typing_mode()
        return True
    if BLINK_SWITCH_RECT.collidepoint(pos):
        toggle_blink_mode()
        return True
    if GAZE_DOT_BTN_RECT.collidepoint(pos):
        toggle_gaze_dot()
        return True
    if CALIBRATE_BTN_RECT.collidepoint(pos):
        start_calibration()
        return True
    if SETTINGS_BTN_RECT.collidepoint(pos):
        global settings_open
        settings_open = True
        return True
    if SIGNOUT_BTN_RECT.collidepoint(pos):
        sign_out()
        return True
    return False


# ---------------------------------------------------------------- settings panel widgets
class Switch:
    def __init__(self, rect):
        self.rect = rect

    def draw(self, surf, on):
        color = ACCENT if on else BORDER
        pygame.draw.rect(surf, color, self.rect, border_radius=self.rect.height // 2)
        knob_r = self.rect.height // 2 - 2
        knob_x = self.rect.right - knob_r - 2 if on else self.rect.left + knob_r + 2
        pygame.draw.circle(surf, PANEL, (knob_x, self.rect.centery), knob_r)


class Slider:
    def __init__(self, rect, min_v, max_v, step=1):
        self.rect = rect
        self.min_v = min_v
        self.max_v = max_v
        self.step = step

    def value_from_x(self, x):
        frac = max(0.0, min(1.0, (x - self.rect.left) / self.rect.width))
        raw = self.min_v + frac * (self.max_v - self.min_v)
        return round(raw / self.step) * self.step

    def x_from_value(self, value):
        frac = (value - self.min_v) / (self.max_v - self.min_v)
        return self.rect.left + frac * self.rect.width

    def draw(self, surf, value):
        pygame.draw.rect(surf, BORDER, self.rect, border_radius=4)
        handle_x = self.x_from_value(value)
        filled = pygame.Rect(self.rect.left, self.rect.top, max(0, handle_x - self.rect.left), self.rect.height)
        pygame.draw.rect(surf, ACCENT, filled, border_radius=4)
        pygame.draw.circle(surf, ACCENT_STRONG, (int(handle_x), self.rect.centery), 9)


PANEL_X = WINDOW_W - SETTINGS_PANEL_W
_pad = 24
_w = SETTINGS_PANEL_W - _pad * 2
settings_widgets = {
    "close": pygame.Rect(WINDOW_W - 48, 16, 24, 24),
    "radio_optimized": pygame.Rect(PANEL_X + _pad, 100, _w, 24),
    "radio_alphabetical": pygame.Rect(PANEL_X + _pad, 128, _w, 24),
    "adaptive_switch": Switch(pygame.Rect(PANEL_X + _pad, 178, 42, 22)),
    "dwell_slider": Slider(pygame.Rect(PANEL_X + _pad, 254, _w, 8), 300, 1500, 25),
    "prediction_switch": Switch(pygame.Rect(PANEL_X + _pad, 322, 42, 22)),
    "strength_slider": Slider(pygame.Rect(PANEL_X + _pad, 386, _w, 8), 20, 150, 10),
    "keyscale_slider": Slider(pygame.Rect(PANEL_X + _pad, 450, _w, 8), 80, 150, 5),
    "gazedot_switch": Switch(pygame.Rect(PANEL_X + _pad, 498, 42, 22)),
    "calibrate_btn": pygame.Rect(PANEL_X + _pad, 552, _w, 40),
    "cleartext_btn": pygame.Rect(PANEL_X + _pad, 600, _w, 40),
    "resetstats_btn": pygame.Rect(PANEL_X + _pad, 648, _w, 40),
    "clearfavorites_btn": pygame.Rect(PANEL_X + _pad, 696, _w, 40),
}


def handle_settings_click(pos):
    global settings_open, dragging_slider
    w = settings_widgets
    if w["close"].collidepoint(pos):
        settings_open = False
        return
    if w["radio_optimized"].collidepoint(pos):
        settings["layout"] = "optimized"
        save_settings(); build_keyboard(); update_predictions()
        return
    if w["radio_alphabetical"].collidepoint(pos):
        settings["layout"] = "alphabetical"
        save_settings(); build_keyboard(); update_predictions()
        return
    if w["adaptive_switch"].rect.collidepoint(pos):
        settings["adaptive"] = not settings["adaptive"]
        if not settings["adaptive"]:
            settings["dwell_time"] = settings["manual_dwell"]
        save_settings()
        return
    if w["dwell_slider"].rect.collidepoint(pos) or w["dwell_slider"].rect.inflate(0, 16).collidepoint(pos):
        dragging_slider = "dwell"
        val = w["dwell_slider"].value_from_x(pos[0])
        settings["manual_dwell"] = val; settings["dwell_time"] = val; settings["adaptive"] = False
        save_settings()
        return
    if w["prediction_switch"].rect.collidepoint(pos):
        settings["prediction_on"] = not settings["prediction_on"]
        save_settings(); update_predictions()
        return
    if w["strength_slider"].rect.collidepoint(pos) or w["strength_slider"].rect.inflate(0, 16).collidepoint(pos):
        dragging_slider = "strength"
        settings["highlight_strength"] = w["strength_slider"].value_from_x(pos[0]) / 100
        save_settings(); update_predictions()
        return
    if w["keyscale_slider"].rect.collidepoint(pos) or w["keyscale_slider"].rect.inflate(0, 16).collidepoint(pos):
        dragging_slider = "keyscale"
        settings["key_scale"] = w["keyscale_slider"].value_from_x(pos[0]) / 100
        save_settings(); build_keyboard(); update_predictions()
        return
    if w["gazedot_switch"].rect.collidepoint(pos):
        settings["show_gaze_dot"] = not settings["show_gaze_dot"]
        save_settings()
        return
    if w["calibrate_btn"].collidepoint(pos):
        settings_open = False
        start_calibration()
        return
    if w["cleartext_btn"].collidepoint(pos):
        handle_clear()
        return
    if w["resetstats_btn"].collidepoint(pos):
        reset_stats()
        return
    if w["clearfavorites_btn"].collidepoint(pos):
        profile_data["favorites"] = []
        save_settings()
        build_suggestions()
        return


def handle_slider_drag(x):
    w = settings_widgets
    if dragging_slider == "dwell":
        val = w["dwell_slider"].value_from_x(x)
        settings["manual_dwell"] = val; settings["dwell_time"] = val
    elif dragging_slider == "strength":
        settings["highlight_strength"] = w["strength_slider"].value_from_x(x) / 100
        update_predictions()
    elif dragging_slider == "keyscale":
        settings["key_scale"] = w["keyscale_slider"].value_from_x(x) / 100
        build_keyboard(); update_predictions()


# ---------------------------------------------------------------- calibration
def start_calibration():
    global calib, prev_typing_mode
    prev_typing_mode = settings["typing_mode"]
    settings["typing_mode"] = True
    settings["blink_mode"] = False
    calib = {
        "phase": "intro", "target_index": 0, "target_results": [], "target_shown_at": 0,
        "target_dwell_start": None, "word_index": 0, "word_buffer": "", "word_results": [],
        "word_start": 0, "word_backspaces": 0, "baseline": 700,
    }


def end_calibration(baseline_dwell):
    global calib, has_calibration
    profile_data["calibration"] = {
        "target_results": calib["target_results"],
        "word_results": calib["word_results"],
        "baseline_dwell": baseline_dwell,
    }
    has_calibration = True
    save_settings()
    if baseline_dwell:
        settings["manual_dwell"] = baseline_dwell
        settings["dwell_time"] = baseline_dwell
        save_settings()
    settings["typing_mode"] = prev_typing_mode
    calib = None
    build_suggestions()


def start_calib_words():
    calib["phase"] = "words"
    calib["word_index"] = 0
    calib["word_buffer"] = ""
    calib["word_start"] = time.time()
    calib["word_backspaces"] = 0


def calib_type_char(ch):
    if ch == " ":
        return
    calib["word_buffer"] += ch
    target = CALIB_WORDS[calib["word_index"]]
    if len(calib["word_buffer"]) >= len(target):
        time_ms = round((time.time() - calib["word_start"]) * 1000)
        calib["word_results"].append({
            "word": target, "typed": calib["word_buffer"],
            "correct": calib["word_buffer"].lower() == target,
            "time_ms": time_ms, "backspaces": calib["word_backspaces"],
        })
        calib["word_index"] += 1
        calib["word_buffer"] = ""
        calib["word_start"] = time.time()
        calib["word_backspaces"] = 0
        if calib["word_index"] >= len(CALIB_WORDS):
            finish_calib_words()


def calib_backspace():
    if not calib["word_buffer"]:
        return
    calib["word_buffer"] = calib["word_buffer"][:-1]
    calib["word_backspaces"] += 1


def finish_calib_words():
    calib["phase"] = "done"
    times = sorted(r["time_to_enter_ms"] for r in calib["target_results"])
    median = times[len(times) // 2] if times else 700
    calib["baseline"] = min(1100, max(450, round(median)))


def calib_button_rect():
    w, h = 240, 54
    return pygame.Rect(WINDOW_W // 2 - w // 2, WINDOW_H // 2 + 90, w, h)


def calib_skip_rect():
    w, h = 160, 30
    return pygame.Rect(WINDOW_W // 2 - w // 2, WINDOW_H // 2 + 160, w, h)


def handle_calib_click(pos):
    if calib["phase"] == "intro":
        if calib_button_rect().collidepoint(pos):
            calib["phase"] = "targets"
            calib["target_index"] = 0
            calib["target_shown_at"] = time.time()
            calib["target_dwell_start"] = None
        elif calib_skip_rect().collidepoint(pos):
            end_calibration(None)
    elif calib["phase"] == "done":
        if calib_button_rect().collidepoint(pos):
            end_calibration(calib["baseline"])


def update_calib_targets(mouse_pos, now):
    idx = calib["target_index"]
    tx_frac, ty_frac, label = CALIB_TARGETS[idx]
    target_pos = (tx_frac * WINDOW_W, ty_frac * WINDOW_H)
    dist = math.hypot(mouse_pos[0] - target_pos[0], mouse_pos[1] - target_pos[1])
    if dist <= CALIB_TARGET_R:
        if calib["target_dwell_start"] is None:
            calib["target_dwell_start"] = now
        elif now - calib["target_dwell_start"] >= CALIB_DWELL:
            time_to_enter = round((now - calib["target_shown_at"]) * 1000)
            calib["target_results"].append({"label": label, "time_to_enter_ms": time_to_enter})
            calib["target_index"] += 1
            calib["target_dwell_start"] = None
            if calib["target_index"] >= len(CALIB_TARGETS):
                start_calib_words()
            else:
                calib["target_shown_at"] = now
    else:
        calib["target_dwell_start"] = None


# ---------------------------------------------------------------- sign-in screen
# Click-only, like Settings - a caregiver or the user themselves picks a
# profile with the mouse before the gaze/dwell keyboard becomes active.
def refresh_signin_profiles():
    global signin_profiles
    signin_profiles = profiles.list_profiles()


def signin_profile_rects():
    w, h, gap = 320, 54, 12
    x = WINDOW_W // 2 - w // 2
    y = WINDOW_H // 2 - 160
    rects = []
    for name in signin_profiles:
        rects.append((name, pygame.Rect(x, y, w, h)))
        y += h + gap
    new_btn_rect = pygame.Rect(x, y, w, h)
    return rects, new_btn_rect


def signin_name_entry_rects():
    w, h = 320, 50
    x = WINDOW_W // 2 - w // 2
    y = WINDOW_H // 2 - 20
    input_rect = pygame.Rect(x, y, w, h)
    create_rect = pygame.Rect(x, y + h + 16, 150, 44)
    cancel_rect = pygame.Rect(x + w - 150, y + h + 16, 150, 44)
    return input_rect, create_rect, cancel_rect


def append_name_text(text):
    global name_input
    if len(name_input) < 24:
        name_input += text


def backspace_name():
    global name_input
    name_input = name_input[:-1]


def cancel_name_entry():
    global entering_name, name_input
    entering_name = False
    name_input = ""


def confirm_name_entry():
    global entering_name, name_input
    chosen = name_input.strip()
    if not chosen:
        return
    entering_name = False
    name_input = ""
    if not profiles.profile_exists(chosen):
        profiles.create_profile(chosen)
        refresh_signin_profiles()
    sign_in(chosen)


def handle_signin_click(pos):
    global entering_name, name_input
    if entering_name:
        _input_rect, create_rect, cancel_rect = signin_name_entry_rects()
        if create_rect.collidepoint(pos):
            confirm_name_entry()
        elif cancel_rect.collidepoint(pos):
            cancel_name_entry()
        return
    rects, new_btn_rect = signin_profile_rects()
    for name, rect in rects:
        if rect.collidepoint(pos):
            sign_in(name)
            return
    if new_btn_rect.collidepoint(pos):
        entering_name = True
        name_input = ""


def draw_signin():
    screen.fill(BG)
    draw_text("Gaze Board", 30, TEXT, (WINDOW_W // 2, WINDOW_H // 2 - 240), "center", bold=True)

    if not entering_name:
        draw_text("Who's using Gaze Board?", 17, MUTED, (WINDOW_W // 2, WINDOW_H // 2 - 195), "center")
        rects, new_btn_rect = signin_profile_rects()
        for name, rect in rects:
            pygame.draw.rect(screen, PANEL, rect, border_radius=12)
            pygame.draw.rect(screen, BORDER, rect, width=1, border_radius=12)
            draw_text(name, 16, TEXT, rect.center, "center", bold=True)
        pygame.draw.rect(screen, ACCENT, new_btn_rect, border_radius=12)
        draw_text("+ New Profile", 16, PANEL, new_btn_rect.center, "center", bold=True)
        if not signin_profiles:
            draw_text("No profiles yet - create one to get started.", 13, MUTED,
                      (WINDOW_W // 2, new_btn_rect.top - 22), "center")
    else:
        draw_text("What's your name?", 17, MUTED, (WINDOW_W // 2, WINDOW_H // 2 - 60), "center")
        input_rect, create_rect, cancel_rect = signin_name_entry_rects()
        pygame.draw.rect(screen, PANEL, input_rect, border_radius=10)
        pygame.draw.rect(screen, ACCENT, input_rect, width=2, border_radius=10)
        cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
        draw_text(name_input + cursor, 18, TEXT, (input_rect.left + 14, input_rect.centery), "midleft")
        can_create = bool(name_input.strip())
        pygame.draw.rect(screen, ACCENT if can_create else BORDER, create_rect, border_radius=10)
        draw_text("Create", 15, PANEL if can_create else MUTED, create_rect.center, "center", bold=True)
        pygame.draw.rect(screen, PANEL_2, cancel_rect, border_radius=10)
        draw_text("Cancel", 15, TEXT, cancel_rect.center, "center")


# ---------------------------------------------------------------- interaction
# This is the one seam a real gaze tracker plugs into: replace
# pygame.mouse.get_pos() (used here and in the main loop) with wherever
# your eye-tracking code publishes its latest (x, y) screen estimate,
# and everything below - hovering, dwelling, blink-clicking - keeps working.
def update_interaction(mouse_pos, now):
    global hovered
    if current_profile is None:
        return  # sign-in screen is click-only, like Settings
    if settings_open:
        return
    if calib and calib["phase"] != "words":
        if calib["phase"] == "targets":
            update_calib_targets(mouse_pos, now)
        return

    pool = list(key_selectables) + list(dock_selectables)
    if not calib:
        pool += suggestion_selectables

    target = None
    for sel in pool:
        if sel.rect.collidepoint(mouse_pos):
            target = sel
            break

    if target is not hovered:
        if hovered is not None:
            hovered.dwell_start = None
        hovered = target
        if hovered is not None and settings["typing_mode"] and not settings["blink_mode"]:
            hovered.dwell_start = now

    if hovered is not None and hovered.dwell_start is not None:
        elapsed = now - hovered.dwell_start
        if elapsed >= settings["dwell_time"] / 1000:
            sel = hovered
            sel.dwell_start = None
            sel.flash_until = now + 0.35
            sel.on_select()


def handle_mouse_down(pos):
    if current_profile is None:
        handle_signin_click(pos)
        return
    if calib:
        if calib["phase"] in ("intro", "done"):
            handle_calib_click(pos)
        return
    if settings_open:
        handle_settings_click(pos)
        return
    if handle_topbar_click(pos):
        return
    if settings["blink_mode"] and hovered is not None:
        hovered.flash_until = time.time() + 0.35
        hovered.on_select()


# ---------------------------------------------------------------- drawing
pygame.init()
pygame.display.set_caption("Gaze Board")
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
clock = pygame.time.Clock()

_font_cache = {}


def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = pygame.font.SysFont("arial", size, bold=bold)
    return _font_cache[key]


def draw_text(text, size, color, pos, anchor="topleft", bold=False):
    surf = get_font(size, bold).render(text, True, color)
    rect = surf.get_rect(**{anchor: pos})
    screen.blit(surf, rect)
    return rect


def blend(base, overlay, alpha):
    return tuple(int(base[i] * (1 - alpha) + overlay[i] * alpha) for i in range(3))


def draw_ring(center, radius, progress, track_color, fill_color, width=3):
    pygame.draw.circle(screen, track_color, center, radius, width)
    if progress > 0:
        start_angle = -math.pi / 2
        sweep = min(1.0, progress) * 2 * math.pi
        steps = max(2, int(sweep / (math.pi / 24)) + 1)
        points = [
            (center[0] + radius * math.cos(start_angle + sweep * i / steps),
             center[1] + radius * math.sin(start_angle + sweep * i / steps))
            for i in range(steps + 1)
        ]
        pygame.draw.lines(screen, fill_color, False, points, width)


def draw_selectable(sel, now, radius_pad=4, rounding=14):
    base = KEY_BG
    if sel.highlight_alpha > 0:
        base = blend(KEY_BG, FOCUS, sel.highlight_alpha)
    pygame.draw.rect(screen, base, sel.rect, border_radius=rounding)
    border_color = ACCENT if sel is hovered else KEY_BORDER
    pygame.draw.rect(screen, border_color, sel.rect, width=2, border_radius=rounding)
    if now < sel.flash_until:
        pygame.draw.rect(screen, ACCENT, sel.rect.inflate(6, 6), width=3, border_radius=rounding + 2)

    radius = min(sel.rect.width, sel.rect.height) // 2 - radius_pad
    if sel.dwell_start is not None:
        progress = (now - sel.dwell_start) / (settings["dwell_time"] / 1000)
        draw_ring(sel.rect.center, radius, progress, KEY_BORDER, ACCENT)
    else:
        pygame.draw.circle(screen, KEY_BORDER, sel.rect.center, radius, 2)

    if sel.align == "left":
        draw_text(sel.label, sel.font_size, TEXT, (sel.rect.left + 14, sel.rect.centery), "midleft", bold=True)
    else:
        draw_text(sel.label, sel.font_size, TEXT, sel.rect.center, "center", bold=True)


def draw_switch_row(rect, label_pos, label, on, hint=None, hint_pos=None):
    settings_widgets_switch = None  # placeholder, actual Switch drawn by caller
    draw_text(label, 15, MUTED, label_pos)


def draw_topbar():
    pygame.draw.rect(screen, PANEL, (0, 0, WINDOW_W, TOPBAR_H))
    pygame.draw.line(screen, BORDER, (0, TOPBAR_H), (WINDOW_W, TOPBAR_H), 1)
    pygame.draw.circle(screen, ACCENT, (34, 37), 18)
    draw_text("Gaze Board", 22, TEXT, (58, 14), bold=True)
    draw_text("mouse-simulated eye-gaze AAC keyboard", 13, MUTED, (58, 42))

    draw_text("Typing mode", 13, MUTED, (TYPING_SWITCH_RECT.left - 8, TYPING_SWITCH_RECT.centery), "midright")
    _switch_typing.draw(screen, settings["typing_mode"])
    draw_text("Blink to select", 13, MUTED, (BLINK_SWITCH_RECT.left - 8, BLINK_SWITCH_RECT.centery), "midright")
    _switch_blink.draw(screen, settings["blink_mode"])

    pygame.draw.rect(screen, PANEL_2, GAZE_DOT_BTN_RECT, border_radius=8)
    draw_text("Gaze dot: " + ("on" if settings["show_gaze_dot"] else "off"), 13, TEXT, GAZE_DOT_BTN_RECT.center, "center")

    pygame.draw.rect(screen, PANEL_2, CALIBRATE_BTN_RECT, border_radius=8)
    draw_text("Calibrate", 13, TEXT, CALIBRATE_BTN_RECT.center, "center")
    if not has_calibration:
        pygame.draw.circle(screen, FOCUS, (CALIBRATE_BTN_RECT.right - 4, CALIBRATE_BTN_RECT.top - 2), 5)

    pygame.draw.rect(screen, PANEL_2, SETTINGS_BTN_RECT, border_radius=8)
    draw_text("⚙", 18, TEXT, SETTINGS_BTN_RECT.center, "center")

    draw_text(f"Signed in as {current_profile}", 12, MUTED,
              (SIGNOUT_BTN_RECT.left - 10, SIGNOUT_BTN_RECT.centery), "midright")
    pygame.draw.rect(screen, PANEL_2, SIGNOUT_BTN_RECT, border_radius=8)
    draw_text("Sign out", 13, TEXT, SIGNOUT_BTN_RECT.center, "center")


_switch_typing = Switch(TYPING_SWITCH_RECT)
_switch_blink = Switch(BLINK_SWITCH_RECT)


def draw_output_strip():
    rect = pygame.Rect(SIDEBAR_W, TOPBAR_H, WINDOW_W - SIDEBAR_W, OUTPUT_H)
    pygame.draw.rect(screen, PANEL, rect)
    pygame.draw.line(screen, BORDER, (SIDEBAR_W, TOPBAR_H + OUTPUT_H), (WINDOW_W, TOPBAR_H + OUTPUT_H), 1)

    box = pygame.Rect(rect.left + 20, rect.top + 14, rect.width - 40, 46)
    pygame.draw.rect(screen, PANEL_2, box, border_radius=10)
    pygame.draw.rect(screen, BORDER, box, width=1, border_radius=10)
    if buffer:
        text = buffer
        if int(time.time() * 2) % 2 == 0:
            text += "|"
        draw_text(text, 22, TEXT, (box.left + 14, box.centery), "midleft")
    else:
        draw_text("Hover a key below to start - the mouse stands in for your eyes.", 15, MUTED,
                  (box.left + 14, box.centery), "midleft")

    stats_y = rect.top + 70
    draw_text(f"{current_wpm():.1f} WPM", 13, MUTED, (rect.left + 20, stats_y))
    draw_text(f"dwell {settings['dwell_time']} ms", 13, MUTED, (rect.left + 150, stats_y))
    draw_text(f"backspace rate {round(backspace_rate()*100)}%", 13, MUTED, (rect.left + 300, stats_y))
    badge = "adaptive: on" if settings["adaptive"] else "adaptive: off"
    draw_text(badge, 13, ACCENT_STRONG, (rect.left + 500, stats_y))


def _draw_wrapped(lines, size, color, x, y, line_gap=20):
    for i, line in enumerate(lines):
        draw_text(line, size, color, (x, y + i * line_gap))


def draw_suggestions():
    rect = pygame.Rect(0, TOPBAR_H, SIDEBAR_W, WINDOW_H - TOPBAR_H)
    pygame.draw.rect(screen, PANEL, rect)
    pygame.draw.line(screen, BORDER, (SIDEBAR_W, TOPBAR_H), (SIDEBAR_W, WINDOW_H), 1)
    header = "SUGGESTED WORDS" if current_word() else "FAVORITES"
    draw_text(header, 12, MUTED, (SIDEBAR_PAD, TOPBAR_H + 20), bold=True)

    if calib:
        _draw_wrapped(["Not available during", "calibration."], 13, MUTED, SIDEBAR_PAD, TOPBAR_H + 56)
        return
    if not suggestion_selectables:
        if current_word():
            lines = ["No matches yet."]
        else:
            lines = ["No favorites yet.", "Type something and tap", "\"favorite\" to save it."]
        _draw_wrapped(lines, 13, MUTED, SIDEBAR_PAD, TOPBAR_H + 56)
        return
    now = time.time()
    for sel in suggestion_selectables:
        draw_selectable(sel, now, radius_pad=8, rounding=10)


def draw_legend():
    y = GRID_ORIGIN[1] - 22
    draw_text("Gold highlight = predicted next letter, stronger = more confident", 14, MUTED,
              (GRID_ORIGIN[0], y))


def draw_board():
    now = time.time()
    draw_legend()
    for sel in key_selectables:
        draw_selectable(sel, now)
    for sel in dock_selectables:
        draw_selectable(sel, now, rounding=16)


def draw_settings_panel():
    rect = pygame.Rect(PANEL_X, 0, SETTINGS_PANEL_W, WINDOW_H)
    pygame.draw.rect(screen, PANEL, rect)
    pygame.draw.line(screen, BORDER, (PANEL_X, 0), (PANEL_X, WINDOW_H), 1)
    draw_text("✕", 16, MUTED, settings_widgets["close"].center, "center")
    draw_text("Settings", 18, TEXT, (PANEL_X + _pad, 20), bold=True)

    draw_text("Layout", 13, MUTED, (PANEL_X + _pad, 76))
    for key, label in (("radio_optimized", "Frequency-optimized (larger, centered)"),
                        ("radio_alphabetical", "Alphabetical grid (uniform size)")):
        r = settings_widgets[key]
        selected = settings["layout"] == key.replace("radio_", "")
        pygame.draw.circle(screen, ACCENT if selected else BORDER, (r.left + 8, r.centery), 7, 0 if selected else 2)
        draw_text(label, 13, TEXT, (r.left + 22, r.centery), "midleft")

    draw_text(f"Adaptive dwell AI: {'on' if settings['adaptive'] else 'off'}", 13, MUTED, (PANEL_X + _pad, 158))
    settings_widgets["adaptive_switch"].draw(screen, settings["adaptive"])
    draw_text("Nudges dwell time from your recent backspace rate.", 11, MUTED, (PANEL_X + _pad, 208))

    draw_text(f"Dwell time: {settings['dwell_time']} ms", 13, MUTED, (PANEL_X + _pad, 232))
    settings_widgets["dwell_slider"].draw(screen, settings["dwell_time"])
    draw_text("Drag to set an exact delay - turns adaptive AI off.", 11, MUTED, (PANEL_X + _pad, 272))

    draw_text(f"Next-letter highlighting: {'on' if settings['prediction_on'] else 'off'}", 13, MUTED,
              (PANEL_X + _pad, 300))
    settings_widgets["prediction_switch"].draw(screen, settings["prediction_on"])

    draw_text(f"Highlight strength: {round(settings['highlight_strength']*100)}%", 13, MUTED, (PANEL_X + _pad, 364))
    settings_widgets["strength_slider"].draw(screen, settings["highlight_strength"] * 100)

    draw_text(f"Key size: {round(settings['key_scale']*100)}%", 13, MUTED, (PANEL_X + _pad, 428))
    settings_widgets["keyscale_slider"].draw(screen, settings["key_scale"] * 100)

    draw_text(f"Show gaze dot: {'on' if settings['show_gaze_dot'] else 'off'}", 13, MUTED, (PANEL_X + _pad, 476))
    settings_widgets["gazedot_switch"].draw(screen, settings["show_gaze_dot"])

    for key, label in (("calibrate_btn", "Run calibration"), ("cleartext_btn", "Clear typed text"),
                        ("resetstats_btn", "Reset adaptive stats"), ("clearfavorites_btn", "Clear favorites")):
        r = settings_widgets[key]
        pygame.draw.rect(screen, PANEL_2, r, border_radius=10)
        pygame.draw.rect(screen, BORDER, r, width=1, border_radius=10)
        draw_text(label, 13, TEXT, r.center, "center")


def draw_calibration(mouse_pos, now):
    phase = calib["phase"]
    if phase == "intro":
        draw_text("Quick calibration", 24, TEXT, (WINDOW_W // 2, WINDOW_H // 2 - 60), "center", bold=True)
        draw_text("This mouse version can't replicate real eye jitter, but it can still learn your pace.",
                  14, MUTED, (WINDOW_W // 2, WINDOW_H // 2 - 20), "center")
        draw_text("1. Move to five dots around the screen and hold.", 13, MUTED,
                  (WINDOW_W // 2, WINDOW_H // 2 + 10), "center")
        draw_text("2. Spell three short words on the keyboard.", 13, MUTED,
                  (WINDOW_W // 2, WINDOW_H // 2 + 32), "center")
        r = calib_button_rect()
        pygame.draw.rect(screen, ACCENT, r, border_radius=10)
        draw_text("Begin", 16, PANEL, r.center, "center", bold=True)
        r2 = calib_skip_rect()
        draw_text("Skip for now", 13, MUTED, r2.center, "center")

    elif phase == "targets":
        idx = calib["target_index"]
        tx_frac, ty_frac, _label = CALIB_TARGETS[idx]
        pos = (int(tx_frac * WINDOW_W), int(ty_frac * WINDOW_H))
        pygame.draw.circle(screen, FOCUS, pos, CALIB_TARGET_R)
        if calib["target_dwell_start"] is not None:
            progress = (now - calib["target_dwell_start"]) / CALIB_DWELL
            draw_ring(pos, CALIB_TARGET_R - 6, progress, (255, 255, 255), (255, 255, 255), width=4)
        draw_text(f"Target {idx+1} of {len(CALIB_TARGETS)}", 14, MUTED, (WINDOW_W // 2, WINDOW_H - 60), "center")

    elif phase == "words":
        word = CALIB_WORDS[calib["word_index"]]
        text = f"Spell {word.upper()}  ({calib['word_index']+1}/{len(CALIB_WORDS)})  -  {calib['word_buffer'].upper()}"
        banner = pygame.Rect(0, 0, 520, 50)
        banner.midtop = (WINDOW_W // 2, TOPBAR_H + 16)
        pygame.draw.rect(screen, PANEL, banner, border_radius=12)
        pygame.draw.rect(screen, BORDER, banner, width=1, border_radius=12)
        draw_text(text, 15, TEXT, banner.center, "center", bold=True)

    elif phase == "done":
        draw_text("Calibration complete", 24, TEXT, (WINDOW_W // 2, WINDOW_H // 2 - 100), "center", bold=True)
        times = [r["time_to_enter_ms"] for r in calib["target_results"]]
        median = sorted(times)[len(times) // 2] if times else 0
        avg_word = round(sum(r["time_ms"] for r in calib["word_results"]) / max(1, len(calib["word_results"])))
        total_bksp = sum(r["backspaces"] for r in calib["word_results"])
        lines = [
            f"Median target reach time: {median} ms",
            f"Avg. time per word: {avg_word} ms",
            f"Backspaces while spelling: {total_bksp}",
            f"Starting dwell time: {calib['baseline']} ms",
        ]
        for i, line in enumerate(lines):
            draw_text(line, 14, MUTED, (WINDOW_W // 2, WINDOW_H // 2 - 50 + i * 24), "center")
        r = calib_button_rect()
        pygame.draw.rect(screen, ACCENT, r, border_radius=10)
        draw_text("Start typing", 16, PANEL, r.center, "center", bold=True)


def draw_gaze_dot(mouse_pos):
    pygame.draw.circle(screen, FOCUS, mouse_pos, 8)
    pygame.draw.circle(screen, ACCENT_STRONG, mouse_pos, 13, 2)


def draw(mouse_pos, now):
    if current_profile is None:
        draw_signin()
        pygame.display.flip()
        return
    screen.fill(BG)
    draw_suggestions()
    draw_output_strip()
    draw_board()
    draw_topbar()
    if settings_open:
        draw_settings_panel()
    if calib:
        overlay = pygame.Surface((WINDOW_W, WINDOW_H))
        overlay.set_alpha(235)
        overlay.fill(BG)
        if calib["phase"] != "words":
            screen.blit(overlay, (0, 0))
        draw_calibration(mouse_pos, now)
    if settings["show_gaze_dot"] and not calib:
        draw_gaze_dot(mouse_pos)
    pygame.display.flip()


# ---------------------------------------------------------------- main loop
def main():
    global dragging_slider
    refresh_signin_profiles()
    running = True
    max_frames = int(os.environ.get("GAZE_BOARD_MAX_FRAMES", "0"))
    frame_count = 0
    while running:
        clock.tick(FPS)
        now = time.time()
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.TEXTINPUT and entering_name:
                append_name_text(event.text)
            elif event.type == pygame.KEYDOWN:
                if entering_name:
                    if event.key == pygame.K_BACKSPACE:
                        backspace_name()
                    elif event.key == pygame.K_RETURN:
                        confirm_name_entry()
                    elif event.key == pygame.K_ESCAPE:
                        cancel_name_entry()
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handle_mouse_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging_slider:
                    save_settings()
                dragging_slider = None
            elif event.type == pygame.MOUSEMOTION and dragging_slider:
                handle_slider_drag(event.pos[0])

        update_interaction(mouse_pos, now)
        draw(mouse_pos, now)

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            running = False

    if current_profile:
        sign_out()
    pygame.quit()


if __name__ == "__main__":
    main()
