# Need to add training mode as well as non-letter keys.

import math

from Quartz.CoreGraphics import CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags, kCGHIDEventTap


LETTER_KEYCODE = {
    "A": 0, "B": 11, "C": 8, "D": 2, "E": 14, "F": 3, "G": 5, "H": 4, "I": 34, "J": 38,
    "K": 40, "L": 37, "M": 46, "N": 45, "O": 31, "P": 35, "Q": 12, "R": 15, "S": 1, "T": 17,
    "U": 32, "V": 9, "W": 13, "X": 7, "Y": 16, "Z": 6,
}
KEYCODE_SPACE = 49
KEYCODE_SHIFT = 56


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _hand_size(lm):
    return _dist(lm[0], lm[9]) + 1e-6


def _near(lm, i, j, scale):
    return (_dist(lm[i], lm[j]) / _hand_size(lm)) <= scale


def detect_typing_entry(finger_states):
    return (
        finger_states["Pinky"] == "extended"
        and finger_states["Thumb"] != "extended"
        and finger_states["Index"] != "extended"
        and finger_states["Middle"] != "extended"
        and finger_states["Ring"] != "extended"
    )


def all_five_extended(finger_states):
    return all(finger_states[name] == "extended" for name in ["Thumb", "Index", "Middle", "Ring", "Pinky"])


def palm_center_y_norm(lm_norm):
    idxs = [0, 5, 9, 13, 17]
    return sum(lm_norm[i][1] for i in idxs) / len(idxs)


def is_fist(finger_states):
    return (
        finger_states["Thumb"] != "extended"
        and finger_states["Index"] != "extended"
        and finger_states["Middle"] != "extended"
        and finger_states["Ring"] != "extended"
        and finger_states["Pinky"] != "extended"
    )


def left_thumb_modifier_active(finger_states):
    return finger_states["Thumb"] == "extended"


def _tap_key(keycode, with_shift=False):
    if with_shift:
        shift_down = CGEventCreateKeyboardEvent(None, KEYCODE_SHIFT, True)
        CGEventSetFlags(shift_down, 0)
        CGEventPost(kCGHIDEventTap, shift_down)
    down = CGEventCreateKeyboardEvent(None, keycode, True)
    up = CGEventCreateKeyboardEvent(None, keycode, False)
    CGEventSetFlags(down, 0)
    CGEventSetFlags(up, 0)
    CGEventPost(kCGHIDEventTap, down)
    CGEventPost(kCGHIDEventTap, up)
    if with_shift:
        shift_up = CGEventCreateKeyboardEvent(None, KEYCODE_SHIFT, False)
        CGEventSetFlags(shift_up, 0)
        CGEventPost(kCGHIDEventTap, shift_up)


def press_letter(letter, uppercase=False):
    keycode = LETTER_KEYCODE.get(letter.upper())
    if keycode is None:
        return False
    _tap_key(keycode, with_shift=uppercase)
    return True


def press_space():
    _tap_key(KEYCODE_SPACE, with_shift=False)
    return True


def classify_asl_letter(lm, finger_states):
    t = finger_states["Thumb"] == "extended"
    i = finger_states["Index"] == "extended"
    m = finger_states["Middle"] == "extended"
    r = finger_states["Ring"] == "extended"
    p = finger_states["Pinky"] == "extended"
    hs = _hand_size(lm)
    idx_mid_sep = _dist(lm[8], lm[12]) / hs

    # Highly distinctive shapes first
    if t and p and not i and not m and not r:
        return "Y"
    if not t and p and not i and not m and not r:
        return "I"
    if i and m and r and not p and not t:
        return "W"
    if i and m and not r and not p:
        if idx_mid_sep <= 0.12:
            return "R"
        if idx_mid_sep <= 0.23:
            return "U"
        return "V"
    if i and not m and not r and not p and t:
        idx_thumb_dist = _dist(lm[4], lm[8]) / hs
        if idx_thumb_dist <= 0.20:
            return "F"
        if lm[8][1] > lm[5][1]:
            return "Q"
        if idx_thumb_dist <= 0.40:
            return "G"
        return "L"
    if i and m and not r and not p and t:
        # K vs P vs H
        if _near(lm, 4, 10, 0.30):
            return "K"
        if lm[8][1] > lm[6][1] and lm[12][1] > lm[10][1]:
            return "P"
        if _dist(lm[8], lm[12]) / hs > 0.18:
            return "H"
        return "H"
    if i and not m and not r and not p and not t:
        # D: index up + thumb to middle area, not near index tip.
        if _near(lm, 4, 12, 0.32) and not _near(lm, 4, 8, 0.24):
            return "D"
        # Q: index pointed downward-like.
        if lm[8][1] > lm[6][1]:
            return "Q"
        # G: index horizontal-ish with thumb near index side.
        if abs(lm[8][1] - lm[6][1]) / hs < 0.20:
            return "G"
        if lm[8][1] > lm[6][1]:
            return "Z"
        return "X"
    if i and m and r and p and not t:
        return "B"

    # Fist-family letters
    if not i and not m and not r and not p:
        if t:
            return "A"
        # O: compact rounded closure.
        if _near(lm, 4, 8, 0.24) and _near(lm, 4, 12, 0.30):
            return "O"
        # T: thumb tucked near index MCP.
        if _near(lm, 4, 5, 0.26):
            return "T"
        # M / N: thumb under 3 or 2 fingers respectively.
        if _near(lm, 4, 14, 0.34):
            return "M"
        if _near(lm, 4, 10, 0.30):
            return "N"
        if _near(lm, 4, 8, 0.27) and _near(lm, 4, 12, 0.33):
            return "E"
        if 0.28 <= (_dist(lm[4], lm[8]) / hs) <= 0.55:
            return "C"
        return "S"

    # J is dynamic in ASL; fallback heuristic near I-like shape
    # Need to fix
    if not t and p and not i and not m and not r and lm[20][1] > lm[18][1]:
        return "J"

    return None
