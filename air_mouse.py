# February 14th, 2026
# Henry Houghton

import time
import math
import subprocess
import numpy as np
import cv2
import mediapipe as mp
import air_mouse_settings as cfg
import typing_mode as tm

from Quartz.CoreGraphics import (
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventSetIntegerValueField,
    CGEventSetFlags,
    CGWarpMouseCursorPosition,
    kCGHIDEventTap,
    kCGEventMouseMoved,
    kCGEventLeftMouseDragged,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGMouseEventClickState,
    kCGMouseButtonLeft,
)
try:
    from Carbon.HIToolbox import kVK_LeftArrow, kVK_RightArrow
except Exception:
    # Fallback virtual keycodes for macOS arrows
    kVK_LeftArrow = 123
    kVK_RightArrow = 124
# macOS mouse helpers
def mouse_move(x, y, dragging=False):
    # to avoids some event throttling
    CGWarpMouseCursorPosition((x, y))
    # some apps respond better with an event ig
    event_type = kCGEventLeftMouseDragged if dragging else kCGEventMouseMoved
    evt = CGEventCreateMouseEvent(None, event_type, (x, y), kCGMouseButtonLeft)
    # Ensure no keyboard modifiers leak into mouse actions
    CGEventSetFlags(evt, 0)
    CGEventPost(kCGHIDEventTap, evt)

def mouse_down(x, y):
    evt = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    CGEventSetFlags(evt, 0)
    CGEventPost(kCGHIDEventTap, evt)

def mouse_up(x, y):
    evt = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    CGEventSetFlags(evt, 0)
    CGEventPost(kCGHIDEventTap, evt)

def mouse_click(x, y, click_state=1):
    down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    CGEventSetFlags(down, 0)
    CGEventSetFlags(up, 0)
    CGEventSetIntegerValueField(down, kCGMouseEventClickState, click_state)
    CGEventSetIntegerValueField(up, kCGMouseEventClickState, click_state)
    CGEventPost(kCGHIDEventTap, down)
    CGEventPost(kCGHIDEventTap, up)

def mouse_double_click(x, y):
    mouse_click(x, y, click_state=1)
    time.sleep(cfg.DOUBLE_CLICK_GAP_S)
    mouse_click(x, y, click_state=2)

def switch_desktop(direction):
    keycode = kVK_RightArrow if direction == "right" else kVK_LeftArrow
    # Use System Events; this is typically more reliable for Mission Control
    # desktop switching than posting raw CG keyboard events
    script = f'tell application "System Events" to key code {keycode} using control down'
    subprocess.run(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

# gesture math stuff
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Index extended heuristic:
# index is extended if tip is farther from wrist than PIP is, by a margin,
# and not curled near palm
def index_extended(lm):
    wrist = lm[0]
    idx_mcp = lm[5]
    idx_pip = lm[6]
    idx_tip = lm[8]

    # Compare distances from wrist (or MCP) to see extension
    d_tip = dist(idx_tip, wrist)
    d_pip = dist(idx_pip, wrist)
    # also check tip away from MCP
    d_tip_mcp = dist(idx_tip, idx_mcp)
    d_pip_mcp = dist(idx_pip, idx_mcp)

    return (d_tip > d_pip * cfg.INDEX_EXT_WRIST_RATIO) and (d_tip_mcp > d_pip_mcp * cfg.INDEX_EXT_MCP_RATIO)

def finger_extended(lm, mcp_idx, pip_idx, tip_idx):
    wrist = lm[0]
    mcp = lm[mcp_idx]
    pip = lm[pip_idx]
    tip = lm[tip_idx]
    d_tip = dist(tip, wrist)
    d_pip = dist(pip, wrist)
    d_tip_mcp = dist(tip, mcp)
    d_pip_mcp = dist(pip, mcp)
    return (d_tip > d_pip * cfg.FINGER_EXT_WRIST_RATIO) and (d_tip_mcp > d_pip_mcp * cfg.FINGER_EXT_MCP_RATIO)

def thumb_extended(lm):
    # Thumb uses landmarks: CMC(1), MCP(2), IP(3), TIP(4)
    th_mcp = lm[2]
    th_ip = lm[3]
    th_tip = lm[4]
    idx_mcp = lm[5]
    idx_tip = lm[8]
    wrist = lm[0]
    d_tip = dist(th_tip, wrist)
    d_ip = dist(th_ip, wrist)
    d_tip_mcp = dist(th_tip, th_mcp)
    d_ip_mcp = dist(th_ip, th_mcp)
    geom_extended = (d_tip > d_ip * cfg.THUMB_EXT_WRIST_RATIO) and (d_tip_mcp > d_ip_mcp * cfg.THUMB_EXT_MCP_RATIO)
    # Crossing test only: if thumb tip crosses the index-finger line
    # (relative to thumb IP side), thumb is closed
    ax, ay = idx_mcp
    bx, by = idx_tip
    tx, ty = th_tip
    ix, iy = th_ip
    thumb_side = (bx - ax) * (ty - ay) - (by - ay) * (tx - ax)
    ip_side = (bx - ax) * (iy - ay) - (by - ay) * (ix - ax)
    tip_not_crossed = (thumb_side * ip_side) >= 0
    return geom_extended and tip_not_crossed

def finger_state(lm, mcp_idx, pip_idx, tip_idx):
    if finger_extended(lm, mcp_idx, pip_idx, tip_idx):
        return "extended"
    wrist = lm[0]
    d_tip = dist(lm[tip_idx], wrist)
    d_pip = dist(lm[pip_idx], wrist)
    if d_tip < d_pip * 0.92:
        return "curled"
    return "closed"

def thumb_state(lm):
    if thumb_extended(lm):
        return "extended"
    wrist = lm[0]
    d_tip = dist(lm[4], wrist)
    d_ip = dist(lm[3], wrist)
    if d_tip < d_ip * 0.92:
        return "curled"
    return "closed"

def peace_sign(lm):
    idx_ok = finger_extended(lm, 5, 6, 8)
    mid_ok = finger_extended(lm, 9, 10, 12)
    ring_curled = not finger_extended(lm, 13, 14, 16)
    pinky_curled = not finger_extended(lm, 17, 18, 20)
    thumb_curled = not thumb_extended(lm)
    split = dist(lm[8], lm[12]) > cfg.PEACE_SPLIT_RATIO * (dist(lm[0], lm[9]) + 1e-6)
    return idx_ok and mid_ok and split and ring_curled and pinky_curled and thumb_curled

def four_fingers_up(lm):
    idx_ok = finger_extended(lm, 5, 6, 8)
    mid_ok = finger_extended(lm, 9, 10, 12)
    ring_ok = finger_extended(lm, 13, 14, 16)
    pinky_ok = finger_extended(lm, 17, 18, 20)
    return idx_ok and mid_ok and ring_ok and pinky_ok

def palm_center_x_norm(lm_norm):
    # Average stable palm landmarks for smoother horizontal swipe tracking
    idxs = [0, 5, 9, 13, 17]
    return sum(lm_norm[i][0] for i in idxs) / len(idxs)

# Pinch heuristic:
# use normalized distance between two fingertips divided by hand size
def pinch_strength_between(lm, a_idx, b_idx):
    a = lm[a_idx]
    b = lm[b_idx]
    # hand size ~ distance wrist to middle MCP (landmark 9) or index MCP (5)
    wrist = lm[0]
    mid_mcp = lm[9]
    hand_size = dist(wrist, mid_mcp) + 1e-6

    pinch = dist(a, b) / hand_size
    return pinch  # smaller = more pinched

# enough pinch math
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam, maybe try enabling in settings")

    # Get screen size (macOS uses origin at bottom-left)
    # Query via OpenCV window properties? Better: use NSScreen, but to avoid extra deps
    # Approximate by asking the user’s primary screen via Quartz:
    from Quartz import CGDisplayPixelsWide, CGDisplayPixelsHigh, CGMainDisplayID
    display_id = CGMainDisplayID()
    screen_w = int(CGDisplayPixelsWide(display_id))
    screen_h = int(CGDisplayPixelsHigh(display_id))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=cfg.MP_MAX_NUM_HANDS,
        model_complexity=cfg.MP_MODEL_COMPLEXITY,
        min_detection_confidence=cfg.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=cfg.MP_MIN_TRACKING_CONFIDENCE,
    )

    # cursor smoothing (might need to change, higher = snappier, 
    # lower = smoother but maybe laggy)
    smoothed = None
    alpha = cfg.CURSOR_SMOOTHING_ALPHA

    # map only within a sort of control box for stability
    # (normalized camera coords)
    box = cfg.CONTROL_BOX

    # Gesture pinch states
    drag_down = False  # index+middle pinch holds left mouse button for drag
    idx_thumb_active = False
    mid_thumb_active = False

    drag_down_thresh = cfg.DRAG_DOWN_THRESH
    drag_up_thresh = cfg.DRAG_UP_THRESH
    click_down_thresh = cfg.CLICK_DOWN_THRESH
    click_up_thresh = cfg.CLICK_UP_THRESH
    click_debounce_s = cfg.CLICK_DEBOUNCE_S
    drag_hold_s = cfg.DRAG_HOLD_S
    click_hold_s = cfg.CLICK_HOLD_S
    last_drag_toggle_time = 0.0
    last_idx_thumb_click_time = 0.0
    last_mid_thumb_click_time = 0.0
    idx_mid_hold_start = None
    idx_th_hold_start = None
    mid_th_hold_start = None

    peace_seen_at = None
    peace_hold_s = cfg.PEACE_HOLD_S
    four_start_x = None
    last_space_switch_time = 0.0
    space_switch_cooldown_s = cfg.SPACE_SWITCH_COOLDOWN_S
    swipe_norm_thresh = cfg.SWIPE_NORM_THRESH
    mode = "browsing"
    last_mode_switch_time = 0.0
    typing_letter_candidate = None
    typing_letter_hold_start = None
    typing_last_key_time = 0.0
    typing_exit_swipe_start_y = None
    typing_letter_history = []

    last_move_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror so hand movement feels natural (move right hand right = cursor right)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # Overlay control box
        cv2.rectangle(
            frame,
            (int(box["xmin"] * w), int(box["ymin"] * h)),
            (int(box["xmax"] * w), int(box["ymax"] * h)),
            cfg.UI_BOX_COLOR,
            1
        )

        chosen = None  # (lm_pixels, handedness_label, landmarks_norm)

        if res.multi_hand_landmarks and res.multi_handedness:
            # pick RIGHT hand
            for hand_lms, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handedness.classification[0].label  # "Right" or "Left" (from camera POV after flip)
                # After frame flipped horizontally, MediaPipe handedness still refers to the hand itself
                if label != "Right":
                    continue

                # get normalized landmarks
                lm_norm = [(p.x, p.y) for p in hand_lms.landmark]
                lm_px = [(int(p.x * w), int(p.y * h)) for p in hand_lms.landmark]
                chosen = (lm_px, label, lm_norm)
                break

        active = False
        finger_states = {
            "Thumb": "no hand",
            "Index": "no hand",
            "Middle": "no hand",
            "Ring": "no hand",
            "Pinky": "no hand",
        }
        action_drag = "no hand"
        action_single = "no hand"
        action_double = "no hand"
        action_peace = "no hand"
        action_swipe = "no hand"
        action_typing = "idle"
        typing_meter_line = "Typing Meter: n/a"
        typing_confidence = 0.0
        typing_stability = 0.0

        if chosen:
            lm_px, _, lm_norm = chosen
            finger_states["Thumb"] = thumb_state(lm_px)
            finger_states["Index"] = finger_state(lm_px, 5, 6, 8)
            finger_states["Middle"] = finger_state(lm_px, 9, 10, 12)
            finger_states["Ring"] = finger_state(lm_px, 13, 14, 16)
            finger_states["Pinky"] = finger_state(lm_px, 17, 18, 20)

            # draw a few landmarks for debugging
            for idx in [0, 4, 5, 8, 9]:
                cv2.circle(frame, lm_px[idx], 5, (0, 255, 0), -1)
            # middle-finger visual tracker (MCP -> PIP -> DIP -> TIP)
            for idx in [9, 10, 11, 12]:
                cv2.circle(frame, lm_px[idx], 6, cfg.UI_MIDDLE_TRACKER_COLOR, -1)
            cv2.line(frame, lm_px[9], lm_px[10], cfg.UI_MIDDLE_TRACKER_COLOR, 2)
            cv2.line(frame, lm_px[10], lm_px[11], cfg.UI_MIDDLE_TRACKER_COLOR, 2)
            cv2.line(frame, lm_px[11], lm_px[12], cfg.UI_MIDDLE_TRACKER_COLOR, 2)
            cv2.line(frame, lm_px[12], lm_px[4], cfg.UI_MIDDLE_TRACKER_COLOR, 1)
            # index-finger tracker + "line going down" toward palm for thumb-side debugging
            for idx in [5, 6, 7, 8]:
                cv2.circle(frame, lm_px[idx], 6, cfg.UI_MIDDLE_TRACKER_COLOR, -1)
            cv2.line(frame, lm_px[5], lm_px[6], cfg.UI_MIDDLE_TRACKER_COLOR, 2)
            cv2.line(frame, lm_px[6], lm_px[7], cfg.UI_MIDDLE_TRACKER_COLOR, 2)
            cv2.line(frame, lm_px[7], lm_px[8], cfg.UI_MIDDLE_TRACKER_COLOR, 2)
            idx_tip_px = lm_px[8]
            idx_mcp_px = lm_px[5]
            dx = idx_mcp_px[0] - idx_tip_px[0]
            dy = idx_mcp_px[1] - idx_tip_px[1]
            down_end = (int(idx_tip_px[0] + 2.0 * dx), int(idx_tip_px[1] + 2.0 * dy))
            cv2.line(frame, idx_tip_px, down_end, cfg.UI_MIDDLE_TRACKER_COLOR, 1)

            # fingertip normalized coords
            ix, iy = lm_norm[8]
            ix = (clamp(ix, box["xmin"], box["xmax"]) - box["xmin"]) / (box["xmax"] - box["xmin"])
            iy = (clamp(iy, box["ymin"], box["ymax"]) - box["ymin"]) / (box["ymax"] - box["ymin"])
            sx = int(ix * screen_w)
            sy = int(iy * screen_h)

            if smoothed is None:
                smoothed = np.array([sx, sy], dtype=np.float32)
            else:
                smoothed = alpha * np.array([sx, sy], dtype=np.float32) + (1 - alpha) * smoothed

            now = time.time()
            if (
                mode == "browsing"
                and (now - last_mode_switch_time >= cfg.MODE_SWITCH_COOLDOWN_S)
                and tm.detect_typing_entry(finger_states)
            ):
                mode = "typing"
                last_mode_switch_time = now
                typing_letter_candidate = None
                typing_letter_hold_start = None
                typing_exit_swipe_start_y = None
                typing_letter_history = []
                if drag_down and smoothed is not None:
                    drag_down = False
                    mouse_up(int(smoothed[0]), int(smoothed[1]))

            if mode == "typing":
                action_drag = "disabled (typing)"
                action_single = "disabled (typing)"
                action_double = "disabled (typing)"
                action_peace = "disabled (typing)"
                action_swipe = "show 5 fingers + swipe down"

                # Keep mouse gestures unreachable in typing mode
                if drag_down and smoothed is not None:
                    drag_down = False
                    mouse_up(int(smoothed[0]), int(smoothed[1]))

                letter = tm.classify_asl_letter(lm_px, finger_states)
                typing_letter_history.append(letter)
                if len(typing_letter_history) > cfg.TYPING_CONFIDENCE_WINDOW_FRAMES:
                    typing_letter_history = typing_letter_history[-cfg.TYPING_CONFIDENCE_WINDOW_FRAMES:]
                if letter is None:
                    typing_letter_candidate = None
                    typing_letter_hold_start = None
                    action_typing = "no letter"
                    typing_meter_line = "Typing Meter: no candidate"
                    typing_confidence = 0.0
                    typing_stability = 0.0
                else:
                    if typing_letter_candidate != letter:
                        typing_letter_candidate = letter
                        typing_letter_hold_start = now
                    hold = 0.0 if typing_letter_hold_start is None else (now - typing_letter_hold_start)
                    non_none = [x for x in typing_letter_history if x is not None]
                    if non_none:
                        typing_confidence = sum(1 for x in non_none if x == letter) / float(len(non_none))
                    else:
                        typing_confidence = 0.0
                    typing_stability = min(1.0, hold / max(1e-6, cfg.TYPING_LETTER_HOLD_S))
                    typing_meter_line = f"Typing Meter: {letter} conf={typing_confidence*100:.0f}% stab={typing_stability*100:.0f}%"
                    action_typing = f"candidate {letter} hold {max(0.0, cfg.TYPING_LETTER_HOLD_S - hold):.2f}s"
                    if (
                        typing_letter_hold_start is not None
                        and hold >= cfg.TYPING_LETTER_HOLD_S
                        and (now - typing_last_key_time) >= cfg.TYPING_REPEAT_COOLDOWN_S
                    ):
                        if tm.press_letter(letter):
                            typing_last_key_time = now
                            action_typing = f"typed {letter}"

                if tm.all_five_extended(finger_states):
                    palm_y = tm.palm_center_y_norm(lm_norm)
                    if typing_exit_swipe_start_y is None:
                        typing_exit_swipe_start_y = palm_y
                    dy_down = palm_y - typing_exit_swipe_start_y
                    action_swipe = f"exit dy={dy_down:+.2f}"
                    if (
                        dy_down >= cfg.TYPING_EXIT_SWIPE_DOWN_THRESH
                        and (now - last_mode_switch_time >= cfg.MODE_SWITCH_COOLDOWN_S)
                    ):
                        mode = "browsing"
                        last_mode_switch_time = now
                        typing_exit_swipe_start_y = None
                        typing_letter_candidate = None
                        typing_letter_hold_start = None
                        typing_letter_history = []
                        action_swipe = "switched browsing"
                else:
                    typing_exit_swipe_start_y = None
            else:
                action_drag = "up"
                action_single = "idle"
                action_double = "idle"
                action_peace = "idle"
                action_swipe = "idle"
                action_typing = "off"

                # Evaluate all three pinch pairings
                p_idx_mid = pinch_strength_between(lm_px, 8, 12)  # drag hold
                p_idx_th = pinch_strength_between(lm_px, 8, 4)    # single click
                p_mid_th = pinch_strength_between(lm_px, 12, 4)   # double click
                want_down = (p_idx_mid <= drag_down_thresh) if (not drag_down) else (p_idx_mid < drag_up_thresh)
                action_drag = "down" if drag_down else ("ready" if want_down else "up")
                action_single = "ready" if p_idx_th <= click_down_thresh else "idle"
                action_double = "ready" if p_mid_th <= click_down_thresh else "idle"

                if index_extended(lm_px) or drag_down:
                    active = True
                    mouse_move(int(smoothed[0]), int(smoothed[1]), dragging=drag_down)

                # index+middle pinch: hold drag
                if (not drag_down):
                    if p_idx_mid <= drag_down_thresh:
                        if idx_mid_hold_start is None:
                            idx_mid_hold_start = now
                        if (now - idx_mid_hold_start >= drag_hold_s) and (now - last_drag_toggle_time > click_debounce_s):
                            drag_down = True
                            action_drag = "down"
                            last_drag_toggle_time = now
                            mouse_down(int(smoothed[0]), int(smoothed[1]))
                    else:
                        idx_mid_hold_start = None
                elif p_idx_mid >= drag_up_thresh:
                    drag_down = False
                    action_drag = "up"
                    last_drag_toggle_time = now
                    mouse_up(int(smoothed[0]), int(smoothed[1]))
                    idx_mid_hold_start = None

                # index+thumb pinch: single click (on pinch-in edge)
                if not idx_thumb_active:
                    if p_idx_th <= click_down_thresh:
                        if idx_th_hold_start is None:
                            idx_th_hold_start = now
                        if now - idx_th_hold_start >= click_hold_s:
                            idx_thumb_active = True
                            if now - last_idx_thumb_click_time > click_debounce_s:
                                mouse_click(int(smoothed[0]), int(smoothed[1]))
                                action_single = "triggered"
                                last_idx_thumb_click_time = now
                    else:
                        idx_th_hold_start = None
                elif idx_thumb_active and (p_idx_th >= click_up_thresh):
                    idx_thumb_active = False
                    idx_th_hold_start = None

                # middle+thumb pinch: double click (on pinch-in edge)
                if not mid_thumb_active:
                    if p_mid_th <= click_down_thresh:
                        if mid_th_hold_start is None:
                            mid_th_hold_start = now
                        if now - mid_th_hold_start >= click_hold_s:
                            mid_thumb_active = True
                            if now - last_mid_thumb_click_time > click_debounce_s:
                                mouse_double_click(int(smoothed[0]), int(smoothed[1]))
                                action_double = "triggered"
                                last_mid_thumb_click_time = now
                    else:
                        mid_th_hold_start = None
                elif mid_thumb_active and (p_mid_th >= click_up_thresh):
                    mid_thumb_active = False
                    mid_th_hold_start = None

                if peace_sign(lm_px):
                    if peace_seen_at is None:
                        peace_seen_at = time.time()
                    hold = time.time() - peace_seen_at
                    action_peace = f"hold {max(0.0, peace_hold_s - hold):.1f}s"
                    if hold >= peace_hold_s:
                        break
                else:
                    peace_seen_at = None

                thumb_out = thumb_extended(lm_px)
                four_up = four_fingers_up(lm_px) and (not thumb_out)
                cooldown_ready = (time.time() - last_space_switch_time) >= space_switch_cooldown_s
                swipe_dx = 0.0
                swipe_left_ready = False
                swipe_right_ready = False
                if four_up:
                    palm_x = palm_center_x_norm(lm_norm)
                    if four_start_x is None:
                        four_start_x = palm_x
                    swipe_dx = palm_x - four_start_x
                    swipe_left_ready = swipe_dx >= swipe_norm_thresh
                    swipe_right_ready = swipe_dx <= -swipe_norm_thresh
                    if cooldown_ready and swipe_left_ready:
                        switch_desktop("left")
                        action_swipe = "switched left"
                        last_space_switch_time = now
                        four_start_x = palm_x
                    elif cooldown_ready and swipe_right_ready:
                        switch_desktop("right")
                        action_swipe = "switched right"
                        last_space_switch_time = now
                        four_start_x = palm_x
                    elif not cooldown_ready:
                        action_swipe = "cooldown"
                    elif swipe_left_ready:
                        action_swipe = "left ready"
                    elif swipe_right_ready:
                        action_swipe = "right ready"
                    else:
                        action_swipe = f"tracking dx={swipe_dx:+.2f}"
                elif thumb_out:
                    action_swipe = "blocked (thumb out)"
                else:
                    four_start_x = None
        else:
            if drag_down:
                drag_down = False
                if smoothed is not None:
                    mouse_up(int(smoothed[0]), int(smoothed[1]))
            idx_thumb_active = False
            mid_thumb_active = False
            idx_mid_hold_start = None
            idx_th_hold_start = None
            mid_th_hold_start = None
            peace_seen_at = None
            four_start_x = None
            typing_letter_candidate = None
            typing_letter_hold_start = None
            typing_exit_swipe_start_y = None
            typing_letter_history = []

        # Left panel: action status
        left_x = 10
        left_y = 60
        left_step = 24
        cv2.putText(frame, "Actions", (left_x, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cfg.UI_TEXT_COLOR, 2)
        action_lines = [
            f"Mode: {mode}",
            f"Drag (Index+Middle): {action_drag}",
            f"Single Click (Index+Thumb): {action_single}",
            f"Double Click (Middle+Thumb): {action_double}",
            f"Typing Letter: {action_typing}",
            typing_meter_line,
            f"Peace Exit: {action_peace}",
            f"4F Desktop Swipe: {action_swipe}",
            f"Mouse Button: {'down' if drag_down else 'up'}",
        ]
        for i, line in enumerate(action_lines, start=1):
            cv2.putText(
                frame,
                line,
                (left_x, left_y + i * left_step),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                cfg.UI_TEXT_COLOR,
                2,
            )

        # Typing confidence/stability bars (active in typing mode)
        if mode == "typing":
            bar_x = left_x
            bar_y = left_y + (len(action_lines) + 1) * left_step
            bar_w = 180
            bar_h = 10
            cv2.putText(frame, "Conf", (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg.UI_TEXT_COLOR, 1)
            cv2.rectangle(frame, (bar_x + 40, bar_y - bar_h), (bar_x + 40 + bar_w, bar_y), cfg.UI_TEXT_COLOR, 1)
            fill_w = int(bar_w * max(0.0, min(1.0, typing_confidence)))
            cv2.rectangle(frame, (bar_x + 40, bar_y - bar_h), (bar_x + 40 + fill_w, bar_y), cfg.UI_TEXT_COLOR, -1)

            bar2_y = bar_y + 18
            cv2.putText(frame, "Stab", (bar_x, bar2_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg.UI_TEXT_COLOR, 1)
            cv2.rectangle(frame, (bar_x + 40, bar2_y - bar_h), (bar_x + 40 + bar_w, bar2_y), cfg.UI_TEXT_COLOR, 1)
            fill2_w = int(bar_w * max(0.0, min(1.0, typing_stability)))
            cv2.rectangle(frame, (bar_x + 40, bar2_y - bar_h), (bar_x + 40 + fill2_w, bar2_y), cfg.UI_TEXT_COLOR, -1)

        # Right panel: finger status
        right_x = max(10, w - 300)
        right_y = 60
        right_step = 24
        cv2.putText(frame, "Fingers", (right_x, right_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cfg.UI_TEXT_COLOR, 2)
        finger_lines = [
            f"Thumb: {finger_states['Thumb']}",
            f"Index: {finger_states['Index']}",
            f"Middle: {finger_states['Middle']}",
            f"Ring: {finger_states['Ring']}",
            f"Pinky: {finger_states['Pinky']}",
        ]
        for i, line in enumerate(finger_lines, start=1):
            cv2.putText(
                frame,
                line,
                (right_x, right_y + i * right_step),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                cfg.UI_TEXT_COLOR,
                2,
            )

        # UI text
        dt = time.time() - last_move_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_move_time = time.time()

        status = "ACTIVE" if active else "idle"
        cv2.putText(frame, f"{status}  FPS={fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cfg.UI_TEXT_COLOR, 2)

        cv2.imshow("Air Mouse (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
