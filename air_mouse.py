# February 14th, 2026
# Henry Houghton

import time
import math
import numpy as np
import cv2
import mediapipe as mp

from Quartz.CoreGraphics import (
    CGEventCreateMouseEvent,
    CGEventPost,
    CGWarpMouseCursorPosition,
    kCGHIDEventTap,
    kCGEventMouseMoved,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGMouseButtonLeft,
)

# macOS mouse helpers
def mouse_move(x, y):
    # to avoids some event throttling
    CGWarpMouseCursorPosition((x, y))
    # some apps respond better with an event ig
    evt = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, evt)

def mouse_down(x, y):
    evt = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, evt)

def mouse_up(x, y):
    evt = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, evt)

# gesture math stuff
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Index extended heuristic:
# index is extended if tip is farther from wrist than PIP is, by a margin,
# and not curled near palm.
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

    return (d_tip > d_pip * 1.08) and (d_tip_mcp > d_pip_mcp * 1.15)

def finger_extended(lm, mcp_idx, pip_idx, tip_idx):
    wrist = lm[0]
    mcp = lm[mcp_idx]
    pip = lm[pip_idx]
    tip = lm[tip_idx]
    d_tip = dist(tip, wrist)
    d_pip = dist(pip, wrist)
    d_tip_mcp = dist(tip, mcp)
    d_pip_mcp = dist(pip, mcp)
    return (d_tip > d_pip * 1.06) and (d_tip_mcp > d_pip_mcp * 1.10)

def peace_sign(lm):
    idx_ok = finger_extended(lm, 5, 6, 8)
    mid_ok = finger_extended(lm, 9, 10, 12)
    ring_curled = not finger_extended(lm, 13, 14, 16)
    pinky_curled = not finger_extended(lm, 17, 18, 20)
    split = dist(lm[8], lm[12]) > 0.28 * (dist(lm[0], lm[9]) + 1e-6)
    return idx_ok and mid_ok and split and ring_curled and pinky_curled

# Pinch heuristic:
# use normalized distance (thumb tip to index tip) divided by hand size.
def pinch_strength(lm):
    thumb_tip = lm[4]
    idx_tip = lm[8]

    # hand size ~ distance wrist to middle MCP (landmark 9) or index MCP (5)
    wrist = lm[0]
    mid_mcp = lm[9]
    hand_size = dist(wrist, mid_mcp) + 1e-6

    pinch = dist(thumb_tip, idx_tip) / hand_size
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
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # cursor smoothing (might need to change, higher = snappier, 
    # lower = smoother but maybe laggy)
    smoothed = None
    alpha = 0.35

    # map only within a sort of control box for stability
    # (normalized camera coords)
    box = {
        "xmin": 0.20,
        "xmax": 0.85,
        "ymin": 0.15,
        "ymax": 0.85
    }

    # Pinch click state
    pinched = False
    pinch_down_thresh = 0.20  # pinch_strength <= this -> press
    pinch_up_thresh   = 0.30  # pinch_strength >= this -> release (hysteresis)
    pinch_down_time = 0.0
    click_debounce_s = 0.08

    peace_seen_at = None
    peace_hold_s = 0.8

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
            (80, 80, 80),
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
        if chosen:
            lm_px, _, lm_norm = chosen

            # draw a few landmarks for debugging
            for idx in [0, 4, 5, 8, 9]:
                cv2.circle(frame, lm_px[idx], 5, (0, 255, 0), -1)

            # fingertip normalized coords
            ix, iy = lm_norm[8]

            # clamp to control box
            ix = (clamp(ix, box["xmin"], box["xmax"]) - box["xmin"]) / (box["xmax"] - box["xmin"])
            iy = (clamp(iy, box["ymin"], box["ymax"]) - box["ymin"]) / (box["ymax"] - box["ymin"])

            # map to screen
            # camera y increases downward; Quartz expects top-left-ish global coords,
            # so keep the same y direction to avoid inverted control.
            sx = int(ix * screen_w)
            sy = int(iy * screen_h)

            # smooth
            if smoothed is None:
                smoothed = np.array([sx, sy], dtype=np.float32)
            else:
                smoothed = alpha * np.array([sx, sy], dtype=np.float32) + (1 - alpha) * smoothed

            # pinch is always evaluated so pinch can act as true mouse press/hold.
            p = pinch_strength(lm_px)  # uses pixel lm for hand_size; good enough
            cv2.putText(frame, f"pinch={p:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if index_extended(lm_px) or pinched:
                active = True
                mouse_move(int(smoothed[0]), int(smoothed[1]))

            now = time.time()
            if (not pinched) and (p <= pinch_down_thresh) and (now - pinch_down_time > click_debounce_s):
                pinched = True
                pinch_down_time = now
                mouse_down(int(smoothed[0]), int(smoothed[1]))
            elif pinched and (p >= pinch_up_thresh):
                pinched = False
                mouse_up(int(smoothed[0]), int(smoothed[1]))

            if peace_sign(lm_px):
                if peace_seen_at is None:
                    peace_seen_at = time.time()
                hold = time.time() - peace_seen_at
                cv2.putText(frame, f"PEACE to quit: {max(0.0, peace_hold_s - hold):.1f}s", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
                if hold >= peace_hold_s:
                    break
            else:
                peace_seen_at = None
        else:
            if pinched:
                pinched = False
                if smoothed is not None:
                    mouse_up(int(smoothed[0]), int(smoothed[1]))
            peace_seen_at = None

        # UI text
        dt = time.time() - last_move_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_move_time = time.time()

        status = "ACTIVE" if active else "idle"
        cv2.putText(frame, f"{status}  FPS={fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Air Mouse (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
