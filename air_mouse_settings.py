"""
Tunable parameters for air_mouse.py.

Might need to make some training/testing data to train this
Also definitely need to update UI
"""

# MediaPipe hand tracking
MP_MAX_NUM_HANDS = 2
MP_MODEL_COMPLEXITY = 1
MP_MIN_DETECTION_CONFIDENCE = 0.6
MP_MIN_TRACKING_CONFIDENCE = 0.6

# Cursor control
CURSOR_SMOOTHING_ALPHA = 0.35
CONTROL_BOX = {
    "xmin": 0.23,
    "xmax": 0.82,
    "ymin": 0.18,
    "ymax": 0.82,
}

# Click/drag timing
DOUBLE_CLICK_GAP_S = 0.03
CLICK_DEBOUNCE_S = 0.12

# Gesture thresholds
DRAG_DOWN_THRESH = 0.22
DRAG_UP_THRESH = 0.32
CLICK_DOWN_THRESH = 0.20
CLICK_UP_THRESH = 0.30

# Four-finger desktop swipe
SPACE_SWITCH_COOLDOWN_S = 1.0
SWIPE_NORM_THRESH = 0.06

# Peace sign quit
PEACE_HOLD_S = 0.8
PEACE_SPLIT_RATIO = 0.28

# Finger extension heuristics
INDEX_EXT_WRIST_RATIO = 1.08
INDEX_EXT_MCP_RATIO = 1.15
FINGER_EXT_WRIST_RATIO = 1.06
FINGER_EXT_MCP_RATIO = 1.10
THUMB_EXT_WRIST_RATIO = 1.03
THUMB_EXT_MCP_RATIO = 1.06

# UI
UI_TEXT_COLOR = (255, 255, 255)
UI_BOX_COLOR = (80, 80, 80)
UI_MIDDLE_TRACKER_COLOR = (255, 200, 0)
