# Quick Configuration Presets
# Copy these settings to demo_video_delectr.py to apply different optimization modes

# ===========================================
# PRESET 1: MAXIMUM ACCURACY (SLOW)
# ===========================================
# Use when accuracy is critical, speed is not important
"""
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
FRAME_SKIP = 0
RESIZE_INFERENCE = False
INFERENCE_SIZE = 640
USE_HALF_PRECISION = False
ENABLE_TRACKING = True
TRACK_BUFFER_SIZE = 7
"""

# ===========================================
# PRESET 2: BALANCED (RECOMMENDED)
# ===========================================
# Best balance between speed and accuracy
"""
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
FRAME_SKIP = 0
RESIZE_INFERENCE = True
INFERENCE_SIZE = 640
USE_HALF_PRECISION = True
ENABLE_TRACKING = True
TRACK_BUFFER_SIZE = 5
"""

# ===========================================
# PRESET 3: HIGH SPEED (REAL-TIME)
# ===========================================
# For real-time applications, lower accuracy acceptable
"""
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
FRAME_SKIP = 1
RESIZE_INFERENCE = True
INFERENCE_SIZE = 480
USE_HALF_PRECISION = True
ENABLE_TRACKING = True
TRACK_BUFFER_SIZE = 3
"""

# ===========================================
# PRESET 4: MAXIMUM SPEED (FAST)
# ===========================================
# Fastest processing, minimum accuracy
"""
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.55
FRAME_SKIP = 2
RESIZE_INFERENCE = True
INFERENCE_SIZE = 320
USE_HALF_PRECISION = True
ENABLE_TRACKING = False
TRACK_BUFFER_SIZE = 3
"""

# ===========================================
# PRESET 5: CPU OPTIMIZED (NO GPU)
# ===========================================
# Best settings when GPU is not available
"""
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
FRAME_SKIP = 1
RESIZE_INFERENCE = True
INFERENCE_SIZE = 416
USE_HALF_PRECISION = False
ENABLE_TRACKING = True
TRACK_BUFFER_SIZE = 3
"""

# ===========================================
# CALIBRATION PARAMETERS
# ===========================================
# Adjust these based on your camera setup

# Camera Parameters (measure your actual setup)
"""
FOCAL_LENGTH_PX = 800      # Typical: 600-1000 for dashcams
CAMERA_HEIGHT_CM = 150     # Measure from ground to camera lens
"""

# Depth Estimation Factors (tune with real measurements)
"""
DRY_DEPTH_FACTOR = 0.06    # Increase if depths too shallow
MUDDY_DEPTH_FACTOR = 0.04  # Increase if depths too shallow
MAX_DEPTH_CM = 35.0        # Maximum realistic pothole depth
"""

# Detection Thresholds
"""
MUDDY_LARGE_THRESH = 0.25  # >25% of frame width = Large
MUDDY_MEDIUM_THRESH = 0.08 # >8% of frame width = Medium
"""

# ===========================================
# TROUBLESHOOTING GUIDE
# ===========================================

# Problem: Low FPS (< 10)
# Solution: Use PRESET 3 or 4, enable FRAME_SKIP=1

# Problem: Missing small potholes
# Solution: Lower CONF_THRESHOLD to 0.25-0.3, use PRESET 1

# Problem: Jittery detections
# Solution: Enable TRACKING, increase TRACK_BUFFER_SIZE to 7

# Problem: Too many false positives
# Solution: Increase CONF_THRESHOLD to 0.4-0.5

# Problem: Inaccurate depth measurements
# Solution: Calibrate CAMERA_HEIGHT_CM and FOCAL_LENGTH_PX with real measurements

# Problem: GPU not being used
# Solution: Check CUDA installation, set USE_HALF_PRECISION=True

# Problem: Out of memory errors
# Solution: Reduce INFERENCE_SIZE to 416 or 320

# ===========================================
# ADVANCED TUNING
# ===========================================

# Fine-tune NMS (Non-Maximum Suppression)
"""
IOU_THRESHOLD = 0.45   # Lower (0.3-0.4) = more aggressive filtering
                        # Higher (0.5-0.6) = keep more overlapping boxes
"""

# Fine-tune Tracking
"""
TRACK_BUFFER_SIZE = 5       # Smoothing window (3-7 recommended)
tracker.iou_threshold = 0.3 # Detection matching threshold
"""

# Frame Processing
"""
FRAME_SKIP = 0  # 0=every frame, 1=every 2nd, 2=every 3rd
FRAME_DELAY_MS = 1  # Display delay (1=fast, 30=normal speed)
"""
