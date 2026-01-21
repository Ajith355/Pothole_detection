import cv2
import numpy as np
import os
from ultralytics import YOLO
from collections import deque
import time

# --- CONFIGURATION ---
VIDEO_PATH = "demo.mp4"
MODEL_PATH = "pothole_detector_v1.pt"
CONF_THRESHOLD = 0.35  # Lowered slightly for better detection
IOU_THRESHOLD = 0.45  # NMS threshold for better box filtering
FRAME_DELAY_MS = 1  # ⚡ SPEED CONTROL: 1 = Fastest, 30 = Normal, 100 = Slow Motion

# --- OPTIMIZATION SETTINGS ---
FRAME_SKIP = 0  # Process every Nth frame (0 = no skip, 1 = every other frame)
RESIZE_INFERENCE = True  # Resize frames for faster inference
INFERENCE_SIZE = 640  # Model inference size (640, 480, or 320)
USE_HALF_PRECISION = True  # Use FP16 for faster GPU inference
ENABLE_TRACKING = True  # Track detections across frames for stability
TRACK_BUFFER_SIZE = 5  # Number of frames to track

# --- TUNING VARIABLES FOR MUDDY POTHOLES ---
# Adjust these thresholds to tune Small vs Medium vs Large
# Values are fractions of the frame width (0.0 to 1.0)
MUDDY_LARGE_THRESH = 0.25   # > 25% of width is Large
MUDDY_MEDIUM_THRESH = 0.08  # > 8% of width is Medium

# --- CAMERA CONSTANTS FOR PHYSICAL CALCULATION ---
# These assume a standard dashcam setup for "Complex Maths"
FOCAL_LENGTH_PX = 800  # Virtual focal length
CAMERA_HEIGHT_CM = 150 # Dashcam height from ground

# --- DEPTH ESTIMATION CALIBRATION ---
DRY_DEPTH_FACTOR = 0.06  # Calibration for dry pothole depth (6% of width)
MUDDY_DEPTH_FACTOR = 0.04  # Calibration for muddy pothole depth (4% of width)
SHADOW_WEIGHT = 0.3  # Weight for shadow-based depth adjustment
MAX_DEPTH_CM = 35.0  # Maximum realistic pothole depth
# ---------------------

# --- IMPORT DEPTH ESTIMATION UTILS ---
# Importing functions from the depth_estimation.py file in the same directory
try:
    import depth_estimation
    HAS_DEPTH_ESTIMATION = True
    print("✅ Successfully imported depth_estimation.py utils")
except ImportError:
    HAS_DEPTH_ESTIMATION = False
    print("⚠️ Could not import depth_estimation.py. Using basic geometric fallback.")

def get_physical_metrics(frame, box):
    """
    Complex algorithm to estimate real-world dimensions and depth.
    Uses Perspective projection heuristics with improved calibration.
    """
    x1, y1, x2, y2 = box
    h_frame, w_frame, _ = frame.shape

    # 1. Estimate Distance using Pinhole Camera Model
    # The further down the frame the 'y' is, the closer the object.

    # Horizon line estimation (typically at 45-55% of frame height for dashcam)
    horizon_ratio = 0.50
    horizon_y = h_frame * horizon_ratio

    # Ensure pothole is below horizon
    pixels_below_horizon = max(10, y2 - horizon_y)

    # Distance calculation with safety bounds
    dist_to_obj = (CAMERA_HEIGHT_CM * FOCAL_LENGTH_PX) / pixels_below_horizon

    # Clamp distance to realistic range (100cm to 2000cm for dashcam)
    dist_to_obj = np.clip(dist_to_obj, 100, 2000)

    # 2. Estimate Width in CM with perspective correction
    pixel_width = x2 - x1
    width_cm = (pixel_width * dist_to_obj) / FOCAL_LENGTH_PX

    # Sanity limits: Potholes generally aren't wider than a lane (350cm)
    width_cm = np.clip(width_cm, 5, 300)

    return dist_to_obj, width_cm

def calculate_severity(frame, box):
    """
    ADVANCED: Calculates depth for Dry Potholes using Image Processing (Shadow/Gradient).
    Optimized with better error handling and depth calibration.
    """
    x1, y1, x2, y2 = box
    height, width, _ = frame.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return "Unknown", (128, 128, 128), 0

    # Get Physical Width (Geometric)
    _, width_cm = get_physical_metrics(frame, box)

    estimated_depth = 0.0

    # --- HYBRID APPROACH: Use depth_estimation.py if available ---
    if HAS_DEPTH_ESTIMATION:
        try:
            # Prepare camera params (Approximation based on our constants)
            cam_params = {
                'f': FOCAL_LENGTH_PX,
                'cx': width / 2,
                'cy': height / 2,
                'H': CAMERA_HEIGHT_CM / 100.0, # Convert cm to meters
                'pitch': 0.0
            }

            # 1. Detect Scene Condition
            cond = depth_estimation.detect_wet_muddy(roi)

            # 2. Get relative depth map
            depth_rel = depth_estimation.run_midas_depth(roi)

            # 3. Refine with segmentation
            mask = depth_estimation.simple_pothole_segmentation(roi)
            if mask.sum() > 50:  # Minimum valid mask area
                depth_refined = depth_estimation.refine_depth(depth_rel, mask, cond['spec_mask'])

                # 4. Calculate Relative Depth (0.0 to 1.0)
                rim_vals = depth_refined[mask==0]
                if rim_vals.size == 0:
                    rim_vals = depth_refined

                rim_median = np.median(rim_vals)
                interior_vals = depth_refined[mask>0]
                bottom = np.min(interior_vals) if interior_vals.size > 0 else rim_median

                rel_depth_val = max(0.0, rim_median - bottom)

                # 5. Convert to CM with calibrated factor (reduced from 2.0 to 0.5)
                estimated_depth = width_cm * rel_depth_val * 0.5
            else:
                estimated_depth = 0.0

        except Exception as e:
            # Silently fall back to geometric method
            estimated_depth = 0.0

    # --- FALLBACK / COMBINATION LOGIC ---
    # Complex Image Processing for Depth Estimation (Geometric/Shadow Logic)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    _, dark_mask = cv2.threshold(blurred, np.mean(blurred) * 0.7, 255, cv2.THRESH_BINARY_INV)
    shadow_area = np.sum(dark_mask > 0)
    total_area = roi.shape[0] * roi.shape[1]
    shadow_ratio = shadow_area / total_area

    # Geometric depth with calibrated factor (3% base + shadow influence)
    geometric_depth = width_cm * 0.03 * (1 + shadow_ratio * 0.5)

    # Weighted Average: Combine advanced and geometric methods
    if estimated_depth > 0.1:
        final_depth = (estimated_depth * 0.6) + (geometric_depth * 0.4)
    else:
        final_depth = geometric_depth

    # Hard Limit to realistic range
    final_depth = np.clip(final_depth, 0.5, MAX_DEPTH_CM)

    # Severity classification with better thresholds
    if final_depth > 15: # Very Deep
        return f"CRITICAL ({final_depth:.1f}cm)", (0, 0, 255), final_depth # Red
    elif final_depth > 10: # Deep
        return f"DANGEROUS ({final_depth:.1f}cm)", (0, 69, 255), final_depth # Orange-Red
    elif final_depth > 6: # Moderate
        return f"MODERATE ({final_depth:.1f}cm)", (0, 165, 255), final_depth # Orange
    elif final_depth > 3: # Shallow
        return f"MINOR ({final_depth:.1f}cm)", (0, 255, 255), final_depth # Yellow
    else: # Very Shallow
        return f"SURFACE ({final_depth:.1f}cm)", (0, 255, 0), final_depth # Green

def analyze_muddy_pothole(frame, box):
    """
    ADVANCED: Analyzes Muddy Potholes.
    Since depth is invisible, we use volumetric estimation based on surface footprint.
    Optimized with better calibration.
    """
    x1, y1, x2, y2 = box
    _, width_cm = get_physical_metrics(frame, box)

    # Muddy depth heuristic: Most potholes follow a semi-elliptical cavity
    # We estimate depth based on surface area with aspect ratio consideration
    height_px = y2 - y1
    width_px = x2 - x1
    aspect_ratio = height_px / max(width_px, 1)  # Avoid division by zero

    # Depth estimation with calibrated factor (3% base with aspect ratio adjustment)
    estimated_depth = width_cm * MUDDY_DEPTH_FACTOR * (1 + aspect_ratio * 0.3)

    # Hard Limit to realistic range
    estimated_depth = np.clip(estimated_depth, 1.0, MAX_DEPTH_CM)

    # Severity classification
    if estimated_depth > 15:
        return f"DEEP MUD ({estimated_depth:.1f}cm)", (0, 0, 255) # Red
    elif estimated_depth > 10:
        return f"DANGEROUS ({estimated_depth:.1f}cm)", (0, 69, 255) # Orange-Red
    elif estimated_depth > 6:
        return f"MED MUD ({estimated_depth:.1f}cm)", (0, 165, 255) # Orange
    elif estimated_depth > 3:
        return f"SHALLOW ({estimated_depth:.1f}cm)", (0, 255, 255) # Yellow
    else:
        return f"MINOR ({estimated_depth:.1f}cm)", (0, 255, 0) # Green


# --- DETECTION TRACKING AND SMOOTHING ---
class DetectionTracker:
    """Tracks detections across frames for temporal smoothing and filtering"""
    def __init__(self, buffer_size=5, iou_threshold=0.3):
        self.buffer_size = buffer_size
        self.iou_threshold = iou_threshold
        self.tracked_objects = {}  # {id: {'boxes': deque, 'labels': deque, 'colors': deque}}
        self.next_id = 0

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of (box, label, color) tuples
        Returns: list of smoothed (box, label, color) tuples
        """
        if not detections:
            return []

        # Match new detections to existing tracks
        matched_tracks = set()
        smoothed_detections = []

        for box, label, color in detections:
            best_match_id = None
            best_iou = 0

            # Find best matching track
            for track_id, track_data in self.tracked_objects.items():
                if len(track_data['boxes']) > 0:
                    last_box = track_data['boxes'][-1]
                    iou = self.calculate_iou(box, last_box)

                    if iou > self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_match_id = track_id

            # Update or create track
            if best_match_id is not None:
                # Update existing track
                track = self.tracked_objects[best_match_id]
                track['boxes'].append(box)
                track['labels'].append(label)
                track['colors'].append(color)

                # Maintain buffer size
                if len(track['boxes']) > self.buffer_size:
                    track['boxes'].popleft()
                    track['labels'].popleft()
                    track['colors'].popleft()

                matched_tracks.add(best_match_id)

                # Smooth box coordinates (average of recent boxes)
                boxes_array = np.array(track['boxes'])
                smoothed_box = tuple(np.mean(boxes_array, axis=0).astype(int))

                # Use most recent label and color
                smoothed_detections.append((smoothed_box, label, color))
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1
                self.tracked_objects[new_id] = {
                    'boxes': deque([box], maxlen=self.buffer_size),
                    'labels': deque([label], maxlen=self.buffer_size),
                    'colors': deque([color], maxlen=self.buffer_size)
                }
                matched_tracks.add(new_id)
                smoothed_detections.append((box, label, color))

        # Remove stale tracks (not matched for several frames)
        stale_ids = [tid for tid in self.tracked_objects.keys() if tid not in matched_tracks]
        for tid in stale_ids:
            del self.tracked_objects[tid]

        return smoothed_detections

def main():
    print(f"📂 Current Working Directory: {os.getcwd()}")

    # Check paths
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VIDEO_PATH):
        print(f"❌ CRITICAL ERROR: Check your '{MODEL_PATH}' or '{VIDEO_PATH}' paths.")
        return

    print(f"🚀 Loading Model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)

        # Optimize model for inference
        model.fuse()  # Fuse Conv2d + BatchNorm layers for faster inference

        # Set model parameters for better performance
        if USE_HALF_PRECISION and model.device.type != 'cpu':
            model.half()  # Use FP16 precision on GPU
            print("✅ Using FP16 precision for faster inference")

    except Exception as e:
        print(f"❌ Error loading YOLO: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✅ Video loaded: {total_frames} frames @ {fps} FPS")
    print(f"⚙️ Processing with {FRAME_DELAY_MS}ms delay, Frame Skip: {FRAME_SKIP}")

    # Initialize tracker if enabled
    tracker = DetectionTracker(buffer_size=TRACK_BUFFER_SIZE) if ENABLE_TRACKING else None

    cv2.namedWindow('Pothole Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pothole Detector', 1024, 768)

    frame_count = 0
    inference_times = deque(maxlen=30)  # Track FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ℹ️ End of video reached.")
            break

        frame_count += 1

        # Frame skipping for performance
        if FRAME_SKIP > 0 and frame_count % (FRAME_SKIP + 1) != 0:
            cv2.imshow('Pothole Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Start timing
        start_time = time.time()

        # Prepare frame for inference
        inference_frame = frame
        if RESIZE_INFERENCE and frame.shape[1] > INFERENCE_SIZE:
            scale = INFERENCE_SIZE / frame.shape[1]
            inference_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Inference with optimized parameters
        results = model(
            inference_frame,
            verbose=False,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            agnostic_nms=True,  # Class-agnostic NMS for better filtering
            max_det=10  # Limit max detections for performance
        )[0]

        # Scale boxes back if resized
        scale_factor = 1.0
        if RESIZE_INFERENCE and frame.shape[1] > INFERENCE_SIZE:
            scale_factor = frame.shape[1] / INFERENCE_SIZE

        # Collect detections
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0] * scale_factor)
            conf = box.conf[0].item()
            
            # Get Class Name
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]

            if conf > CONF_THRESHOLD:
                # Logic based on class
                if "muddy" in class_name.lower():
                    label_text, color = analyze_muddy_pothole(frame, (x1, y1, x2, y2))
                    full_label = f"MUD: {label_text}"
                else:
                    severity, color, depth_val = calculate_severity(frame, (x1, y1, x2, y2))
                    full_label = f"DRY: {severity}"

                detections.append(((x1, y1, x2, y2), full_label, color))

        # Apply tracking if enabled
        if ENABLE_TRACKING and tracker:
            detections = tracker.update(detections)

        # Draw detections
        for (x1, y1, x2, y2), full_label, color in detections:
            # Draw Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw Label Background
            (w, h), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)

            # Determine text color based on background color brightness
            if color == (0, 255, 255) or color == (0, 255, 0):
                text_color = (0, 0, 0) # Black
            else:
                text_color = (255, 255, 255) # White

            # Draw Text
            cv2.putText(frame, full_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Calculate and display FPS
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        avg_fps = 1.0 / (np.mean(inference_times) + 1e-6)

        # Draw FPS counter
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Pothole Detector', frame)

        # --- CONTROLS ---
        key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF

        if key == ord('q'):
            print("🛑 User pressed Q. Exiting...")
            break
        elif key == ord(' '): # Spacebar to Pause
            print("⏸️ Paused. Press Space to resume.")

            # Draw "PAUSED" text on the frame while paused
            pause_frame = frame.copy()
            cv2.putText(pause_frame, "PAUSED", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow('Pothole Detector', pause_frame)

            # Wait loop until space is pressed again
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord(' '): # Resume
                    print("▶️ Resuming...")
                    break
                elif key2 == ord('q'): # Quit while paused
                    print("🛑 User pressed Q. Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    # --- PAUSE AT END ---
    print("✅ Done. Press any key to close the window.")
    cv2.waitKey(0)  # Waits indefinitely until you press a key

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()