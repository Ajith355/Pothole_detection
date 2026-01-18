import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = "demo.mp4"
MODEL_PATH = "pothole_detector_v1.pt"
CONF_THRESHOLD = 0.4
FRAME_DELAY_MS = 1  # ⚡ SPEED CONTROL: 1 = Fastest, 30 = Normal, 100 = Slow Motion

# --- TUNING VARIABLES FOR MUDDY POTHOLES ---
# Adjust these thresholds to tune Small vs Medium vs Large
# Values are fractions of the frame width (0.0 to 1.0)
MUDDY_LARGE_THRESH = 0.30   # > 30% of width is Large
MUDDY_MEDIUM_THRESH = 0.10  # > 10% of width is Medium (Lowered from 0.15 to catch more medium ones)

# --- CAMERA CONSTANTS FOR PHYSICAL CALCULATION ---
# These assume a standard dashcam setup for "Complex Maths"
FOCAL_LENGTH_PX = 800  # Virtual focal length
CAMERA_HEIGHT_CM = 150 # Dashcam height from ground
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
    Uses Perspective projection heuristics.
    """
    x1, y1, x2, y2 = box
    h_frame, w_frame, _ = frame.shape

    # 1. Estimate Distance using Pinhole Camera Model
    # Assuming the bottom of the pothole is on the ground plane.
    # The further down the frame the 'y' is, the closer the object.

    # SAFETY: Ensure we don't divide by zero if object is at or above horizon line
    # We assume horizon is roughly at 50% height (h_frame / 2)
    pixels_below_horizon = max(10, y2 - (h_frame * 0.53)) # Adjusted horizon slightly down

    dist_to_obj = (CAMERA_HEIGHT_CM * FOCAL_LENGTH_PX) / pixels_below_horizon

    # 2. Estimate Width in CM
    pixel_width = x2 - x1
    width_cm = (pixel_width * dist_to_obj) / FOCAL_LENGTH_PX

    # Sanity limits: Potholes generally aren't wider than a lane (350cm)
    width_cm = min(width_cm, 300)

    return dist_to_obj, width_cm

def calculate_severity(frame, box):
    """
    ADVANCED: Calculates depth for Dry Potholes using Image Processing (Shadow/Gradient).
    """
    x1, y1, x2, y2 = box
    height, width, _ = frame.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return "Unknown", (128, 128, 128), 0

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
                'H': CAMERA_HEIGHT_CM / 100.0, # Convert cm to meters for the depth module
                'pitch': 0.0
            }

            # The depth module expects a path, but we have a frame in memory.
            # We can refactor or just save a temp file for now to reuse the exact function,
            # OR we can call the internal functions directly if we pass the array.
            # Here we will adapt the logic from estimate_pothole_depth but for an in-memory ROI.

            # 1. Detect Scene Condition
            cond = depth_estimation.detect_wet_muddy(roi)

            # 2. Get relative depth map (Simple gradient fallback in current depth_estimation file)
            depth_rel = depth_estimation.run_midas_depth(roi)

            # 3. Refine
            # We need a mask for the ROI. We can use the simple segmenter from the module
            mask = depth_estimation.simple_pothole_segmentation(roi)
            if mask.sum() > 0:
                depth_refined = depth_estimation.refine_depth(depth_rel, mask, cond['spec_mask'])

                # 4. Calculate Relative Depth (0.0 to 1.0)
                # Rim vs Bottom logic from the module
                rim_vals = depth_refined[mask==0] # approximate rim as outside mask in ROI or edge
                if rim_vals.size == 0: rim_vals = depth_refined

                rim_median = np.median(rim_vals)
                interior_vals = depth_refined[mask>0]
                bottom = np.min(interior_vals) if interior_vals.size > 0 else rim_median

                rel_depth_val = max(0.0, rim_median - bottom)

                # 5. Convert to CM using our geometric width as a scaler
                # If width is W, and relative depth is D_rel, we estimate:
                # Real Depth ≈ Width * (RelDepth / RelWidth)
                # Simplified: Depth = Width * RelDepth * Factor
                # MODIFIED: Reduced Calibration factor from 2.0 to 0.8
                estimated_depth = width_cm * rel_depth_val * 0.8
            else:
                # Fallback if segmentation fails in module
                estimated_depth = 0.0

        except Exception as e:
            print(f"Depth module error: {e}")
            estimated_depth = 0.0

    # --- FALLBACK / COMBINATION LOGIC ---
    # If the module failed or returned 0, or we just want to average with our fast geometric method:

    # Complex Image Processing for Depth Estimation (Original Geometric/Shadow Logic)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    _, dark_mask = cv2.threshold(blurred, np.mean(blurred) * 0.7, 255, cv2.THRESH_BINARY_INV)
    shadow_area = np.sum(dark_mask > 0)
    total_area = roi.shape[0] * roi.shape[1]
    shadow_ratio = shadow_area / total_area

    # MODIFIED: Reduced multiplier from 0.15 to 0.05
    # This assumes depth is approx 5% of width + shadow cues, yielding 5-15cm depths
    geometric_depth = (width_cm * 0.05) * (shadow_ratio + 0.2)

    # Weighted Average: Give more weight to the advanced module if it worked
    if estimated_depth > 0.1:
        final_depth = (estimated_depth * 0.5) + (geometric_depth * 0.5) # Balanced weight
    else:
        final_depth = geometric_depth

    # Hard Limit
    final_depth = min(final_depth, 40.0)

    if final_depth > 12: # Deep
        return f"CRITICAL ({final_depth:.1f}cm)", (0, 0, 255), final_depth # Red
    elif final_depth > 8: # Medium-Deep
        return f"DANGEROUS ({final_depth:.1f}cm)", (0, 69, 255), final_depth # Orange-Red
    elif final_depth > 4: # Moderate
        return f"MODERATE ({final_depth:.1f}cm)", (0, 165, 255), final_depth # Orange
    elif final_depth > 2: # Shallow
        return f"MINOR ({final_depth:.1f}cm)", (0, 255, 255), final_depth # Yellow
    else: # Very Shallow / Surface
        return f"SURFACE ({final_depth:.1f}cm)", (0, 255, 0), final_depth # Green

def analyze_muddy_pothole(frame, box):
    """
    ADVANCED: Analyzes Muddy Potholes.
    Since depth is invisible, we use volumetric estimation based on surface footprint.
    """
    x1, y1, x2, y2 = box
    _, width_cm = get_physical_metrics(frame, box)

    # Muddy depth heuristic: Most potholes follow a semi-elliptical cavity
    # We estimate depth as approx 1/3 to 1/2 of the shortest radius to be safe
    height_px = y2 - y1
    width_px = x2 - x1
    aspect_ratio = height_px / width_px

    # Depth = width_cm * shape_factor (0.2 to 0.5 for typical road holes)
    # REDUCED FACTOR: Reduced from 0.10 to 0.05 for more realistic estimates
    estimated_depth = width_cm * 0.05 * (1 + aspect_ratio)

    # Hard Limit
    estimated_depth = min(estimated_depth, 45.0)

    if estimated_depth > 12:
        return f"DEEP MUD ({estimated_depth:.1f}cm)", (0, 0, 255) # Red
    elif estimated_depth > 8:
        return f"DANGEROUS ({estimated_depth:.1f}cm)", (0, 69, 255) # Orange-Red
    elif estimated_depth > 5:
        return f"MED MUD ({estimated_depth:.1f}cm)", (0, 165, 255) # Orange
    else:
        return f"SHALLOW ({estimated_depth:.1f}cm)", (0, 255, 255) # Yellow


# ---------------------

def main():
    print(f"📂 Current Working Directory: {os.getcwd()}")

    # Check paths
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VIDEO_PATH):
        print(f"❌ CRITICAL ERROR: Check your '{MODEL_PATH}' or '{VIDEO_PATH}' paths.")
        return

    print(f"🚀 Loading Model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading YOLO: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    print(f"✅ Video loaded. Playing with {FRAME_DELAY_MS}ms delay...")

    cv2.namedWindow('Pothole Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pothole Detector', 1024, 768)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ℹ️ End of video reached.")
            break

        # Inference
        results = model(frame, verbose=False)[0]

        # Drawing
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
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

                # Draw Rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw Label Background
                (w, h), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)
                
                # Determine text color based on background color brightness
                # Yellow (0, 255, 255) and Green (0, 255, 0) are bright, so use Black text
                if color == (0, 255, 255) or color == (0, 255, 0):
                    text_color = (0, 0, 0) # Black
                else:
                    text_color = (255, 255, 255) # White

                # Draw Text
                cv2.putText(frame, full_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        cv2.imshow('Pothole Detector', frame)

        # --- CONTROLS ---
        # Uses FRAME_DELAY_MS from configuration
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