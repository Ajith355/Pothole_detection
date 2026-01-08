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


# ---------------------

def calculate_severity(frame, box):
    """
    Calculates severity for Dry Potholes based on size and brightness (depth cue).
    """
    x1, y1, x2, y2 = box
    height, width, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return "Unknown", (128, 128, 128)

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_roi)

    pothole_area = (x2 - x1) * (y2 - y1)
    frame_area = width * height
    relative_size = pothole_area / frame_area

    if relative_size > 0.015 or avg_brightness < 75:
        return "HIGH SEVERITY", (0, 0, 255) # Red
    elif relative_size > 0.005:
        return "MODERATE", (0, 165, 255) # Orange
    else:
        return "LOW", (0, 255, 0) # Green


def analyze_muddy_pothole(frame, box):
    """
    Analyzes Muddy Potholes based on Risk Area (Width coverage).
    Refractive index math is invalid for opaque muddy water.
    """
    x1, y1, x2, y2 = box
    height, width, _ = frame.shape
    
    # Calculate dimensions
    pothole_width = x2 - x1
    
    # Estimate Lane Width (Assuming camera captures mostly the lane, or using frame width as proxy)
    # In a real scenario, you'd use lane detection. Here we use frame width.
    lane_width = width 
    
    # Calculate Risk Score (Fraction of lane width occupied)
    risk_score = pothole_width / lane_width
    
    # Convert to percentage for display
    risk_pct = int(risk_score * 100)
    
    # --- NEW LOGIC: Small, Medium, Large + Approximate Value (Size %) ---
    if risk_score > MUDDY_LARGE_THRESH: 
        return f"Large Hazard (Size: {risk_pct}%)", (0, 0, 255) # Red
    elif risk_score > MUDDY_MEDIUM_THRESH: 
        return f"Medium Hazard (Size: {risk_pct}%)", (0, 165, 255) # Orange
    else:
        return f"Small Hazard (Size: {risk_pct}%)", (0, 255, 255) # Yellow


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
                    # Append class name to label
                    full_label = f"Muddy: {label_text}"
                else:
                    # Assume Dry Pothole
                    severity, color = calculate_severity(frame, (x1, y1, x2, y2))
                    full_label = f"Dry: {severity}"

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

        # --- SPEED CONTROL ---
        # Uses FRAME_DELAY_MS from configuration
        if cv2.waitKey(FRAME_DELAY_MS) & 0xFF == ord('q'):
            print("🛑 User pressed Q. Exiting...")
            break

    # --- PAUSE AT END ---
    print("✅ Done. Press any key to close the window.")
    cv2.waitKey(0)  # Waits indefinitely until you press a key

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()