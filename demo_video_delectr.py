import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = "demo.mp4"
MODEL_PATH = "best _2.pt"
CONF_THRESHOLD = 0.4


# ---------------------

def calculate_severity(frame, box):
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
        return "HIGH SEVERITY", (0, 0, 255)
    elif relative_size > 0.005:
        return "MODERATE", (0, 165, 255)
    else:
        return "LOW", (0, 255, 0)


def main():
    print(f"📂 Current Working Directory: {os.getcwd()}")

    # Check paths
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VIDEO_PATH):
        print("❌ CRITICAL ERROR: Check your 'best.pt' or 'demo.mp4' paths.")
        return

    print(f"🚀 Loading Model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading YOLO: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    print("✅ Video loaded. Playing in SLOW MOTION...")

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

            if conf > CONF_THRESHOLD:
                severity, color = calculate_severity(frame, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{severity}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Pothole Detector', frame)

        # --- SLOW MOTION CONTROL ---
        # 100ms delay = approx 10 FPS (Slow Motion)
        # Press 'q' to quit early
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("🛑 User pressed Q. Exiting...")
            break

    # --- PAUSE AT END ---
    print("✅ Done. Press any key to close the window.")
    cv2.waitKey(0)  # Waits indefinitely until you press a key

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()