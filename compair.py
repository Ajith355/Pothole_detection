import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = "demo.mp4"

# Put your model paths here
MODELS = {
    "New Model": "pothole_detector_v1.pt",
    "Model A": "best_new_1.pt",
    "Model B": "best _2.pt"
}
CONF_THRESHOLD = 0.4


# ---------------------

def main():
    # 1. Load all models
    loaded_models = []
    model_names = []

    # Dictionary to hold statistics: { "Model A": {'count': 0, 'sum_conf': 0.0} }
    stats = {}

    print("🚀 Loading Models...")
    for name, path in MODELS.items():
        if os.path.exists(path):
            try:
                print(f"   Load: {name}...")
                loaded_models.append(YOLO(path))
                model_names.append(name)
                stats[name] = {'count': 0, 'sum_conf': 0.0}
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
        else:
            print(f"❌ File not found: {path}")

    if not loaded_models:
        print("No models loaded. Exiting.")
        return

    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video.")
        return

    # Window Setup
    cv2.namedWindow('Statistical Comparison', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Statistical Comparison', 1400, 500)

    print("✅ Starting Comparison with Stats. Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Resize for display (fit 3 side-by-side)
        height, width = frame.shape[:2]
        new_width = 400
        new_height = int(height * (new_width / width))
        small_frame = cv2.resize(frame, (new_width, new_height))

        frame_list = []

        # 3. Run Inference for each model
        for i, model in enumerate(loaded_models):
            name = model_names[i]
            current_view = small_frame.copy()

            # Run Prediction
            results = model(current_view, verbose=False)[0]

            # --- STATISTICS LOGIC ---
            # We track detections in THIS specific frame to update the global counters
            frame_detections = 0

            for box in results.boxes:
                conf = box.conf[0].item()

                # Only count valid detections
                if conf > CONF_THRESHOLD:
                    # Update Global Stats
                    stats[name]['count'] += 1
                    stats[name]['sum_conf'] += conf

                    # Get Class Name
                    cls_id = int(box.cls[0])
                    class_name = results.names[cls_id]

                    # Draw Box (Manual draw to control color/thickness)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Color based on class
                    color = (0, 255, 0) # Default Green
                    if "muddy" in class_name.lower():
                        color = (0, 0, 255) # Red for muddy
                    elif "dry" in class_name.lower():
                        color = (0, 255, 255) # Yellow for dry

                    cv2.rectangle(current_view, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw Label
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(current_view, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- CALCULATE AVERAGE ---
            total_count = stats[name]['count']
            if total_count > 0:
                avg_conf = (stats[name]['sum_conf'] / total_count) * 100
            else:
                avg_conf = 0.0

            # --- DRAW TEXT ON SCREEN ---
            # 1. Model Name
            cv2.putText(current_view, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 2. Total Count
            cv2.putText(current_view, f"Total: {total_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 3. Average Confidence
            cv2.putText(current_view, f"Avg Conf: {avg_conf:.1f}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2)

            frame_list.append(current_view)

        # 4. Combine frames side-by-side
        if frame_list:
            combined_view = np.hstack(frame_list)
            # 5. Show
            cv2.imshow('Statistical Comparison', combined_view)

        # Slow motion control (50ms)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # --- FINAL REPORT ---
    print("\n" + "=" * 40)
    print("📊 FINAL STATISTICS REPORT")
    print("=" * 40)
    print(f"{'Model Name':<20} | {'Detections':<10} | {'Avg Confidence'}")
    print("-" * 50)

    for name in model_names:
        count = stats[name]['count']
        if count > 0:
            avg = (stats[name]['sum_conf'] / count) * 100
        else:
            avg = 0.0
        print(f"{name:<20} | {count:<10} | {avg:.2f}%")

    print("=" * 40)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()