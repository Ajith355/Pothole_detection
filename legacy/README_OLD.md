q# Pothole Deep Vision 🕳️🚗

An advanced computer vision system for real-time pothole detection and risk assessment. This project combines state-or-the-art **Object Detection (YOLO)** with **Hybrid Depth Estimation** algorithms (Geometric Physics + Monocular Depth Transformers) to accurately gauge the severity of road hazards.

## 🌟 Features

*   **Real-time Detection:** Identifies potholes instantly in video feeds.
*   **Hybrid Depth Engine:** Fuses geometric perspective mathematics with Deep Learning (MiDaS) to estimate depth in centimeters.
*   **Material Classification:** Distinguishes between **Dry** and **Muddy/Wet** potholes.
*   **Severity Grading:** Color-coded risk assessment (Surface, Minor, Moderate, Dangerous, Critical).
*   **Playback Controls:** Pause/Resume/Quit controls for video analysis.

---

## 🧠 How It Works

The system operates in a multi-stage pipeline for every video frame:

### 1. Object Detection (YOLOv8)
The frame is passed through a custom-trained **YOLO (You Only Look Once)** model (`pothole_detector_v1.pt`). This neural network detects objects and classifies them as either:
*   `pothole` (Dry)
*   `muddy_pothole` (Wet/Filled with water)

### 2. The Hybrid Depth Estimation Algorithm 📏

Once a pothole is detected, its bounding box is sent to a complex depth calculator that uses two simultaneous methods for maximum accuracy:

#### A. Geometric Physics (The "Pinhole" Model)
*   **Distance Estimation:** By analyzing where the pothole appears vertically in the frame (Y-axis), the system calculates how far away it is using perspective projection logic.
*   **Physical Width:** Using the estimated distance and the camera's focal length, it converts the pixel width of the pothole into real-world centimeters.
*   **Base Depth:** A heuristic formula estimates depth based on the width (assuming a typical width-to-depth ratio for road erosion).

#### B. AI Monocular Depth (MiDaS / DPT)
*   The Region of Interest (ROI) is fed into the **MiDaS (Monocular Depth Estimation in the Wild)** neural network (specifically `DPT_Large` or `DPT_Hybrid`).
*   This Transformer-based model analyzes shadows, lighting gradients, and texture to generate a dense **Depth Map**.
*   **CLAHE Enhancement:** This map is post-processed with Contrast Limited Adaptive Histogram Equalization to sharpen the details of the pothole's cavity.
*   **Relative Scoring:** The system compares the depth values at the "rim" of the pothole versus the "bottom" to calculate a precise relative depth factor.

#### C. The Fusion Strategy
The final depth in centimeters is a **Weighted Average** of the Geometric result and the AI result.
> `Final Depth = (AI_Depth * 0.5) + (Geometric_Depth * 0.5)`

This ensures that if the AI hallucinates, the physics keeps it grounded, and if the physics is off due to camera angle, the AI corrects it based on visual appearance.

### 3. Muddy Pothole Analysis 🌊
Since muddy water is opaque, optical depth estimation is impossible. The system switches to a **Volumetric Risk Heuristic**:
*   It analyzes the **Surface Area** and **Aspect Ratio** of the water puddle.
*   It assumes that larger, rounder puddles hide deeper cavities (following standard road erosion patterns).

---

## 🎨 Severity Classification

The system assigns a risk level based on the calculated depth:

| Severity | Depth Range | Color Code | Description |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | > 12 cm | 🔴 **Red** | High risk of tire/rim damage or accident. |
| **DANGEROUS**| > 8 cm | 🟠 **Orange-Red** | Significant hazard; requires immediate caution. |
| **MODERATE** | > 4 cm | 🟠 **Orange** | Standard pothole; noticeable bump. |
| **MINOR** | > 2 cm | 🟡 **Yellow** | Small hole; minor ride discomfort. |
| **SURFACE** | < 2 cm | 🟢 **Green** | Shallow unevenness; broadly safe. |

---

## 🎮 Controls

While the video is playing:
*   `SPACEBAR` : **Pause / Resume** video.
*   `Q` : **Quit** the application.

---

## 🛠️ Requirements

*   Python 3.8+
*   OpenCV (`opencv-python`)
*   Ultralytics YOLO (`ultralytics`)
*   PyTorch (`torch`, `torchvision`)
*   Timm (`timm` - for the Transformer backbone)

Install dependencies:
```bash
pip install opencv-python ultralytics torch torchvision timm
```

## �� Running the Project

1.  Ensure your video file is named `demo.mp4` and placed in the project folder.
2.  Run the main script:
    ```bash
    python demo_video_delectr.py
    ```
3.  The first run will download the Depth Estimation models (approx. 1GB).


