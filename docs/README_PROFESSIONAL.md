# Pothole Detection System

A professional, modular pothole detection and analysis system using YOLO object detection combined with depth estimation for severity classification.

## 🌟 Features

- **Real-time Detection**: Fast YOLO-based pothole detection
- **Temporal Tracking**: IoU-based object tracking for stable detections
- **Depth Estimation**: Hybrid neural + geometric depth estimation
- **Severity Classification**: Automatic severity level assignment (Critical/Dangerous/Moderate/Minor/Surface)
- **Performance Optimized**: Multiple optimization strategies for different hardware
- **Modular Architecture**: Clean, professional code structure
- **Configurable**: Extensive configuration options and presets

## 📋 Requirements

```
python >= 3.8
opencv-python >= 4.5.0
numpy >= 1.19.0
ultralytics >= 8.0.0
torch >= 1.10.0
torchvision >= 0.11.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Pothole_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download or place your YOLO model file in the project directory

## 💻 Usage

### Basic Usage

Run with default settings:
```bash
python main.py
```

### Advanced Usage

```bash
# Use specific video and model
python main.py --video input.mp4 --model best.pt

# Use speed preset for real-time processing
python main.py --preset speed

# Use accuracy preset for best results
python main.py --preset accuracy

# Custom configuration
python main.py --conf-threshold 0.3 --frame-skip 1 --inference-size 480

# Disable tracking
python main.py --no-tracking
```

### Command Line Arguments

```
--video, -v          Path to input video file
--model, -m          Path to YOLO model file
--preset, -p         Configuration preset (accuracy/balanced/speed/cpu)
--no-tracking        Disable temporal tracking
--conf-threshold     Confidence threshold (0.0-1.0)
--frame-skip         Skip every N frames
--inference-size     Model inference size (320/416/480/640)
```

## ⚙️ Configuration Presets

### Accuracy Mode
Best detection quality, slower processing
- Confidence: 0.25
- Frame Skip: 0 (process all frames)
- Full resolution inference
- Extended tracking buffer

### Balanced Mode (Default)
Good balance of speed and accuracy
- Confidence: 0.35
- Frame Skip: 0
- Resized inference (640px)
- Standard tracking

### Speed Mode
Fast processing, acceptable accuracy
- Confidence: 0.4
- Frame Skip: 1 (every other frame)
- Resized inference (480px)
- Reduced tracking buffer

### CPU Mode
Optimized for systems without GPU
- Confidence: 0.4
- FP16 disabled
- Reduced inference size (416px)
- Frame skipping enabled

## 📁 Project Structure

```
Pothole_detection/
├── main.py                    # Main application entry point
├── config.py                  # Configuration management
├── detector.py                # Pothole detection pipeline
├── video_processor.py         # Video input/output handling
├── tracker.py                 # Detection tracking module
├── severity_estimator.py      # Depth estimation and classification
├── utils.py                   # Utility functions
├── depth_estimation.py        # Advanced depth estimation (MiDaS)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎯 Module Overview

### `main.py`
Application entry point with argument parsing and orchestration.

### `config.py`
Centralized configuration using dataclasses:
- `VideoConfig`: Video input/display settings
- `ModelConfig`: YOLO model parameters
- `OptimizationConfig`: Performance optimization settings
- `CameraConfig`: Camera calibration parameters
- `DepthEstimationConfig`: Depth estimation settings
- `ClassificationConfig`: Severity thresholds

### `detector.py`
Main detection pipeline:
- Model loading and optimization
- Inference execution
- Detection processing and annotation

### `video_processor.py`
Video handling:
- Video file I/O
- Frame skipping logic
- Display and controls

### `tracker.py`
Temporal tracking:
- IoU-based object matching
- Multi-frame smoothing
- Track lifecycle management

### `severity_estimator.py`
Depth estimation and classification:
- Physical metrics calculation
- Neural depth estimation (MiDaS)
- Geometric depth estimation
- Hybrid approach
- Severity level assignment

### `utils.py`
Helper functions and utilities:
- Color utilities
- Validation functions
- Performance monitoring

## 🎨 Severity Levels

| Level | Depth Range | Color | Description |
|-------|-------------|-------|-------------|
| CRITICAL | > 15 cm | 🔴 Red | Severe damage, immediate attention |
| DANGEROUS | 10-15 cm | 🟠 Orange-Red | Significant damage |
| MODERATE | 6-10 cm | 🟠 Orange | Moderate damage |
| MINOR | 3-6 cm | 🟡 Yellow | Minor damage |
| SURFACE | < 3 cm | 🟢 Green | Surface irregularity |

## ⌨️ Keyboard Controls

- `Q` - Quit application
- `SPACE` - Pause/Resume video playback
- `Any key` - Close window (at end of video)

## 🔧 Customization

### Creating Custom Configurations

```python
from config import Config

# Create custom configuration
config = Config()
config.model.confidence_threshold = 0.3
config.optimization.frame_skip = 2
config.optimization.inference_size = 416

# Or start from preset and modify
config = Config.from_preset('speed')
config.model.confidence_threshold = 0.35
```

### Adjusting Camera Calibration

Edit values in `config.py`:
```python
@dataclass
class CameraConfig:
    focal_length_px: float = 800.0      # Measure from your camera
    camera_height_cm: float = 150.0     # Measure actual height
    horizon_ratio: float = 0.50         # Adjust based on angle
```

### Tuning Depth Estimation

Adjust factors in `config.py`:
```python
@dataclass
class DepthEstimationConfig:
    dry_depth_factor: float = 0.06      # Increase if depths too shallow
    muddy_depth_factor: float = 0.04    # Tune based on testing
    neural_weight: float = 0.6          # Neural vs geometric balance
```

## 📊 Performance Tips

1. **For higher FPS**: Use speed or CPU preset, enable frame skipping
2. **For better accuracy**: Use accuracy preset, disable frame skipping
3. **GPU acceleration**: Ensure CUDA is installed and available
4. **Memory issues**: Reduce inference size to 416 or 320
5. **Jittery detections**: Increase tracking buffer size

## 🐛 Troubleshooting

### Low FPS
- Enable frame skipping: `--frame-skip 1`
- Reduce inference size: `--inference-size 480`
- Use speed preset: `--preset speed`

### Missing Detections
- Lower confidence threshold: `--conf-threshold 0.25`
- Use accuracy preset: `--preset accuracy`
- Disable frame skipping

### Inaccurate Depth
- Calibrate camera parameters in `config.py`
- Adjust depth factors based on real measurements
- Ensure MiDaS model is loading correctly

## 📝 License

[Your License Here]

## 👥 Contributors

- AI Assistant - Initial refactoring and optimization

## 🙏 Acknowledgments

- Ultralytics YOLO for object detection
- Intel MiDaS for depth estimation
- OpenCV for computer vision utilities

## 📧 Contact

[Your Contact Information]

---

**Version**: 2.0.0  
**Last Updated**: January 2026  
**Status**: ✅ Production Ready
