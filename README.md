# Pothole Detection System 🚗💥

> Professional pothole detection and analysis using YOLO and depth estimation

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![Status](https://img.shields.io/badge/status-production-success)

## 🚀 Quick Start

### Backend (Python Detection System)
```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt

# Run application
python run.py

# See all options
python run.py --help
```

### Frontend (Flutter Mobile App)
```bash
# Navigate to frontend
cd frontend

# Install dependencies
flutter pub get

# Run application
flutter run
```

## 📁 Project Structure

```
Pothole_detection/
├── backend/               # 🐍 Python Backend - Detection System
│   ├── run.py            # ⭐ Main entry point - START HERE
│   ├── requirements.txt  # Python dependencies
│   │
│   ├── src/              # Source code modules
│   │   ├── config/       # Configuration management
│   │   ├── detector.py   # Detection pipeline
│   │   ├── tracker.py    # Temporal tracking
│   │   ├── severity_estimator.py  # Depth estimation
│   │   ├── video_processor.py     # Video I/O
│   │   ├── websocket_transmitter.py  # WebSocket communication
│   │   └── main.py       # Application logic
│   │
│   ├── docs/             # Documentation
│   │   ├── README_PROFESSIONAL.md   # Complete guide
│   │   ├── MIGRATION_GUIDE.md       # Migration guide
│   │   └── REFACTORING_SUMMARY.md   # Changes summary
│   │
│   ├── models/           # YOLO model files (*.pt)
│   ├── videos/           # Video files for testing
│   ├── legacy/           # Old code (reference)
│   ├── paper/            # Research paper and LaTeX files
│   └── depth_estimation.py  # Advanced depth module
│
└── frontend/             # 📱 Flutter Frontend - Mobile App
    ├── lib/              # Dart source code
    │   ├── main.dart     # App entry point
    │   ├── models/       # Data models
    │   ├── screens/      # UI screens
    │   └── services/     # WebSocket & services
    │
    ├── android/          # Android platform files
    ├── windows/          # Windows platform files
    ├── test/             # Unit tests
    ├── pubspec.yaml      # Flutter dependencies
    └── README.md         # Frontend documentation
```

## 💻 Usage Examples

### Basic Usage
```bash
cd backend
python run.py
```

### With Configuration
```bash
cd backend
python run.py --video videos/demo.mp4 --preset speed
```

### All Options
```bash
cd backend
python run.py \
    --video videos/input.mp4 \
    --model models/best.pt \
    --preset balanced \
    --conf-threshold 0.3 \
    --frame-skip 1
```

## ⚙️ Presets

| Preset | Use Case | Speed | Accuracy |
|--------|----------|-------|----------|
| **accuracy** | Best quality | ⚡ | ⭐⭐⭐⭐⭐ |
| **balanced** | Default (recommended) | ⚡⚡ | ⭐⭐⭐⭐ |
| **speed** | Real-time | ⚡⚡⚡ | ⭐⭐⭐ |
| **cpu** | No GPU | ⚡⚡⚡ | ⭐⭐⭐ |

## 🎯 Severity Levels

- 🔴 **CRITICAL** (> 15 cm) - Urgent attention required
- 🟠 **DANGEROUS** (10-15 cm) - High priority
- 🟠 **MODERATE** (6-10 cm) - Medium priority
- 🟡 **MINOR** (3-6 cm) - Low priority
- 🟢 **SURFACE** (< 3 cm) - Informational

## ⌨️ Controls

- `Q` - Quit
- `SPACE` - Pause/Resume
- `Any key` - Close (at end)

## 📚 Full Documentation

- **Backend**: See [backend/docs/README_PROFESSIONAL.md](backend/docs/README_PROFESSIONAL.md) for complete documentation
- **Frontend**: See [frontend/README.md](frontend/README.md) for Flutter app documentation
- **WebSocket Setup**: See [WEBSOCKET_SETUP.md](WEBSOCKET_SETUP.md) for backend-frontend communication

## 🔧 Requirements

### Backend
- Python 3.8+
- OpenCV 4.5+
- NumPy 1.19+
- Ultralytics YOLO 8.0+
- PyTorch 1.10+
- WebSockets 12.0+

Install all: `cd backend && pip install -r requirements.txt`

### Frontend
- Flutter SDK 3.0+
- Dart 2.17+

Install all: `cd frontend && flutter pub get`

## 🐛 Quick Troubleshooting

### Backend Issues

**Low FPS?**
```bash
cd backend
python run.py --preset speed --frame-skip 1
```

**Missing detections?**
```bash
cd backend
python run.py --preset accuracy --conf-threshold 0.25
```

**GPU not working?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
# Add backend to path
import sys
sys.path.insert(0, 'backend/src')

from config import Config
from detector import PotholeDetector

config = Config.from_preset('balanced')
detector = PotholeDetector(config)
detector.load_model()

# Process frame
detections, time_taken = detector.detect(frame)
annotated = detector.draw_detections(frame, detections)
```

---

**Version**: 2.0.0 | **Status**: ✅ Production Ready | **Updated**: March 2026

**Quick Start Backend**: `cd backend && python run.py`
**Quick Start Frontend**: `cd frontend && flutter run

# Process frame
detections, time_taken = detector.detect(frame)
annotated = detector.draw_detections(frame, detections)
```

---

**Version**: 2.0.0 | **Status**: ✅ Production Ready | **Updated**: Jan 2026

**Quick Start**: `python run.py`
