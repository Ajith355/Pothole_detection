"""
Quick Test Script for Optimized Pothole Detection
Run this to verify all optimizations are working correctly
"""

import sys
import os

print("=" * 60)
print("POTHOLE DETECTION OPTIMIZATION VERIFICATION")
print("=" * 60)

# Test 1: Check Python version
print("\n[1/8] Checking Python version...")
python_version = sys.version_info
print(f"   ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("   ⚠️ Warning: Python 3.8+ recommended")

# Test 2: Check required packages
print("\n[2/8] Checking required packages...")
required_packages = {
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'ultralytics': 'ultralytics',
    'torch': 'torch'
}

missing_packages = []
for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - MISSING")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   Install missing packages: pip install {' '.join(missing_packages)}")
else:
    print("   ✓ All required packages installed")

# Test 3: Check CUDA availability
print("\n[3/8] Checking GPU/CUDA support...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print("   → FP16 optimization will be enabled")
    else:
        print("   ⚠️ CUDA not available - will use CPU")
        print("   → FP16 optimization will be disabled")
except ImportError:
    print("   ⚠️ PyTorch not installed - cannot check CUDA")

# Test 4: Check model file
print("\n[4/8] Checking model files...")
model_files = [
    "pothole_detector_v1.pt",
    "best.pt",
    "best_.pt",
    "best_2.pt"
]

found_models = []
for model in model_files:
    if os.path.exists(model):
        size_mb = os.path.getsize(model) / (1024 * 1024)
        print(f"   ✓ {model} ({size_mb:.1f} MB)")
        found_models.append(model)

if not found_models:
    print("   ✗ No model files found!")
else:
    print(f"   ✓ Found {len(found_models)} model(s)")

# Test 5: Check video file
print("\n[5/8] Checking video files...")
video_files = ["demo.mp4"]
found_videos = []

for video in video_files:
    if os.path.exists(video):
        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"   ✓ {video} ({size_mb:.1f} MB)")
        found_videos.append(video)

if not found_videos:
    print("   ✗ No video files found!")
else:
    print(f"   ✓ Found {len(found_videos)} video(s)")

# Test 6: Check depth estimation module
print("\n[6/8] Checking depth estimation module...")
try:
    import depth_estimation
    print("   ✓ depth_estimation.py found")

    # Check if MiDaS can be loaded
    if hasattr(depth_estimation, 'HAS_TORCH') and depth_estimation.HAS_TORCH:
        print("   ✓ PyTorch available for MiDaS")
    else:
        print("   ⚠️ PyTorch not available - will use geometric fallback")
except ImportError:
    print("   ⚠️ depth_estimation.py not found or has errors")

# Test 7: Verify optimization features
print("\n[7/8] Checking optimization features...")
try:
    with open('demo_video_delectr.py', 'r', encoding='utf-8') as f:
        content = f.read()

    features = {
        'model.fuse()': 'Model fusion',
        'USE_HALF_PRECISION': 'FP16 support',
        'ENABLE_TRACKING': 'Detection tracking',
        'DetectionTracker': 'Tracker class',
        'FRAME_SKIP': 'Frame skipping',
        'RESIZE_INFERENCE': 'Inference resizing',
        'agnostic_nms': 'Agnostic NMS'
    }

    for feature, description in features.items():
        if feature in content:
            print(f"   ✓ {description}")
        else:
            print(f"   ✗ {description} - MISSING")

except FileNotFoundError:
    print("   ✗ demo_video_delectr.py not found!")

# Test 8: Configuration check
print("\n[8/8] Checking configuration files...")
config_files = [
    'OPTIMIZATION_GUIDE.md',
    'OPTIMIZATION_SUMMARY.md',
    'config_presets.py'
]

for config in config_files:
    if os.path.exists(config):
        print(f"   ✓ {config}")
    else:
        print(f"   ⚠️ {config} - not found")

# Summary
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)

if missing_packages:
    print("⚠️ ACTION REQUIRED: Install missing packages")
    print(f"   pip install {' '.join(missing_packages)}")
else:
    print("✅ All core dependencies satisfied")

if not found_models:
    print("⚠️ WARNING: No model files found")
    print("   Place your YOLO model (.pt file) in this directory")

if not found_videos:
    print("⚠️ WARNING: No video files found")
    print("   Place your test video (demo.mp4) in this directory")

print("\nReady to run:")
print("   python demo_video_delectr.py")
print("\nFor configuration help, see:")
print("   - OPTIMIZATION_GUIDE.md (detailed guide)")
print("   - OPTIMIZATION_SUMMARY.md (quick reference)")
print("   - config_presets.py (preset configurations)")
print("=" * 60)
