"""
Configuration module for Pothole Detection System
Contains all configurable parameters for the detection pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoConfig:
    """Video input and display configuration."""
    video_path: str = "videos/demo.mp4"
    frame_delay_ms: int = 1  # 1 = Fastest, 30 = Normal, 100 = Slow Motion
    display_window_width: int = 1024
    display_window_height: int = 768


@dataclass
class ModelConfig:
    """YOLO model configuration."""
    model_path: str = "models/pothole_detector_v1.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 10
    use_half_precision: bool = True
    agnostic_nms: bool = True


@dataclass
class OptimizationConfig:
    """Performance optimization settings."""
    frame_skip: int = 0  # 0 = no skip, 1 = every other frame
    resize_inference: bool = True
    inference_size: int = 640  # 640, 480, or 320
    enable_tracking: bool = True
    track_buffer_size: int = 5
    tracking_iou_threshold: float = 0.3


@dataclass
class CameraConfig:
    """Camera calibration parameters."""
    focal_length_px: float = 800.0  # Virtual focal length in pixels
    camera_height_cm: float = 150.0  # Camera height from ground
    horizon_ratio: float = 0.50  # Horizon line as fraction of frame height
    min_distance_cm: float = 100.0  # Minimum realistic distance
    max_distance_cm: float = 2000.0  # Maximum realistic distance


@dataclass
class DepthEstimationConfig:
    """Depth estimation calibration parameters."""
    dry_depth_factor: float = 0.06  # 6% of width for dry potholes
    muddy_depth_factor: float = 0.04  # 4% of width for muddy potholes
    shadow_weight: float = 0.3  # Weight for shadow-based adjustment
    max_depth_cm: float = 35.0  # Maximum realistic pothole depth
    min_depth_cm: float = 0.5  # Minimum depth threshold
    neural_weight: float = 0.6  # Weight for neural depth in hybrid approach
    geometric_weight: float = 0.4  # Weight for geometric depth
    min_mask_area: int = 50  # Minimum mask area for valid segmentation


@dataclass
class ClassificationConfig:
    """Pothole severity classification thresholds."""
    muddy_large_threshold: float = 0.25  # > 25% of frame width
    muddy_medium_threshold: float = 0.08  # > 8% of frame width

    # Severity thresholds in cm
    critical_threshold: float = 15.0
    dangerous_threshold: float = 10.0
    moderate_threshold: float = 6.0
    minor_threshold: float = 3.0


class Config:
    """Main configuration container."""

    def __init__(self):
        self.video = VideoConfig()
        self.model = ModelConfig()
        self.optimization = OptimizationConfig()
        self.camera = CameraConfig()
        self.depth = DepthEstimationConfig()
        self.classification = ClassificationConfig()

    @classmethod
    def from_preset(cls, preset: str) -> 'Config':
        """
        Create configuration from preset.

        Args:
            preset: One of 'accuracy', 'balanced', 'speed', 'cpu'

        Returns:
            Configured Config instance
        """
        config = cls()

        if preset == 'accuracy':
            config.model.confidence_threshold = 0.25
            config.optimization.frame_skip = 0
            config.optimization.resize_inference = False
            config.optimization.track_buffer_size = 7

        elif preset == 'balanced':
            # Default settings
            pass

        elif preset == 'speed':
            config.model.confidence_threshold = 0.4
            config.optimization.frame_skip = 1
            config.optimization.inference_size = 480
            config.optimization.track_buffer_size = 3

        elif preset == 'cpu':
            config.model.confidence_threshold = 0.4
            config.model.use_half_precision = False
            config.optimization.frame_skip = 1
            config.optimization.inference_size = 416
            config.optimization.track_buffer_size = 3

        else:
            raise ValueError(f"Unknown preset: {preset}")

        return config


# Default configuration instance
default_config = Config()
