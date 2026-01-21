"""
Utility functions and helpers for pothole detection system.
"""

import cv2
import numpy as np
from typing import Tuple


def get_text_color_for_background(
    bg_color: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """
    Determine appropriate text color based on background brightness.

    Args:
        bg_color: Background color in BGR format

    Returns:
        Text color in BGR format (black or white)
    """
    # Calculate perceived brightness
    b, g, r = bg_color
    brightness = (0.299 * r + 0.587 * g + 0.114 * b)

    # Return black for bright backgrounds, white for dark
    return (0, 0, 0) if brightness > 128 else (255, 255, 255)


def validate_box(
    box: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Validate and clamp bounding box coordinates to frame boundaries.

    Args:
        box: Bounding box (x1, y1, x2, y2)
        frame_shape: Frame shape (height, width)

    Returns:
        Validated bounding box
    """
    x1, y1, x2, y2 = box
    height, width = frame_shape

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return (x1, y1, x2, y2)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def calculate_distance_between_boxes(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """
    Calculate Euclidean distance between box centers.

    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)

    Returns:
        Distance between centers
    """
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2

    return np.sqrt(
        (x1_center - x2_center) ** 2 +
        (y1_center - y2_center) ** 2
    )


def create_color_map(num_colors: int = 10) -> list:
    """
    Generate distinct colors for visualization.

    Args:
        num_colors: Number of colors to generate

    Returns:
        List of BGR colors
    """
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        color = cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]),
            cv2.COLOR_HSV2BGR
        )[0][0]
        colors.append(tuple(map(int, color)))

    return colors


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        self.metrics = {
            'total_frames': 0,
            'detected_frames': 0,
            'total_detections': 0,
            'processing_time': 0.0
        }

    def update(
        self,
        num_detections: int,
        processing_time: float
    ):
        """Update metrics with new frame data."""
        self.metrics['total_frames'] += 1
        if num_detections > 0:
            self.metrics['detected_frames'] += 1
        self.metrics['total_detections'] += num_detections
        self.metrics['processing_time'] += processing_time

    def get_summary(self) -> dict:
        """Get performance summary."""
        total_frames = self.metrics['total_frames']
        if total_frames == 0:
            return self.metrics.copy()

        summary = self.metrics.copy()
        summary['avg_fps'] = total_frames / (
            self.metrics['processing_time'] + 1e-6
        )
        summary['detection_rate'] = (
            self.metrics['detected_frames'] / total_frames
        )
        summary['avg_detections_per_frame'] = (
            self.metrics['total_detections'] / total_frames
        )

        return summary

    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total Frames Processed: {summary['total_frames']}")
        print(f"Frames with Detections: {summary['detected_frames']}")
        print(f"Total Detections: {summary['total_detections']}")
        print(f"Average FPS: {summary.get('avg_fps', 0):.2f}")
        print(f"Detection Rate: {summary.get('detection_rate', 0)*100:.1f}%")
        print(f"Avg Detections/Frame: {summary.get('avg_detections_per_frame', 0):.2f}")
        print(f"Total Processing Time: {format_time(summary['processing_time'])}")
        print("=" * 60)
