"""
Severity classification module for pothole detection.
Handles depth estimation and severity level assignment.
"""

from typing import Tuple, Optional
from enum import Enum
import cv2
import numpy as np

from .config import Config


class SeverityLevel(Enum):
    """Pothole severity levels with associated colors."""
    CRITICAL = ("CRITICAL", (0, 0, 255))  # Red
    DANGEROUS = ("DANGEROUS", (0, 69, 255))  # Orange-Red
    MODERATE = ("MODERATE", (0, 165, 255))  # Orange
    MINOR = ("MINOR", (0, 255, 255))  # Yellow
    SURFACE = ("SURFACE", (0, 255, 0))  # Green
    UNKNOWN = ("UNKNOWN", (128, 128, 128))  # Gray


class PhysicsCalculator:
    """Handles physical measurements using camera geometry."""

    def __init__(self, config: Config):
        self.config = config
        self.camera = config.camera

    def estimate_physical_metrics(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """
        Estimate real-world distance and width using perspective projection.

        Args:
            frame: Input frame
            box: Bounding box (x1, y1, x2, y2)

        Returns:
            Tuple of (distance_cm, width_cm)
        """
        x1, y1, x2, y2 = box
        h_frame, w_frame, _ = frame.shape

        # Calculate horizon line
        horizon_y = h_frame * self.camera.horizon_ratio

        # Ensure pothole is below horizon
        pixels_below_horizon = max(10, y2 - horizon_y)

        # Distance calculation using pinhole camera model
        distance_cm = (
            self.camera.camera_height_cm * self.camera.focal_length_px
        ) / pixels_below_horizon

        # Clamp to realistic range
        distance_cm = np.clip(
            distance_cm,
            self.camera.min_distance_cm,
            self.camera.max_distance_cm
        )

        # Estimate width with perspective correction
        pixel_width = x2 - x1
        width_cm = (pixel_width * distance_cm) / self.camera.focal_length_px

        # Sanity limits: potholes generally aren't wider than a lane
        width_cm = np.clip(width_cm, 5, 300)

        return distance_cm, width_cm


class DepthEstimator:
    """Handles depth estimation for potholes."""

    def __init__(self, config: Config, has_depth_module: bool = False):
        self.config = config
        self.depth_config = config.depth
        self.has_depth_module = has_depth_module
        self.physics = PhysicsCalculator(config)

        if has_depth_module:
            try:
                import depth_estimation
                self.depth_module = depth_estimation
            except ImportError:
                self.has_depth_module = False
                self.depth_module = None

    def _validate_roi(
        self,
        roi: np.ndarray,
        min_size: int = 5
    ) -> bool:
        """Validate region of interest has minimum size."""
        return (
            roi.size > 0 and
            roi.shape[0] >= min_size and
            roi.shape[1] >= min_size
        )

    def _estimate_neural_depth(
        self,
        roi: np.ndarray,
        width_cm: float,
        frame_shape: Tuple[int, int, int]
    ) -> float:
        """Estimate depth using neural network (MiDaS)."""
        if not self.has_depth_module:
            return 0.0

        try:
            # Prepare camera parameters
            cam_params = {
                'f': self.config.camera.focal_length_px,
                'cx': frame_shape[1] / 2,
                'cy': frame_shape[0] / 2,
                'H': self.config.camera.camera_height_cm / 100.0,
                'pitch': 0.0
            }

            # Detect scene condition
            cond = self.depth_module.detect_wet_muddy(roi)

            # Get relative depth map
            depth_rel = self.depth_module.run_midas_depth(roi)

            # Refine with segmentation
            mask = self.depth_module.simple_pothole_segmentation(roi)

            if mask.sum() < self.depth_config.min_mask_area:
                return 0.0

            depth_refined = self.depth_module.refine_depth(
                depth_rel, mask, cond['spec_mask']
            )

            # Calculate relative depth
            rim_vals = depth_refined[mask == 0]
            if rim_vals.size == 0:
                rim_vals = depth_refined

            rim_median = np.median(rim_vals)
            interior_vals = depth_refined[mask > 0]
            bottom = np.min(interior_vals) if interior_vals.size > 0 else rim_median

            rel_depth_val = max(0.0, rim_median - bottom)

            # Convert to cm with calibrated factor
            return width_cm * rel_depth_val * 0.5

        except Exception:
            return 0.0

    def _estimate_geometric_depth(
        self,
        roi: np.ndarray,
        width_cm: float
    ) -> float:
        """Estimate depth using geometric/shadow analysis."""
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        threshold_val = int(np.mean(blurred) * 0.7)
        _, dark_mask = cv2.threshold(
            blurred, threshold_val, 255, cv2.THRESH_BINARY_INV
        )

        shadow_area = np.sum(dark_mask > 0)
        total_area = roi.shape[0] * roi.shape[1]
        shadow_ratio = shadow_area / total_area

        # Geometric depth with calibrated factor
        return width_cm * 0.03 * (1 + shadow_ratio * 0.5)

    def estimate_dry_pothole_depth(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> Tuple[str, Tuple[int, int, int], float]:
        """
        Estimate depth for dry potholes using hybrid approach.

        Args:
            frame: Input frame
            box: Bounding box (x1, y1, x2, y2)

        Returns:
            Tuple of (label, color, depth_cm)
        """
        x1, y1, x2, y2 = box
        height, width, _ = frame.shape

        # Validate and clamp box coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        roi = frame[y1:y2, x1:x2]

        if not self._validate_roi(roi):
            return (
                SeverityLevel.UNKNOWN.value[0],
                SeverityLevel.UNKNOWN.value[1],
                0.0
            )

        # Get physical width
        _, width_cm = self.physics.estimate_physical_metrics(frame, box)

        # Neural depth estimation
        neural_depth = self._estimate_neural_depth(
            roi, width_cm, frame.shape
        )

        # Geometric depth estimation
        geometric_depth = self._estimate_geometric_depth(roi, width_cm)

        # Combine estimates
        if neural_depth > 0.1:
            final_depth = (
                neural_depth * self.depth_config.neural_weight +
                geometric_depth * self.depth_config.geometric_weight
            )
        else:
            final_depth = geometric_depth

        # Clamp to realistic range
        final_depth = np.clip(
            final_depth,
            self.depth_config.min_depth_cm,
            self.depth_config.max_depth_cm
        )

        return self._classify_severity(final_depth, "DRY")

    def estimate_muddy_pothole_depth(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> Tuple[str, Tuple[int, int, int]]:
        """
        Estimate depth for muddy potholes using volumetric heuristics.

        Args:
            frame: Input frame
            box: Bounding box (x1, y1, x2, y2)

        Returns:
            Tuple of (label, color)
        """
        x1, y1, x2, y2 = box

        # Get physical width
        _, width_cm = self.physics.estimate_physical_metrics(frame, box)

        # Calculate aspect ratio
        height_px = y2 - y1
        width_px = x2 - x1
        aspect_ratio = height_px / max(width_px, 1)

        # Depth estimation with aspect ratio consideration
        depth_cm = (
            width_cm *
            self.depth_config.muddy_depth_factor *
            (1 + aspect_ratio * 0.3)
        )

        # Clamp to realistic range
        depth_cm = np.clip(
            depth_cm,
            1.0,
            self.depth_config.max_depth_cm
        )

        label, color, _ = self._classify_severity(depth_cm, "MUD")
        return label, color

    def _classify_severity(
        self,
        depth_cm: float,
        pothole_type: str = "DRY"
    ) -> Tuple[str, Tuple[int, int, int], float]:
        """
        Classify pothole severity based on depth.

        Args:
            depth_cm: Estimated depth in centimeters
            pothole_type: "DRY" or "MUD"

        Returns:
            Tuple of (label, color, depth)
        """
        clf = self.config.classification

        if depth_cm > clf.critical_threshold:
            level = SeverityLevel.CRITICAL
        elif depth_cm > clf.dangerous_threshold:
            level = SeverityLevel.DANGEROUS
        elif depth_cm > clf.moderate_threshold:
            level = SeverityLevel.MODERATE
        elif depth_cm > clf.minor_threshold:
            level = SeverityLevel.MINOR
        else:
            level = SeverityLevel.SURFACE

        label = f"{level.value[0]} ({depth_cm:.1f}cm)"
        return label, level.value[1], depth_cm
