"""
Pothole detection and analysis pipeline.
Handles YOLO inference, tracking, and visualization.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
from collections import deque
import time

from ultralytics import YOLO
from .config import Config
from .tracker import DetectionTracker
from .severity_estimator import DepthEstimator


class PotholeDetector:
    """
    Main pothole detection pipeline.

    Handles model loading, inference, tracking, and visualization.
    """

    def __init__(self, config: Config):
        """
        Initialize detector.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model: Optional[YOLO] = None
        self.tracker: Optional[DetectionTracker] = None
        self.depth_estimator: Optional[DepthEstimator] = None
        self.inference_times = deque(maxlen=30)

        # Try to import depth estimation module
        self.has_depth_module = False
        try:
            import depth_estimation
            self.has_depth_module = True
            print("✅ Successfully imported depth_estimation.py")
        except ImportError:
            print("⚠️ Could not import depth_estimation.py. Using geometric fallback.")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load YOLO model and initialize components.

        Args:
            model_path: Path to model file (uses config if None)

        Returns:
            True if successful, False otherwise
        """
        if model_path is None:
            model_path = self.config.model.model_path

        try:
            print(f"🚀 Loading model: {model_path}")
            self.model = YOLO(model_path)

            # Optimize model
            self.model.fuse()

            # Enable half precision if configured and GPU available
            if (self.config.model.use_half_precision and
                self.model.device.type != 'cpu'):
                self.model.half()
                print("✅ Using FP16 precision for faster inference")

            # Initialize tracker
            if self.config.optimization.enable_tracking:
                self.tracker = DetectionTracker(
                    buffer_size=self.config.optimization.track_buffer_size,
                    iou_threshold=self.config.optimization.tracking_iou_threshold
                )

            # Initialize depth estimator
            self.depth_estimator = DepthEstimator(
                self.config,
                self.has_depth_module
            )

            print("✅ Model loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def _prepare_inference_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Prepare frame for inference with optional resizing.

        Args:
            frame: Original frame

        Returns:
            Tuple of (inference_frame, scale_factor)
        """
        if not self.config.optimization.resize_inference:
            return frame, 1.0

        if frame.shape[1] <= self.config.optimization.inference_size:
            return frame, 1.0

        scale = self.config.optimization.inference_size / frame.shape[1]
        inference_frame = cv2.resize(
            frame, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

        return inference_frame, scale

    def _run_inference(
        self,
        frame: np.ndarray
    ) -> Tuple[List, float]:
        """
        Run YOLO inference on frame.

        Args:
            frame: Input frame

        Returns:
            Tuple of (results, inference_time)
        """
        start_time = time.time()

        # Prepare frame
        inference_frame, scale_factor = self._prepare_inference_frame(frame)

        # Run inference
        results = self.model(
            inference_frame,
            verbose=False,
            conf=self.config.model.confidence_threshold,
            iou=self.config.model.iou_threshold,
            agnostic_nms=self.config.model.agnostic_nms,
            max_det=self.config.model.max_detections
        )[0]

        inference_time = time.time() - start_time

        return results, scale_factor, inference_time

    def _process_detections(
        self,
        frame: np.ndarray,
        results,
        scale_factor: float
    ) -> List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]]:
        """
        Process YOLO results into detections with severity analysis.

        Args:
            frame: Original frame
            results: YOLO results
            scale_factor: Scale factor from resizing

        Returns:
            List of (box, label, color) tuples
        """
        detections = []

        for box in results.boxes:
            # Scale box coordinates back
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].cpu().numpy() * (1.0 / scale_factor)
            )
            conf = box.conf[0].item()

            if conf <= self.config.model.confidence_threshold:
                continue

            # Get class name
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]

            # Analyze based on pothole type
            if "muddy" in class_name.lower():
                label, color = self.depth_estimator.estimate_muddy_pothole_depth(
                    frame, (x1, y1, x2, y2)
                )
                full_label = f"MUD: {label}"
            else:
                label, color, _ = self.depth_estimator.estimate_dry_pothole_depth(
                    frame, (x1, y1, x2, y2)
                )
                full_label = f"DRY: {label}"

            detections.append(((x1, y1, x2, y2), full_label, color))

        return detections

    def detect(
        self,
        frame: np.ndarray
    ) -> Tuple[List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]], float]:
        """
        Detect potholes in frame.

        Args:
            frame: Input frame

        Returns:
            Tuple of (detections, inference_time)
            detections: List of (box, label, color) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference
        results, scale_factor, inference_time = self._run_inference(frame)
        self.inference_times.append(inference_time)

        # Process detections
        detections = self._process_detections(frame, results, scale_factor)

        # Apply tracking if enabled
        if self.tracker is not None:
            detections = self.tracker.update(detections)

        return detections, inference_time

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]],
        show_fps: bool = True
    ) -> np.ndarray:
        """
        Draw detections on frame.

        Args:
            frame: Input frame
            detections: List of (box, label, color) tuples
            show_fps: Whether to show FPS counter

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        for (x1, y1, x2, y2), label, color in detections:
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - 30),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Determine text color based on background brightness
            text_color = (0, 0, 0) if color in [(0, 255, 255), (0, 255, 0)] else (255, 255, 255)

            # Draw label text
            cv2.putText(
                annotated_frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
            )

        # Draw FPS counter
        if show_fps and len(self.inference_times) > 0:
            avg_fps = 1.0 / (np.mean(self.inference_times) + 1e-6)
            cv2.putText(
                annotated_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        return annotated_frame

    @property
    def average_fps(self) -> float:
        """Get average FPS over recent frames."""
        if len(self.inference_times) == 0:
            return 0.0
        return 1.0 / (np.mean(self.inference_times) + 1e-6)
