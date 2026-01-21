"""
Detection tracking module for temporal smoothing.
Implements IoU-based object tracking across frames.
"""

from typing import List, Tuple, Dict
from collections import deque
import numpy as np


class DetectionTracker:
    """
    Tracks detections across frames for temporal smoothing and filtering.

    Uses Intersection over Union (IoU) for matching detections between frames
    and maintains a rolling buffer for smoothing bounding box coordinates.
    """

    def __init__(self, buffer_size: int = 5, iou_threshold: float = 0.3):
        """
        Initialize tracker.

        Args:
            buffer_size: Number of frames to keep in history for smoothing
            iou_threshold: Minimum IoU for matching detections between frames
        """
        self.buffer_size = buffer_size
        self.iou_threshold = iou_threshold
        self.tracked_objects: Dict[int, Dict] = {}
        self.next_id = 0

    def calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)

        Returns:
            IoU value between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection area
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height

        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        # Prevent division by zero
        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _find_best_match(
        self,
        box: Tuple[int, int, int, int]
    ) -> Tuple[int, float]:
        """
        Find best matching tracked object for a detection.

        Args:
            box: Bounding box to match

        Returns:
            Tuple of (track_id, iou_score) or (None, 0.0) if no match
        """
        best_match_id = None
        best_iou = 0.0

        for track_id, track_data in self.tracked_objects.items():
            if len(track_data['boxes']) > 0:
                last_box = track_data['boxes'][-1]
                iou = self.calculate_iou(box, last_box)

                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id

        return best_match_id, best_iou

    def _create_new_track(
        self,
        box: Tuple[int, int, int, int],
        label: str,
        color: Tuple[int, int, int]
    ) -> int:
        """
        Create a new tracked object.

        Args:
            box: Initial bounding box
            label: Detection label
            color: Display color

        Returns:
            New track ID
        """
        track_id = self.next_id
        self.next_id += 1

        self.tracked_objects[track_id] = {
            'boxes': deque([box], maxlen=self.buffer_size),
            'labels': deque([label], maxlen=self.buffer_size),
            'colors': deque([color], maxlen=self.buffer_size)
        }

        return track_id

    def _update_track(
        self,
        track_id: int,
        box: Tuple[int, int, int, int],
        label: str,
        color: Tuple[int, int, int]
    ):
        """
        Update existing tracked object.

        Args:
            track_id: ID of track to update
            box: New bounding box
            label: New label
            color: New color
        """
        track = self.tracked_objects[track_id]
        track['boxes'].append(box)
        track['labels'].append(label)
        track['colors'].append(color)

    def _smooth_box(self, boxes: deque) -> Tuple[int, int, int, int]:
        """
        Smooth bounding box coordinates by averaging recent boxes.

        Args:
            boxes: Deque of recent bounding boxes

        Returns:
            Smoothed bounding box
        """
        boxes_array = np.array(boxes)
        smoothed = np.mean(boxes_array, axis=0).astype(int)
        return tuple(smoothed)

    def _remove_stale_tracks(self, matched_tracks: set):
        """
        Remove tracks that were not matched in current frame.

        Args:
            matched_tracks: Set of track IDs that were matched
        """
        stale_ids = [
            tid for tid in self.tracked_objects.keys()
            if tid not in matched_tracks
        ]
        for tid in stale_ids:
            del self.tracked_objects[tid]

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]]
    ) -> List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]]:
        """
        Update tracker with new detections and return smoothed results.

        Args:
            detections: List of (box, label, color) tuples

        Returns:
            List of smoothed (box, label, color) tuples
        """
        if not detections:
            # Clear all tracks if no detections
            self.tracked_objects.clear()
            return []

        matched_tracks = set()
        smoothed_detections = []

        for box, label, color in detections:
            # Find best matching track
            best_match_id, _ = self._find_best_match(box)

            if best_match_id is not None:
                # Update existing track
                self._update_track(best_match_id, box, label, color)
                matched_tracks.add(best_match_id)

                # Get smoothed box
                track = self.tracked_objects[best_match_id]
                smoothed_box = self._smooth_box(track['boxes'])
                smoothed_detections.append((smoothed_box, label, color))
            else:
                # Create new track
                new_id = self._create_new_track(box, label, color)
                matched_tracks.add(new_id)
                smoothed_detections.append((box, label, color))

        # Remove stale tracks
        self._remove_stale_tracks(matched_tracks)

        return smoothed_detections

    def reset(self):
        """Clear all tracked objects."""
        self.tracked_objects.clear()
        self.next_id = 0

    @property
    def active_tracks(self) -> int:
        """Get number of currently active tracks."""
        return len(self.tracked_objects)
