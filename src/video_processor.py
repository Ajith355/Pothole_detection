"""
Video processing module for pothole detection.
Handles video input, frame processing, and display.
"""

import cv2
import os
from typing import Optional, Callable

from .config import Config
from .detector import PotholeDetector


class VideoProcessor:
    """
    Handles video input and processing for pothole detection.
    """

    def __init__(self, config: Config, detector: PotholeDetector):
        """
        Initialize video processor.

        Args:
            config: Configuration object
            detector: PotholeDetector instance
        """
        self.config = config
        self.detector = detector
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0

    def open_video(self, video_path: Optional[str] = None) -> bool:
        """
        Open video file.

        Args:
            video_path: Path to video file (uses config if None)

        Returns:
            True if successful, False otherwise
        """
        if video_path is None:
            video_path = self.config.video.video_path

        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return False

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"✅ Video loaded: {self.total_frames} frames @ {self.fps} FPS")
        print(f"⚙️ Processing with {self.config.video.frame_delay_ms}ms delay, "
              f"Frame Skip: {self.config.optimization.frame_skip}")

        return True

    def _should_process_frame(self) -> bool:
        """Check if current frame should be processed based on skip setting."""
        if self.config.optimization.frame_skip == 0:
            return True
        return self.frame_count % (self.config.optimization.frame_skip + 1) == 0

    def _create_window(self):
        """Create display window."""
        cv2.namedWindow('Pothole Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            'Pothole Detector',
            self.config.video.display_window_width,
            self.config.video.display_window_height
        )

    def _show_pause_screen(self, frame):
        """Display pause screen overlay."""
        pause_frame = frame.copy()
        h, w = frame.shape[:2]

        # Add semi-transparent overlay
        overlay = pause_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        pause_frame = cv2.addWeighted(pause_frame, 0.7, overlay, 0.3, 0)

        # Add "PAUSED" text
        cv2.putText(
            pause_frame, "PAUSED", (w // 2 - 100, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3
        )
        cv2.putText(
            pause_frame, "Press SPACE to resume, Q to quit",
            (w // 2 - 250, h // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        return pause_frame

    def _handle_pause(self, frame) -> bool:
        """
        Handle pause state.

        Args:
            frame: Current frame

        Returns:
            True to continue, False to quit
        """
        print("⏸️ Paused. Press Space to resume.")
        pause_frame = self._show_pause_screen(frame)
        cv2.imshow('Pothole Detector', pause_frame)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord(' '):  # Resume
                print("▶️ Resuming...")
                return True
            elif key == ord('q'):  # Quit
                print("🛑 User pressed Q. Exiting...")
                return False

    def process_video(
        self,
        on_frame_callback: Optional[Callable] = None
    ) -> None:
        """
        Process video file with pothole detection.

        Args:
            on_frame_callback: Optional callback function called after each frame
                             Signature: callback(frame, detections, frame_number)
        """
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open_video() first.")

        self._create_window()
        self.frame_count = 0

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    print("ℹ️ End of video reached.")
                    break

                self.frame_count += 1

                # Skip frame if configured
                if not self._should_process_frame():
                    cv2.imshow('Pothole Detector', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Detect potholes
                detections, _ = self.detector.detect(frame)

                # Draw annotations
                annotated_frame = self.detector.draw_detections(frame, detections)

                # Call callback if provided
                if on_frame_callback is not None:
                    on_frame_callback(annotated_frame, detections, self.frame_count)

                # Display frame
                cv2.imshow('Pothole Detector', annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(self.config.video.frame_delay_ms) & 0xFF

                if key == ord('q'):
                    print("🛑 User pressed Q. Exiting...")
                    break
                elif key == ord(' '):  # Pause
                    if not self._handle_pause(annotated_frame):
                        break

            # Show final frame
            print("✅ Done. Press any key to close the window.")
            cv2.waitKey(0)

        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
