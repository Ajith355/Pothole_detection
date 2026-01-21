"""
Pothole Detection System - Main Application

A professional pothole detection and analysis system using YOLO and depth estimation.
Supports real-time video processing with temporal tracking and severity classification.

Author: AI Assistant
Date: January 2026
Version: 2.0.0
"""

import sys
import argparse
from pathlib import Path

from .config import Config
from .detector import PotholeDetector
from .video_processor import VideoProcessor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pothole Detection and Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run with default settings
  %(prog)s --video input.mp4        # Process specific video
  %(prog)s --model best.pt          # Use specific model
  %(prog)s --preset speed           # Use speed preset
  %(prog)s --no-tracking            # Disable tracking
        """
    )

    parser.add_argument(
        '--video', '-v',
        type=str,
        help='Path to input video file (default: demo.mp4)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to YOLO model file (default: pothole_detector_v1.pt)'
    )

    parser.add_argument(
        '--preset', '-p',
        type=str,
        choices=['accuracy', 'balanced', 'speed', 'cpu'],
        default='balanced',
        help='Configuration preset (default: balanced)'
    )

    parser.add_argument(
        '--no-tracking',
        action='store_true',
        help='Disable temporal tracking'
    )

    parser.add_argument(
        '--conf-threshold',
        type=float,
        help='Confidence threshold (0.0-1.0)'
    )

    parser.add_argument(
        '--frame-skip',
        type=int,
        help='Skip every N frames (0 = process all)'
    )

    parser.add_argument(
        '--inference-size',
        type=int,
        choices=[320, 416, 480, 640],
        help='Model inference size'
    )

    return parser.parse_args()


def apply_arguments(config: Config, args):
    """Apply command line arguments to configuration."""
    if args.video:
        config.video.video_path = args.video

    if args.model:
        config.model.model_path = args.model

    if args.no_tracking:
        config.optimization.enable_tracking = False

    if args.conf_threshold is not None:
        config.model.confidence_threshold = args.conf_threshold

    if args.frame_skip is not None:
        config.optimization.frame_skip = args.frame_skip

    if args.inference_size is not None:
        config.optimization.inference_size = args.inference_size


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           POTHOLE DETECTION & ANALYSIS SYSTEM                ║
║                      Version 2.0.0                           ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_configuration(config: Config):
    """Print current configuration."""
    print("\n📋 Configuration:")
    print(f"  Video: {config.video.video_path}")
    print(f"  Model: {config.model.model_path}")
    print(f"  Confidence Threshold: {config.model.confidence_threshold}")
    print(f"  Frame Skip: {config.optimization.frame_skip}")
    print(f"  Inference Size: {config.optimization.inference_size}")
    print(f"  Tracking: {'Enabled' if config.optimization.enable_tracking else 'Disabled'}")
    print(f"  FP16: {'Enabled' if config.model.use_half_precision else 'Disabled'}")
    print()


def print_controls():
    """Print keyboard controls."""
    print("⌨️  Keyboard Controls:")
    print("  Q       - Quit application")
    print("  SPACE   - Pause/Resume video")
    print()


def main():
    """Main application entry point."""
    # Print banner
    print_banner()

    # Parse arguments
    args = parse_arguments()

    # Create configuration
    try:
        config = Config.from_preset(args.preset)
        apply_arguments(config, args)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return 1

    # Print configuration
    print_configuration(config)
    print_controls()

    # Initialize detector
    detector = PotholeDetector(config)

    # Load model
    if not detector.load_model():
        print("❌ Failed to load model. Exiting.")
        return 1

    # Initialize video processor
    processor = VideoProcessor(config, detector)

    # Open video
    if not processor.open_video():
        print("❌ Failed to open video. Exiting.")
        return 1

    # Process video
    try:
        processor.process_video()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        processor.cleanup()

    print("\n✅ Processing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
