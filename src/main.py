"""
Main application for video tracking with living being detection.
"""

import cv2
import argparse
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Union

from .config import (
    DEFAULT_VIDEO_SOURCE, MAX_FRAME_SIZE, PROCESS_EVERY_N_FRAMES,
    OUTPUT_VIDEO_FPS, LOG_LEVEL, LOG_DETECTIONS
)
from .detection import create_detector, DetectionTracker
from .utils import FPSMeter, draw_detections, draw_info_panel, resize_frame, validate_frame
from .audio_feedback import create_audio_feedback, create_detection_announcer

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoTracker:
    """
    Main video tracking application.
    """
    
    def __init__(self, video_source: Union[int, str] = None, 
                 model_path: str = None, enable_audio: bool = True,
                 enable_tts: bool = False, output_path: str = None):
        """
        Initialize the video tracker.
        
        Args:
            video_source: Video source (camera index or file path)
            model_path: Path to YOLOv8 model
            enable_audio: Enable audio feedback
            enable_tts: Enable text-to-speech announcements
            output_path: Path to save output video
        """
        self.video_source = video_source or DEFAULT_VIDEO_SOURCE
        self.output_path = output_path
        self.cap = None
        self.out_writer = None
        
        # Initialize components
        self.detector = create_detector(model_path)
        self.tracker = DetectionTracker()
        self.fps_meter = FPSMeter()
        self.audio_feedback = create_audio_feedback(enable_audio)
        self.announcer = create_detection_announcer(enable_tts)
        
        # Processing state
        self.frame_count = 0
        self.running = False
        self.paused = False
        
        logger.info("Video tracker initialized")
    
    def initialize_video_source(self) -> bool:
        """
        Initialize the video source.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.video_source}")
                return False
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or OUTPUT_VIDEO_FPS
            
            logger.info(f"Video source opened: {width}x{height} @ {fps}fps")
            
            # Initialize output writer if specified
            if self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out_writer = cv2.VideoWriter(
                    self.output_path, fourcc, fps, (width, height)
                )
                logger.info(f"Output video writer initialized: {self.output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize video source: {e}")
            return False
    
    def process_frame(self, frame) -> tuple:
        """
        Process a single frame for detections.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, detections, detection_summary)
        """
        if not validate_frame(frame):
            return frame, [], {}
        
        # Resize frame for better performance
        frame = resize_frame(frame, MAX_FRAME_SIZE)
        
        detections = []
        detection_summary = {}
        
        # Process every N frames for performance
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Detect living beings
            detections = self.detector.detect_living_beings(frame)
            
            # Update tracker
            detections = self.tracker.update(detections)
            
            # Get detection summary
            detection_summary = self.detector.get_detection_summary(detections)
            
            # Log detections if enabled
            if LOG_DETECTIONS and detections:
                logger.info(f"Frame {self.frame_count}: {len(detections)} detections")
                for det in detections:
                    logger.info(f"  - {det['class_name']}: {det['confidence']:.2f}")
        
        # Draw detections on frame
        if detections:
            frame = draw_detections(frame, detections)
        
        # Draw information panel
        frame = draw_info_panel(
            frame,
            fps=self.fps_meter.get_fps(),
            detection_count=len(detections),
            detection_summary=detection_summary
        )
        
        return frame, detections, detection_summary
    
    def run(self):
        """Run the main video tracking loop."""
        if not self.initialize_video_source():
            return False
        
        self.running = True
        logger.info("Starting video tracking...")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame from video source")
                    break
                
                # Handle pause
                if self.paused:
                    cv2.waitKey(30)
                    continue
                
                # Process frame
                processed_frame, detections, detection_summary = self.process_frame(frame)
                
                # Audio feedback
                if detections:
                    self.audio_feedback.play_detection_alert(detections)
                    self.announcer.announce_detections(detections)
                
                # Save frame to output video
                if self.out_writer:
                    self.out_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('Morph1x - Living Being Tracker', processed_frame)
                
                # Update FPS
                self.fps_meter.update()
                self.frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('p'):  # 'p' for pause
                    self.paused = not self.paused
                    logger.info(f"Video {'paused' if self.paused else 'resumed'}")
                elif key == ord('r'):  # 'r' for reset
                    self.tracker.clear_history()
                    self.fps_meter.reset()
                    logger.info("Tracker reset")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        if self.out_writer:
            self.out_writer.release()
        
        cv2.destroyAllWindows()
        
        if self.audio_feedback:
            self.audio_feedback.cleanup()
        
        logger.info("Cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Morph1x - Video Tracking for Living Beings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                    # Use default webcam
  python -m src.main --source 0         # Use webcam index 0
  python -m src.main --source video.mp4 # Process video file
  python -m src.main --output result.mp4 # Save output video
  python -m src.main --no-audio         # Disable audio feedback
  python -m src.main --tts              # Enable text-to-speech
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=str(DEFAULT_VIDEO_SOURCE),
        help='Video source (camera index or file path)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to YOLOv8 model file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output video file path'
    )
    
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Disable audio feedback'
    )
    
    parser.add_argument(
        '--tts',
        action='store_true',
        help='Enable text-to-speech announcements'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert source to int if it's a number
    video_source = args.source
    if video_source.isdigit():
        video_source = int(video_source)
    
    # Create and run tracker
    try:
        tracker = VideoTracker(
            video_source=video_source,
            model_path=args.model,
            enable_audio=not args.no_audio,
            enable_tts=args.tts,
            output_path=args.output
        )
        
        success = tracker.run()
        
        if success:
            logger.info("Video tracking completed successfully")
        else:
            logger.error("Video tracking failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
