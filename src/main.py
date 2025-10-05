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
        
        self.video_source = video_source or DEFAULT_VIDEO_SOURCE
        self.output_path = output_path
        self.cap = None
        self.out_writer = None
        
        self.detector = create_detector(model_path)
        self.tracker = DetectionTracker()
        self.fps_meter = FPSMeter()
        self.audio_feedback = create_audio_feedback(enable_audio)
        self.announcer = create_detection_announcer(enable_tts)
        
        self.frame_count = 0
        self.running = False
        self.paused = False
        
        logger.info("Video tracker initialized")
    
    def initialize_video_source(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.video_source}")
                return False
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or OUTPUT_VIDEO_FPS
            
            logger.info(f"Video source opened: {width}x{height} @ {fps}fps")
            
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
        
        frame = resize_frame(frame, MAX_FRAME_SIZE)
        
        detections = []
        detection_summary = {}
        
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            detections = self.detector.detect_living_beings(frame)
            
            detections = self.tracker.update(detections)
            
            detection_summary = self.detector.get_detection_summary(detections)
            
            if LOG_DETECTIONS and detections:
                logger.info(f"Frame {self.frame_count}: {len(detections)} detections")
                for det in detections:
                    logger.info(f"  - {det['class_name']}: {det['confidence']:.2f}")
        
        if detections:
            frame = draw_detections(frame, detections)
        
        frame = draw_info_panel(
            frame,
            fps=self.fps_meter.get_fps(),
            detection_count=len(detections),
            detection_summary=detection_summary
        )
        
        return frame, detections, detection_summary
    
    def run(self):
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
                
               
                if self.paused:
                    cv2.waitKey(30)
                    continue
                
                processed_frame, detections, detection_summary = self.process_frame(frame)
                
                if detections:
                    self.audio_feedback.play_detection_alert(detections)
                    self.announcer.announce_detections(detections)
                
                if self.out_writer:
                    self.out_writer.write(processed_frame)
                
                cv2.imshow('Morph1x - Living Being Tracker', processed_frame)
                
                self.fps_meter.update()
                self.frame_count += 1
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: 
                    break
                elif key == ord('p'):  
                    self.paused = not self.paused
                    logger.info(f"Video {'paused' if self.paused else 'resumed'}")
                elif key == ord('r'): 
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
    parser = argparse.ArgumentParser(
        description="Morph1x",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Press 'q' or 'ESC' to quit, 'p' to pause/resume, 'r' to reset tracker"
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
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    video_source = args.source
    if video_source.isdigit():
        video_source = int(video_source)
    
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
