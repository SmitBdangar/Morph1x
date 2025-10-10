"""
Video processing script for Morph1x.
Processes video files with detection and saves output.
"""

import cv2
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import ObjectDetector, HUDRenderer
from src.utils import FPSMeter, validate_frame, resize_frame, load_config

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process video files with detection."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize video processor.
        
        Args:
            config_path: Path to model configuration file.
        """
        self.config = load_config(config_path)
        
        self.detector = ObjectDetector(
            model_path=self.config["model"]["path"],
            conf_threshold=self.config["inference"]["confidence_threshold"],
            iou_threshold=self.config["inference"]["iou_threshold"]
        )
        
        self.renderer = HUDRenderer(config={
            "colors": self.config["visualization"]["colors"],
            "panel_bg_color": self.config["visualization"]["hud"]["panel_bg_color"],
            "header_color": self.config["visualization"]["hud"]["header_color"],
            "accent_color": self.config["visualization"]["hud"]["accent_color"],
            "divider_color": self.config["visualization"]["hud"]["divider_color"]
        })
        
        self.fps_meter = FPSMeter()
    
    def process(self, video_path: str, output_path: str = None, display: bool = True) -> None:
        """
        Process video file with detection.
        
        Args:
            video_path: Path to input video.
            output_path: Path to save output video (optional).
            display: Whether to display the video in real-time.
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video info: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            logger.info(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                if not validate_frame(frame):
                    logger.warning(f"Invalid frame at {frame_count}")
                    continue
                
                # Resize for processing
                proc_frame = resize_frame(frame, (
                    self.config["visualization"]["frame_resize"]["max_width"],
                    self.config["visualization"]["frame_resize"]["max_height"]
                ))
                
                # Detect
                detections = self.detector.detect(
                    proc_frame,
                    self.config["classes"]["allowed"]
                )
                
                # Draw detections
                output_frame = self.renderer.draw_detections(proc_frame.copy(), detections)
                
                # Add FPS
                self.fps_meter.update()
                fps_val = self.fps_meter.get_fps()
                cv2.putText(
                    output_frame,
                    f"FPS: {fps_val}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Display if requested
                if display:
                    cv2.imshow("Morph1x - Video Processing", output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Processing stopped by user")
                        break
                
                # Write to output video
                if out:
                    out.write(output_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            logger.info(f"Processing complete. Total frames: {frame_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process video with Morph1x detection")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-c", "--config", default="config/model_config.yaml", help="Config file path")
    parser.add_argument("--no-display", action="store_true", help="Disable real-time display")
    
    args = parser.parse_args()
    
    processor = VideoProcessor(args.config)
    processor.process(args.input, args.output, display=not args.no_display)


if __name__ == "__main__":
    main()