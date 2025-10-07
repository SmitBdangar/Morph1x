import cv2
import argparse
import sys
import logging
from pathlib import Path
from typing import Union

from .config import (
    DEFAULT_VIDEO_SOURCE, MAX_FRAME_SIZE, PROCESS_EVERY_N_FRAMES,
    OUTPUT_VIDEO_FPS, LOG_LEVEL, LOG_DETECTIONS
)
from .detection import create_detector, DetectionTracker
from .utils import FPSMeter, draw_detections, resize_frame, validate_frame
from .ui import apply_hud
 

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoTracker:
    
    def __init__(self, video_source: Union[int, str] = None, 
                 model_path: str = None, output_path: str = None):
        
        self.video_source = video_source or DEFAULT_VIDEO_SOURCE
        self.output_path = output_path
        self.cap = None
        self.out_writer = None
        
        self.detector = create_detector(model_path)
        self.tracker = DetectionTracker()
        self.fps_meter = FPSMeter()
        self.audio_feedback = None
        self.announcer = None
        
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
        if not validate_frame(frame):
            return frame, [], {}
        
        frame = resize_frame(frame, MAX_FRAME_SIZE)
        
        detections = []
        detection_summary = {}
        
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            detections = self.detector.detect_living_beings(frame)
            
            current_fps = self.fps_meter.get_fps() or OUTPUT_VIDEO_FPS
            detections = self.tracker.update(detections, fps=current_fps)
            
            detection_summary = self.detector.get_detection_summary(detections)
            
            if LOG_DETECTIONS and detections:
                logger.info(f"Frame {self.frame_count}: {len(detections)} detections")
                for det in detections:
                    logger.info(f"  - {det['class_name']}: {det['confidence']:.2f}")
        
        if detections:
            frame = draw_detections(frame, detections)
        
        frame = apply_hud(
            frame,
            fps=self.fps_meter.get_fps(),
            det_count=len(detections),
            summary=detection_summary,
            objects=detections
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
                
                # Audio removed
                
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


def apply_config_preset(preset: str):
    """Apply configuration preset by modifying config module variables."""
    import src.config as config
    
    if preset == 'balanced':
        # Default balanced settings (already in config.py)
        print("Using balanced configuration preset")
        
    elif preset == 'high-accuracy':
        # Lower thresholds for more detections
        config.CONFIDENCE_THRESHOLD = 0.3
        config.IOU_THRESHOLD = 0.3
        config.MAX_DETECTIONS = 200
        config.PROCESS_EVERY_N_FRAMES = 1
        config.MAX_FRAME_SIZE = (1920, 1080)
        config.LOG_LEVEL = "INFO"
        print("Using high-accuracy configuration preset")
        
    elif preset == 'performance':
        # Higher thresholds for faster processing
        config.CONFIDENCE_THRESHOLD = 0.7
        config.IOU_THRESHOLD = 0.6
        config.MAX_DETECTIONS = 50
        config.PROCESS_EVERY_N_FRAMES = 3
        config.MAX_FRAME_SIZE = (640, 480)
        config.LOG_LEVEL = "WARNING"
        print("Using performance configuration preset")
    
    elif preset == 'long-range':
        # Tuned for small/distant objects (more sensitive, higher resolution)
        config.CONFIDENCE_THRESHOLD = 0.25
        config.IOU_THRESHOLD = 0.35
        config.MAX_DETECTIONS = 300
        config.PROCESS_EVERY_N_FRAMES = 1
        config.MAX_FRAME_SIZE = (1920, 1080)
        config.LOG_LEVEL = "INFO"
        print("Using long-range configuration preset")
        
    elif preset == 'development':
        # Debug settings
        config.LOG_LEVEL = "DEBUG"
        config.LOG_DETECTIONS = True
        config.SHOW_FPS = True
        config.ENABLE_AUDIO_FEEDBACK = False
        print("Using development configuration preset")


def main():
    parser = argparse.ArgumentParser(
        description="Morph1x - Real-time detection and tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use webcam (default)
  python -m src.main
  
  # Process video file
  python -m src.main --source "video.mp4"
  
  # High accuracy mode with TTS
  python -m src.main --preset high-accuracy --tts
  
  # Save processed video
  python -m src.main --source "input.mp4" --output "output.mp4"
  
  # Performance mode for low-end devices
  python -m src.main --preset performance
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=str(DEFAULT_VIDEO_SOURCE),
        help='Video source: camera index (0,1,2...) or video file path (default: 0 for webcam)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to YOLOv8 model file (default: uses yolov8n.pt from models/)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output video file path (optional, for saving processed video)'
    )
    
    parser.add_argument(
        '--preset', '-p',
        choices=['balanced', 'high-accuracy', 'performance', 'development', 'long-range'],
        default='balanced',
        help='Configuration preset (default: balanced)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        help='Detection confidence threshold (0.0-1.0, overrides preset)'
    )

    parser.add_argument(
        '--resolution', '-r',
        type=str,
        help='Processing resolution WIDTHxHEIGHT (e.g., 1280x720). Overrides preset.'
    )

    parser.add_argument(
        '--meters-per-pixel', '--mpp',
        dest='meters_per_pixel',
        type=float,
        help='Calibration scale: meters per pixel (e.g., 0.002).'
    )

    parser.add_argument(
        '--pixels-per-meter', '--ppm',
        dest='pixels_per_meter',
        type=float,
        help='Calibration scale: pixels per meter (e.g., 50).'
    )
    
    # Audio options removed
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--list-classes',
        action='store_true',
        help='List all detectable classes and exit'
    )
    
    args = parser.parse_args()
    
    # Handle list-classes option
    if args.list_classes:
        from .config import PRIMARY_LIVING_BEINGS
        print("Morph1x - Detectable Living Beings:")
        print("=" * 40)
        for class_id, class_name in PRIMARY_LIVING_BEINGS.items():
            print(f"  {class_id:2d}. {class_name}")
        print(f"\nTotal classes: {len(PRIMARY_LIVING_BEINGS)}")
        return 0
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Apply configuration presets
    apply_config_preset(args.preset)
    
    # Override confidence if specified
    if args.confidence is not None:
        if not 0.0 <= args.confidence <= 1.0:
            print(f"Error: Confidence must be between 0.0 and 1.0, got {args.confidence}")
            sys.exit(1)
        import src.config as config
        config.CONFIDENCE_THRESHOLD = args.confidence
        print(f"Confidence threshold set to: {args.confidence}")

    # Override processing resolution if specified
    if args.resolution:
        try:
            width_str, height_str = args.resolution.lower().split('x')
            width, height = int(width_str), int(height_str)
            if width <= 0 or height <= 0:
                raise ValueError("Resolution must be positive")
            import src.config as config
            config.MAX_FRAME_SIZE = (width, height)
            print(f"Processing resolution set to: {width}x{height}")
        except Exception as e:
            print(f"Error: Invalid --resolution '{args.resolution}'. Use format WIDTHxHEIGHT, e.g., 1280x720")
            sys.exit(1)

    # Calibration scale overrides
    if args.meters_per_pixel is not None and args.pixels_per_meter is not None:
        print("Error: Use either --meters-per-pixel or --pixels-per-meter, not both.")
        sys.exit(1)
    if args.meters_per_pixel is not None:
        import src.config as config
        if args.meters_per_pixel <= 0:
            print("Error: --meters-per-pixel must be > 0")
            sys.exit(1)
        config.METERS_PER_PIXEL = float(args.meters_per_pixel)
        print(f"Meters per pixel set to: {config.METERS_PER_PIXEL}")
    if args.pixels_per_meter is not None:
        import src.config as config
        if args.pixels_per_meter <= 0:
            print("Error: --pixels-per-meter must be > 0")
            sys.exit(1)
        config.METERS_PER_PIXEL = 1.0 / float(args.pixels_per_meter)
        print(f"Meters per pixel set to: {config.METERS_PER_PIXEL} (from pixels per meter)")
    
    # Parse video source
    video_source = args.source
    if video_source.isdigit():
        video_source = int(video_source)
        print(f"Using camera {video_source}")
    else:
        # Check if file exists
        if not Path(video_source).exists():
            print(f"Error: Video file '{video_source}' not found!")
            sys.exit(1)
        print(f"Processing video file: {video_source}")
    
    # Set default model path if not provided
    model_path = args.model
    if not model_path:
        default_model = Path(__file__).parent.parent / "models" / "yolov8n.pt"
        if default_model.exists():
            model_path = str(default_model)
            print(f"Using default model: {model_path}")
        else:
            print("Warning: No model file found, using YOLOv8 default")
    
    try:
        print("\nStarting Morph1x...")
        print("Controls: 'q' to quit, 'p' to pause, 'r' to reset")
        
        tracker = VideoTracker(
            video_source=video_source,
            model_path=model_path,
            
            output_path=args.output
        )
        
        success = tracker.run()
        
        if success:
            logger.info("Video tracking completed successfully")
            if args.output:
                print(f"Output saved to: {args.output}")
        else:
            logger.error("Video tracking failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
