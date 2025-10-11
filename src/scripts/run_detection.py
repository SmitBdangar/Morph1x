import cv2
import sys
import logging
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import ObjectDetector, HUDRenderer
from src.utils import FPSMeter, validate_frame, resize_frame, load_config

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionRunner:
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        logger.info("start")
        
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
        self.hud_width = self.config["visualization"]["hud"]["panel_width"]
    
    def run(self, video_source: str) -> None:
        logger.info(f"Starting detection on: {video_source}")
        
        if video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
        else:
            if not Path(video_source).exists():
                logger.error(f"Video file not found: {video_source}")
                return
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {video_source}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video resolution: {frame_width}x{frame_height}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                if not validate_frame(frame):
                    logger.warning("Invalid frame skipped")
                    continue
                
                frame = resize_frame(frame, (
                    self.config["visualization"]["frame_resize"]["max_width"],
                    self.config["visualization"]["frame_resize"]["max_height"]
                ))

                detections = self.detector.detect(
                    frame,
                    self.config["classes"]["allowed"]
                )

                frame_with_boxes = self.renderer.draw_detections(frame.copy(), detections)

                active_ids = list(set([d["unique_id"] for d in detections]))
                active_ids.sort()
                
                panel = np.zeros(
                    (frame_with_boxes.shape[0], self.hud_width, 3),
                    dtype=np.uint8
                )
                panel = self.renderer.draw_panel(
                    panel,
                    "MORPH1X HUD",
                    len(detections),
                    active_ids
                )
                
                combined = np.hstack((frame_with_boxes, panel))
                
                self.fps_meter.update()
                fps = self.fps_meter.get_fps()
                
                cv2.putText(
                    combined,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("Morph1x", combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit")
                    break
        
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("stop successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Morph1x",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("source", help="Video file path or camera index (0 for webcam)")
    parser.add_argument("-c", "--config", default="config/model_config.yaml", help="Model config file")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("=" * 60)
    
    try:
        runner = DetectionRunner(args.config)
        runner.run(args.source)
    
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()