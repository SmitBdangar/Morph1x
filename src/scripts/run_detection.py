import cv2
import sys
import logging
import argparse
from pathlib import Path
import numpy as np
from typing import List, Dict, Set

# --- Ensure project root is on path ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Simple Package Imports ---
from src.core import ObjectDetector, HUDRenderer
from src.utils import FPSMeter, validate_frame, resize_frame, load_config

# Set logging level to DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CRITICAL SANITIZATION FUNCTION (Layer 2) ---
def sanitize_detections(detections: List[Dict]) -> List[Dict]:
    """
    Performs an aggressive cleanup on all 'unique_id' keys in the detections list.
    """
    for det in detections:
        unique_id = det.get("unique_id", "").strip()
        
        if "???" in unique_id:
            cleaned_id = unique_id.replace("??? ", "ID ").replace("???", "ID").strip()
            logger.debug(f"Main Loop Sanitized ID (Layer 2): From '{unique_id}' to '{cleaned_id}'")
            det["unique_id"] = cleaned_id
            
    return detections


class DetectionRunner:
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        logger.info("start")
        
        # Resolve config path relative to the project root
        absolute_config_path = PROJECT_ROOT / config_path
        self.config = load_config(str(absolute_config_path))
        
        # Resolve model path relative to the project root
        absolute_model_path = PROJECT_ROOT / self.config["model"]["path"]
        
        self.detector = ObjectDetector(
            model_path=str(absolute_model_path),
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
            # Resolve video path to absolute path
            video_path = Path(video_source)
            if not video_path.is_absolute():
                video_path = PROJECT_ROOT / video_path
            
            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return
            cap = cv2.VideoCapture(str(video_path))
        
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

                # Step 1: Detect objects (Layer 1 cleanup in ObjectDetector)
                detections: List[Dict] = self.detector.detect(
                    frame,
                    self.config["classes"]["allowed"]
                )

                # Step 2: SANITIZE THE ENTIRE LIST (Layer 2 cleanup)
                detections = sanitize_detections(detections)
                
                # --- FINAL SANITY CHECK PRINT ---
                for det in detections:
                    if "???" in det["unique_id"]:
                        print(f"!!! CRITICAL FAIL: ID is still '{det['unique_id']}' right before render!")
                # --- END SANITY CHECK ---
                    
                # Step 3: Draw bounding boxes (Layer 3 cleanup in HUDRenderer)
                frame_with_boxes = self.renderer.draw_detections(frame.copy(), detections)

                # Step 4: Create the active_ids list
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