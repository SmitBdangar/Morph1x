import argparse
import logging
import cv2
import numpy as np
from src.core.object_detector import ObjectDetector, VideoCapture, VideoWriter
from src.core.renderer import HUDRenderer, FPSMeter
from src.core.Post_processing import PostProcessor

def main():
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="YOLO Object Detection Runner")
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source (0 for webcam or path to video file)"
    )
    parser.add_argument(
        "--model", type=str, default="model/yolov8n.pt",
        help="Path to YOLO model file"
    )
    parser.add_argument(
        "--classes", nargs="+", default=["person", "car", "dog"],
        help="Allowed classes for detection"
    )
    parser.add_argument(
        "--output", type=str, default="output/output.mp4",
        help="Optional output video path"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold"
    )
    args = parser.parse_args()

    # --- Logging setup ---
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("runner")

    logger.info("Starting YOLO Object Detector...")
    detector = ObjectDetector(args.model, args.conf, args.iou)
    video = VideoCapture(args.source)

    if not video.open():
        logger.error("Failed to open video source.")
        return

    # --- Initialization ---
    renderer = HUDRenderer()
    fps_meter = FPSMeter()

    panel_width = 100
    combined_width = video.frame_width + panel_width
    
    writer = VideoWriter(args.output, combined_width, video.frame_height, video.fps)

    while True:
        ret, frame = video.read()
        if not ret:
            logger.info("End of stream or read error.")
            break

        # 1. Get detections
        detections = detector.detect(frame, set(args.classes))

        # 2. Draw detection boxes and labels on the frame
        frame = renderer.draw_detections(frame, detections)
        
        # 3. Update and draw FPS
        fps_meter.update()
        frame = renderer.draw_fps(frame, fps_meter.get_fps())

        # 4. Create a blank panel
        panel = np.zeros((video.frame_height, panel_width, 3), dtype=np.uint8)
        
        # 5. Draw the table-style HUD on the panel
        panel = renderer.draw_panel(panel, detections)

        # 6. Combine the video frame and the panel
        combined_frame = np.hstack((frame, panel))

        # 7. Show and write the combined frame
        cv2.imshow("YOLO Detection", combined_frame)
        writer.write(combined_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()
    logger.info("Detection finished.")

if __name__ == "__main__":
    main()