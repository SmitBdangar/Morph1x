import argparse
import logging
import cv2
from src.core.object_detector import ObjectDetector, VideoCapture, VideoWriter

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

    writer = VideoWriter(args.output, video.frame_width, video.frame_height, video.fps)

    while True:
        ret, frame = video.read()
        if not ret:
            logger.info("End of stream or read error.")
            break

        detections = detector.detect(frame, set(args.classes))

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()
    logger.info("Detection finished.")

if __name__ == "__main__":
    main()
