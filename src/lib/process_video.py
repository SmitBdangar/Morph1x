import argparse
import logging
from pathlib import Path
from ..core.object_detector import ObjectDetector, VideoCapture, VideoWriter
from ..core.renderer import HUDRenderer, FPSMeter

def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection - Save Marked Video")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output video file"
    )
    parser.add_argument(
        "--model", type=str, default="model/yolov8n.pt",
        help="Path to YOLO model file"
    )
    parser.add_argument(
        "--classes", nargs="+", default=["person"],
        help="Allowed classes for detection"
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

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("video_processor")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    logger.info(f"Processing video: {args.input}")
    
    detector = ObjectDetector(args.model, args.conf, args.iou)
    video = VideoCapture(args.input)

    if not video.open():
        logger.error("Failed to open video source.")
        return
    renderer = HUDRenderer()
    fps_meter = FPSMeter()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = VideoWriter(args.output, video.frame_width, video.frame_height, video.fps)

    logger.info(f"Output will be saved to: {args.output}")
    logger.info(f"Processing {video.total_frames} frames at {video.fps} FPS")
    
    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = video.read()
        if not ret:
            logger.info("End of stream reached.")
            break
        
        detections = detector.detect(frame, set(args.classes))
        total_detections += len(detections)

        frame = renderer.draw_detections(frame, detections)
        
        fps_meter.update()
        frame = renderer.draw_fps(frame, fps_meter.get_fps())

        writer.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            logger.info(f"Processed {frame_count}/{video.total_frames} frames - "
                       f"Found {len(detections)} objects")

    video.release()
    writer.release()
    
    logger.info("=" * 50)
    logger.info("Processing complete!")
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Output file saved: {args.output}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()