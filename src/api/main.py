import cv2
import numpy as np
import yaml
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from src.core import ObjectDetector, HUDRenderer, filter_detections, format_detections
from src.utils import FPSMeter, validate_frame, resize_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path("config/model_config.yaml")
deploy_config_path = Path("config/deployment.yaml")

with open(config_path) as f:
    MODEL_CONFIG = yaml.safe_load(f)

with open(deploy_config_path) as f:
    DEPLOY_CONFIG = yaml.safe_load(f)

class DetectionResponse(BaseModel):
    total_detections: int
    detections: list
    fps: float

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    version: str

app = FastAPI(
    title="Morph1x",
)

detector = ObjectDetector(
    model_path=MODEL_CONFIG["model"]["path"],
    conf_threshold=MODEL_CONFIG["inference"]["confidence_threshold"],
    iou_threshold=MODEL_CONFIG["inference"]["iou_threshold"]
)

renderer = HUDRenderer(config={
    "colors": MODEL_CONFIG["visualization"]["colors"],
    "panel_bg_color": MODEL_CONFIG["visualization"]["hud"]["panel_bg_color"],
    "header_color": MODEL_CONFIG["visualization"]["hud"]["header_color"],
    "accent_color": MODEL_CONFIG["visualization"]["hud"]["accent_color"],
    "divider_color": MODEL_CONFIG["visualization"]["hud"]["divider_color"]
})

fps_meter = FPSMeter()

logger.info("successfully")

@app.get("/health", response_model=HealthResponse)
async def health():
    model_info = detector.get_model_info()
    return HealthResponse(
        status="healthy",
        model=model_info["model_name"],
        device=MODEL_CONFIG["inference"]["device"],
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if not validate_frame(frame):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        frame = resize_frame(frame, (
            MODEL_CONFIG["visualization"]["frame_resize"]["max_width"],
            MODEL_CONFIG["visualization"]["frame_resize"]["max_height"]
        ))
        
        detections = detector.detect(
            frame,
            MODEL_CONFIG["classes"]["allowed"]
        )
        
        detections = filter_detections(
            detections,
            conf_threshold=MODEL_CONFIG["inference"]["confidence_threshold"],
            max_detections=MODEL_CONFIG["inference"]["max_detections"]
        )
        
        fps_meter.update()
        
        return DetectionResponse(
            total_detections=len(detections),
            detections=format_detections(detections)["detections"],
            fps=fps_meter.get_fps()
        )
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream")
async def stream_video():
    try:
        video_source = DEPLOY_CONFIG["video"]["source"]
        
        if video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise HTTPException(status_code=404, detail="Video not available")
        
        def generate():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if not validate_frame(frame):
                    continue
                
                frame = resize_frame(frame, (
                    MODEL_CONFIG["visualization"]["frame_resize"]["max_width"],
                    MODEL_CONFIG["visualization"]["frame_resize"]["max_height"]
                ))
                
                detections = detector.detect(
                    frame,
                    MODEL_CONFIG["classes"]["allowed"]
                )
                
                frame = renderer.draw_detections(frame, detections)
                
                fps_meter.update()
                fps = fps_meter.get_fps()
                cv2.putText(
                    frame,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + f'{len(frame_bytes)}'.encode() + b'\r\n\r\n'
                       + frame_bytes + b'\r\n')
        
        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        cap.release()


@app.get("/model/info")
async def model_info():
    return detector.get_model_info()

@app.get("/config/model")
async def get_model_config():
    return MODEL_CONFIG


@app.get("/config/deployment")
async def get_deployment_config():
    return DEPLOY_CONFIG


@app.post("/config/update")
async def update_config(config: dict):
    try:
        detector.conf_threshold = config.get("conf_threshold", detector.conf_threshold)
        detector.iou_threshold = config.get("iou_threshold", detector.iou_threshold)
        
        logger.info(f"Configuration updated: {config}")
        return {"status": "success", "message": "Configuration updated"}
    
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "application": "Morph1x - Intelligent Vision System",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /detect": "Detect objects in image",
            "GET /stream": "Stream video with detections",
            "GET /model/info": "Get model information",
            "GET /config/model": "Get model configuration",
            "GET /config/deployment": "Get deployment configuration",
            "POST /config/update": "Update configuration",
            "GET /docs": "Interactive API documentation"
        }
    }

@app.on_event("startup")
async def startup():
    logger.info("Morph1x API starting")
    logger.info(f"Environment: {DEPLOY_CONFIG.get('environment', 'development')}")
    logger.info(f"Model: {MODEL_CONFIG['model']['name']}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Morph1x API shutting down")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=DEPLOY_CONFIG["api"]["host"],
        port=DEPLOY_CONFIG["api"]["port"],
        workers=DEPLOY_CONFIG["api"]["workers"],
        reload=DEPLOY_CONFIG["api"]["reload"],
        log_level=DEPLOY_CONFIG["api"]["log_level"]
    )