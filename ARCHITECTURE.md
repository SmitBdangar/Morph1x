# Morph1x Architecture

Enterprise-grade system architecture for Morph1x Intelligent Vision System.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT SOURCES                              │
├─────────────────┬──────────────────┬──────────────────┬─────────┤
│   Video Files   │   Video Stream   │   Webcam/IP Cam  │  Images │
│   (MP4, AVI)    │   (RTSP, HTTP)   │   (USB/Network)  │ (JPG)   │
└────────┬────────┴────────┬─────────┴────────┬─────────┴────┬────┘
         │                 │                  │              │
         └─────────────────┴──────────────────┴──────────────┘
                          │
         ┌────────────────▼────────────────┐
         │   CONFIGURATION LAYER            │
         │  ┌──────────────────────────┐  │
         │  │ model_config.yaml        │  │
         │  │ deployment.yaml          │  │
         │  │ • Thresholds             │  │
         │  │ • Model paths            │  │
         │  │ • Device settings        │  │
         │  └──────────────────────────┘  │
         └────────────────┬────────────────┘
                          │
         ┌────────────────▼────────────────────────────────────┐
         │              MORPH1X CORE LAYER                     │
         │                                                     │
         │  ┌──────────────────────────────────────────────┐  │
         │  │         src/core/detection.py                │  │
         │  │      ObjectDetector Class                    │  │
         │  │ ┌──────────────────────────────────────────┐ │  │
         │  │ │ YOLOv8 Model (Ultralytics)               │ │  │
         │  │ │ • Inference Engine                       │ │  │
         │  │ │ • ByteTrack Integration                  │ │  │
         │  │ │ • Frame Processing                       │ │  │
         │  │ └──────────────────────────────────────────┘ │  │
         │  │                    │                         │  │
         │  │                    ▼                         │  │
         │  │ ┌──────────────────────────────────────────┐ │  │
         │  │ │ src/core/postprocessing/utils.py         │ │  │
         │  │ │ • NMS (Non-Maximum Suppression)          │ │  │
         │  │ │ • Confidence Filtering                   │ │  │
         │  │ │ • Result Formatting                      │ │  │
         │  │ └──────────────────────────────────────────┘ │  │
         │  │                    │                         │  │
         │  │                    ▼                         │  │
         │  │ ┌──────────────────────────────────────────┐ │  │
         │  │ │ src/core/visualization/renderer.py       │ │  │
         │  │ │ • HUD Panel Rendering                    │ │  │
         │  │ │ • Bounding Box Drawing                   │ │  │
         │  │ │ • Label Overlays                         │ │  │
         │  │ └──────────────────────────────────────────┘ │  │
         │  └──────────────────────────────────────────────┘  │
         │                                                     │
         └────────────────┬────────────────────────────────────┘
                          │
         ┌────────────────┴───────────────────────────────┐
         │                                                 │
    ┌────▼──────────────┐                   ┌────────────▼─────┐
    │  VISUALIZATION    │                   │   API SERVER     │
    │  (CLI Runner)     │                   │  (FastAPI)       │
    │                  │                   │                  │
    │ run_detection.py │                   │ src/api/main.py  │
    │ • Real-time HUD  │                   │ • /detect        │
    │ • FPS Counter    │                   │ • /stream        │
    │ • Live Display   │                   │ • /health        │
    │                  │                   │ • /model/info    │
    └────┬─────────────┘                   └────────┬─────────┘
         │                                          │
         │                                  ┌───────▼────────┐
         │                                  │  REST Clients  │
         │                                  │                │
         │                                  │ • Web Browser  │
         │                                  │ • Python App   │
         │                                  │ • Mobile App   │
         │                                  │ • Curl/Postman │
         │                                  └────────────────┘
         │
    ┌────▼──────────────────┐
    │  BATCH PROCESSING     │
    │                       │
    │ process_video.py      │
    │ • Multi-file Support  │
    │ • Video Output        │
    │ • Progress Tracking   │
    │                       │
    └────┬──────────────────┘
         │
         ▼
    ┌────────────────┐
    │ Output Files   │
    │ • Processed    │
    │ • Logs         │
    │ • Results      │
    └────────────────┘
```

## Data Flow Architecture

```
INPUT VIDEO FRAME
       │
       ▼
┌─────────────────┐
│ Frame Validation│ ◄─── src/utils.py::validate_frame()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Frame Resizing  │ ◄─── src/utils.py::resize_frame()
└────────┬────────┘      (Maintains aspect ratio)
         │
         ▼
┌──────────────────────────┐
│ YOLOv8 Inference         │ ◄─── src/core/detection.py::ObjectDetector
│ • Model.track()          │      • ByteTrack enabled
│ • Returns: boxes, ids    │      • Confidence & IOU thresholds applied
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Postprocessing           │ ◄─── src/core/postprocessing/utils.py
│ • NMS                    │      • filter_detections()
│ • Confidence Filtering   │      • apply_nms()
│ • Format Results         │      • format_detections()
└────────┬─────────────────┘
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
┌──────────────────┐     ┌─────────────────┐
│ Visualization    │     │ API Response    │
│                  │     │ (JSON)          │
│ HUD Panel        │     │ • Detections    │
│ Bounding Boxes   │     │ • Confidence    │
│ Labels           │     │ • Tracking IDs  │
│ FPS Counter      │     │ • Timestamps    │
└────┬─────────────┘     └────────┬────────┘
     │                           │
     ▼                           ▼
DISPLAY OUTPUT              API CLIENTS
(OpenCV Window)         (Web/Mobile/Desktop)
```

## Module Dependencies Graph

```
┌─────────────────────┐
│  config/            │
│  • model_config     │
│  • deployment       │
└──────┬──────────────┘
       │
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 ▼
┌──────────────────┐            ┌─────────────────┐
│  src/utils.py    │            │  src/core/      │
│                  │            │  • detection    │
│ • FPSMeter       │            │  • postproc     │
│ • validate_frame │            │  • visualization│
│ • resize_frame   │            │                 │
│ • load_config    │            │ ObjectDetector  │
└────┬─────┬──────┘            │ HUDRenderer     │
     │     │                   │ Postprocessing  │
     │     └─────────┬─────────┘                 │
     │               │                          │
     │               ▼                          │
     │        ┌───────────────┐                │
     │        │ src/scripts/  │                │
     │        │               │                │
     │        │ • run_detect  │────────────────┘
     │        │ • process_vid │
     │        └───────┬───────┘
     │                │
     ▼                ▼
┌──────────────────────────┐
│  CLI / Script Usage      │
│                          │
│  Display Output          │
│  File Output             │
└──────────────────────────┘


     ┌────────────────┐
     │ src/utils.py   │
     │ src/core/      │
     └────┬───────────┘
          │
          ▼
    ┌────────────────┐
    │  src/api/      │
    │  main.py       │
    │                │
    │  FastAPI App   │
    │  • Endpoints   │
    │  • Config      │
    └────┬───────────┘
         │
         ▼
    REST API Clients
```

## Component Interactions

### 1. Detection Pipeline

```
Video/Image Input
       ↓
Validation & Resizing (utils.py)
       ↓
YOLOv8 Inference (detection.py)
       ↓
Track Assignment (ByteTrack)
       ↓
Postprocessing (postprocessing/utils.py)
       ↓
Visualization (visualization/renderer.py)
       ↓
Output (Display/File/API)
```

### 2. CLI Execution Flow

```
run_detection.py
    ↓
Load Config
    ↓
Initialize ObjectDetector
    ↓
Initialize HUDRenderer
    ↓
Open Video Source
    ↓
LOOP:
  • Read Frame
  • Detect Objects
  • Render HUD
  • Display
  • Handle Input
    ↓
Release Resources
```

### 3. API Execution Flow

```
FastAPI Server Starts
    ↓
Load Configuration
    ↓
Initialize ObjectDetector
    ↓
Initialize HUDRenderer
    ↓
Listen on /0.0.0.0:8000
    ↓
CLIENT REQUEST
    ↓
Process Request
  • Validate Input
  • Run Detection
  • Format Response
    ↓
Send JSON Response
```

## Technology Stack

```
┌─────────────────────────────────────┐
│  PRESENTATION LAYER                  │
├─────────────────────────────────────┤
│ • OpenCV (Display)                   │
│ • FastAPI (REST API)                 │
│ • Swagger UI (API Docs)              │
└─────────────────────────────────────┘
           ↑
           │
┌─────────────────────────────────────┐
│  APPLICATION LAYER                   │
├─────────────────────────────────────┤
│ • Python 3.8+                        │
│ • Pydantic (Validation)              │
│ • PyYAML (Config)                    │
└─────────────────────────────────────┘
           ↑
           │
┌─────────────────────────────────────┐
│  DETECTION LAYER                     │
├─────────────────────────────────────┤
│ • YOLOv8 (Ultralytics)               │
│ • ByteTrack (Tracking)               │
│ • NumPy (Array Operations)           │
└─────────────────────────────────────┘
           ↑
           │
┌─────────────────────────────────────┐
│  DEEP LEARNING FRAMEWORKS            │
├─────────────────────────────────────┤
│ • PyTorch (Inference Engine)         │
│ • TorchVision (Computer Vision)      │
│ • CUDA (GPU Acceleration)            │
└─────────────────────────────────────┘
```

## Scalability Architecture

```
Single Instance (Current)
┌──────────────────┐
│  Morph1x App     │
│  • Detector      │
│  • Renderer      │
│  • API           │
└──────────────────┘

Load Balanced (Future)
┌────────────────┬────────────────┬────────────────┐
│ Morph1x #1     │ Morph1x #2     │ Morph1x #3     │
│ • Detector     │ • Detector     │ • Detector     │
│ • Renderer     │ • Renderer     │ • Renderer     │
│ • API          │ • API          │ • API          │
└────────┬───────┴────────┬───────┴────────┬───────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                   Load Balancer
                   (Nginx/HAProxy)

Distributed (Future)
┌────────────────────┐
│  Message Queue     │
│  (RabbitMQ/Redis)  │
└────────┬───────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
    ▼         ▼        ▼        ▼
  Worker   Worker   Worker   Worker
  (GPU)    (GPU)    (CPU)    (CPU)
```

## Configuration & Dependency Hierarchy

```
deployment.yaml (Top-level)
├── Environment settings
├── API configuration
├── Logging configuration
└── Video source settings

model_config.yaml (Mid-level)
├── Model selection
├── Inference parameters
├── Class definitions
└── Visualization settings

Python Code (Implementation)
├── src/core/* (Core logic)
├── src/scripts/* (CLI tools)
├── src/api/* (API server)
└── src/utils.py (Utilities)
```

## Security Architecture

```
Input Validation
    ↓
File Validation (frames, configs)
    ↓
Bounds Checking (array access)
    ↓
Type Checking (Pydantic models)
    ↓
Error Handling (exceptions caught)
    ↓
Logging (audit trail)
    ↓
Safe Output
```

## Performance Optimization Strategy

```
PERFORMANCE LAYERS
├─ Level 1: Input Optimization
│  ├── Frame resizing
│  ├── Format conversion
│  └── Memory efficiency
│
├─ Level 2: Model Optimization
│  ├── Model selection (nano→large)
│  ├── Quantization (optional)
│  ├── Batch processing
│  └── GPU acceleration
│
├─ Level 3: Processing Optimization
│  ├── Confidence threshold tuning
│  ├── IOU threshold adjustment
│  ├── NMS optimization
│  └── Caching
│
└─ Level 4: Output Optimization
   ├── Async processing
   ├── Streaming compression
   ├── Format optimization
   └── Buffer management
```

## Monitoring & Logging Architecture

```
┌─────────────────────────────────────┐
│  APPLICATION METRICS                 │
├─────────────────────────────────────┤
│ • FPS (Frames per second)            │
│ • Detection count                    │
│ • Processing time                    │
│ • Memory usage                       │
│ • GPU utilization                    │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  LOGGING SYSTEM                      │
├─────────────────────────────────────┤
│ • INFO: Normal operations            │
│ • WARNING: Potential issues          │
│ • ERROR: Failures                    │
│ • DEBUG: Detailed information        │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  LOG OUTPUT                          │
├─────────────────────────────────────┤
│ • Console (stdout/stderr)            │
│ • File (logs/morph1x.log)            │
│ • Cloud logging (optional)           │
└─────────────────────────────────────┘
```

## Testing Architecture

```
┌─────────────────────────────────────┐
│  TEST PYRAMID                        │
├─────────────────────────────────────┤
│                △                     │
│               ╱ ╲ E2E Tests          │
│              ╱   ╲                   │
│             ╱─────╲                  │
│            ╱       ╲ Integration     │
│           ╱─────────╲               │
│          ╱           ╲              │
│         ╱─────────────╲            │
│        ╱               ╲           │
│       ╱─────────────────╲         │
│      ╱ Unit Tests        ╲        │
│     ╱─────────────────────╲      │
└────────────────────────────────────┘

Unit Tests (tests/unit/)
├── test_detector.py
│   ├── Detection output format
│   ├── Invalid frame handling
│   ├── Confidence filtering
│   └── Class filtering
│
├── test_utils.py
│   ├── FPS calculation
│   ├── Frame validation
│   └── Frame resizing
│
└── test_postprocessing.py
    ├── NMS algorithm
    ├── Filtering logic
    └── Result formatting

Integration Tests (tests/integration/)
├── test_api.py
│   ├── /detect endpoint
│   ├── /stream endpoint
│   ├── /health endpoint
│   └── Configuration endpoints
```

## Deployment Architecture

```
LOCAL DEVELOPMENT
┌──────────────────┐
│  Desktop/Laptop  │
│  • Python venv   │
│  • Local config  │
│  • Direct files  │
└──────────────────┘

CONTAINER (Docker)
┌──────────────────┐
│  Docker Image    │
│  • Isolated env  │
│  • Versioned     │
│  • Reproducible  │
└────────┬─────────┘
         │
    Docker Registry

CLOUD DEPLOYMENT
┌──────────────────────────────────┐
│ Cloud Platform (AWS/GCP/Azure)   │
├──────────────────────────────────┤
│ • Container Service (ECS/GKE)    │
│ • Load Balancer                  │
│ • Auto-scaling                   │
│ • Monitoring                     │
│ • Logging                        │
└──────────────────────────────────┘
```

## Data Flow for Different Use Cases

### Use Case 1: Real-time Detection CLI

```
Video File/Webcam
    ↓
run_detection.py
    ↓
Load Config → ObjectDetector → HUDRenderer
    ↓
LOOP:
  Read Frame → Detect → Render → Display → Handle Input
    ↓
Release Resources
```

### Use Case 2: Batch Video Processing

```
Multiple Videos
    ↓
process_video.py
    ↓
FOR each video:
  Open → LOOP:
    Read Frame → Detect → Render → Write Output
  Close
    ↓
Complete
```

### Use Case 3: REST API Integration

```
Client Request (Image/Stream)
    ↓
FastAPI Server
    ↓
Validate Input → Load Frame → Detect → Postprocess
    ↓
Format Response (JSON)
    ↓
Send to Client
```

### Use Case 4: Web Application Integration

```
Web Browser/Mobile App
    ↓
Send Image/Video URL
    ↓
REST API Endpoint
    ↓
Download Resource → Process → Return Detections
    ↓
Display Results in UI
```

## Error Handling & Recovery

```
┌─────────────────┐
│  Error Occurs   │
└────────┬────────┘
         │
    ┌────▼────────────────────────┐
    │  Error Classification       │
    └┬────────────────────────────┘
     │
   ┌─┴──────────────────────────────────┐
   │                                    │
   ▼                                    ▼
RECOVERABLE                     FATAL ERROR
│                               │
├─ Retry Logic                  ├─ Log Error
├─ Fallback Options             ├─ Notify Admin
├─ Graceful Degradation         ├─ Clean Shutdown
└─ Continue Processing          └─ Exit (1)
```

## Resource Management

```
Memory Management
├── Frame buffers (pre-allocated)
├── Model weights (GPU memory)
├── Detection cache
└── Result buffers

CPU/GPU Management
├── Batch size optimization
├── Worker thread allocation
├── GPU VRAM monitoring
└── Thermal management

Disk I/O
├── Input video streaming
├── Output video writing
├── Log file rotation
└── Cache management
```

## API Gateway Pattern

```
Client Requests
    ↓
┌────────────────┐
│ API Gateway    │
│ (Nginx/HAProxy)│
└────┬───────────┘
     │
   ┌─┴──────────┬──────────┬──────────┐
   │            │          │          │
   ▼            ▼          ▼          ▼
 API-1        API-2      API-3      API-N
(Instance)   (Instance) (Instance) (Instance)
   │            │          │          │
   └─┬──────────┴──────────┴──────────┘
     │
Shared Database / Cache
```

---

**This architecture ensures:**
- ✅ Scalability
- ✅ Reliability
- ✅ Maintainability
- ✅ Performance
- ✅ Security
- ✅ Extensibility