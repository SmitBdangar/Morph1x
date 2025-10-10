# Morph1x Project Structure

project structure for Morph1x - Intelligent Vision System.

## Directory Tree

```
MORPH1X/
│
├── data/                           # Data management
│   ├── raw/                        # Original, unmodified data
│   │   └── .gitkeep
│   ├── processed/                  # Transformed/processed data
│   │   └── .gitkeep
│   └── external/                   # Third-party data, pre-trained weights
│       └── .gitkeep
│
├── notebooks/                      # Jupyter notebooks for experimentation
│   ├── .gitkeep
│   ├── 01_data_exploration.ipynb
│   └── 02_model_analysis.ipynb
│
├── models/                         # Model storage
│   ├── current/                    # Active model in use
│   │   └── yolov8n.pt             # YOLOv8 model (auto-downloaded)
│   ├── archives/                   # Previous model versions
│   │   └── .gitkeep
│   └── .gitkeep
│
├── src/                            # Source code
│   ├── core/                       # Core business logic
│   │   ├── __init__.py            # Package exports
│   │   ├── detection.py           # ObjectDetector class
│   │   │
│   │   ├── postprocessing/        # Output transformation
│   │   │   ├── __init__.py
│   │   │   └── utils.py           # NMS, filtering utilities
│   │   │
│   │   └── visualization/         # Drawing and visualization
│   │       ├── __init__.py
│   │       └── renderer.py        # HUDRenderer class
│   │
│   ├── api/                        # Application serving layer
│   │   ├── __init__.py
│   │   └── main.py                # FastAPI application
│   │
│   ├── scripts/                    # Utility scripts
│   │   ├── run_detection.py       # CLI runner with HUD
│   │   ├── process_video.py       # Batch video processing
│   │   └── .gitkeep
│   │
│   ├── __init__.py
│   └── utils.py                    # Shared utilities (FPS, config, etc.)
│
├── config/                         # Configuration files
│   ├── model_config.yaml           # Model hyperparameters & paths
│   ├── deployment.yaml             # Environment-specific settings
│   └── .gitkeep
│
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   │   ├── __init__.py
│   │   ├── test_detector.py
│   │   ├── test_utils.py
│   │   └── test_postprocessing.py
│   │
│   ├── integration/                # Integration tests
│   │   ├── __init__.py
│   │   └── test_api.py
│   │
│   ├── conftest.py                 # Pytest configuration
│   └── .gitkeep
│
├── logs/                           # Application logs
│   └── .gitkeep
│
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup configuration
├── pytest.ini                      # Pytest configuration
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation

```

## File Descriptions

### `/data` - Data Management

- **raw/** - Original data files (videos, images) in their native format
- **processed/** - Transformed data ready for training/testing
- **external/** - Pre-trained models, third-party datasets

### `/notebooks` - Experimentation

Jupyter notebooks for data exploration, model analysis, and prototyping.

### `/models` - Model Storage

- **current/** - Active production model(s)
- **archives/** - Version history of models for rollback

### `/src` - Source Code

#### `core/` - Core Logic

**detection.py**
- `ObjectDetector` class - YOLOv8 inference with tracking
- Methods: `detect()`, `get_model_info()`

**postprocessing/**
- **utils.py** - NMS, filtering, formatting functions
- `filter_detections()` - Confidence-based filtering
- `apply_nms()` - Non-maximum suppression
- `format_detections()` - Response formatting

**visualization/**
- **renderer.py** - `HUDRenderer` class for drawing
- Methods: `draw_panel()`, `draw_detections()`, `draw_detection_box()`

#### `api/` - API Server

**main.py**
- FastAPI application
- Endpoints: `/health`, `/detect`, `/stream`, `/config/*`
- Startup/shutdown handlers

#### `scripts/` - Utility Scripts

**run_detection.py**
- CLI runner with real-time HUD visualization
- Command: `python src/scripts/run_detection.py video.mp4`

**process_video.py**
- Batch video processing with output saving
- Command: `python src/scripts/process_video.py input.mp4 -o output.mp4`

#### `utils.py`

Shared utilities:
- `FPSMeter` - Real-time FPS calculation
- `validate_frame()` - Frame validation
- `resize_frame()` - Aspect-ratio preserving resize
- `load_config()` - YAML configuration loader
- `save_config()` - Configuration saver

### `/config` - Configuration

**model_config.yaml**
```yaml
model:
  name: yolov8n
  path: models/current/yolov8n.pt

inference:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: cuda

classes:
  allowed: [person, car, dog, ...]
```

**deployment.yaml**
```yaml
environment: development
api:
  host: 0.0.0.0
  port: 8000

video:
  source: video.mp4
```

### `/tests` - Test Suite

**unit/**
- `test_detector.py` - ObjectDetector tests
- `test_utils.py` - Utility function tests
- `test_postprocessing.py` - Postprocessing tests

**integration/**
- `test_api.py` - API endpoint tests

### Root Files

**requirements.txt**
```
opencv-python>=4.8.0
ultralytics>=8.0.0
torch>=2.0.0
fastapi>=0.100.0
...
```

**setup.py**
- Package metadata
- Dependencies
- Console script entry points

**pytest.ini**
- Pytest configuration
- Test discovery patterns
- Coverage settings

**.gitignore**
- Python cache and build artifacts
- Virtual environments
- IDE settings
- Large data/model files
- Logs

## Development Workflow

### Setup

```bash
# Clone and setup
git clone <repo>
cd MORPH1X
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Development

1. **Add feature to `src/core/`**
   ```python
   # src/core/detection.py
   def new_method(self, ...):
       ...
   ```

2. **Update configuration in `config/`**
   ```yaml
   # config/model_config.yaml
   new_param: value
   ```

3. **Write tests in `tests/`**
   ```python
   # tests/unit/test_new_feature.py
   def test_new_method():
       ...
   ```

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

### Production

1. **Update version in `setup.py`**
2. **Create release notes**
3. **Archive model in `models/archives/`**
4. **Deploy using `config/deployment.yaml`**

## Import Patterns

### Import from Core
```python
from src.core import ObjectDetector, HUDRenderer
from src.core.postprocessing import filter_detections
```

### Import Utilities
```python
from src.utils import FPSMeter, load_config, validate_frame
```

### In Scripts
```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import ObjectDetector
```

## Configuration Management

### Model Settings
Edit `config/model_config.yaml`:
- Model path and parameters
- Inference thresholds
- Allowed classes
- Visualization colors

### Deployment Settings
Edit `config/deployment.yaml`:
- Environment (dev/staging/prod)
- API settings
- Video source
- Logging configuration

## Adding New Features

### 1. New Detector Method

```python
# src/core/detection.py
def new_method(self, ...):
    """Docstring"""
    pass
```

### 2. New Visualization

```python
# src/core/visualization/renderer.py
def draw_custom(self, frame, ...):
    """Docstring"""
    pass
```

### 3. New API Endpoint

```python
# src/api/main.py
@app.post("/new_endpoint")
async def new_endpoint(...):
    """Docstring"""
    pass
```

### 4. New Script

```python
# src/scripts/new_script.py
if __name__ == "__main__":
    main()
```

## Best Practices

- ✅ Keep core logic in `src/core/`
- ✅ Keep configuration in `config/`
- ✅ Write tests for new features
- ✅ Document with docstrings
- ✅ Use type hints
- ✅ Handle errors gracefully
- ✅ Log important events
- ✅ Update README when adding features

## Deployment

```bash
# Package installation
pip install -e .

# Console scripts available
morph1x video.mp4
morph1x-process input.mp4 -o output.mp4

# Or run directly
python src/api/main.py
python src/scripts/run_detection.py video.mp4
```