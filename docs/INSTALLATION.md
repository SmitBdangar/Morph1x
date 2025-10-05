# Morph1x Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Camera**: USB webcam or built-in camera (for live detection)

### Recommended Requirements
- **GPU**: NVIDIA GPU with CUDA support (for faster processing)
- **RAM**: 16GB or more
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

## Installation Methods

### Method 1: Quick Installation (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Morph1x
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**:
   ```bash
   python demo.py
   ```

### Method 2: Manual Installation

1. **Install Python dependencies**:
   ```bash
   pip install ultralytics>=8.0.0
   pip install opencv-python>=4.8.0
   pip install numpy>=1.24.0
   pip install pygame>=2.5.0
   ```

2. **Install optional dependencies** (for enhanced features):
   ```bash
   # For text-to-speech
   pip install pyttsx3>=2.90
   
   # For Windows audio (Windows only)
   pip install pywin32>=306
   
   # For development and testing
   pip install pytest>=7.0.0
   pip install jupyter>=1.0.0
   ```

### Method 3: Using Virtual Environment (Recommended for Development)

1. **Create virtual environment**:
   ```bash
   python -m venv morph1x_env
   
   # Windows
   morph1x_env\Scripts\activate
   
   # macOS/Linux
   source morph1x_env/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Deactivate when done**:
   ```bash
   deactivate
   ```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU with CUDA

1. **Install CUDA Toolkit**:
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
   - Install version 11.8 or 12.1 (compatible with PyTorch)

2. **Install PyTorch with CUDA support**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify CUDA installation**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

### Apple Silicon (M1/M2) Macs

1. **Install PyTorch for Apple Silicon**:
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Verify Metal Performance Shaders (MPS)**:
   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

## Verification

### Test Installation

1. **Run the test suite**:
   ```bash
   python -m pytest tests/ -v
   ```

2. **Test detection system**:
   ```bash
   python -c "from src.detection import test_detection; test_detection()"
   ```

3. **Run the demo**:
   ```bash
   python demo.py
   ```

### Expected Output

You should see:
- ✅ Model loading successfully
- ✅ Detection system initialized
- ✅ Video window opening (if webcam is available)
- ✅ Green boxes around detected objects

## Troubleshooting

### Common Issues

#### 1. Model Download Fails
**Error**: `Failed to load model: Ran out of input`

**Solution**:
```bash
# Manually download the model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

#### 2. Camera Not Found
**Error**: `Failed to open video source: 0`

**Solutions**:
- Check camera permissions
- Try different camera indices (1, 2, 3...)
- Test with video file instead:
  ```bash
  python -m src.main --source "path/to/video.mp4"
  ```

#### 3. Import Errors
**Error**: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution**:
```bash
pip install -r requirements.txt
```

#### 4. Audio Issues
**Error**: Audio feedback not working

**Solutions**:
- Install pygame: `pip install pygame`
- Disable audio: `python -m src.main --no-audio`
- Check system audio settings

#### 5. Performance Issues
**Symptoms**: Low FPS, laggy video

**Solutions**:
- Reduce frame size in `src/config.py`:
  ```python
  MAX_FRAME_SIZE = (640, 480)  # Smaller frames
  ```
- Skip frames for processing:
  ```python
  PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
  ```
- Use GPU acceleration (see GPU Setup section)

### Platform-Specific Issues

#### Windows
- **Issue**: `pywin32` installation fails
- **Solution**: `pip install --upgrade pip setuptools wheel`

#### macOS
- **Issue**: Camera permissions denied
- **Solution**: Grant camera access in System Preferences → Security & Privacy

#### Linux
- **Issue**: OpenCV GUI not working
- **Solution**: Install GUI dependencies:
  ```bash
  sudo apt-get install python3-opencv
  sudo apt-get install libgl1-mesa-glx
  ```

## Development Setup

### For Contributors

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd Morph1x
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov black flake8
   ```

3. **Run tests**:
   ```bash
   pytest tests/ -v --cov=src
   ```

4. **Format code**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

### IDE Setup

#### VS Code
1. Install Python extension
2. Open project folder
3. Select Python interpreter
4. Press F5 to run debug configurations

#### PyCharm
1. Open project folder
2. Configure Python interpreter
3. Set working directory to project root
4. Run configurations from Run menu

## Uninstallation

### Remove Dependencies
```bash
pip uninstall ultralytics opencv-python numpy pygame pyttsx3 pywin32
```

### Remove Project
```bash
# Delete project folder
rm -rf Morph1x  # Linux/macOS
rmdir /s Morph1x  # Windows
```

## Getting Help

### Documentation
- [API Documentation](API.md)
- [README](../README.md)
- [Jupyter Notebook Demo](../notebooks/object_detection_demo.ipynb)

### Support
- Check [Issues](https://github.com/your-repo/issues) for known problems
- Create new issue for bugs or feature requests
- Check logs for detailed error messages

### Performance Tips
- Use GPU acceleration for better performance
- Adjust confidence threshold for your use case
- Process every N frames for lower-end systems
- Use smaller frame sizes for faster processing

## Next Steps

After successful installation:

1. **Run the demo**: `python demo.py`
2. **Process your video**: `python -m src.main --source "your_video.mp4"`
3. **Explore the notebook**: Open `notebooks/object_detection_demo.ipynb`
4. **Customize settings**: Edit `src/config.py`
5. **Run tests**: `python -m pytest tests/ -v`

Welcome to Morph1x! 🎯
