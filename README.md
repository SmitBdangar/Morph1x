# Morph1x - Living Being Tracker

A real-time video tracking system that detects and tracks living beings (humans, animals) using YOLOv8 object detection. The system draws green bounding boxes around detected living beings and displays their names.

## Features

- 🎯 **Real-time Detection**: Detects humans, cats, dogs, horses, and other animals
- 🟢 **Visual Feedback**: Green bounding boxes with labels
- 🔊 **Audio Alerts**: Optional sound notifications for new detections
- 📊 **Live Statistics**: FPS counter and detection summary
- 📹 **Multiple Sources**: Webcam, video files, or any OpenCV-compatible source
- 💾 **Video Recording**: Save processed video with detections
- 🎤 **Text-to-Speech**: Optional voice announcements (Windows)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Morph1x
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model** (if not already present):
   ```bash
   # The model will be downloaded automatically on first run
   # Or download manually to models/yolov8n.pt
   ```

## Quick Start

### Basic Usage

Run with default webcam:
```bash
python demo.py
```

Or use the main module:
```bash
python -m src.main
```

### Advanced Usage

```bash
# Use specific camera
python -m src.main --source 1

# Process video file
python -m src.main --source video.mp4

# Save output video
python -m src.main --output result.mp4

# Disable audio feedback
python -m src.main --no-audio

# Enable text-to-speech announcements
python -m src.main --tts

# Verbose logging
python -m src.main --verbose
```

## Controls

When running the application:
- **'q' or ESC**: Quit
- **'p'**: Pause/Resume video
- **'r'**: Reset tracker statistics

## Detected Living Beings

The system detects the following living beings:
- 👤 **Person** (Human)
- 🐱 **Cat**
- 🐶 **Dog**
- 🐴 **Horse**
- 🐑 **Sheep**
- 🐄 **Cow**
- 🐘 **Elephant**
- 🐻 **Bear**
- 🦓 **Zebra**
- 🦒 **Giraffe**

## Configuration

Edit `src/config.py` to customize:
- Detection confidence threshold
- Box colors and styling
- Audio settings
- Performance options

## Project Structure

```
Morph1x/
├── src/                    # Source code
│   ├── main.py            # Main application
│   ├── detection.py       # YOLOv8 detection logic
│   ├── utils.py           # Utility functions
│   ├── audio_feedback.py  # Audio system
│   └── config.py          # Configuration
├── models/                # YOLOv8 model files
├── data/                  # Data directory
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks
├── demo.py               # Demo script
└── requirements.txt      # Dependencies
```

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- NumPy
- Pygame (for audio feedback)
- PyTTSx3 (for text-to-speech, optional)

## Performance Tips

- Use a GPU for better performance (CUDA-compatible)
- Adjust `PROCESS_EVERY_N_FRAMES` in config for lower-end systems
- Reduce `MAX_FRAME_SIZE` for faster processing
- Disable audio feedback if not needed

## Troubleshooting

### Common Issues

1. **Camera not found**: Try different camera indices (0, 1, 2...)
2. **Model download fails**: Check internet connection or download manually
3. **Audio not working**: Install pygame or disable audio feedback
4. **Low FPS**: Reduce frame size or skip frames in config

### Getting Help

- Check the logs for detailed error messages
- Use `--verbose` flag for debug information
- Ensure all dependencies are installed correctly

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

**Morph1x** - Track living beings in real-time with AI-powered detection! 🎯
