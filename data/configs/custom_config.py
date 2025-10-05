"""
Custom configuration examples for Morph1x.
Copy and modify these settings in src/config.py for your specific needs.
"""

# Example 1: High Accuracy Configuration
# Use this for maximum detection accuracy (slower processing)
HIGH_ACCURACY_CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.3,  # Lower threshold = more detections
    'IOU_THRESHOLD': 0.3,         # Lower threshold = more overlapping boxes
    'MAX_DETECTIONS': 200,        # Allow more detections
    'PROCESS_EVERY_N_FRAMES': 1,  # Process every frame
    'MAX_FRAME_SIZE': (1920, 1080),  # Full HD processing
}

# Example 2: High Performance Configuration
# Use this for maximum speed (lower accuracy)
HIGH_PERFORMANCE_CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.7,  # Higher threshold = fewer false positives
    'IOU_THRESHOLD': 0.6,         # Higher threshold = fewer overlapping boxes
    'MAX_DETECTIONS': 50,         # Limit detections for speed
    'PROCESS_EVERY_N_FRAMES': 3,  # Process every 3rd frame
    'MAX_FRAME_SIZE': (640, 480),  # Smaller frames for speed
}

# Example 3: Balanced Configuration
# Use this for good balance between speed and accuracy
BALANCED_CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.5,  # Standard threshold
    'IOU_THRESHOLD': 0.45,        # Standard threshold
    'MAX_DETECTIONS': 100,        # Standard limit
    'PROCESS_EVERY_N_FRAMES': 1,  # Process every frame
    'MAX_FRAME_SIZE': (1280, 720),  # HD processing
}

# Example 4: Custom Living Beings
# Modify this to detect only specific types of living beings
CUSTOM_LIVING_BEINGS = {
    0: "person",      # Human
    15: "cat",        # Cat
    16: "dog",        # Dog
    # Remove other animals if you only want to detect humans and pets
}

# Example 5: Custom Display Settings
# Modify colors and appearance
CUSTOM_DISPLAY_CONFIG = {
    'BOX_COLOR': (0, 255, 0),      # Green boxes (BGR format)
    'TEXT_COLOR': (255, 255, 255), # White text
    'BOX_THICKNESS': 3,            # Thicker boxes
    'TEXT_THICKNESS': 2,           # Thicker text
    'FONT_SCALE': 1.0,             # Larger font
    'TEXT_PADDING': 10,            # More padding
}

# Example 6: Audio Settings
# Customize audio feedback
CUSTOM_AUDIO_CONFIG = {
    'ENABLE_AUDIO_FEEDBACK': True,
    'AUDIO_ALERT_FILE': "custom_alert.wav",
    'ALERT_COOLDOWN': 1.0,  # 1 second between alerts
}

# Example 7: Video Settings
# Customize video processing
CUSTOM_VIDEO_CONFIG = {
    'DEFAULT_VIDEO_SOURCE': 0,     # Webcam
    'OUTPUT_VIDEO_FPS': 30,        # Output FPS
    'SHOW_FPS': True,              # Show FPS counter
    'SHOW_DETECTION_COUNT': True,  # Show detection count
}

# Example 8: Logging Settings
# Customize logging behavior
CUSTOM_LOGGING_CONFIG = {
    'LOG_LEVEL': "INFO",           # DEBUG, INFO, WARNING, ERROR
    'LOG_DETECTIONS': True,        # Log each detection
}

# Example 9: Complete Custom Configuration
# Copy this to src/config.py and modify as needed
COMPLETE_CUSTOM_CONFIG = {
    # Detection settings
    'CONFIDENCE_THRESHOLD': 0.6,
    'IOU_THRESHOLD': 0.5,
    'MAX_DETECTIONS': 75,
    
    # Performance settings
    'PROCESS_EVERY_N_FRAMES': 2,
    'MAX_FRAME_SIZE': (960, 540),
    
    # Display settings
    'BOX_COLOR': (0, 255, 0),      # Green
    'TEXT_COLOR': (255, 255, 255), # White
    'BOX_THICKNESS': 2,
    'TEXT_THICKNESS': 2,
    'FONT_SCALE': 0.8,
    'TEXT_PADDING': 5,
    
    # Audio settings
    'ENABLE_AUDIO_FEEDBACK': True,
    'AUDIO_ALERT_FILE': "detection_alert.wav",
    
    # Video settings
    'DEFAULT_VIDEO_SOURCE': 0,
    'OUTPUT_VIDEO_FPS': 30,
    'SHOW_FPS': True,
    'SHOW_DETECTION_COUNT': True,
    
    # Logging settings
    'LOG_LEVEL': "INFO",
    'LOG_DETECTIONS': False,
}

# Example 10: Environment-specific Configurations

# Development configuration
DEVELOPMENT_CONFIG = {
    'LOG_LEVEL': "DEBUG",
    'LOG_DETECTIONS': True,
    'SHOW_FPS': True,
    'ENABLE_AUDIO_FEEDBACK': False,  # Disable audio in development
}

# Production configuration
PRODUCTION_CONFIG = {
    'LOG_LEVEL': "WARNING",
    'LOG_DETECTIONS': False,
    'SHOW_FPS': False,
    'ENABLE_AUDIO_FEEDBACK': True,
    'CONFIDENCE_THRESHOLD': 0.7,  # Higher confidence for production
}

# Testing configuration
TESTING_CONFIG = {
    'LOG_LEVEL': "ERROR",
    'LOG_DETECTIONS': False,
    'SHOW_FPS': False,
    'ENABLE_AUDIO_FEEDBACK': False,
    'PROCESS_EVERY_N_FRAMES': 5,  # Skip frames for faster testing
    'MAX_FRAME_SIZE': (320, 240),  # Small frames for testing
}

# Usage Instructions:
# 1. Copy the configuration you want to use
# 2. Paste it into src/config.py, replacing the existing values
# 3. Restart the application to apply changes
# 4. Test with your specific use case
# 5. Adjust values as needed

# Example usage in code:
# from data.configs.custom_config import HIGH_ACCURACY_CONFIG
# 
# # Apply configuration
# for key, value in HIGH_ACCURACY_CONFIG.items():
#     setattr(config, key, value)
