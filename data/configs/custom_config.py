"""
Custom configuration examples for Morph1x.
Copy and modify these settings in src/config.py for your specific needs.
"""

HIGH_ACCURACY_CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.3,  # Lower threshold
    'IOU_THRESHOLD': 0.3,         # Lower threshold
    'MAX_DETECTIONS': 200,       
    'PROCESS_EVERY_N_FRAMES': 1,
    'MAX_FRAME_SIZE': (1920, 1080),
}

HIGH_PERFORMANCE_CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.7, 
    'IOU_THRESHOLD': 0.6,         
    'MAX_DETECTIONS': 50,         
    'PROCESS_EVERY_N_FRAMES': 3,  
    'MAX_FRAME_SIZE': (640, 480),  
}

BALANCED_CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.5,  
    'IOU_THRESHOLD': 0.45,     
    'MAX_DETECTIONS': 100,        
    'PROCESS_EVERY_N_FRAMES': 1,  
    'MAX_FRAME_SIZE': (1280, 720),  
}

CUSTOM_LIVING_BEINGS = {
    0: "person",      # Human
    15: "cat",        # Cat
    16: "dog",        # Dog
}

CUSTOM_DISPLAY_CONFIG = {
    'BOX_COLOR': (0, 255, 0),
    'TEXT_COLOR': (255, 255, 255), 
    'BOX_THICKNESS': 3,            
    'TEXT_THICKNESS': 2,           
    'FONT_SCALE': 1.0,             
    'TEXT_PADDING': 10,            
}

CUSTOM_AUDIO_CONFIG = {
    'ENABLE_AUDIO_FEEDBACK': True,
    'AUDIO_ALERT_FILE': "custom_alert.wav",
    'ALERT_COOLDOWN': 1.0,  # 1 second between alerts
}

CUSTOM_VIDEO_CONFIG = {
    'DEFAULT_VIDEO_SOURCE': 0,    
    'OUTPUT_VIDEO_FPS': 30,     
    'SHOW_FPS': True,           
    'SHOW_DETECTION_COUNT': True,  
}

CUSTOM_LOGGING_CONFIG = {
    'LOG_LEVEL': "INFO",      
    'LOG_DETECTIONS': True,       
}

COMPLETE_CUSTOM_CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.6,
    'IOU_THRESHOLD': 0.5,
    'MAX_DETECTIONS': 75,
    
    'PROCESS_EVERY_N_FRAMES': 2,
    'MAX_FRAME_SIZE': (960, 540),
    
    'BOX_COLOR': (0, 255, 0),      # Green
    'TEXT_COLOR': (255, 255, 255), # White
    'BOX_THICKNESS': 2,
    'TEXT_THICKNESS': 2,
    'FONT_SCALE': 0.8,
    'TEXT_PADDING': 5,
    
    'ENABLE_AUDIO_FEEDBACK': True,
    'AUDIO_ALERT_FILE': "detection_alert.wav",
    
    'DEFAULT_VIDEO_SOURCE': 0,
    'OUTPUT_VIDEO_FPS': 30,
    'SHOW_FPS': True,
    'SHOW_DETECTION_COUNT': True,
    
    'LOG_LEVEL': "INFO",
    'LOG_DETECTIONS': False,
}

DEVELOPMENT_CONFIG = {
    'LOG_LEVEL': "DEBUG",
    'LOG_DETECTIONS': True,
    'SHOW_FPS': True,
    'ENABLE_AUDIO_FEEDBACK': False,
}

PRODUCTION_CONFIG = {
    'LOG_LEVEL': "WARNING",
    'LOG_DETECTIONS': False,
    'SHOW_FPS': False,
    'ENABLE_AUDIO_FEEDBACK': True,
    'CONFIDENCE_THRESHOLD': 0.7,
}

TESTING_CONFIG = {
    'LOG_LEVEL': "ERROR",
    'LOG_DETECTIONS': False,
    'SHOW_FPS': False,
    'ENABLE_AUDIO_FEEDBACK': False,
    'PROCESS_EVERY_N_FRAMES': 5,  
    'MAX_FRAME_SIZE': (320, 240), 
}

