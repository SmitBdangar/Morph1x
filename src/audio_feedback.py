import os
import sys
import threading
import time
import logging
from typing import Optional, Dict, List
import numpy as np

from .config import ENABLE_AUDIO_FEEDBACK, AUDIO_ALERT_FILE, PROJECT_ROOT

logger = logging.getLogger(__name__)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("pygame not available. Audio feedback will be disabled.")

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class AudioFeedback:

    def __init__(self, enabled: bool = None):
        self.enabled = enabled if enabled is not None else ENABLE_AUDIO_FEEDBACK
        self.audio_system = None
        self.alert_sound = None
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # Minimum seconds between alerts
        self.detection_history = []
        
        if self.enabled:
            self._initialize_audio()
    
    def _initialize_audio(self):
        """Initialize the audio system."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.audio_system = "pygame"
                logger.info("Audio system initialized with pygame")
                self._create_alert_sound()
            except Exception as e:
                logger.error(f"Failed to initialize pygame audio: {e}")
                self.audio_system = None
        
        if self.audio_system is None and WINSOUND_AVAILABLE:
            self.audio_system = "winsound"
            logger.info("Audio system initialized with winsound")
        
        if self.audio_system is None:
            logger.warning("No audio system available. Audio feedback disabled.")
            self.enabled = False
    
    def _create_alert_sound(self):
        """Create a simple alert sound."""
        if self.audio_system == "pygame":
            try:
                duration = 0.5  # seconds
                sample_rate = 22050
                frequency = 800  # Hz
                
                frames = int(duration * sample_rate)
                arr = np.zeros((frames, 2))
                
                for i in range(frames):
                    arr[i][0] = 32767 * np.sin(2 * np.pi * frequency * i / sample_rate)
                    arr[i][1] = arr[i][0]
                
                sound = pygame.sndarray.make_sound(arr.astype(np.int16))
                self.alert_sound = sound
                logger.info("Alert sound created")
                
            except Exception as e:
                logger.error(f"Failed to create alert sound: {e}")
                self.alert_sound = None
    
    def play_detection_alert(self, detections: List[Dict] = None):
        if not self.enabled or self.audio_system is None:
            return
        
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        if detections and self._has_new_detections(detections):
            self._play_alert()
            self.last_alert_time = current_time
            self.detection_history = detections.copy()
    
    def _has_new_detections(self, current_detections: List[Dict]) -> bool:
        if not self.detection_history:
            return len(current_detections) > 0
        
        return len(current_detections) > len(self.detection_history)
    
    def _play_alert(self):
        """Play the alert sound."""
        try:
            if self.audio_system == "pygame" and self.alert_sound:
                self.alert_sound.play()
            elif self.audio_system == "winsound":
                winsound.Beep(800, 500)  # 800 Hz, 500 ms
        except Exception as e:
            logger.error(f"Failed to play alert: {e}")
    
    def play_custom_alert(self, sound_file: str):
        if not self.enabled or self.audio_system is None:
            return
        
        try:
            if self.audio_system == "pygame":
                if os.path.exists(sound_file):
                    sound = pygame.mixer.Sound(sound_file)
                    sound.play()
                else:
                    logger.warning(f"Sound file not found: {sound_file}")
        except Exception as e:
            logger.error(f"Failed to play custom alert: {e}")
    
    def set_alert_cooldown(self, seconds: float):

        self.alert_cooldown = max(0.1, seconds)
    
    def enable(self):
        self.enabled = True
        if self.audio_system is None:
            self._initialize_audio()
    
    def disable(self):
        self.enabled = False
    
    def cleanup(self):
        if self.audio_system == "pygame":
            try:
                pygame.mixer.quit()
            except:
                pass

class DetectionAnnouncer:  
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.tts_available = False
        self.last_announcement = {}
        self.announcement_cooldown = 5.0  # seconds
        
        if self.enabled:
            self._initialize_tts()
    
    def _initialize_tts(self):
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_available = True
            logger.info("TTS system initialized with pyttsx3")
        except ImportError:
            try:
                import win32com.client
                self.tts_engine = win32com.client.Dispatch("SAPI.SpVoice")
                self.tts_available = True
                logger.info("TTS system initialized with Windows SAPI")
            except ImportError:
                logger.warning("No TTS system available")
                self.tts_available = False
    
    def announce_detections(self, detections: List[Dict]):
        if not self.enabled or not self.tts_available:
            return
        
        current_time = time.time()
        
        detection_summary = {}
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            detection_summary[class_name] = detection_summary.get(class_name, 0) + 1
        
        for class_name, count in detection_summary.items():
            last_time = self.last_announcement.get(class_name, 0)
            if current_time - last_time > self.announcement_cooldown:
                self._announce_detection(class_name, count)
                self.last_announcement[class_name] = current_time
    
    def _announce_detection(self, class_name: str, count: int):
        try:
            if count == 1:
                message = f"{class_name} detected"
            else:
                message = f"{count} {class_name}s detected"
            
            if hasattr(self.tts_engine, 'say'):
                # pyttsx3
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            else:
                # Windows SAPI
                self.tts_engine.Speak(message)
                
        except Exception as e:
            logger.error(f"Failed to announce detection: {e}")


def create_audio_feedback(enabled: bool = None) -> AudioFeedback:
    return AudioFeedback(enabled)


def create_detection_announcer(enabled: bool = False) -> DetectionAnnouncer:
    return DetectionAnnouncer(enabled)


if __name__ == "__main__":
    audio = create_audio_feedback()
    print(f"Audio feedback enabled: {audio.enabled}")
    print(f"Audio system: {audio.audio_system}")
    
    test_detections = [
        {'class_name': 'person', 'confidence': 0.9},
        {'class_name': 'dog', 'confidence': 0.8}
    ]
    
    audio.play_detection_alert(test_detections)
    time.sleep(1)
    
    audio.cleanup()
