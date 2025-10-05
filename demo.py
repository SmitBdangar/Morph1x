#!/usr/bin/env python3
"""
Demo script for Morph1x video tracking system.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import VideoTracker

def main():
    """Run the demo."""
    print("Morph1x - Living Being Tracker Demo")
    print("=" * 40)
    print("Controls:")
    print("  'q' or ESC - Quit")
    print("  'p' - Pause/Resume")
    print("  'r' - Reset tracker")
    print("=" * 40)
    
    # Create tracker with default settings
    tracker = VideoTracker(
        video_source=0,  # Default webcam
        enable_audio=True,
        enable_tts=False
    )
    
    try:
        # Run the tracker
        success = tracker.run()
        
        if success:
            print("Demo completed successfully!")
        else:
            print("Demo failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
