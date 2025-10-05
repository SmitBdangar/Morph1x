#!/usr/bin/env python3
"""
Simple runner script for Morph1x video tracking system.
This script provides an easy way to run the system with different options.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import VideoTracker

def main():
    """Main runner function with interactive options."""
    print("🎯 Morph1x - Living Being Tracker")
    print("=" * 50)
    
    # Check if we're in VS Code
    if os.environ.get('VSCODE_PID'):
        print("📝 Running in VS Code environment")
    
    print("\nChoose an option:")
    print("1. Run with webcam (default)")
    print("2. Process video file")
    print("3. Run demo")
    print("4. Custom options")
    
    try:
        choice = input("\nEnter your choice (1-4) [default: 1]: ").strip()
        if not choice:
            choice = "1"
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    
    if choice == "1":
        # Webcam
        print("\n🎥 Starting webcam tracking...")
        print("Press 'q' to quit, 'p' to pause, 'r' to reset")
        tracker = VideoTracker(video_source=0, enable_audio=True)
        return tracker.run()
    
    elif choice == "2":
        # Video file
        video_path = input("\n📁 Enter video file path: ").strip()
        if not video_path:
            print("No video path provided!")
            return 1
        
        output_path = input("💾 Enter output file path [processed_output.mp4]: ").strip()
        if not output_path:
            output_path = "processed_output.mp4"
        
        print(f"\n🎬 Processing video: {video_path}")
        print(f"💾 Output will be saved as: {output_path}")
        
        tracker = VideoTracker(
            video_source=video_path,
            output_path=output_path,
            enable_audio=True
        )
        return tracker.run()
    
    elif choice == "3":
        # Demo
        print("\n🎮 Running demo...")
        from demo import main as demo_main
        return demo_main()
    
    elif choice == "4":
        # Custom options
        print("\n⚙️ Custom options:")
        
        # Video source
        source = input("Video source (0 for webcam, or file path): ").strip()
        if source.isdigit():
            source = int(source)
        
        # Output path
        output = input("Output path (optional): ").strip()
        if not output:
            output = None
        
        # Audio
        audio_input = input("Enable audio feedback? (y/n) [y]: ").strip().lower()
        enable_audio = audio_input != 'n'
        
        # TTS
        tts_input = input("Enable text-to-speech? (y/n) [n]: ").strip().lower()
        enable_tts = tts_input == 'y'
        
        print(f"\n🚀 Starting with custom settings...")
        tracker = VideoTracker(
            video_source=source,
            output_path=output,
            enable_audio=enable_audio,
            enable_tts=enable_tts
        )
        return tracker.run()
    
    else:
        print("❌ Invalid choice!")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
