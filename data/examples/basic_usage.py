#!/usr/bin/env python3
"""
Basic usage examples for Morph1x video tracking system.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.detection import create_detector
from src.utils import draw_detections, draw_info_panel
from src.audio_feedback import create_audio_feedback


def example_1_basic_detection():
    """Example 1: Basic image detection."""
    print("🎯 Example 1: Basic Image Detection")
    print("-" * 40)
    
    # Initialize detector
    detector = create_detector()
    
    # Create a test image (replace with your own image)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Test Image", (250, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Run detection
    detections = detector.detect_living_beings(test_image)
    
    # Print results
    print(f"Found {len(detections)} detections:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']} (confidence: {det['confidence']:.2f})")
    
    return detections


def example_2_draw_detections():
    """Example 2: Drawing detections on image."""
    print("\n🎨 Example 2: Drawing Detections")
    print("-" * 40)
    
    # Initialize detector
    detector = create_detector()
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Detection Test", (200, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Run detection
    detections = detector.detect_living_beings(test_image)
    
    # Draw detections
    result_image = draw_detections(test_image.copy(), detections)
    result_image = draw_info_panel(result_image, detection_count=len(detections))
    
    # Save result
    cv2.imwrite("detection_result.jpg", result_image)
    print(f"✅ Result saved as 'detection_result.jpg'")
    print(f"🟢 Drew {len(detections)} detection boxes")
    
    return result_image


def example_3_video_processing():
    """Example 3: Video processing."""
    print("\n🎬 Example 3: Video Processing")
    print("-" * 40)
    
    # Initialize detector
    detector = create_detector()
    
    # Create a simple video (you can replace with your own video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (640, 480))
    
    # Generate test frames
    for i in range(30):  # 1 second at 30 FPS
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i+1}", (250, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)
    
    out.release()
    print("✅ Created test video: 'test_video.mp4'")
    
    # Process the video
    cap = cv2.VideoCapture('test_video.mp4')
    frame_count = 0
    
    while cap.isOpened() and frame_count < 10:  # Process first 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        detections = detector.detect_living_beings(frame)
        
        # Draw detections
        result_frame = draw_detections(frame, detections)
        result_frame = draw_info_panel(result_frame, detection_count=len(detections))
        
        print(f"Frame {frame_count + 1}: {len(detections)} detections")
        frame_count += 1
    
    cap.release()
    print("✅ Video processing completed")


def example_4_audio_feedback():
    """Example 4: Audio feedback system."""
    print("\n🔊 Example 4: Audio Feedback")
    print("-" * 40)
    
    # Initialize audio feedback
    audio = create_audio_feedback(enabled=True)
    
    if audio.enabled:
        print("✅ Audio feedback enabled")
        
        # Simulate detections
        mock_detections = [
            {'class_name': 'person', 'confidence': 0.8},
            {'class_name': 'dog', 'confidence': 0.9}
        ]
        
        # Play alert
        audio.play_detection_alert(mock_detections)
        print("🔊 Played detection alert")
        
        # Cleanup
        audio.cleanup()
    else:
        print("❌ Audio feedback not available")


def example_5_custom_configuration():
    """Example 5: Custom configuration."""
    print("\n⚙️ Example 5: Custom Configuration")
    print("-" * 40)
    
    # Import config
    from src.config import CONFIDENCE_THRESHOLD, BOX_COLOR, PRIMARY_LIVING_BEINGS
    
    print(f"Current confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Current box color (BGR): {BOX_COLOR}")
    print(f"Detectable classes: {len(PRIMARY_LIVING_BEINGS)}")
    
    # Show all detectable classes
    print("\nDetectable living beings:")
    for class_id, class_name in PRIMARY_LIVING_BEINGS.items():
        print(f"  {class_id:2d}. {class_name}")
    
    # You can modify these in src/config.py for permanent changes
    print("\n💡 To customize settings, edit src/config.py")


def example_6_performance_test():
    """Example 6: Performance testing."""
    print("\n⚡ Example 6: Performance Test")
    print("-" * 40)
    
    import time
    
    # Initialize detector
    detector = create_detector()
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test multiple runs
    num_runs = 10
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        detections = detector.detect_living_beings(test_image)
        end_time = time.time()
        
        times.append(end_time - start_time)
        print(f"Run {i+1}: {(end_time - start_time)*1000:.2f} ms, {len(detections)} detections")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Performance Statistics:")
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Min time: {min_time*1000:.2f} ms")
    print(f"  Max time: {max_time*1000:.2f} ms")
    print(f"  FPS: {1/avg_time:.1f}")


def main():
    """Run all examples."""
    print("🎯 Morph1x - Basic Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_1_basic_detection()
        example_2_draw_detections()
        example_3_video_processing()
        example_4_audio_feedback()
        example_5_custom_configuration()
        example_6_performance_test()
        
        print("\n✅ All examples completed successfully!")
        print("\n💡 Next steps:")
        print("  1. Try with your own images and videos")
        print("  2. Customize settings in src/config.py")
        print("  3. Run the full application: python demo.py")
        print("  4. Explore the Jupyter notebook: notebooks/object_detection_demo.ipynb")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
