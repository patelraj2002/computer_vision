import os
import sys
from detection_engine import VideoDetectionEngine, analyze_video_simple

def find_first_video():
    """Find the first video file in current directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            return file
    
    return None

def main():
    print("🚀 Quick Test - Video Object Detection")
    print("=" * 50)
    
    # Find video
    video_path = os.path.join('test_videos', 'test.mp4')

    if not os.path.exists(video_path):
        print(f"❌ Video not found at: {video_path}")
        return

    print(f"📹 Using video: {video_path}")
    
    try:
        # Run simple analysis
        print("\n🔍 Starting object detection...")
        results = analyze_video_simple(video_path, 'quick_test_output')
        
        print(f"\n✅ Quick test completed!")
        print(f"📁 Results in: quick_test_output/")
        print(f"📊 Objects detected: {results['total_objects']}")
        print(f"🏷️  Unique classes: {len(results['class_counts'])}")
        
        # Show top 5 detected classes
        if results['class_counts']:
            print(f"\n🔝 Top detected objects:")
            sorted_classes = sorted(results['class_counts'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:5]:
                print(f"   {class_name}: {count}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"1. Check if video file is valid")
        print(f"2. Install dependencies: pip install torch torchvision opencv-python matplotlib")
        print(f"3. Check internet connection (for YOLOv5 download)")

if __name__ == "__main__":
    main()