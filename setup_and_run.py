#!/usr/bin/env python3
"""
Setup and Run Script for Video Object Detection Engine
This script handles installation of dependencies and runs the detection system
"""

import subprocess
import sys
import os
import platform

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_dependencies():
    """Check for required packages and install if missing"""
    required_packages = [
        'torch',
        'torchvision', 
        'opencv-python',
        'matplotlib',
        'numpy',
        'Pillow'
    ]
    
    print("🔧 Checking and installing dependencies...")
    print("=" * 50)
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} - Already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            try:
                install_package(package)
                print(f"✅ {package} - Installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    # Special handling for YOLOv5 dependencies
    try:
        print("📦 Installing YOLOv5 dependencies...")
        install_package('ultralytics')
        print("✅ YOLOv5 dependencies installed")
    except:
        print("⚠️  YOLOv5 will be downloaded automatically on first run")
    
    return True

def find_video_files():
    """Find available video files in current directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    return video_files

def run_detection_engine():
    """Run the detection engine with user input"""
    print("\n🎬 Video Object Detection Engine")
    print("=" * 50)
    
    # Find available videos
    video_files = find_video_files()
    
    if video_files:
        print(f"📁 Found {len(video_files)} video file(s):")
        for i, video in enumerate(video_files, 1):
            print(f"  {i}. {video}")
        
        if len(video_files) == 1:
            video_path = video_files[0]
            print(f"\n🎯 Using: {video_path}")
        else:
            while True:
                try:
                    choice = int(input(f"\nSelect video (1-{len(video_files)}): ")) - 1
                    if 0 <= choice < len(video_files):
                        video_path = video_files[choice]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
    else:
        print("❌ No video files found in current directory!")
        print("\n📋 Please:")
        print("1. Download a test video from:")
        print("   - https://file-examples.com/index.php/sample-video-files/sample-mp4-files/")
        print("   - https://www.pexels.com/search/videos/traffic/")
        print("2. Place the video file in the same directory as this script")
        print("3. Run this script again")
        return
    
    # Get user preferences
    print(f"\n⚙️  Configuration Options:")
    
    # Output directory
    output_dir = input("Output directory (press Enter for 'output'): ").strip()
    if not output_dir:
        output_dir = 'output'
    
    # Confidence threshold
    confidence_input = input("Confidence threshold 0.1-1.0 (press Enter for 0.5): ").strip()
    try:
        confidence = float(confidence_input) if confidence_input else 0.5
        confidence = max(0.1, min(1.0, confidence))
    except ValueError:
        confidence = 0.5
    
    # Model selection
    print("\nModel options:")
    print("1. yolov5s (fastest, smaller)")
    print("2. yolov5m (balanced)")
    print("3. yolov5l (more accurate)")
    print("4. yolov5x (most accurate, slower)")
    
    model_choice = input("Select model (1-4, press Enter for 1): ").strip()
    models = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    try:
        model = models[int(model_choice) - 1] if model_choice else 'yolov5s'
    except (ValueError, IndexError):
        model = 'yolov5s'
    
    # Annotated frames option
    annotate = input("Generate annotated frames? (y/N): ").strip().lower() == 'y'
    
    # Build command
    cmd = [
        sys.executable, 'detection_engine.py', video_path,
        '--output', output_dir,
        '--confidence', str(confidence),
        '--model', model
    ]
    
    if annotate:
        cmd.append('--annotate')
    
    print(f"\n🚀 Starting detection with:")
    print(f"   Video: {video_path}")
    print(f"   Model: {model}")
    print(f"   Confidence: {confidence}")
    print(f"   Output: {output_dir}")
    print(f"   Annotated frames: {'Yes' if annotate else 'No'}")
    print("\n" + "=" * 50)
    
    # Run the detection
    try:
        subprocess.run(cmd, check=True)
        print(f"\n🎉 Detection completed successfully!")
        print(f"📁 Results saved in: {output_dir}/")
        print(f"📊 Check the bar chart: {output_dir}/object_frequency_chart.png")
        print(f"📄 Full results: {output_dir}/detection_results.json")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Detection failed: {e}")
    except KeyboardInterrupt:
        print(f"\n⏹️  Detection interrupted by user")

def main():
    """Main execution function"""
    print("🎯 Video Object Detection Setup & Runner")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # Check if detection_engine.py exists
    if not os.path.exists('detection_engine.py'):
        print("❌ detection_engine.py not found!")
        print("Please ensure detection_engine.py is in the same directory as this script.")
        return
    
    # Install dependencies
    if not check_and_install_dependencies():
        print("❌ Failed to install required dependencies")
        return
    
    print("\n✅ All dependencies installed successfully!")
    
    # Run detection engine
    run_detection_engine()

if __name__ == "__main__":
    main()