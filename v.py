import cv2
import numpy as np
import requests
import os
from urllib.parse import urlparse
import random

class VideoDownloader:
    """Download and create test videos for object detection"""
    
    def __init__(self):
        self.test_video_urls = {
            # Popular test videos for computer vision
            'sample_mp4_small': 'https://file-examples.com/storage/feb68d0cd5d93a0b0862e3b/2017/10/file_example_MP4_480_1_5MG.mp4',
            'sample_mp4_medium': 'https://file-examples.com/storage/feb68d0cd5d93a0b0862e3b/2017/10/file_example_MP4_1280_10MG.mp4',
            'traffic_sample': 'https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4',
        }
    
    def download_video(self, url, filename=None, output_dir='test_videos'):
        """
        Download a video from URL
        Args:
            url: Video URL
            filename: Optional custom filename
            output_dir: Directory to save video
        Returns:
            Path to downloaded video
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            if filename is None:
                filename = os.path.basename(urlparse(url).path)
                if not filename.endswith('.mp4'):
                    filename += '.mp4'
            
            filepath = os.path.join(output_dir, filename)
            
            print(f"Downloading video from: {url}")
            print(f"Saving to: {filepath}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Download complete: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error downloading video: {str(e)}")
            return None
    
    def download_sample_videos(self, output_dir='test_videos'):
        """Download all sample videos"""
        downloaded = []
        
        for name, url in self.test_video_urls.items():
            filename = f"{name}.mp4"
            filepath = self.download_video(url, filename, output_dir)
            if filepath:
                downloaded.append(filepath)
        
        return downloaded
    
    def create_synthetic_video(self, filename='synthetic_test.mp4', 
                             duration=15, fps=30, output_dir='test_videos'):
        """
        Create a synthetic test video with moving objects
        Args:
            filename: Output filename
            duration: Video duration in seconds
            fps: Frames per second
            output_dir: Output directory
        Returns:
            Path to created video
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # Video properties
        width, height = 1280, 720
        total_frames = duration * fps
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        print(f"Creating synthetic video: {filepath}")
        print(f"Duration: {duration}s, FPS: {fps}, Frames: {total_frames}")
        
        # Create moving objects
        objects = [
            {'pos': [100, 100], 'vel': [3, 2], 'color': (0, 255, 0), 'size': 50},  # Green circle
            {'pos': [200, 300], 'vel': [-2, 3], 'color': (255, 0, 0), 'size': 40}, # Blue rectangle
            {'pos': [400, 200], 'vel': [4, -1], 'color': (0, 0, 255), 'size': 60}, # Red circle
            {'pos': [600, 400], 'vel': [-3, -2], 'color': (255, 255, 0), 'size': 35}, # Cyan rectangle
            {'pos': [800, 150], 'vel': [1, 4], 'color': (255, 0, 255), 'size': 45},  # Magenta circle
        ]
        
        for frame_num in range(total_frames):
            # Create blank frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add background gradient
            for y in range(height):
                intensity = int(50 + (y / height) * 50)
                frame[y, :] = [intensity, intensity//2, intensity//3]
            
            # Draw and move objects
            for i, obj in enumerate(objects):
                x, y = obj['pos']
                vx, vy = obj['vel']
                color = obj['color']
                size = obj['size']
                
                # Draw object (alternating circles and rectangles)
                if i % 2 == 0:  # Circle
                    cv2.circle(frame, (int(x), int(y)), size, color, -1)
                    cv2.circle(frame, (int(x), int(y)), size, (255, 255, 255), 2)
                else:  # Rectangle
                    pt1 = (int(x - size), int(y - size))
                    pt2 = (int(x + size), int(y + size))
                    cv2.rectangle(frame, pt1, pt2, color, -1)
                    cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 2)
                
                # Update position
                obj['pos'][0] += vx
                obj['pos'][1] += vy
                
                # Bounce off walls
                if obj['pos'][0] <= size or obj['pos'][0] >= width - size:
                    obj['vel'][0] *= -1
                if obj['pos'][1] <= size or obj['pos'][1] >= height - size:
                    obj['vel'][1] *= -1
                
                # Keep within bounds
                obj['pos'][0] = max(size, min(width - size, obj['pos'][0]))
                obj['pos'][1] = max(size, min(height - size, obj['pos'][1]))
            
            # Add frame number
            cv2.putText(frame, f'Frame: {frame_num}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add timestamp
            timestamp = frame_num / fps
            cv2.putText(frame, f'Time: {timestamp:.2f}s', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(frame)
            
            if frame_num % (fps * 2) == 0:  # Progress every 2 seconds
                print(f"Progress: {frame_num}/{total_frames} frames ({timestamp:.1f}s)")
        
        out.release()
        print(f"✅ Synthetic video created: {filepath}")
        return filepath

def get_test_video():
    """
    Main function to get a test video - tries multiple methods
    Returns:
        Path to a usable test video
    """
    downloader = VideoDownloader()
    
    print("🎬 Getting test video for object detection...")
    print("=" * 50)
    
    # Method 1: Try to download a sample video
    print("\n📥 Method 1: Downloading sample video...")
    try:
        sample_url = 'https://file-examples.com/storage/feb68d0cd5d93a0b0862e3b/2017/10/file_example_MP4_480_1_5MG.mp4'
        video_path = downloader.download_video(sample_url, 'test_sample.mp4')
        if video_path and os.path.exists(video_path):
            print(f"✅ Sample video ready: {video_path}")
            return video_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
    
    # Method 2: Create synthetic video
    print("\n🎨 Method 2: Creating synthetic test video...")
    try:
        video_path = downloader.create_synthetic_video()
        if video_path and os.path.exists(video_path):
            print(f"✅ Synthetic video ready: {video_path}")
            return video_path
    except Exception as e:
        print(f"❌ Synthetic video creation failed: {e}")
    
    # Method 3: Instructions for manual download
    print("\n📋 Method 3: Manual Download Instructions")
    print("Please download a test video manually from one of these sources:")
    print("1. https://file-examples.com/index.php/sample-video-files/sample-mp4-files/")
    print("2. https://www.pexels.com/search/videos/traffic/ (free, no login required)")
    print("3. https://sample-videos.com/")
    print("\nRecommended videos:")
    print("- Traffic scenes (cars, pedestrians)")
    print("- Street scenes (multiple objects)")
    print("- Indoor scenes (people, furniture)")
    print("- Duration: 15-20 seconds")
    print("- Format: MP4")
    
    return None

def verify_video(video_path):
    """
    Verify that a video file is valid and get its properties
    Args:
        video_path: Path to video file
    Returns:
        Dictionary with video properties or None if invalid
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        props = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        props['duration'] = props['frame_count'] / props['fps']
        props['valid'] = True
        
        cap.release()
        return props
        
    except Exception as e:
        print(f"Error verifying video: {e}")
        return None

if __name__ == "__main__":
    # Get a test video
    video_path = get_test_video()
    
    if video_path:
        # Verify the video
        props = verify_video(video_path)
        if props:
            print(f"\n✅ Video verified successfully!")
            print(f"   Path: {video_path}")
            print(f"   Duration: {props['duration']:.2f} seconds")
            print(f"   Resolution: {props['width']}x{props['height']}")
            print(f"   FPS: {props['fps']}")
            print(f"   Total frames: {props['frame_count']}")
            print(f"\n🚀 You can now run the detection engine:")
            print(f"   python detection_engine.py {video_path}")
        else:
            print("❌ Video verification failed")
    else:
        print("❌ No video obtained. Please download manually.")