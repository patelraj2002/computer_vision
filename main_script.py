import cv2
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os
from pathlib import Path
import argparse
from datetime import datetime

class VideoDetectionEngine:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.5):
        """
        Initialize the detection engine with YOLOv5 model
        Args:
            model_name: YOLOv5 model variant ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.conf = confidence_threshold
        self.results_data = []
        self.class_counts = defaultdict(int)
        
    def process_video(self, video_path, output_dir='output'):
        """
        Process video and detect objects in every 5th frame
        Args:
            video_path: Path to input video file
            output_dir: Directory to save outputs
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video Info:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Processing every 5th frame...")
        
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame
            if frame_count % 5 == 0:
                timestamp = frame_count / fps
                detections = self._detect_objects(frame, frame_count, timestamp)
                self.results_data.append(detections)
                processed_frames += 1
                
                print(f"Processed frame {frame_count} ({processed_frames} total)")
                
            frame_count += 1
        
        cap.release()
        
        # Save results and generate analysis
        self._save_json_results(output_dir)
        max_diversity_frame = self._analyze_class_diversity()
        self._generate_visualizations(output_dir)
        
        print(f"\nProcessing complete!")
        print(f"Processed {processed_frames} frames")
        print(f"Frame with maximum class diversity: Frame {max_diversity_frame}")
        
        return self.results_data
    
    def _detect_objects(self, frame, frame_number, timestamp):
        """
        Run object detection on a single frame
        Args:
            frame: OpenCV frame
            frame_number: Frame index
            timestamp: Time in seconds
        Returns:
            Dictionary with detection results
        """
        # Run inference
        results = self.model(frame)
        
        # Parse results
        detections = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'detections': []
        }
        
        # Extract detection data
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                class_name = self.model.names[int(cls)]
                
                detection = {
                    'class': class_name,
                    'confidence': float(conf),
                    'bounding_box': {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': x2 - x1, 'height': y2 - y1
                    }
                }
                
                detections['detections'].append(detection)
                self.class_counts[class_name] += 1
        
        # Add frame-level statistics
        frame_classes = [d['class'] for d in detections['detections']]
        detections['frame_stats'] = {
            'total_objects': len(detections['detections']),
            'unique_classes': len(set(frame_classes)),
            'class_counts': dict(Counter(frame_classes))
        }
        
        return detections
    
    def _save_json_results(self, output_dir):
        """Save per-frame detection results to JSON file"""
        output_file = os.path.join(output_dir, 'detection_results.json')
        
        # Create summary data
        summary = {
            'metadata': {
                'total_frames_processed': len(self.results_data),
                'confidence_threshold': self.confidence_threshold,
                'processing_timestamp': datetime.now().isoformat()
            },
            'overall_statistics': {
                'total_objects_detected': sum(self.class_counts.values()),
                'unique_classes_found': len(self.class_counts),
                'class_totals': dict(self.class_counts)
            },
            'per_frame_results': self.results_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return output_file
    
    def _analyze_class_diversity(self):
        """Find frame with maximum class diversity"""
        max_diversity = 0
        max_diversity_frame = 0
        
        print(f"\nClass Diversity Analysis:")
        for frame_data in self.results_data:
            diversity = frame_data['frame_stats']['unique_classes']
            frame_num = frame_data['frame_number']
            
            if diversity > max_diversity:
                max_diversity = diversity
                max_diversity_frame = frame_num
            
            print(f"  Frame {frame_num}: {diversity} unique classes")
        
        print(f"Maximum diversity: {max_diversity} classes in frame {max_diversity_frame}")
        return max_diversity_frame
    
    def _generate_visualizations(self, output_dir):
        """Generate bar chart visualization of object frequencies"""
        if not self.class_counts:
            print("No objects detected for visualization")
            return
        
        # Prepare data for plotting
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        # Sort by count for better visualization
        sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_data)
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(classes)), counts, color='skyblue', alpha=0.7)
        
        # Customize the plot
        plt.xlabel('Object Classes', fontsize=12)
        plt.ylabel('Detection Count', fontsize=12)
        plt.title('Object Detection Frequency Analysis', fontsize=14, fontweight='bold')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, 'object_frequency_chart.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_file}")
        
        # Also save as PDF for better quality
        pdf_file = os.path.join(output_dir, 'object_frequency_chart.pdf')
        plt.savefig(pdf_file, bbox_inches='tight')
        
        plt.show()
    
    def generate_annotated_frames(self, video_path, output_dir='output/annotated_frames'):
        """
        Optional: Generate annotated frames showing detections
        Args:
            video_path: Path to input video
            output_dir: Directory to save annotated frames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        print("Generating annotated frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 5 == 0:
                # Run detection
                results = self.model(frame)
                
                # Draw annotations
                annotated_frame = results.render()[0]
                
                # Save annotated frame
                output_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(output_path, annotated_frame)
                
            frame_count += 1
        
        cap.release()
        print(f"Annotated frames saved to: {output_dir}")
    
    def print_summary(self):
        """Print comprehensive summary of detection results"""
        print("\n" + "="*60)
        print("DETECTION SUMMARY REPORT")
        print("="*60)
        
        print(f"Total frames processed: {len(self.results_data)}")
        print(f"Total objects detected: {sum(self.class_counts.values())}")
        print(f"Unique classes found: {len(self.class_counts)}")
        
        print(f"\nClass Distribution:")
        for class_name, count in sorted(self.class_counts.items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")
        
        # Frame-by-frame summary
        print(f"\nPer-Frame Summary:")
        for i, frame_data in enumerate(self.results_data[:5]):  # Show first 5 frames
            frame_num = frame_data['frame_number']
            total_objects = frame_data['frame_stats']['total_objects']
            unique_classes = frame_data['frame_stats']['unique_classes']
            print(f"  Frame {frame_num}: {total_objects} objects, {unique_classes} classes")
        
        if len(self.results_data) > 5:
            print(f"  ... and {len(self.results_data) - 5} more frames")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Video Object Detection Analysis')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', default='output', 
                       help='Output directory (default: output)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--model', '-m', default='yolov5s',
                       choices=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
                       help='YOLOv5 model variant (default: yolov5s)')
    parser.add_argument('--annotate', action='store_true',
                       help='Generate annotated frames')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    print("Initializing Video Detection Engine...")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    
    # Initialize detection engine
    engine = VideoDetectionEngine(
        model_name=args.model,
        confidence_threshold=args.confidence
    )
    
    try:
        # Process video
        results = engine.process_video(args.video_path, args.output)
        
        # Generate optional annotated frames
        if args.annotate:
            engine.generate_annotated_frames(args.video_path, 
                                           os.path.join(args.output, 'annotated_frames'))
        
        # Print summary
        engine.print_summary()
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return


if __name__ == "__main__":
    # Example usage when run directly
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python detection_engine.py path/to/video.mp4")
        print("python detection_engine.py path/to/video.mp4 --output results --confidence 0.6 --annotate")
    else:
        main()


# Alternative: Simple function-based usage
def analyze_video_simple(video_path, output_dir='output'):
    """
    Simplified function for easy integration
    Args:
        video_path: Path to video file
        output_dir: Output directory
    Returns:
        Dictionary with results
    """
    engine = VideoDetectionEngine()
    results = engine.process_video(video_path, output_dir)
    engine.print_summary()
    return {
        'results': results,
        'class_counts': dict(engine.class_counts),
        'total_objects': sum(engine.class_counts.values())
    }