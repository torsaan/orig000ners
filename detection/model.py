#!/usr/bin/env python3
"""
YOLOv8 Object Detection Model
Handles traffic sign and vegetation detection using YOLOv8.
"""

import os
import json
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import cv2
import logging

# Import our utilities and config
from config import (
    YOLO_MODEL_NAME, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD,
    YOLO_CLASSES_FILE, ENABLE_MODEL_CACHING
)
from utils import (
    ModelLoadingError, handle_streamlit_errors, log_performance,
    load_model_cached, logger, retry_on_exception, safe_execute
)


class Detector:
    """YOLO-based object detector for traffic signs and vegetation"""
    
    def __init__(self, 
                 model_path: str = None, 
                 confidence_threshold: float = None,
                 iou_threshold: float = None):
        """
        Initialize the detector
        
        Args:
            model_path: Path to YOLO model weights (uses config default if None)
            confidence_threshold: Minimum confidence for detections (uses config default if None)
            iou_threshold: IoU threshold for NMS (uses config default if None)
        """
        self.model_path = model_path or YOLO_MODEL_NAME
        self.confidence_threshold = confidence_threshold or YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or YOLO_IOU_THRESHOLD
        self.model = None
        self.class_names = None
        
        # Initialize components
        self._load_class_names()
        self._initialize_model()
    
    @retry_on_exception(max_retries=3, exceptions=(Exception,))
    def _load_class_names(self) -> None:
        """Load YOLO class names from JSON file or use defaults"""
        self.class_names = safe_execute(
            lambda: self._load_class_names_from_file(),
            default_return=self._get_default_class_names(),
            error_message="Failed to load class names from file, using defaults"
        )
        
        logger.info(f"📋 Loaded {len(self.class_names)} YOLO class names")
    
    def _load_class_names_from_file(self) -> Dict[int, str]:
        """Load class names from JSON file"""
        if not YOLO_CLASSES_FILE.exists():
            raise FileNotFoundError(f"Class names file not found: {YOLO_CLASSES_FILE}")
        
        with open(YOLO_CLASSES_FILE, 'r') as f:
            data = json.load(f)
            # Convert string keys to integers
            return {int(k): v for k, v in data["class"].items()}
    
    def _get_default_class_names(self) -> Dict[int, str]:
        """Get default YOLO COCO class names as fallback"""
        return {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
            10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
            35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
            39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
            44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
            49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
            54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
            59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
            64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
            69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
            74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
            79: "toothbrush"
        }
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO model"""
        try:
            print(f"🔄 Initializing YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print(f"✅ Loaded YOLO model: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model {self.model_path}: {e}")
            print("📥 Downloading pretrained YOLOv8n model...")
            try:
                self.model = YOLO("yolov8n.pt")  # This will download if not exists
                print("✅ Loaded fallback YOLOv8n model")
            except Exception as e2:
                print(f"❌ Failed to load fallback model: {e2}")
                self.model = None
    
    def detect(self, image_path: str) -> List[Dict]:
        """
        Detect objects in an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detection dictionaries with bbox, class, confidence
        """
        if not self.model:
            print("❌ Model not initialized. Attempting to reinitialize...")
            self._initialize_model()
            if not self.model:
                raise RuntimeError("Model not initialized and cannot be reinitialized")
        
        try:
            # Run inference
            results = self.model(image_path, conf=self.confidence_threshold)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        
                        # Convert to YOLO format (center_x, center_y, width, height)
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Get class name
                        class_id = int(cls)
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        detection = {
                            "class": class_name,
                            "confidence": float(conf),
                            "box": {
                                "center_x": float(center_x),
                                "center_y": float(center_y), 
                                "width": float(width),
                                "height": float(height)
                            },
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection failed for {image_path}: {e}")
            return []
    
    def detect_batch(self, image_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        Detect objects in multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary mapping image paths to detections
        """
        results = {}
        for image_path in image_paths:
            results[image_path] = self.detect(image_path)
        return results
    
    def annotate_image(self, image_path: str, detections: List[Dict], 
                      output_path: Optional[str] = None) -> np.ndarray:
        """
        Annotate image with detection boxes and labels
        
        Args:
            image_path: Path to input image
            detections: List of detections
            output_path: Optional path to save annotated image
            
        Returns:
            Annotated image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Define colors for different classes (cycling through a palette)
            color_palette = [
                (0, 0, 255),      # Red
                (0, 255, 0),      # Green  
                (255, 0, 0),      # Blue
                (255, 255, 0),    # Cyan
                (255, 0, 255),    # Magenta
                (0, 255, 255),    # Yellow
                (128, 0, 128),    # Purple
                (255, 165, 0),    # Orange
                (255, 192, 203),  # Pink
                (0, 128, 0),      # Dark Green
                (128, 128, 0),    # Olive
                (0, 0, 128),      # Navy
                (128, 0, 0),      # Maroon
                (255, 20, 147),   # Deep Pink
                (0, 191, 255),    # Deep Sky Blue
                (50, 205, 50),    # Lime Green
            ]
            
            def get_color_for_class(class_name: str) -> Tuple[int, int, int]:
                """Get consistent color for a class name"""
                # Use hash of class name to get consistent color
                hash_val = hash(class_name) % len(color_palette)
                return color_palette[hash_val]
            
            # Draw detections
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                class_name = detection["class"]
                confidence = detection["confidence"]
                
                # Get color
                color = get_color_for_class(class_name)
                
                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background rectangle for text
                cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)), color, -1)
                
                # Text
                cv2.putText(image, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save if output path provided
            if output_path:
                cv2.imwrite(output_path, image)
                print(f"✅ Annotated image saved: {output_path}")
            
            return image
            
        except Exception as e:
            print(f"❌ Failed to annotate image {image_path}: {e}")
            return np.array([])
    
    def get_class_statistics(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Get statistics of detected classes
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with class counts
        """
        stats = {}
        for detection in detections:
            class_name = detection["class"]
            stats[class_name] = stats.get(class_name, 0) + 1
        return stats


def main():
    """Test the detector"""
    print("🔍 Testing YOLO Detector...")
    
    # Initialize detector
    detector = Detector()
    
    # Test with sample image if available
    test_images_dir = "images"
    if os.path.exists(test_images_dir):
        image_files = [f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            test_image = os.path.join(test_images_dir, image_files[0])
            print(f"🖼️  Testing with: {test_image}")
            
            detections = detector.detect(test_image)
            print(f"📊 Found {len(detections)} detections")
            
            for i, detection in enumerate(detections):
                print(f"  {i+1}. {detection['class']} (conf: {detection['confidence']:.2f})")
            
            # Create annotated image
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"annotated_{image_files[0]}")
            detector.annotate_image(test_image, detections, output_path)
        else:
            print("📂 No test images found in images/ directory")
    else:
        print("📂 Images directory not found")


if __name__ == "__main__":
    main()
