#!/usr/bin/env python3
"""
Refactored YOLOv8 Object Detection Model
Handles traffic sign and vegetation detection using YOLOv8 with improved structure and performance.
"""

import json
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple, Union
import cv2
import logging
from pathlib import Path

# Import our utilities and config
from config import (
    YOLO_MODEL_NAME, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD,
    YOLO_CLASSES_FILE, ENABLE_MODEL_CACHING, YOLO_MAX_DETECTIONS
)
from utils import (
    ModelLoadingError, handle_streamlit_errors, log_performance,
    load_model_cached, logger, retry_on_exception, safe_execute,
    validate_input
)


class YOLODetector:
    """Improved YOLO-based object detector with better error handling and performance"""
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 confidence_threshold: Optional[float] = None,
                 iou_threshold: Optional[float] = None):
        """
        Initialize the detector
        
        Args:
            model_path: Path to YOLO model weights (uses config default if None)
            confidence_threshold: Minimum confidence for detections (uses config default if None)
            iou_threshold: IoU threshold for NMS (uses config default if None)
        """
        # Validate inputs
        if model_path is not None:
            validate_input(model_path, str, required=True)
        if confidence_threshold is not None:
            validate_input(confidence_threshold, float, min_value=0.0, max_value=1.0)
        if iou_threshold is not None:
            validate_input(iou_threshold, float, min_value=0.0, max_value=1.0)
        
        # Set parameters with defaults from config
        self.model_path = model_path or YOLO_MODEL_NAME
        self.confidence_threshold = confidence_threshold or YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or YOLO_IOU_THRESHOLD
        self.max_detections = YOLO_MAX_DETECTIONS
        
        # Initialize state
        self.model = None
        self.class_names = None
        self.is_initialized = False
        
        # Initialize components
        self._load_class_names()
        self._initialize_model()
    
    @retry_on_exception(max_retries=2, exceptions=(FileNotFoundError, json.JSONDecodeError))
    def _load_class_names(self) -> None:
        """Load YOLO class names from JSON file or use defaults"""
        def load_from_file():
            """Load class names from JSON file"""
            if not YOLO_CLASSES_FILE.exists():
                raise FileNotFoundError(f"Class names file not found: {YOLO_CLASSES_FILE}")
            
            with open(YOLO_CLASSES_FILE, 'r') as f:
                data = json.load(f)
                return {int(k): v for k, v in data["class"].items()}
        
        # Try loading from file, fallback to defaults
        self.class_names = safe_execute(
            load_from_file,
            default_return=self._get_default_class_names(),
            error_message="Failed to load class names from file, using defaults"
        )
        
        logger.info(f"📋 Loaded {len(self.class_names)} YOLO class names")
    
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
    
    @retry_on_exception(max_retries=3, exceptions=(Exception,))
    def _initialize_model(self) -> None:
        """Initialize the YOLO model with caching support"""
        try:
            logger.info(f"🔄 Initializing YOLO model: {self.model_path}")
            
            if ENABLE_MODEL_CACHING:
                # Use cached model loading from Streamlit
                self.model = load_model_cached(self.model_path, "yolo")
            else:
                # Direct loading
                self.model = YOLO(self.model_path)
            
            if self.model is None:
                raise ModelLoadingError(f"Failed to load YOLO model: {self.model_path}")
            
            self.is_initialized = True
            logger.info(f"✅ Loaded YOLO model: {self.model_path}")
            
        except Exception as e:
            # Try fallback to default model
            try:
                logger.warning(f"Primary model failed, trying fallback: {e}")
                self.model = YOLO("yolov8n.pt")  # This will download if needed
                self.is_initialized = True
                logger.info("✅ Loaded fallback YOLOv8n model")
            except Exception as e2:
                error_msg = f"Failed to initialize YOLO model {self.model_path}: {e}. Fallback also failed: {e2}"
                logger.error(error_msg)
                self.is_initialized = False
                raise ModelLoadingError(error_msg) from e
    
    @log_performance
    @handle_streamlit_errors
    def detect_objects(self, 
                      image: Union[np.ndarray, str], 
                      confidence_threshold: Optional[float] = None,
                      iou_threshold: Optional[float] = None) -> List[Dict]:
        """
        Detect objects in image using YOLO
        
        Args:
            image: Input image as numpy array or file path
            confidence_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            
        Returns:
            List of detection dictionaries
            
        Raises:
            ModelLoadingError: If model is not properly loaded
        """
        if not self.is_initialized or self.model is None:
            raise ModelLoadingError("Model not initialized. Call _initialize_model() first.")
        
        if self.class_names is None:
            raise ModelLoadingError("Class names not loaded")
        
        # Validate inputs
        if isinstance(image, str):
            validate_input(image, str, required=True)
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")
        elif isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Empty image array provided")
        else:
            raise ValueError("Image must be numpy array or file path string")
        
        # Use provided thresholds or defaults
        conf_thresh = confidence_threshold or self.confidence_threshold
        iou_thresh = iou_threshold or self.iou_threshold
        
        # Validate thresholds
        validate_input(conf_thresh, float, min_value=0.0, max_value=1.0)
        validate_input(iou_thresh, float, min_value=0.0, max_value=1.0)
        
        try:
            # Run YOLO inference
            results = self.model(
                image,
                conf=conf_thresh,
                iou=iou_thresh,
                max_det=self.max_detections,
                verbose=False
            )
            
            detections = []
            
            # Process results
            if results and len(results) > 0:
                result = results[0]  # First (and only) image
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                    
                    for i in range(len(boxes)):
                        class_id = class_ids[i]
                        confidence = float(confidences[i])
                        bbox = boxes[i].tolist()  # [x1, y1, x2, y2]
                        
                        # Get class name
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        detection = {
                            'class_id': int(class_id),
                            'class': class_name,
                            'confidence': confidence,
                            'bbox_xyxy': bbox,
                            'bbox_xywh': self._xyxy_to_xywh(bbox)
                        }
                        
                        detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} objects with confidence >= {conf_thresh}")
            return detections
            
        except Exception as e:
            error_msg = f"Error during object detection: {e}"
            logger.error(error_msg)
            raise ModelLoadingError(error_msg) from e
    
    @staticmethod
    def _xyxy_to_xywh(bbox_xyxy: List[float]) -> List[float]:
        """
        Convert bounding box from xyxy to xywh format
        
        Args:
            bbox_xyxy: [x1, y1, x2, y2] format
            
        Returns:
            [x_center, y_center, width, height] format
        """
        x1, y1, x2, y2 = bbox_xyxy
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        return [x_center, y_center, width, height]
    
    @staticmethod
    def _xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
        """
        Convert bounding box from xywh to xyxy format
        
        Args:
            bbox_xywh: [x_center, y_center, width, height] format
            
        Returns:
            [x1, y1, x2, y2] format
        """
        x_center, y_center, width, height = bbox_xywh
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[Dict],
                       draw_confidence: bool = True,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        Draw detection bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            draw_confidence: Whether to draw confidence scores
            color: Box color in BGR format
            thickness: Line thickness
            
        Returns:
            Image with drawn detections
        """
        annotated_image = image.copy()
        
        for detection in detections:
            try:
                bbox = detection['bbox_xyxy']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Draw bounding box
                pt1 = (int(bbox[0]), int(bbox[1]))
                pt2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(annotated_image, pt1, pt2, color, thickness)
                
                # Draw label
                if draw_confidence:
                    label = f"{class_name}: {confidence:.2f}"
                else:
                    label = class_name
                
                # Calculate label size and position
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_image, 
                            (pt1[0], pt1[1] - label_h - 10),
                            (pt1[0] + label_w, pt1[1]),
                            color, -1)
                cv2.putText(annotated_image, label, 
                          (pt1[0], pt1[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
            except Exception as e:
                logger.warning(f"Error drawing detection: {e}")
                continue
        
        return annotated_image
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Get summary statistics of detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with class counts
        """
        summary = {}
        for detection in detections:
            class_name = detection['class']
            summary[class_name] = summary.get(class_name, 0) + 1
        
        return summary
    
    def filter_detections_by_class(self, 
                                  detections: List[Dict], 
                                  allowed_classes: List[str]) -> List[Dict]:
        """
        Filter detections by allowed class names
        
        Args:
            detections: List of detection dictionaries
            allowed_classes: List of allowed class names
            
        Returns:
            Filtered list of detections
        """
        return [d for d in detections if d['class'] in allowed_classes]
    
    def filter_detections_by_confidence(self, 
                                       detections: List[Dict], 
                                       min_confidence: float) -> List[Dict]:
        """
        Filter detections by minimum confidence
        
        Args:
            detections: List of detection dictionaries
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of detections
        """
        validate_input(min_confidence, float, min_value=0.0, max_value=1.0)
        return [d for d in detections if d['confidence'] >= min_confidence]
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_detections,
            'num_classes': len(self.class_names) if self.class_names else 0,
            'is_initialized': self.is_initialized
        }


# Backward compatibility
class Detector(YOLODetector):
    """Backward compatibility alias for YOLODetector"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.warning("Detector class is deprecated. Use YOLODetector instead.")
    
    def detect(self, image_path: str) -> List[Dict]:
        """Legacy detect method for backward compatibility"""
        return self.detect_objects(image_path)
