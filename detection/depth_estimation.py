#!/usr/bin/env python3
"""
MiDaS Depth Estimation Module
Provides monocular depth estimation for enhanced issue analysis.
"""

import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from typing import Optional, Dict, Tuple
import timm


class DepthEstimator:
    """MiDaS-based depth estimation for single images"""
    
    def __init__(self, model_name: str = "MiDaS_small", device: Optional[str] = None):
        """
        Initialize depth estimator
        
        Args:
            model_name: MiDaS model variant (MiDaS_small, MiDaS, DPT_Large, etc.)
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize MiDaS model and transforms"""
        try:
            # Try to load MiDaS model
            self.model = torch.hub.load('intel-isl/MiDaS', self.model_name, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            
            if self.model_name in ['MiDaS_small']:
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.default_transform
                
            print(f"✅ Loaded MiDaS model: {self.model_name} on {self.device}")
            
        except Exception as e:
            print(f"⚠️  Failed to load MiDaS model: {e}")
            print("💡 Depth estimation will be disabled")
            self.model = None
            self.transform = None
    
    def estimate_depth(self, image_path: str) -> Optional[np.ndarray]:
        """
        Estimate depth map for an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Depth map as numpy array (higher values = closer objects)
            None if estimation fails
        """
        if not self.model or not self.transform:
            return None
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_batch = self.transform(img_rgb).to(self.device)
            
            # Predict depth
            with torch.no_grad():
                prediction = self.model(input_batch)
                
                # Convert to numpy
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                depth_map = prediction.cpu().numpy()
                
            return depth_map
            
        except Exception as e:
            print(f"❌ Depth estimation failed for {image_path}: {e}")
            return None
    
    def get_object_depth(self, depth_map: np.ndarray, bbox: Dict) -> Optional[float]:
        """
        Get average depth for an object bounding box
        
        Args:
            depth_map: Depth map from estimate_depth
            bbox: Bounding box dictionary with center_x, center_y, width, height
            
        Returns:
            Average depth value for the object region
        """
        if depth_map is None:
            return None
        
        try:
            # Get bounding box coordinates
            center_x = int(bbox["center_x"])
            center_y = int(bbox["center_y"])
            width = int(bbox["width"])
            height = int(bbox["height"])
            
            # Calculate region bounds
            x1 = max(0, center_x - width // 2)
            x2 = min(depth_map.shape[1], center_x + width // 2)
            y1 = max(0, center_y - height // 2)
            y2 = min(depth_map.shape[0], center_y + height // 2)
            
            # Extract region and calculate mean depth
            region = depth_map[y1:y2, x1:x2]
            if region.size > 0:
                return float(np.mean(region))
            else:
                return None
                
        except Exception as e:
            print(f"❌ Failed to get object depth: {e}")
            return None
    
    def analyze_depth_occlusion(self, depth_map: np.ndarray, 
                               sign_bbox: Dict, vegetation_bbox: Dict) -> Dict:
        """
        Analyze if vegetation is in front of a sign based on depth
        
        Args:
            depth_map: Depth map from estimate_depth
            sign_bbox: Sign bounding box
            vegetation_bbox: Vegetation bounding box
            
        Returns:
            Analysis results with occlusion status and depth values
        """
        if depth_map is None:
            return {"occlusion_detected": False, "reason": "No depth map"}
        
        try:
            sign_depth = self.get_object_depth(depth_map, sign_bbox)
            veg_depth = self.get_object_depth(depth_map, vegetation_bbox)
            
            if sign_depth is None or veg_depth is None:
                return {"occlusion_detected": False, "reason": "Could not extract depths"}
            
            # In MiDaS, higher values typically mean closer objects
            # But this can vary, so we also consider the depth difference
            depth_difference = abs(veg_depth - sign_depth)
            depth_threshold = 0.1  # Adjust based on testing
            
            # If vegetation has significantly higher depth value than sign,
            # it's likely in front
            is_occluded = veg_depth > sign_depth and depth_difference > depth_threshold
            
            return {
                "occlusion_detected": is_occluded,
                "sign_depth": sign_depth,
                "vegetation_depth": veg_depth,
                "depth_difference": depth_difference,
                "confidence": min(depth_difference / depth_threshold, 1.0) if is_occluded else 0.0
            }
            
        except Exception as e:
            print(f"❌ Depth occlusion analysis failed: {e}")
            return {"occlusion_detected": False, "reason": f"Analysis error: {e}"}
    
    def save_depth_visualization(self, depth_map: np.ndarray, output_path: str):
        """
        Save depth map as a visualization
        
        Args:
            depth_map: Depth map to visualize
            output_path: Path to save the visualization
        """
        if depth_map is None:
            return
        
        try:
            # Normalize depth map to 0-255 range
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # Apply colormap for better visualization
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
            
            # Save
            cv2.imwrite(output_path, depth_colored)
            print(f"✅ Depth visualization saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to save depth visualization: {e}")


def main():
    """Test depth estimation"""
    print("🔍 Testing Depth Estimation...")
    
    # Initialize estimator
    estimator = DepthEstimator()
    
    # Test with sample image if available
    test_images_dir = "images"
    if os.path.exists(test_images_dir):
        image_files = [f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files and estimator.model:
            test_image = os.path.join(test_images_dir, image_files[0])
            print(f"🖼️  Testing with: {test_image}")
            
            # Estimate depth
            depth_map = estimator.estimate_depth(test_image)
            
            if depth_map is not None:
                print(f"📊 Depth map shape: {depth_map.shape}")
                print(f"📊 Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
                
                # Save visualization
                output_dir = "outputs"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"depth_{image_files[0]}")
                estimator.save_depth_visualization(depth_map, output_path)
            else:
                print("❌ Depth estimation failed")
        else:
            if not image_files:
                print("📂 No test images found in images/ directory")
            if not estimator.model:
                print("🚫 Depth estimation model not available")
    else:
        print("📂 Images directory not found")


if __name__ == "__main__":
    main()
