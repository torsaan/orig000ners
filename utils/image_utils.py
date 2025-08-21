#!/usr/bin/env python3
"""
Image Processing Utilities
Centralized image handling, validation, and preprocessing functions.
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import GPSTAGS
from typing import Tuple, Optional, List, Dict, Any, Union
import tempfile
import os
from pathlib import Path
import logging
import streamlit as st

from config import (
    MAX_IMAGE_SIZE, IMAGE_QUALITY, SUPPORTED_IMAGE_FORMATS,
    BATCH_SIZE, MAX_CONCURRENT_PROCESSES
)

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Centralized image processing utilities"""
    
    def __init__(self):
        """Initialize image processor"""
        self.supported_formats = SUPPORTED_IMAGE_FORMATS
        self.max_size = MAX_IMAGE_SIZE
        self.quality = IMAGE_QUALITY
    
    @staticmethod
    def validate_image_format(file_path: str) -> bool:
        """
        Validate if image format is supported
        
        Args:
            file_path: Path to image file
            
        Returns:
            bool: True if format is supported
        """
        try:
            file_extension = Path(file_path).suffix.lower()
            return file_extension in SUPPORTED_IMAGE_FORMATS
        except Exception as e:
            logger.error(f"Error validating image format for {file_path}: {e}")
            return False
    
    @staticmethod
    def load_image_safely(file_path: str) -> Optional[np.ndarray]:
        """
        Safely load image with error handling
        
        Args:
            file_path: Path to image file
            
        Returns:
            np.ndarray or None: Loaded image array or None if failed
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Image file not found: {file_path}")
                return None
            
            if not ImageProcessor.validate_image_format(file_path):
                logger.error(f"Unsupported image format: {file_path}")
                return None
            
            # Load with OpenCV
            image = cv2.imread(file_path)
            if image is None:
                logger.error(f"Failed to load image with OpenCV: {file_path}")
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.debug(f"Successfully loaded image: {file_path}")
            return image_rgb
            
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None
    
    @staticmethod
    def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
        """
        Load image from Streamlit uploaded file
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            np.ndarray or None: Loaded image array or None if failed
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load image
            image = ImageProcessor.load_image_safely(tmp_path)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading uploaded image {uploaded_file.name}: {e}")
            return None
    
    @staticmethod
    def resize_image_if_needed(image: np.ndarray, max_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Resize image if it exceeds maximum dimensions
        
        Args:
            image: Input image array
            max_size: Maximum (width, height) or None to use default
            
        Returns:
            np.ndarray: Resized image
        """
        try:
            if max_size is None:
                max_size = MAX_IMAGE_SIZE
            
            height, width = image.shape[:2]
            max_width, max_height = max_size
            
            # Check if resizing is needed
            if width <= max_width and height <= max_height:
                return image
            
            # Calculate scaling factor
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            return resized
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image
    
    @staticmethod
    def preprocess_image_for_detection(image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO detection
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Resize if needed
            processed = ImageProcessor.resize_image_if_needed(image)
            
            # Ensure correct color space (RGB)
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                # Already RGB, no conversion needed
                pass
            elif len(processed.shape) == 3 and processed.shape[2] == 4:
                # RGBA to RGB
                processed = processed[:, :, :3]
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    @staticmethod
    def array_to_base64(image_array: np.ndarray) -> str:
        """
        Convert numpy image array to base64 string
        
        Args:
            image_array: Input image array
            
        Returns:
            str: Base64 encoded image
        """
        try:
            if image_array is None:
                return ""
            
            # Convert to PIL Image
            if len(image_array.shape) == 3:
                pil_image = Image.fromarray(image_array.astype('uint8'))
            else:
                pil_image = Image.fromarray(image_array.astype('uint8'), mode='L')
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
            img_data = buffer.getvalue()
            
            return base64.b64encode(img_data).decode()
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""
    
    @staticmethod
    def base64_to_array(base64_string: str) -> Optional[np.ndarray]:
        """
        Convert base64 string to numpy image array
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            np.ndarray or None: Image array or None if failed
        """
        try:
            if not base64_string:
                return None
            
            # Decode base64
            img_data = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to numpy array
            return np.array(img)
            
        except Exception as e:
            logger.error(f"Error converting base64 to image array: {e}")
            return None
    
    @staticmethod
    def extract_gps_coords(image_path: str) -> Optional[Tuple[float, float]]:
        """
        Extract GPS coordinates from image EXIF data
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple[float, float] or None: (latitude, longitude) or None if not found
        """
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                
                if not exif_data:
                    return None
                
                gps_info = None
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    if tag_name == "GPSInfo":
                        gps_info = value
                        break
                
                if not gps_info:
                    return None
                
                # Extract GPS coordinates
                lat = ImageProcessor._get_decimal_from_dms(
                    gps_info.get(GPSTAGS.get(2, 2)), 
                    gps_info.get(GPSTAGS.get(1, 1))
                )
                lon = ImageProcessor._get_decimal_from_dms(
                    gps_info.get(GPSTAGS.get(4, 4)), 
                    gps_info.get(GPSTAGS.get(3, 3))
                )
                
                if lat is not None and lon is not None:
                    return lat, lon
                
                return None
                
        except Exception as e:
            logger.error(f"Error extracting GPS from {image_path}: {e}")
            return None
    
    @staticmethod
    def _get_decimal_from_dms(dms: tuple, ref: str) -> Optional[float]:
        """
        Convert DMS (Degrees, Minutes, Seconds) to decimal degrees
        
        Args:
            dms: Tuple of (degrees, minutes, seconds)
            ref: Reference direction ('N', 'S', 'E', 'W')
            
        Returns:
            float or None: Decimal degrees or None if conversion failed
        """
        try:
            if not dms or not ref:
                return None
            
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            if ref in ['S', 'W']:
                decimal = -decimal
            
            return decimal
            
        except Exception as e:
            logger.error(f"Error converting DMS to decimal: {e}")
            return None
    
    @staticmethod
    def batch_process_images(image_paths: List[str], 
                           process_function: callable,
                           batch_size: int = None) -> List[Any]:
        """
        Process multiple images in batches
        
        Args:
            image_paths: List of image file paths
            process_function: Function to apply to each image
            batch_size: Number of images per batch
            
        Returns:
            List[Any]: Results from processing function
        """
        try:
            if batch_size is None:
                batch_size = BATCH_SIZE
            
            results = []
            
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                batch_results = []
                
                for image_path in batch:
                    try:
                        result = process_function(image_path)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing image {image_path}: {e}")
                        batch_results.append(None)
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return []
    
    @staticmethod
    def create_image_thumbnail(image: np.ndarray, size: int = 150) -> np.ndarray:
        """
        Create thumbnail of image
        
        Args:
            image: Input image array
            size: Maximum dimension for thumbnail
            
        Returns:
            np.ndarray: Thumbnail image
        """
        try:
            height, width = image.shape[:2]
            
            # Calculate scaling factor
            scale = min(size / width, size / height)
            
            if scale >= 1:
                return image
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return thumbnail
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return image


# UI Display Utilities
def create_streamlit_thumbnail(image: Union[Image.Image, str, bytes, np.ndarray], 
                              max_size: Tuple[int, int] = (200, 200)) -> Image.Image:
    """
    Create a thumbnail from an image while preserving aspect ratio for Streamlit display
    
    Args:
        image: PIL Image, file path string, bytes, or numpy array
        max_size: Maximum dimensions (width, height)
    
    Returns:
        PIL Image thumbnail
    """
    try:
        if isinstance(image, str):
            # File path
            img = Image.open(image)
        elif isinstance(image, bytes):
            # Bytes data
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            # Numpy array
            img = Image.fromarray(image.astype('uint8'))
        elif isinstance(image, Image.Image):
            # Already a PIL Image
            img = image.copy()
        else:
            raise ValueError("Image must be PIL Image, file path, bytes, or numpy array")
        
        # Convert to RGB if necessary (for PNG with transparency, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Create thumbnail preserving aspect ratio
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        logger.error(f"Error creating streamlit thumbnail: {e}")
        # Return a placeholder image
        placeholder = Image.new('RGB', max_size, color='gray')
        return placeholder


def get_optimal_display_width(images_per_row: int, max_width: int = 400) -> int:
    """
    Calculate optimal image width based on grid layout
    
    Args:
        images_per_row: Number of images per row
        max_width: Maximum width for any image
    
    Returns:
        Optimal width in pixels
    """
    # Width mapping for different grid layouts
    width_mapping = {
        1: min(max_width, 600),  # Single image can be larger but not too large
        2: min(max_width, 400),  # Two images per row
        3: min(max_width, 300),  # Three images per row
        4: min(max_width, 250),  # Four images per row (ideal size)
        5: min(max_width, 200),  # Five images per row
        6: min(max_width, 180),  # Six images per row
    }
    
    return width_mapping.get(images_per_row, max_width)


class UIImageManager:
    """Manager class for consistent image display across the Streamlit UI"""
    
    def __init__(self, thumbnail_size: Tuple[int, int] = (200, 200), max_display_width: int = 400):
        self.thumbnail_size = thumbnail_size
        self.max_display_width = max_display_width
    
    def display_thumbnail(self, image: Union[Image.Image, str, bytes, np.ndarray], 
                         caption: str = "", container_width: bool = False) -> None:
        """Display a standardized thumbnail"""
        try:
            import streamlit as st
            thumbnail = create_streamlit_thumbnail(image, self.thumbnail_size)
            st.image(thumbnail, caption=caption, use_container_width=container_width)
        except ImportError:
            logger.warning("Streamlit not available for image display")
    
    def get_display_width(self, images_per_row: int) -> int:
        """Get optimal display width for grid layout"""
        return get_optimal_display_width(images_per_row, self.max_display_width)
    
    def resize_for_grid(self, image: Union[Image.Image, str, bytes, np.ndarray], 
                       images_per_row: int) -> Image.Image:
        """Resize image optimally for grid display"""
        optimal_width = self.get_display_width(images_per_row)
        
        # Convert to PIL if needed
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype('uint8'))
        elif isinstance(image, Image.Image):
            img = image.copy()
        else:
            raise ValueError("Image must be PIL Image, file path, bytes, or numpy array")
        
        # Calculate height based on aspect ratio
        aspect_ratio = img.height / img.width
        target_height = int(optimal_width * aspect_ratio)
        
        # Resize with high quality
        resized_img = img.resize((optimal_width, target_height), Image.Resampling.LANCZOS)
        
        return resized_img
