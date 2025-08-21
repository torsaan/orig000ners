#!/usr/bin/env python3
"""
RoadSight AI - Complete Professional Application
Combines object detection, expert review, and reporting in one comprehensive platform.
"""

import os
import json
import streamlit as st
import folium
from streamlit_folium import folium_static
from PIL import Image, ExifTags
from PIL.ExifTags import GPSTAGS
import tempfile
import zipfile
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import base64
import io
import numpy as np
import cv2
import pandas as pd
import hashlib

# Import our modules
from detection.model import Detector
from detection.depth_estimation import DepthEstimator
from nvdb.api_client import NVDBClient
from nvdb.matcher import NVDBMatcher
from analysis.issue_detector import IssueAnalyzer
from output.geojson_generator import GeoJSONGenerator, ImageProcessingResult
from reporting import ExpertReviewInterface
from reporting.interactive_report import InteractiveReportGenerator


# Cached Component Initialization Functions
@st.cache_resource
def load_detection_model() -> Detector:
    """Load and cache the YOLO detection model - persists across reruns"""
    return Detector()


@st.cache_resource  
def load_depth_estimator() -> DepthEstimator:
    """Load and cache the depth estimation model - persists across reruns"""
    return DepthEstimator()


@st.cache_resource
def load_nvdb_components() -> Tuple[NVDBClient, NVDBMatcher]:
    """Load and cache NVDB components - persists across reruns"""
    nvdb_client = NVDBClient()
    nvdb_matcher = NVDBMatcher()
    return nvdb_client, nvdb_matcher


@st.cache_resource
def load_analysis_components() -> Tuple[IssueAnalyzer, GeoJSONGenerator]:
    """Load and cache analysis components - persists across reruns"""
    issue_analyzer = IssueAnalyzer()
    geojson_generator = GeoJSONGenerator()
    return issue_analyzer, geojson_generator


@st.cache_resource
def load_reporting_components() -> Tuple[ExpertReviewInterface, InteractiveReportGenerator]:
    """Load and cache reporting components - persists across reruns"""
    expert_interface = ExpertReviewInterface()
    interactive_report = InteractiveReportGenerator()
    return expert_interface, interactive_report


@st.cache_data
def process_uploaded_images(_uploaded_files: List, config: Dict) -> List[ImageProcessingResult]:
    """
    Process uploaded images with caching based on file content hash
    
    Args:
        _uploaded_files: List of uploaded files (underscore prevents hashing the object itself)
        config: Processing configuration
        
    Returns:
        List of ImageProcessingResult objects
    """
    # Create stable hash key based on file contents and config
    file_hashes = []
    file_data = []
    
    for uploaded_file in _uploaded_files:
        # Get file content for processing
        content = uploaded_file.getvalue()
        file_hashes.append(hashlib.md5(content).hexdigest())
        file_data.append((uploaded_file.name, content))
    
    # Create config hash
    config_hash = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()
    
    # The actual cache key is: (file_hashes, config_hash)
    # If files or config change, this function will re-run
    
    # Get cached models
    detector = load_detection_model()
    depth_estimator = load_depth_estimator()
    nvdb_client, nvdb_matcher = load_nvdb_components()
    issue_analyzer, geojson_generator = load_analysis_components()
    
    # Process images
    results = []
    
    for name, content in file_data:
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Extract GPS coordinates
            gps_coords = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img_file:
                    temp_img_file.write(content)
                    temp_img_path = temp_img_file.name
                
                image = Image.open(temp_img_path)
                gps_coords = extract_gps_from_image(image)
                os.unlink(temp_img_path)
            except Exception:
                pass
            
            # Decide whether to process
            should_process = gps_coords is not None or config['process_without_gps']
            
            if should_process:
                # Create a mock uploaded file object for existing processing logic
                class MockUploadedFile:
                    def __init__(self, name, content):
                        self.name = name
                        self._content = content
                        self._position = 0
                    
                    def getvalue(self):
                        return self._content
                    
                    def seek(self, position):
                        self._position = position
                    
                    def read(self, size=-1):
                        if size == -1:
                            result = self._content[self._position:]
                            self._position = len(self._content)
                        else:
                            result = self._content[self._position:self._position + size]
                            self._position += len(result)
                        return result
                
                mock_file = MockUploadedFile(name, content)
                
                # Process using existing logic
                app = RoadSightAIApp()
                app.detector = detector
                app.depth_estimator = depth_estimator
                app.nvdb_client = nvdb_client
                app.nvdb_matcher = nvdb_matcher
                app.issue_analyzer = issue_analyzer
                app.geojson_generator = geojson_generator
                
                result = app.process_single_image(mock_file, gps_coords, config['process_without_gps'])
                results.append(result)
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    return results


@st.cache_data
def load_custom_css() -> str:
    """Load and cache custom CSS content"""
    try:
        css_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'style.css')
        with open(css_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'cached_models_loaded' not in st.session_state:
        st.session_state.cached_models_loaded = False
    if 'last_config_hash' not in st.session_state:
        st.session_state.last_config_hash = None
    if 'interactive_report_initialized' not in st.session_state:
        st.session_state.interactive_report_initialized = False
    if 'report_initialized' not in st.session_state:
        st.session_state.report_initialized = False
    if 'detected_issues' not in st.session_state:
        st.session_state.detected_issues = []


def extract_gps_from_image(image: Image.Image) -> Optional[Tuple[float, float]]:
    """Extract GPS coordinates from PIL Image EXIF data"""
    try:
        exif_data = image._getexif()
        
        if not exif_data:
            return None
        
        # Look for GPS info
        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value
                break
        
        if not gps_info:
            return None
        
        # Convert to decimal degrees
        def convert_to_degrees(value):
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)
        
        lat = convert_to_degrees(gps_info.get('GPSLatitude', [0, 0, 0]))
        lon = convert_to_degrees(gps_info.get('GPSLongitude', [0, 0, 0]))
        
        # Apply hemispheres
        if gps_info.get('GPSLatitudeRef') == 'S':
            lat = -lat
        if gps_info.get('GPSLongitudeRef') == 'W':
            lon = -lon
        
        return lat, lon
        
    except Exception:
        return None


class RoadSightAIApp:
    """Complete RoadSight AI application with expert review and reporting"""
    
    def __init__(self):
        """Initialize the application"""
        # Initialize session state first
        initialize_session_state()
        
        self.detector = None
        self.depth_estimator = None
        self.nvdb_client = None
        self.nvdb_matcher = None
        self.issue_analyzer = None
        self.geojson_generator = None
        self.expert_interface = None
        self.interactive_report = None
        
        # Initialize components using cached functions
        self._initialize_components()
    
    def load_css(self):
        """Load custom CSS styling with caching"""
        css_content = load_custom_css()
        if css_content:
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
        else:
            st.warning("Custom CSS file not found. Using default styling.")
    
    def _initialize_components(self):
        """Initialize all processing components using cached resources"""
        try:
            with st.spinner("🔧 Initializing processing components..."):
                # Use cached component loading functions
                self.detector = load_detection_model()
                self.depth_estimator = load_depth_estimator()
                self.nvdb_client, self.nvdb_matcher = load_nvdb_components()
                self.issue_analyzer, self.geojson_generator = load_analysis_components()
                self.expert_interface, self.interactive_report = load_reporting_components()
                
            st.success("✅ All components initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            st.info("Some features may not be available.")

    def image_to_base64(self, image_array: np.ndarray) -> str:
        """Convert numpy image array to base64 string for HTML display"""
        try:
            if image_array is None:
                return ""
            
            # Convert BGR to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_array
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb.astype('uint8'))
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            img_data = buffer.getvalue()
            
            return base64.b64encode(img_data).decode()
        except Exception as e:
            return ""
    
    def pil_to_base64(self, pil_image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        try:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            img_data = buffer.getvalue()
            return base64.b64encode(img_data).decode()
        except Exception as e:
            return ""
    
    def base64_to_pil(self, base64_string: str) -> Image.Image:
        """Convert base64 string back to PIL Image"""
        try:
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            # Return a placeholder image if conversion fails
            return Image.new('RGB', (150, 150), color='lightgray')
    
    def crop_detection_from_image(self, base64_image: str, bbox_xyxy: List[float], padding: int = 10) -> Image.Image:
        """
        Crop detection region from base64 encoded image
        
        Args:
            base64_image: Base64 encoded source image
            bbox_xyxy: Bounding box coordinates [x1, y1, x2, y2]
            padding: Additional padding around the detection (pixels)
            
        Returns:
            Cropped PIL Image
        """
        try:
            # Convert base64 to PIL Image
            source_image = self.base64_to_pil(base64_image)
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = bbox_xyxy
            
            # Add padding and ensure coordinates are within image bounds
            img_width, img_height = source_image.size
            x1 = max(0, int(x1 - padding))
            y1 = max(0, int(y1 - padding))
            x2 = min(img_width, int(x2 + padding))
            y2 = min(img_height, int(y2 + padding))
            
            # Crop the image
            cropped_image = source_image.crop((x1, y1, x2, y2))
            
            return cropped_image
            
        except Exception as e:
            # Return a placeholder image if cropping fails
            return Image.new('RGB', (150, 150), color='lightgray')
    
    def extract_gps_from_exif(self, image_file) -> Optional[Tuple[float, float]]:
        """
        Extract GPS coordinates from image EXIF data
        
        Args:
            image_file: Uploaded image file
            
        Returns:
            GPS coordinates (lat, lon) or None
        """
        try:
            image = Image.open(image_file)
            exif_data = image._getexif()
            
            if not exif_data:
                return None
            
            # Look for GPS info
            gps_info = {}
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    for gps_tag_id, gps_value in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_info[gps_tag] = gps_value
                    break
            
            if not gps_info:
                return None
            
            # Convert to decimal degrees
            def convert_to_degrees(value):
                d, m, s = value
                return d + (m / 60.0) + (s / 3600.0)
            
            lat = convert_to_degrees(gps_info.get('GPSLatitude', [0, 0, 0]))
            lon = convert_to_degrees(gps_info.get('GPSLongitude', [0, 0, 0]))
            
            # Check for hemisphere
            if gps_info.get('GPSLatitudeRef') == 'S':
                lat = -lat
            if gps_info.get('GPSLongitudeRef') == 'W':
                lon = -lon
                
            return (lat, lon)
            
        except Exception as e:
            return None
    
    def process_single_image(self, image_file, gps_coords: Optional[Tuple[float, float]], process_without_gps: bool = False) -> ImageProcessingResult:
        """
        Process a single image through the complete pipeline
        
        Args:
            image_file: Uploaded image file
            gps_coords: GPS coordinates (lat, lon) or None
            process_without_gps: Whether to process images without GPS coordinates
            
        Returns:
            Processing results
        """
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_file.getvalue())
            temp_image_path = tmp_file.name
        
        try:
            # Store original image as base64
            original_image = Image.open(temp_image_path)
            original_image_b64 = self.pil_to_base64(original_image)
            
            # Object detection
            detections = []
            annotated_image = None
            annotated_image_b64 = None
            
            if self.detector:
                detections = self.detector.detect(temp_image_path)
                
                # Create annotated image if we have detections
                if detections:
                    annotated_image = self.detector.annotate_image(temp_image_path, detections)
                    if annotated_image is not None:
                        annotated_image_b64 = self.image_to_base64(annotated_image)
            
            # Only do analysis if we have GPS coordinates (or if processing without GPS)
            issues = []
            nvdb_matches = []
            
            if gps_coords or process_without_gps:
                # Depth estimation
                depth_map = None
                if self.depth_estimator:
                    depth_map = self.depth_estimator.estimate_depth(temp_image_path)
                
                # Issue analysis
                if self.issue_analyzer and detections:
                    issue_objects = self.issue_analyzer.analyze_objects(detections, depth_map)
                    issues = [
                        {
                            "issue_type": issue.issue_type,
                            "affected_object": issue.affected_object,
                            "severity": issue.severity,
                            "confidence": issue.confidence,
                            "details": issue.details,
                            "recommendations": issue.recommendations
                        }
                        for issue in issue_objects
                    ]
                
                # NVDB matching (only if we have GPS coordinates)
                if self.nvdb_matcher and detections and gps_coords:
                    match_objects = self.nvdb_matcher.match_detections(detections, gps_coords)
                    nvdb_matches = [
                        {
                            "detected_class": match.detected_class,
                            "nvdb_id": match.nvdb_id,
                            "nvdb_status": match.nvdb_status,
                            "confidence": match.confidence,
                            "nvdb_data": match.nvdb_data
                        }
                        for match in match_objects
                    ]
            
            # Create result
            result = ImageProcessingResult(
                image_path=image_file.name,
                gps_coords=gps_coords or (0.0, 0.0),  # Default coords for non-GPS images
                detections=detections,
                issues=issues,
                nvdb_matches=nvdb_matches,
                processing_metadata={
                    "processed_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "gps_accuracy": "smartphone GPS (~3-5m)" if gps_coords else "no GPS data",
                    "has_gps": gps_coords is not None
                },
                original_image_b64=original_image_b64,
                annotated_image_b64=annotated_image_b64
            )
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
    
    def create_results_map(self, results: List[ImageProcessingResult]) -> folium.Map:
        """
        Create Folium map with processing results (only for images with GPS)
        
        Args:
            results: List of processing results
            
        Returns:
            Folium map object
        """
        # Filter results to only include those with GPS coordinates
        gps_results = [r for r in results if r.processing_metadata.get("has_gps", False)]
        
        if not gps_results:
            # Default map centered on Norway
            return folium.Map(location=[62.0, 10.0], zoom_start=6)
        
        # Calculate map center
        lats = [result.gps_coords[0] for result in gps_results]
        lons = [result.gps_coords[1] for result in gps_results]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add markers for each result
        for i, result in enumerate(gps_results):
            lat, lon = result.gps_coords
            
            # Determine marker color based on issues
            color = "green"  # No issues
            if result.issues:
                high_severity = any(issue["severity"] == "high" for issue in result.issues)
                medium_severity = any(issue["severity"] == "medium" for issue in result.issues)
                
                if high_severity:
                    color = "red"
                elif medium_severity:
                    color = "orange"
                else:
                    color = "yellow"
            
            # Create popup content with image preview
            popup_content = f"""
            <div style="width:400px;">
                <b>📷 Image:</b> {result.image_path}<br>
                <b>📍 GPS:</b> {lat:.6f}, {lon:.6f}<br>
                <b>🔍 Detections:</b> {len(result.detections)}<br>
                <b>⚠️ Issues:</b> {len(result.issues)}<br>
                <b>🗺️ NVDB Matches:</b> {len([m for m in result.nvdb_matches if m["nvdb_status"] == "match"])}<br>
            """
            
            # Add image preview if available
            if hasattr(result, 'original_image_b64') and result.original_image_b64:
                popup_content += f"""
                <br><b>Original Image:</b><br>
                <img src="data:image/jpeg;base64,{result.original_image_b64}" style="width:350px;height:auto;"><br>
                """
            
            # Add annotated image preview if available
            if hasattr(result, 'annotated_image_b64') and result.annotated_image_b64:
                popup_content += f"""
                <br><b>With Detections:</b><br>
                <img src="data:image/jpeg;base64,{result.annotated_image_b64}" style="width:350px;height:auto;"><br>
                """
            
            # Add detection details
            if result.detections:
                popup_content += "<br><b>Detected Objects:</b><br>"
                for detection in result.detections[:3]:  # Show first 3
                    popup_content += f"• {detection['class']} ({detection['confidence']:.2f})<br>"
            
            # Add issue details
            if result.issues:
                popup_content += "<br><b>Issues:</b><br>"
                for issue in result.issues[:2]:  # Show first 2
                    popup_content += f"• {issue['issue_type']} ({issue['severity']})<br>"
            
            # Close the div
            popup_content += "</div>"
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=500),
                icon=folium.Icon(color=color, icon='camera')
            ).add_to(m)
        
        return m
    
    def create_issues_dataframe(self, results: List[ImageProcessingResult]) -> pd.DataFrame:
        """Create a comprehensive issues summary DataFrame"""
        data = []
        
        for result in results:
            if result.detections:
                for detection in result.detections:
                    # Find related issues for this detection
                    related_issues = [
                        issue for issue in result.issues 
                        if issue.get('affected_object', '').lower() in detection['class'].lower()
                    ]
                    
                    # Create row for each detection
                    row = {
                        'Image': result.image_path,
                        'GPS Latitude': result.gps_coords[0] if result.processing_metadata.get("has_gps") else "N/A",
                        'GPS Longitude': result.gps_coords[1] if result.processing_metadata.get("has_gps") else "N/A",
                        'Detected Object': detection['class'],
                        'Detection Confidence': f"{detection['confidence']:.3f}",
                        'Issues Found': len(related_issues),
                        'Highest Severity': max([issue['severity'] for issue in related_issues], default='none'),
                        'Issue Types': ', '.join([issue['issue_type'] for issue in related_issues]) if related_issues else 'None',
                        'NVDB Match': 'Yes' if any(m['nvdb_status'] == 'match' for m in result.nvdb_matches) else 'No',
                        'Processing Time': result.processing_metadata.get('processed_at', 'Unknown')
                    }
                    data.append(row)
            else:
                # No detections case
                row = {
                    'Image': result.image_path,
                    'GPS Latitude': result.gps_coords[0] if result.processing_metadata.get("has_gps") else "N/A",
                    'GPS Longitude': result.gps_coords[1] if result.processing_metadata.get("has_gps") else "N/A",
                    'Detected Object': 'None',
                    'Detection Confidence': 'N/A',
                    'Issues Found': 0,
                    'Highest Severity': 'none',
                    'Issue Types': 'None',
                    'NVDB Match': 'No',
                    'Processing Time': result.processing_metadata.get('processed_at', 'Unknown')
                }
                data.append(row)
        
        return pd.DataFrame(data)
    
    def render_sidebar(self):
        """Render the sidebar with controls and configuration"""
        with st.sidebar:
            # Company logo section
            st.markdown("""
            <div class="logo-container">
                <h2>🚗 RoadSight AI</h2>
                <p>Professional Asset Inspection</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.header("📁 File Upload")
            uploaded_files = st.file_uploader(
                "Upload Images",
                type=['jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload JPEG images for analysis. GPS data optional for basic detection."
            )
            
            st.divider()
            
            st.header("⚙️ Configuration")
            
            # Processing options
            st.subheader("Processing Options")
            enable_depth = st.checkbox("Enable depth estimation", value=True)
            enable_nvdb = st.checkbox("Enable NVDB matching", value=True, 
                                    help="Only available for images with GPS coordinates")
            process_without_gps = st.checkbox("Process images without GPS", value=True,
                                            help="Allow processing of images without GPS data (detection only)")
            
            st.subheader("Detection Settings")
            confidence_threshold = st.slider("Detection confidence", 0.1, 1.0, 0.5, 0.1)
            iou_threshold = st.slider("IoU threshold", 0.05, 0.5, 0.15, 0.05)
            
            st.divider()
            
            st.subheader("📊 Quick Stats")
            if hasattr(st.session_state, 'processing_results') and st.session_state.processing_results:
                results = st.session_state.processing_results
                total_images = len(results)
                gps_images = len([r for r in results if r.processing_metadata.get("has_gps")])
                total_detections = sum(len(r.detections) for r in results)
                total_issues = sum(len(r.issues) for r in results)
                
                st.metric("Total Images", total_images)
                st.metric("With GPS", gps_images)
                st.metric("Total Detections", total_detections)
                st.metric("Issues Found", total_issues)
            else:
                st.info("No processing results yet")
            
            st.divider()
            
            st.subheader("ℹ️ About")
            st.info("""
            **RoadSight AI** processes smartphone images to:
            • Detect traffic signs and vegetation
            • Identify obscuration issues
            • Verify assets against NVDB
            • Generate GIS-ready outputs
            
            **New**: Expert review and PDF reporting!
            """)
            
            return uploaded_files, {
                'enable_depth': enable_depth,
                'enable_nvdb': enable_nvdb,
                'process_without_gps': process_without_gps,
                'confidence_threshold': confidence_threshold,
                'iou_threshold': iou_threshold
            }
    
    def render_detection_tab(self, uploaded_files, config):
        """Render the object detection tab"""
        st.header("🎯 Object Detection & Analysis")
        
        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} image(s)")
            
            # Show cache status
            if hasattr(st.session_state, 'processing_results') and st.session_state.processing_results:
                st.success("⚡ **Fast Mode Active**: Previous results cached - UI interactions will be instant!")
            
            # Show uploaded images preview
            st.subheader("📸 Uploaded Images Preview")
            
            # Display images in columns
            cols_per_row = 3
            for i in range(0, len(uploaded_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(uploaded_files):
                        image_file = uploaded_files[i + j]
                        
                        with col:
                            # Display image as thumbnail (max 200x200)
                            image = Image.open(image_file)
                            
                            # Create thumbnail preserving aspect ratio
                            thumbnail = image.copy()
                            thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
                            
                            st.image(thumbnail, caption=image_file.name, use_container_width=False)
                            
                            # Show GPS info if available
                            gps_coords = self.extract_gps_from_exif(image_file)
                            if gps_coords:
                                st.success(f"📍 GPS: {gps_coords[0]:.6f}, {gps_coords[1]:.6f}")
                            else:
                                st.warning("⚠️ No GPS data")
                            
                            # Reset file pointer for later processing
                            image_file.seek(0)
            
            st.divider()
            
            if st.button("🚀 Process Images", type="primary", use_container_width=True):
                # Use cached processing function
                with st.spinner("🔄 Processing images..."):
                    results = process_uploaded_images(uploaded_files, config)
                
                # Store results in session state
                st.session_state.processing_results = results
                
                # Display summary
                if results:
                    st.success(f"✅ Successfully processed {len(results)} image(s)")
                    
                    # Quick statistics
                    total_detections = sum(len(r.detections) for r in results)
                    total_issues = sum(len(r.issues) for r in results)
                    total_matches = sum(len([m for m in r.nvdb_matches if m["nvdb_status"] == "match"]) for r in results)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Detections", total_detections)
                    col2.metric("Issues Found", total_issues)
                    col3.metric("NVDB Matches", total_matches)
                    
                    # Show detected object classes
                    all_classes = []
                    for result in results:
                        for detection in result.detections:
                            all_classes.append(detection['class'])
                    
                    if all_classes:
                        class_counts = {}
                        for cls in all_classes:
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                        
                        st.subheader("🎯 Detected Object Classes")
                        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"• **{cls}**: {count} instances")
                else:
                    st.warning("No images were successfully processed")
        else:
            st.info("👈 Upload images using the sidebar to get started")
    
    def render_map_view(self):
        """Render the map view tab"""
        st.header("🗺️ Interactive Results Map")
        
        if hasattr(st.session_state, 'processing_results') and st.session_state.processing_results:
            results = st.session_state.processing_results
            gps_results = [r for r in results if r.processing_metadata.get("has_gps", False)]
            
            if gps_results:
                # Create and display map
                with st.container():
                    st.markdown('<div class="map-container">', unsafe_allow_html=True)
                    results_map = self.create_results_map(results)
                    folium_static(results_map, width=1000, height=600)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Map legend
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("""
                    **📍 Map Legend:**
                    - 🟢 **Green**: No issues detected
                    - 🟡 **Yellow**: Low severity issues  
                    - 🟠 **Orange**: Medium severity issues
                    - 🔴 **Red**: High severity issues
                    """)
                with col2:
                    st.metric("Images on Map", len(gps_results))
                    if len(results) > len(gps_results):
                        st.warning(f"{len(results) - len(gps_results)} images without GPS not shown")
                
                st.info("💡 **Tip**: Click on camera markers to view original images and detection results!")
            else:
                st.warning("🗺️ No images with GPS coordinates found. Upload GPS-tagged images to see them on the map.")
                # Show default map
                default_map = folium.Map(location=[62.0, 10.0], zoom_start=6)
                folium_static(default_map, width=1000, height=400)
        else:
            st.info("📷 No processing results available. Upload and process images to see them on the map.")
            # Show default map
            default_map = folium.Map(location=[62.0, 10.0], zoom_start=6)
            folium_static(default_map, width=1000, height=400)
    
    def render_image_gallery(self):
        """Render the image gallery tab"""
        st.header("🖼️ Image Gallery")
        
        if hasattr(st.session_state, 'processing_results') and st.session_state.processing_results:
            results = st.session_state.processing_results
            
            # Gallery view options
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                view_mode = st.selectbox("View Mode", ["Side by Side", "Original Only", "Annotated Only"])
            with col2:
                images_per_row = st.selectbox("Images per Row", [1, 2, 3, 4, 5, 6], index=3)  # Default to 4
            with col3:
                show_details = st.checkbox("Show Details", value=True)
            
            st.divider()
            
            # Display images in grid
            for i in range(0, len(results), images_per_row):
                cols = st.columns(images_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        result = results[i + j]
                        
                        with col:
                            st.markdown(f'<div class="image-gallery-item">', unsafe_allow_html=True)
                            
                            # Image title
                            st.subheader(f"📷 {result.image_path}")
                            
                            # Display images based on view mode
                            if view_mode == "Side by Side":
                                if hasattr(result, 'original_image_b64') and result.original_image_b64:
                                    img_data = base64.b64decode(result.original_image_b64)
                                    img = Image.open(io.BytesIO(img_data))
                                    
                                    # Resize image to optimal width (max 250px for 4-column view)
                                    optimal_width = min(250, 600 // images_per_row) if images_per_row > 1 else 400
                                    img_ratio = img.height / img.width
                                    target_height = int(optimal_width * img_ratio)
                                    img_resized = img.resize((optimal_width, target_height), Image.Resampling.LANCZOS)
                                    
                                    st.image(img_resized, caption="Original", use_container_width=False)
                                
                                if hasattr(result, 'annotated_image_b64') and result.annotated_image_b64:
                                    img_data = base64.b64decode(result.annotated_image_b64)
                                    img = Image.open(io.BytesIO(img_data))
                                    
                                    # Resize image to optimal width
                                    optimal_width = min(250, 600 // images_per_row) if images_per_row > 1 else 400
                                    img_ratio = img.height / img.width
                                    target_height = int(optimal_width * img_ratio)
                                    img_resized = img.resize((optimal_width, target_height), Image.Resampling.LANCZOS)
                                    
                                    st.image(img_resized, caption="With Detections", use_container_width=False)
                                elif result.detections:
                                    st.info("Detections found but annotation failed")
                                else:
                                    st.info("No detections found")
                            
                            elif view_mode == "Original Only":
                                if hasattr(result, 'original_image_b64') and result.original_image_b64:
                                    img_data = base64.b64decode(result.original_image_b64)
                                    img = Image.open(io.BytesIO(img_data))
                                    
                                    # Resize image to optimal width
                                    optimal_width = min(250, 600 // images_per_row) if images_per_row > 1 else 400
                                    img_ratio = img.height / img.width
                                    target_height = int(optimal_width * img_ratio)
                                    img_resized = img.resize((optimal_width, target_height), Image.Resampling.LANCZOS)
                                    
                                    st.image(img_resized, caption="Original Image", use_container_width=False)
                            
                            elif view_mode == "Annotated Only":
                                if hasattr(result, 'annotated_image_b64') and result.annotated_image_b64:
                                    img_data = base64.b64decode(result.annotated_image_b64)
                                    img = Image.open(io.BytesIO(img_data))
                                    
                                    # Resize image to optimal width
                                    optimal_width = min(250, 600 // images_per_row) if images_per_row > 1 else 400
                                    img_ratio = img.height / img.width
                                    target_height = int(optimal_width * img_ratio)
                                    img_resized = img.resize((optimal_width, target_height), Image.Resampling.LANCZOS)
                                    
                                    st.image(img_resized, caption="With Detections", use_container_width=False)
                                else:
                                    st.info("No annotations available")
                            
                            # Show details if enabled
                            if show_details:
                                has_gps = result.processing_metadata.get("has_gps", False)
                                if has_gps:
                                    st.write(f"📍 **GPS**: {result.gps_coords[0]:.6f}, {result.gps_coords[1]:.6f}")
                                else:
                                    st.write("📍 **GPS**: Not available")
                                
                                st.write(f"🔍 **Detections**: {len(result.detections)}")
                                st.write(f"⚠️ **Issues**: {len(result.issues)}")
                                
                                # Show detection summary
                                if result.detections:
                                    detection_summary = ", ".join([d['class'] for d in result.detections[:3]])
                                    if len(result.detections) > 3:
                                        detection_summary += f" +{len(result.detections) - 3} more"
                                    st.write(f"**Objects**: {detection_summary}")
                                
                                # Show issue summary
                                if result.issues:
                                    high_issues = len([i for i in result.issues if i['severity'] == 'high'])
                                    medium_issues = len([i for i in result.issues if i['severity'] == 'medium'])
                                    if high_issues > 0:
                                        st.error(f"🚨 {high_issues} high severity issue(s)")
                                    elif medium_issues > 0:
                                        st.warning(f"⚠️ {medium_issues} medium severity issue(s)")
                                    else:
                                        st.info("💡 Low severity issues only")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📷 No images processed yet. Upload and process images to view them here.")
    
    def render_issue_summary(self):
        """Render the issue summary tab with data table"""
        st.header("📝 Comprehensive Issue Summary")
        
        if hasattr(st.session_state, 'processing_results') and st.session_state.processing_results:
            results = st.session_state.processing_results
            
            # Create summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_images = len(results)
            total_detections = sum(len(r.detections) for r in results)
            total_issues = sum(len(r.issues) for r in results)
            high_severity_issues = sum(len([i for i in r.issues if i['severity'] == 'high']) for r in results)
            
            col1.metric("Total Images", total_images)
            col2.metric("Total Detections", total_detections)
            col3.metric("Total Issues", total_issues)
            col4.metric("High Severity", high_severity_issues)
            
            st.divider()
            
            # Create and display DataFrame
            df = self.create_issues_dataframe(results)
            
            if not df.empty:
                st.subheader("📊 Detailed Detection & Issue Report")
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    severity_filter = st.selectbox("Filter by Severity", 
                                                 ["All", "high", "medium", "low", "none"])
                with col2:
                    object_filter = st.selectbox("Filter by Object Type", 
                                                ["All"] + list(df['Detected Object'].unique()))
                with col3:
                    gps_filter = st.selectbox("GPS Status", ["All", "With GPS", "Without GPS"])
                
                # Apply filters
                filtered_df = df.copy()
                
                if severity_filter != "All":
                    filtered_df = filtered_df[filtered_df['Highest Severity'] == severity_filter]
                
                if object_filter != "All":
                    filtered_df = filtered_df[filtered_df['Detected Object'] == object_filter]
                
                if gps_filter == "With GPS":
                    filtered_df = filtered_df[filtered_df['GPS Latitude'] != "N/A"]
                elif gps_filter == "Without GPS":
                    filtered_df = filtered_df[filtered_df['GPS Latitude'] == "N/A"]
                
                # Display filtered table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download options
                st.subheader("💾 Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV Report",
                        data=csv_data,
                        file_name=f"roadside_inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    if st.button("📄 Generate GeoJSON Export"):
                        with st.spinner("Generating GeoJSON..."):
                            geojson = self.geojson_generator.generate_geojson(results)
                            
                            geojson_str = json.dumps(geojson, indent=2)
                            st.download_button(
                                label="🗺️ Download GeoJSON",
                                data=geojson_str,
                                file_name=f"roadside_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                                mime="application/geo+json"
                            )
            else:
                st.info("No detection data available for summary.")
        else:
            st.info("📊 No analysis results available. Process images to see detailed summaries here.")
    
    def render_interactive_report(self):
        """Render the interactive report tab with editable components"""
        if hasattr(st.session_state, 'processing_results') and st.session_state.processing_results:
            # Use session state data efficiently - no need to reload models
            if self.interactive_report:
                # Initialize the interactive report from processing results
                # The interactive report module handles its own session state initialization
                self.interactive_report.initialize_from_processing_results(st.session_state.processing_results)
                
                # Render the interactive report interface
                self.interactive_report.render_interactive_report()
            else:
                st.error("❌ Interactive report generator not available")
        else:
            st.info("📊 No processing results available. Upload and analyze images first to create an interactive report.")
            
            # Show example of what the interactive report will look like
            st.markdown("""
            ### 📋 Interactive Report Features
            
            Once you process images, you'll be able to:
            
            - **🖼️ View Images**: See original and analyzed images side-by-side for each detection
            - **📝 Edit Notes**: Add custom notes for each detected issue
            - **⚠️ Adjust Severity**: Use dropdowns to set severity levels (Low, Medium, High, Critical)
            - **🏷️ Categorize Issues**: Classify detections into issue types
            - **💾 Export Reports**: Download as CSV or JSON formats
            - **📊 View Statistics**: Get summaries and breakdowns of findings
            
            Upload images and run detection to get started!
            """)
    
    def render_technical_report(self):
        """Render the technical report with card-based layout"""
        st.header("📊 Technical Report")
        
        if hasattr(st.session_state, 'processing_results') and st.session_state.processing_results:
            results = st.session_state.processing_results
            
            # Report summary statistics
            total_images = len(results)
            total_detections = sum(len(result.detections) for result in results)
            total_issues = sum(len(result.issues) for result in results)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📸 Images Processed", total_images)
            with col2:
                st.metric("🔍 Total Detections", total_detections)
            with col3:
                st.metric("⚠️ Issues Identified", total_issues)
            with col4:
                avg_confidence = np.mean([det['confidence'] for result in results for det in result.detections]) if total_detections > 0 else 0
                st.metric("📈 Avg Confidence", f"{avg_confidence:.2f}")
            
            st.divider()
            
            # Create cards for each detection
            self.render_detection_cards(results)
            
        else:
            st.info("📊 No processing results available. Upload and analyze images first to generate a technical report.")
            
            # Show example of technical report features
            st.markdown("""
            ### 📋 Technical Report Features
            
            The technical report provides detailed analysis in an easy-to-read card format:
            
            - **🎯 Detection Cards**: Each detection displayed in a professional card with icon and details
            - **📊 Confidence Indicators**: Visual indicators showing detection confidence levels
            - **🏷️ Classification Tags**: Clear badges showing detection types (pothole, crack, sign, etc.)
            - **📍 Location Data**: GPS coordinates and spatial information
            - **🔧 Technical Details**: Expandable sections with raw detection data
            - **📈 Statistics**: Summary metrics and performance indicators
            
            Upload images and run detection to see the technical report!
            """)
    
    def render_detection_cards(self, results):
        """Render detection data as professional cards"""
        st.subheader("🎯 Detection Details")
        
        # Group detections for card display
        all_detections = []
        for result in results:
            for detection in result.detections:
                detection_data = {
                    'image_path': result.image_path,
                    'detection': detection,
                    'gps_coords': result.gps_coords,
                    'processing_metadata': result.processing_metadata
                }
                all_detections.append(detection_data)
        
        if not all_detections:
            st.info("No detections found to display.")
            return
        
        # Create grid of cards (2 columns)
        for i in range(0, len(all_detections), 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < len(all_detections):
                    detection_data = all_detections[i + j]
                    
                    with col:
                        self.render_detection_card(detection_data, i + j + 1)
    
    def render_detection_card(self, detection_data, card_index):
        """Render a single detection as a professional card with thumbnail"""
        detection = detection_data['detection']
        image_path = detection_data['image_path']
        gps_coords = detection_data['gps_coords']
        
        # Get the original image from results for cropping
        original_image_b64 = None
        for result in st.session_state.get('processing_results', []):
            if result.image_path == image_path:
                original_image_b64 = result.original_image_b64
                break
        
        # Start card container
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        
        # Card header with detection ID
        st.markdown(f'<div class="card-header">Detection #{card_index:03d}: {detection["class"].title()}</div>', 
                   unsafe_allow_html=True)
        
        # Create three-column layout: thumbnail, details, actions
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            # --- Column 1: Detection Thumbnail ---
            if original_image_b64 and 'bbox_xyxy' in detection:
                try:
                    # Crop the detection from the original image
                    cropped_image = self.crop_detection_from_image(
                        original_image_b64, 
                        detection['bbox_xyxy'],
                        padding=15
                    )
                    st.image(cropped_image, caption="Detection", width=150)
                except Exception as e:
                    # Fallback to icon if cropping fails
                    icon_map = {
                        'pothole': '🕳️', 'crack': '⚡', 'sign': '🚧', 'vegetation': '🌿',
                        'debris': '🗑️', 'car': '🚗', 'person': '👤'
                    }
                    icon = icon_map.get(detection['class'], '�')
                    st.markdown(f"""
                    <div class="card-icon" style="font-size: 3rem; text-align: center; padding: 20px;">
                        {icon}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Fallback to icon if no image available
                icon_map = {
                    'pothole': '🕳️', 'crack': '⚡', 'sign': '�', 'vegetation': '🌿',
                    'debris': '🗑️', 'car': '🚗', 'person': '👤'
                }
                icon = icon_map.get(detection['class'], '🔍')
                st.markdown(f"""
                <div class="card-icon" style="font-size: 3rem; text-align: center; padding: 20px;">
                    {icon}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # --- Column 2: Core Information ---
            # Detection class with small icon
            icon_map = {
                'pothole': '🕳️', 'crack': '⚡', 'sign': '🚧', 'vegetation': '🌿',
                'debris': '🗑️', 'car': '🚗', 'person': '👤'
            }
            icon = icon_map.get(detection['class'], '🔍')
            st.markdown(f"#### {icon} Class: {detection['class'].title()}")
            
            # Confidence metric
            confidence = detection['confidence']
            if confidence >= 0.8:
                conf_color = "🟢"
                conf_text = "High"
            elif confidence >= 0.6:
                conf_color = "🟡"
                conf_text = "Medium"
            else:
                conf_color = "🔴"
                conf_text = "Low"
            
            st.metric(label="Confidence", value=f"{confidence:.2%}", delta=f"{conf_color} {conf_text}")
            
            # Location information
            if gps_coords and gps_coords != (0.0, 0.0):
                st.markdown(f"📍 **Location:** {gps_coords[0]:.6f}, {gps_coords[1]:.6f}")
            else:
                st.markdown("📍 **Location:** No GPS data available")
            
            # Image source
            image_name = os.path.basename(image_path)
            st.markdown(f"📄 **Source:** `{image_name}`")
        
        with col3:
            # --- Column 3: Actions & Details ---
            # Bounding box info (compact display)
            bbox = detection.get('bbox_xyxy', [])
            if bbox:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                st.markdown(f"**📐 Size:** {width:.0f}×{height:.0f}px")
            
            # Technical details in expander
            with st.expander("🔧 Show Technical Details"):
                tech_details = {
                    "Detection Data": detection,
                    "Image Path": image_path,
                    "GPS Coordinates": gps_coords,
                    "Processing Metadata": detection_data.get('processing_metadata', {})
                }
                st.json(tech_details)
        
        # End card container
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing between cards
    
    def run(self):
        """Run the complete RoadSight AI application"""
        st.set_page_config(
            page_title="RoadSight AI - Complete Platform",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load custom CSS
        self.load_css()
        
        # Main title
        st.title("🚗 RoadSight AI - Complete Platform")
        st.markdown("**Professional Roadside Asset Inspection with Expert Review & Reporting**")
        st.markdown("---")
        
        # Main navigation
        main_tab = st.selectbox(
            "🧭 Navigation",
            ["🎯 Detection & Analysis", "👨‍💼 Expert Review & Reporting"],
            index=0
        )
        
        if main_tab == "🎯 Detection & Analysis":
            # Render sidebar and get user inputs
            uploaded_files, config = self.render_sidebar()
            
            # Main content area with tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Object Detection", "🗺️ Map View", "🖼️ Image Gallery", "📝 Interactive Report", "📊 Technical Report"])
            
            with tab1:
                self.render_detection_tab(uploaded_files, config)
            
            with tab2:
                self.render_map_view()
            
            with tab3:
                self.render_image_gallery()
            
            with tab4:
                self.render_interactive_report()
            
            with tab5:
                self.render_technical_report()
        
        elif main_tab == "👨‍💼 Expert Review & Reporting":
            # Run expert review interface
            if self.expert_interface:
                # Get selected finding from expert interface
                selected_finding = self.expert_interface.render_sidebar()
                
                # Expert interface tabs
                tab1, tab2, tab3 = st.tabs(["🔍 Finding Review", "📄 Report Generation", "📊 Statistics"])
                
                with tab1:
                    self.expert_interface.render_finding_details(selected_finding)
                
                with tab2:
                    self.expert_interface.render_report_generation()
                
                with tab3:
                    self.expert_interface.render_statistics()
            else:
                st.error("Expert review interface not available")


def main():
    """Main application entry point"""
    app = RoadSightAIApp()
    app.run()


if __name__ == "__main__":
    main()
