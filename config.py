#!/usr/bin/env python3
"""
Central Configuration File
Contains all hardcoded values, constants, and configuration settings.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

# Base paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"
LOGS_DIR = PROJECT_ROOT / "logs"

# Image directories
IMAGES_DIR = PROJECT_ROOT / "images"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

# Documentation and reference files
DOCS_DIR = PROJECT_ROOT / "YOLODOCS"
YOLO_CLASSES_FILE = DOCS_DIR / "YOLO8_ORGOBJS.json"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# YOLO Model Settings
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_MAX_DETECTIONS = 300

# Depth Estimation Settings
DEPTH_MODEL_TYPE = "MiDaS_small"
DEPTH_DEVICE = "cpu"  # "cuda" for GPU, "cpu" for CPU

# Model caching settings
ENABLE_MODEL_CACHING = True
MODEL_CACHE_TTL = 3600  # seconds

# =============================================================================
# API CONFIGURATION
# =============================================================================

# NVDB API Settings
NVDB_BASE_URL = "https://nvdbapiles-v3.atlas.vegvesen.no"
NVDB_API_VERSION = "v3"
NVDB_TIMEOUT = 30  # seconds
NVDB_MAX_RETRIES = 3
NVDB_RETRY_DELAY = 1  # seconds

# NVDB Matching Parameters
NVDB_SEARCH_RADIUS = 15  # meters
NVDB_MAX_RESULTS = 50

# =============================================================================
# IMAGE PROCESSING CONFIGURATION
# =============================================================================

# Image processing settings
MAX_IMAGE_SIZE = (1920, 1080)  # (width, height)
IMAGE_QUALITY = 85  # JPEG quality (1-100)
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Batch processing settings
BATCH_SIZE = 5
MAX_CONCURRENT_PROCESSES = 4

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

# Issue Detection Settings
ISSUE_IOU_THRESHOLD = 0.15
ISSUE_CONFIDENCE_THRESHOLD = 0.5

# Severity Mapping (confidence-based)
SEVERITY_THRESHOLDS = {
    'low': (0.0, 0.5),
    'medium': (0.5, 0.75),
    'high': (0.75, 0.9),
    'critical': (0.9, 1.0)
}

# Issue Types Configuration
ISSUE_TYPES = [
    'Pothole',
    'Alligator Crack', 
    'Longitudinal Crack',
    'Obscured Sign',
    'Vegetation Overgrowth',
    'Damaged Sign',
    'Missing Sign',
    'Poor Visibility',
    'Road Marking Issue',
    'Other'
]

SEVERITY_LEVELS = ['Low', 'Medium', 'High', 'Critical']

# YOLO Class to Issue Type Mapping
YOLO_TO_ISSUE_MAPPING = {
    'stop sign': 'Obscured Sign',
    'traffic light': 'Poor Visibility',
    'car': 'Other',
    'truck': 'Other',
    'bus': 'Other',
    'person': 'Other',
    'bicycle': 'Other',
    'potted plant': 'Vegetation Overgrowth',
    'tree': 'Vegetation Overgrowth',
    'flower': 'Vegetation Overgrowth'
}

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Streamlit App Settings
APP_TITLE = "RoadSight AI - Complete Platform"
APP_ICON = "🚗"
PAGE_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Server Settings
DEFAULT_PORT = 8505
DEBUG_MODE = False

# Image Display Settings
THUMBNAIL_SIZE = 150
GALLERY_IMAGES_PER_ROW = 3
MAP_DEFAULT_LOCATION = [62.0, 10.0]  # Norway center
MAP_DEFAULT_ZOOM = 6

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

# File Export Settings
EXPORT_FORMATS = ['csv', 'json', 'geojson', 'pdf']
DEFAULT_EXPORT_FORMAT = 'csv'

# CSV Export Columns
CSV_EXPORT_COLUMNS = [
    'ID', 'Image', 'Detection Class', 'Confidence',
    'Issue Type', 'Severity', 'Notes',
    'GPS Latitude', 'GPS Longitude', 'Timestamp'
]

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# SQLite Database Settings
DATABASE_PATH = PROJECT_ROOT / "findings.db"
DATABASE_TIMEOUT = 30  # seconds
DATABASE_BACKUP_ENABLED = True
DATABASE_BACKUP_INTERVAL = 3600  # seconds

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging Settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "roadsight.log"
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Console logging
ENABLE_CONSOLE_LOGGING = True
ENABLE_FILE_LOGGING = True

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Caching Settings
CACHE_ENABLED = True
CACHE_TTL = 3600  # seconds
MAX_CACHE_SIZE = 100  # items

# Memory Management
MAX_MEMORY_USAGE = 2048  # MB
GARBAGE_COLLECTION_THRESHOLD = 0.8  # 80% memory usage

# =============================================================================
# ERROR HANDLING CONFIGURATION
# =============================================================================

# Retry Settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
EXPONENTIAL_BACKOFF = True

# Timeout Settings
DEFAULT_TIMEOUT = 30  # seconds
MODEL_LOADING_TIMEOUT = 120  # seconds
API_REQUEST_TIMEOUT = 30  # seconds

# =============================================================================
# COORDINATE SYSTEM CONFIGURATION
# =============================================================================

# Norwegian Coordinate Systems
SUPPORTED_CRS = {
    'WGS84': 'EPSG:4326',
    'UTM33N': 'EPSG:25833',
    'UTM32N': 'EPSG:25832',
    'UTM35N': 'EPSG:25835'
}

DEFAULT_CRS = 'EPSG:4326'  # WGS84
NVDB_CRS = 'EPSG:25833'  # UTM Zone 33N

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Development flags
DEVELOPMENT_MODE = False
ENABLE_DEBUG_LOGGING = False
ENABLE_PROFILING = False
MOCK_API_RESPONSES = False

# Testing settings
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
ENABLE_UNIT_TESTS = True
COVERAGE_THRESHOLD = 80  # percentage

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create required directories if they don't exist"""
    directories = [
        DATA_DIR, MODELS_DIR, OUTPUT_DIR, TEMP_DIR, 
        LOGS_DIR, TEST_DATA_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_confidence_based_severity(confidence: float) -> str:
    """Get severity level based on confidence score"""
    for severity, (min_conf, max_conf) in SEVERITY_THRESHOLDS.items():
        if min_conf <= confidence < max_conf:
            return severity.capitalize()
    return 'Low'


def get_issue_type_from_yolo_class(yolo_class: str) -> str:
    """Map YOLO detection class to issue type"""
    return YOLO_TO_ISSUE_MAPPING.get(yolo_class.lower(), 'Other')


def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required files exist
    if not YOLO_CLASSES_FILE.exists():
        errors.append(f"YOLO classes file not found: {YOLO_CLASSES_FILE}")
    
    # Validate thresholds
    if not 0 <= YOLO_CONFIDENCE_THRESHOLD <= 1:
        errors.append("YOLO confidence threshold must be between 0 and 1")
    
    if not 0 <= YOLO_IOU_THRESHOLD <= 1:
        errors.append("YOLO IoU threshold must be between 0 and 1")
    
    # Validate severity thresholds
    for severity, (min_val, max_val) in SEVERITY_THRESHOLDS.items():
        if min_val >= max_val:
            errors.append(f"Invalid severity threshold for {severity}: {min_val} >= {max_val}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True


# Initialize directories on import
ensure_directories()

# Validate configuration on import
if not DEVELOPMENT_MODE:
    validate_config()
