#!/usr/bin/env python3
"""
Utils Package
Centralized utility functions for the RoadSight AI project.
"""

from .error_handling import (
    RoadSightError, ModelLoadingError, ImageProcessingError, 
    APIError, ConfigurationError, setup_logging, retry_on_exception,
    safe_execute, validate_input, ErrorContext, handle_streamlit_errors,
    log_performance, logger
)

from .performance import (
    MemoryManager, SimpleCache, cache_result, load_model_cached,
    cache_api_response, BatchProcessor, optimize_dataframe_operations,
    ProgressTracker, clear_cache, memory_efficient_image_processing
)

# Note: ImageProcessor import commented out due to dependency issues
# Uncomment when PIL, cv2, numpy are properly installed
# from .image_utils import ImageProcessor

__all__ = [
    # Error handling
    'RoadSightError', 'ModelLoadingError', 'ImageProcessingError',
    'APIError', 'ConfigurationError', 'setup_logging', 'retry_on_exception',
    'safe_execute', 'validate_input', 'ErrorContext', 'handle_streamlit_errors',
    'log_performance', 'logger',
    
    # Performance
    'MemoryManager', 'SimpleCache', 'cache_result', 'load_model_cached',
    'cache_api_response', 'BatchProcessor', 'optimize_dataframe_operations',
    'ProgressTracker', 'clear_cache', 'memory_efficient_image_processing',
    
    # Image processing (when available)
    # 'ImageProcessor'
]
