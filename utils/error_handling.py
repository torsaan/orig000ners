#!/usr/bin/env python3
"""
Error Handling Utilities
Centralized error handling, logging, and retry mechanisms.
"""

import logging
import time
import functools
from typing import Any, Callable, Optional, Dict, Type
import traceback
import sys
from pathlib import Path

from config import (
    DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY, EXPONENTIAL_BACKOFF,
    LOG_LEVEL, LOG_FORMAT, LOG_FILE, ENABLE_CONSOLE_LOGGING, ENABLE_FILE_LOGGING
)


class RoadSightError(Exception):
    """Base exception class for RoadSight application"""
    pass


class ModelLoadingError(RoadSightError):
    """Raised when model loading fails"""
    pass


class ImageProcessingError(RoadSightError):
    """Raised when image processing fails"""
    pass


class APIError(RoadSightError):
    """Raised when API calls fail"""
    pass


class ConfigurationError(RoadSightError):
    """Raised when configuration is invalid"""
    pass


def setup_logging() -> logging.Logger:
    """
    Setup centralized logging configuration
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('roadsight')
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    if ENABLE_CONSOLE_LOGGING:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if ENABLE_FILE_LOGGING:
        try:
            # Ensure log directory exists
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(LOG_FILE)
            file_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    return logger


def retry_on_exception(max_retries: int = DEFAULT_MAX_RETRIES,
                      delay: float = DEFAULT_RETRY_DELAY,
                      exponential_backoff: bool = EXPONENTIAL_BACKOFF,
                      exceptions: tuple = (Exception,)):
    """
    Decorator for retrying functions on exceptions
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        exponential_backoff: Whether to use exponential backoff
        exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger('roadsight')
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    current_delay = delay * (2 ** attempt) if exponential_backoff else delay
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {current_delay}s: {e}")
                    time.sleep(current_delay)
            
        return wrapper
    return decorator


def safe_execute(func: Callable, 
                default_return: Any = None,
                log_errors: bool = True,
                error_message: str = None) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        default_return: Value to return on error
        log_errors: Whether to log errors
        error_message: Custom error message
        
    Returns:
        Function result or default_return on error
    """
    logger = logging.getLogger('roadsight')
    
    try:
        return func()
    except Exception as e:
        if log_errors:
            message = error_message or f"Error executing {func.__name__}"
            logger.error(f"{message}: {e}")
            if logger.level <= logging.DEBUG:
                logger.debug(traceback.format_exc())
        
        return default_return


def validate_input(value: Any, 
                  expected_type: Type,
                  min_value: Any = None,
                  max_value: Any = None,
                  allowed_values: list = None,
                  required: bool = True) -> bool:
    """
    Validate input parameters
    
    Args:
        value: Value to validate
        expected_type: Expected type
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allowed_values: List of allowed values
        required: Whether value is required
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        if required:
            raise ValueError("Required value is None")
        return True
    
    # Type check
    if not isinstance(value, expected_type):
        raise ValueError(f"Expected {expected_type.__name__}, got {type(value).__name__}")
    
    # Range checks
    if min_value is not None and value < min_value:
        raise ValueError(f"Value {value} is below minimum {min_value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"Value {value} is above maximum {max_value}")
    
    # Allowed values check
    if allowed_values is not None and value not in allowed_values:
        raise ValueError(f"Value {value} not in allowed values: {allowed_values}")
    
    return True


class ErrorContext:
    """Context manager for error handling and logging"""
    
    def __init__(self, operation_name: str, 
                 logger: logging.Logger = None,
                 reraise: bool = True,
                 default_return: Any = None):
        """
        Initialize error context
        
        Args:
            operation_name: Name of operation for logging
            logger: Logger instance
            reraise: Whether to reraise exceptions
            default_return: Default return value on error
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger('roadsight')
        self.reraise = reraise
        self.default_return = default_return
        self.exception = None
    
    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            self.exception = exc_value
            self.logger.error(f"Operation failed: {self.operation_name} - {exc_value}")
            
            if self.logger.level <= logging.DEBUG:
                self.logger.debug(traceback.format_exc())
            
            if not self.reraise:
                return True  # Suppress exception
        else:
            self.logger.debug(f"Operation completed: {self.operation_name}")
        
        return False


def handle_streamlit_errors(func: Callable) -> Callable:
    """
    Decorator for handling errors in Streamlit applications
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import streamlit as st
            
            logger = logging.getLogger('roadsight')
            logger.error(f"Streamlit error in {func.__name__}: {e}")
            
            # Show user-friendly error message
            if isinstance(e, ModelLoadingError):
                st.error("🔴 Model loading failed. Please check your model files and try again.")
            elif isinstance(e, ImageProcessingError):
                st.error("🔴 Image processing failed. Please check your image files and try again.")
            elif isinstance(e, APIError):
                st.error("🔴 API request failed. Please check your internet connection and try again.")
            elif isinstance(e, ConfigurationError):
                st.error("🔴 Configuration error. Please check your settings.")
            else:
                st.error(f"🔴 An unexpected error occurred: {str(e)}")
            
            # Show detailed error in debug mode
            if logger.level <= logging.DEBUG:
                st.exception(e)
            
            return None
    
    return wrapper


def log_performance(func: Callable) -> Callable:
    """
    Decorator for logging function performance
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('roadsight')
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Function {func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


# Initialize logging on module import
logger = setup_logging()
