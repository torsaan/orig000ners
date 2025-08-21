#!/usr/bin/env python3
"""
Performance Optimization Utilities
Caching, batch processing, and memory management utilities.
"""

import streamlit as st
import time
import functools
import gc
import psutil
import os
from typing import Any, Dict, List, Callable, Optional
import logging
from threading import Lock
import json
import hashlib

from config import (
    CACHE_ENABLED, CACHE_TTL, MAX_CACHE_SIZE,
    MAX_MEMORY_USAGE, GARBAGE_COLLECTION_THRESHOLD,
    BATCH_SIZE, MAX_CONCURRENT_PROCESSES
)

logger = logging.getLogger('roadsight')


class MemoryManager:
    """Memory management utilities"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """
        Get current memory usage percentage
        
        Returns:
            float: Memory usage percentage (0-100)
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            return memory_mb
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0
    
    @staticmethod
    def should_garbage_collect() -> bool:
        """
        Check if garbage collection should be triggered
        
        Returns:
            bool: True if GC should be triggered
        """
        memory_usage = MemoryManager.get_memory_usage()
        return memory_usage > (MAX_MEMORY_USAGE * GARBAGE_COLLECTION_THRESHOLD)
    
    @staticmethod
    def cleanup_memory():
        """Force garbage collection and memory cleanup"""
        try:
            gc.collect()
            logger.debug("Performed garbage collection")
        except Exception as e:
            logger.warning(f"Error during garbage collection: {e}")


class SimpleCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, max_size: int = MAX_CACHE_SIZE, default_ttl: int = CACHE_TTL):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = Lock()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else None
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            item = self._cache[key]
            if time.time() > item['expires']:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            self._access_times[key] = time.time()
            return item['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set item in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            # Remove expired items
            self._cleanup_expired()
            
            # Remove LRU items if at capacity
            if len(self._cache) >= self.max_size:
                self._remove_lru()
            
            expires = time.time() + ttl
            self._cache[key] = {'value': value, 'expires': expires}
            self._access_times[key] = time.time()
    
    def _cleanup_expired(self):
        """Remove expired items from cache"""
        current_time = time.time()
        expired_keys = [
            key for key, item in self._cache.items()
            if current_time > item['expires']
        ]
        
        for key in expired_keys:
            del self._cache[key]
            del self._access_times[key]
    
    def _remove_lru(self):
        """Remove least recently used item"""
        if self._access_times:
            lru_key = min(self._access_times, key=self._access_times.get)
            del self._cache[lru_key]
            del self._access_times[lru_key]
    
    def clear(self):
        """Clear all cached items"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


# Global cache instance
_cache = SimpleCache() if CACHE_ENABLED else None


def cache_result(ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time-to-live in seconds
        key_func: Function to generate custom cache key
    """
    def decorator(func: Callable) -> Callable:
        if not CACHE_ENABLED or _cache is None:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _cache._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


@st.cache_resource
def load_model_cached(model_path: str, model_type: str = "yolo"):
    """
    Cache model loading using Streamlit's caching
    
    Args:
        model_path: Path to model file
        model_type: Type of model ('yolo', 'depth')
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading {model_type} model: {model_path}")
    
    if model_type == "yolo":
        from ultralytics import YOLO
        model = YOLO(model_path)
        logger.info(f"✅ Loaded YOLO model: {model_path}")
        return model
    elif model_type == "depth":
        import torch
        model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        model.eval()
        logger.info(f"✅ Loaded depth model: {model_path}")
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@st.cache_data(ttl=CACHE_TTL)
def cache_api_response(url: str, params: dict = None):
    """
    Cache API responses using Streamlit's caching
    
    Args:
        url: API URL
        params: Request parameters
        
    Returns:
        API response data
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    # Setup retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    return response.json()


class BatchProcessor:
    """Efficient batch processing utilities"""
    
    def __init__(self, batch_size: int = BATCH_SIZE):
        """
        Initialize batch processor
        
        Args:
            batch_size: Size of each batch
        """
        self.batch_size = batch_size
    
    def process_in_batches(self, 
                          items: List[Any], 
                          process_func: Callable,
                          progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        Process items in batches with optional progress callback
        
        Args:
            items: List of items to process
            process_func: Function to apply to each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processed results
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for i, batch_start in enumerate(range(0, len(items), self.batch_size)):
            batch_end = min(batch_start + self.batch_size, len(items))
            batch = items[batch_start:batch_end]
            
            try:
                # Process batch
                batch_results = process_func(batch)
                results.extend(batch_results)
                
                # Memory management
                if MemoryManager.should_garbage_collect():
                    MemoryManager.cleanup_memory()
                
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / total_batches
                    progress_callback(progress, i + 1, total_batches)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i + 1}: {e}")
                # Continue with next batch
                continue
        
        return results


def optimize_dataframe_operations(df):
    """
    Optimize pandas DataFrame operations
    
    Args:
        df: Input DataFrame
        
    Returns:
        Optimized DataFrame
    """
    try:
        # Optimize data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to categorical if few unique values
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'int64':
                # Downcast integers
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'float64':
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        logger.debug(f"Optimized DataFrame memory usage")
        return df
        
    except Exception as e:
        logger.warning(f"DataFrame optimization failed: {e}")
        return df


class ProgressTracker:
    """Progress tracking for long-running operations"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker
        
        Args:
            total_steps: Total number of steps
            description: Description of operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.progress_bar = None
        self.status_text = None
    
    def __enter__(self):
        """Setup progress display"""
        try:
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
            self.update(0, "Starting...")
        except:
            pass  # Fallback if Streamlit not available
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup progress display"""
        try:
            if self.progress_bar:
                self.progress_bar.progress(1.0)
            if self.status_text:
                elapsed = time.time() - self.start_time
                self.status_text.success(f"✅ {self.description} completed in {elapsed:.2f}s")
        except:
            pass
    
    def update(self, step: int, message: str = ""):
        """
        Update progress
        
        Args:
            step: Current step number
            message: Status message
        """
        self.current_step = step
        progress = min(step / self.total_steps, 1.0)
        
        try:
            if self.progress_bar:
                self.progress_bar.progress(progress)
            
            if self.status_text:
                elapsed = time.time() - self.start_time
                if step > 0:
                    eta = (elapsed / step) * (self.total_steps - step)
                    status = f"{self.description}: {step}/{self.total_steps} - {message} (ETA: {eta:.1f}s)"
                else:
                    status = f"{self.description}: {message}"
                self.status_text.text(status)
                
        except:
            # Fallback to console logging
            logger.info(f"Progress: {step}/{self.total_steps} - {message}")


def clear_cache():
    """Clear all caches"""
    try:
        if _cache:
            _cache.clear()
        
        # Clear Streamlit caches
        st.cache_data.clear()
        st.cache_resource.clear()
        
        logger.info("All caches cleared")
    except Exception as e:
        logger.warning(f"Error clearing caches: {e}")


def memory_efficient_image_processing(images: List, process_func: Callable) -> List:
    """
    Process images in a memory-efficient manner
    
    Args:
        images: List of images or image paths
        process_func: Processing function
        
    Returns:
        List of processed results
    """
    results = []
    
    for i, image in enumerate(images):
        try:
            # Process single image
            result = process_func(image)
            results.append(result)
            
            # Memory cleanup every few images
            if (i + 1) % 5 == 0:
                if MemoryManager.should_garbage_collect():
                    MemoryManager.cleanup_memory()
                    
        except Exception as e:
            logger.error(f"Error processing image {i}: {e}")
            results.append(None)
    
    return results
