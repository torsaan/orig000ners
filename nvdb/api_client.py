#!/usr/bin/env python3
"""
NVDB API Client
Handles communication with the Norwegian National Road Database API.
"""

import json
import requests
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class NVDBResponse:
    """Container for NVDB API response"""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class NVDBClient:
    """Client for NVDB API operations"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize NVDB client
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.base_url = self.config["nvdb_api"]["base_url"]
        self.fallback_url = self.config["nvdb_api"]["fallback_url"]
        self.timeout = self.config["nvdb_api"]["timeout"]
        
        # Setup headers (simplified for V3 API)
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "DemoAssetInspection/1.0"
        }
        
        # Cache for responses
        self.cache = {}
        self.use_cache = self.config["nvdb_api"]["cache_responses"]
        
        print(f"🔗 NVDB Client initialized: {self.base_url}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            # Return default config
            return {
                "nvdb_api": {
                    "base_url": "https://nvdbapiles-v3.atlas.vegvesen.no",
                    "fallback_url": "https://nvdbapiles-v4.atlas.vegvesen.no",
                    "client_email": "demo.asset.inspection@test.com",
                    "user_agent": "DemoAssetInspection/1.0",
                    "timeout": 15,
                    "cache_responses": True
                }
            }
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> NVDBResponse:
        """
        Make HTTP request to NVDB API
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            NVDBResponse object
        """
        # Check cache first
        cache_key = f"{endpoint}_{params}"
        if self.use_cache and cache_key in self.cache:
            print(f"📋 Using cached response for {endpoint}")
            return self.cache[cache_key]
        
        # Try primary URL first
        urls_to_try = [self.base_url, self.fallback_url]
        
        for url in urls_to_try:
            try:
                full_url = f"{url}{endpoint}"
                print(f"🌐 Requesting: {full_url}")
                
                response = requests.get(
                    full_url,
                    headers=self.headers,
                    params=params,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = NVDBResponse(
                        success=True,
                        data=response.json(),
                        status_code=response.status_code
                    )
                    
                    # Cache successful response
                    if self.use_cache:
                        self.cache[cache_key] = result
                    
                    return result
                    
                elif response.status_code == 404:
                    # Not found - try next URL or return error
                    continue
                else:
                    print(f"⚠️  HTTP {response.status_code}: {response.text[:200]}")
                    
            except requests.RequestException as e:
                print(f"❌ Request failed for {url}: {e}")
                continue
        
        # All URLs failed
        return NVDBResponse(
            success=False,
            error="All API endpoints failed",
            status_code=None
        )
    
    def test_connectivity(self) -> NVDBResponse:
        """Test NVDB API connectivity"""
        print("🔍 Testing NVDB connectivity...")
        return self._make_request("/status")
    
    def get_feature_types(self) -> NVDBResponse:
        """Get available feature types from NVDB"""
        print("🔍 Fetching NVDB feature types...")
        return self._make_request("/vegobjekttyper")
    
    def get_objects_by_bbox(self, feature_id: int, bbox: str, 
                           include_attrs: bool = True) -> NVDBResponse:
        """
        Get objects within a bounding box
        
        Args:
            feature_id: NVDB feature type ID
            bbox: Bounding box as "xmin,ymin,xmax,ymax"
            include_attrs: Include object attributes
            
        Returns:
            NVDBResponse with objects
        """
        params = {
            "kartutsnitt": bbox,
            "srid": 25833  # UTM Zone 33N (ETRS89) - most common for Norway
        }
        
        if include_attrs:
            params["inkluder"] = "alle"
        
        endpoint = f"/vegobjekter/{feature_id}"
        return self._make_request(endpoint, params)
    
    def get_objects_near_point(self, feature_id: int, lat: float, lon: float, 
                              radius: float = 15.0) -> NVDBResponse:
        """
        Get objects near a GPS point
        
        Args:
            feature_id: NVDB feature type ID
            lat: Latitude in WGS84
            lon: Longitude in WGS84
            radius: Search radius in meters
            
        Returns:
            NVDBResponse with objects
        """
        from pyproj import Transformer
        
        try:
            # Transform to UTM (EPSG:25833 for most of Norway)
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
            x, y = transformer.transform(lon, lat)
            
            # Create bounding box
            bbox = f"{x-radius},{y-radius},{x+radius},{y+radius}"
            
            print(f"🗺️  Searching around ({lat:.6f}, {lon:.6f}) -> UTM({x:.1f}, {y:.1f}), radius={radius}m")
            
            return self.get_objects_by_bbox(feature_id, bbox)
            
        except Exception as e:
            return NVDBResponse(
                success=False,
                error=f"Coordinate transformation failed: {e}"
            )
    
    def search_sign_types(self) -> List[Dict]:
        """
        Search for sign-related feature types
        
        Returns:
            List of sign feature types
        """
        response = self.get_feature_types()
        
        if not response.success:
            print("❌ Failed to get feature types")
            return []
        
        # Filter for sign-related types
        sign_keywords = ["skilt", "signal", "sign", "merke", "fartsgrense", "stopp"]
        sign_types = []
        
        if "objekter" in response.data:
            for feature_type in response.data["objekter"]:
                name = feature_type.get("navn", "").lower()
                if any(keyword in name for keyword in sign_keywords):
                    sign_types.append(feature_type)
        
        print(f"🔍 Found {len(sign_types)} sign-related feature types")
        return sign_types
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cached_requests": len(self.cache),
            "cache_enabled": self.use_cache
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
        print("🗑️  NVDB cache cleared")


def main():
    """Test NVDB client"""
    print("🔍 Testing NVDB Client...")
    
    # Initialize client
    client = NVDBClient()
    
    # Test connectivity
    conn_result = client.test_connectivity()
    if conn_result.success:
        print("✅ NVDB connectivity OK")
    else:
        print(f"❌ NVDB connectivity failed: {conn_result.error}")
    
    # Test feature types
    types_result = client.get_feature_types()
    if types_result.success:
        print(f"✅ Retrieved {len(types_result.data.get('objekter', []))} feature types")
    
    # Search for signs
    sign_types = client.search_sign_types()
    for sign_type in sign_types[:5]:  # Show first 5
        print(f"  📋 {sign_type.get('id', 'N/A')}: {sign_type.get('navn', 'N/A')}")
    
    # Test geospatial query (Oslo center)
    oslo_lat, oslo_lon = 59.9139, 10.7522
    objects_result = client.get_objects_near_point(105, oslo_lat, oslo_lon)  # Feature ID 105
    
    if objects_result.success:
        object_count = len(objects_result.data.get("objekter", []))
        print(f"✅ Found {object_count} objects near Oslo center")
    else:
        print(f"❌ Geospatial query failed: {objects_result.error}")
    
    # Cache stats
    cache_stats = client.get_cache_stats()
    print(f"📋 Cache: {cache_stats['cached_requests']} requests cached")


if __name__ == "__main__":
    main()
