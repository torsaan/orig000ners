#!/usr/bin/env python3
"""
NVDB Matcher
Matches detected objects with NVDB database records.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from nvdb.api_client import NVDBClient


@dataclass
class MatchResult:
    """Result of NVDB matching"""
    detected_class: str
    nvdb_id: Optional[str] = None
    nvdb_status: str = "no_match"  # match, no_match, multiple_matches
    nvdb_data: Optional[Dict] = None
    confidence: float = 0.0
    distance: Optional[float] = None  # Distance to matched object


class NVDBMatcher:
    """Matches detected objects with NVDB records"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize NVDB matcher
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.sign_mapping = self._load_sign_mapping()
        self.client = NVDBClient(config_path)
        self.search_radius = self.config.get("nvdb_search_radius", 15)
        
        print(f"🔗 NVDB Matcher initialized (radius: {self.search_radius}m)")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return {"nvdb_search_radius": 15}
    
    def _load_sign_mapping(self) -> Dict:
        """Load sign mapping from JSON file"""
        try:
            with open("config/sign_mapping.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load sign mapping: {e}")
            # Return default mapping
            return {
                "stop_sign": {"nvdb_feature_id": 96},
                "speed_sign_30": {"nvdb_feature_id": 105, "speed_value": 30},
                "speed_sign_50": {"nvdb_feature_id": 105, "speed_value": 50},
                "speed_sign_80": {"nvdb_feature_id": 105, "speed_value": 80},
                "warning_sign": {"nvdb_feature_id": 107},
                "regulatory_sign": {"nvdb_feature_id": 109},
                "vegetation": {"nvdb_feature_id": None}
            }
    
    def match_detections(self, detections: List[Dict], gps_coords: Tuple[float, float]) -> List[MatchResult]:
        """
        Match detected objects with NVDB records
        
        Args:
            detections: List of detected objects
            gps_coords: GPS coordinates (lat, lon)
            
        Returns:
            List of match results
        """
        lat, lon = gps_coords
        results = []
        
        print(f"🔍 Matching {len(detections)} detections at ({lat:.6f}, {lon:.6f})")
        
        for detection in detections:
            class_name = detection["class"]
            
            # Skip vegetation - no NVDB matching
            if class_name == "vegetation":
                results.append(MatchResult(
                    detected_class=class_name,
                    nvdb_status="no_nvdb_mapping"
                ))
                continue
            
            # Get NVDB feature ID for this class
            feature_info = self.sign_mapping.get(class_name)
            if not feature_info or not feature_info.get("nvdb_feature_id"):
                results.append(MatchResult(
                    detected_class=class_name,
                    nvdb_status="no_feature_mapping"
                ))
                continue
            
            feature_id = feature_info["nvdb_feature_id"]
            
            # Query NVDB for objects near this location
            nvdb_response = self.client.get_objects_near_point(
                feature_id, lat, lon, self.search_radius
            )
            
            if not nvdb_response.success:
                results.append(MatchResult(
                    detected_class=class_name,
                    nvdb_status="api_error"
                ))
                continue
            
            # Process NVDB objects
            nvdb_objects = nvdb_response.data.get("objekter", []) if nvdb_response.data else []
            
            if not nvdb_objects:
                results.append(MatchResult(
                    detected_class=class_name,
                    nvdb_status="no_match"
                ))
                continue
            
            # Find best match
            best_match = self._find_best_match(detection, nvdb_objects, feature_info)
            results.append(best_match)
        
        return results
    
    def _find_best_match(self, detection: Dict, nvdb_objects: List[Dict], 
                        feature_info: Dict) -> MatchResult:
        """
        Find the best matching NVDB object
        
        Args:
            detection: Detected object
            nvdb_objects: List of NVDB objects
            feature_info: Feature information from sign mapping
            
        Returns:
            Best match result
        """
        class_name = detection["class"]
        
        if len(nvdb_objects) == 1:
            # Single match - validate if it's correct
            nvdb_obj = nvdb_objects[0]
            if self._validate_match(detection, nvdb_obj, feature_info):
                return MatchResult(
                    detected_class=class_name,
                    nvdb_id=str(nvdb_obj.get("id", "")),
                    nvdb_status="match",
                    nvdb_data=nvdb_obj,
                    confidence=1.0
                )
            else:
                return MatchResult(
                    detected_class=class_name,
                    nvdb_status="validation_failed"
                )
        
        elif len(nvdb_objects) > 1:
            # Multiple objects - need to find best match
            print(f"🔍 Found {len(nvdb_objects)} potential matches for {class_name}")
            
            valid_matches = []
            for nvdb_obj in nvdb_objects:
                if self._validate_match(detection, nvdb_obj, feature_info):
                    valid_matches.append(nvdb_obj)
            
            if len(valid_matches) == 1:
                return MatchResult(
                    detected_class=class_name,
                    nvdb_id=str(valid_matches[0].get("id", "")),
                    nvdb_status="match",
                    nvdb_data=valid_matches[0],
                    confidence=0.8  # Lower confidence due to multiple candidates
                )
            elif len(valid_matches) > 1:
                return MatchResult(
                    detected_class=class_name,
                    nvdb_status="multiple_matches",
                    confidence=0.5
                )
            else:
                return MatchResult(
                    detected_class=class_name,
                    nvdb_status="no_valid_match"
                )
        
        return MatchResult(
            detected_class=class_name,
            nvdb_status="no_match"
        )
    
    def _validate_match(self, detection: Dict, nvdb_obj: Dict, feature_info: Dict) -> bool:
        """
        Validate if a detected object matches an NVDB object
        
        Args:
            detection: Detected object
            nvdb_obj: NVDB object
            feature_info: Feature information from sign mapping
            
        Returns:
            True if match is valid
        """
        class_name = detection["class"]
        
        # For speed signs, check the speed value
        if "speed_sign" in class_name:
            expected_speed = feature_info.get("speed_value")
            if expected_speed:
                # Look for speed value in NVDB object attributes
                egenskaper = nvdb_obj.get("egenskaper", [])
                for egenskap in egenskaper:
                    if "fartsgrense" in egenskap.get("navn", "").lower():
                        try:
                            nvdb_speed = int(egenskap.get("verdi", 0))
                            if nvdb_speed == expected_speed:
                                return True
                        except (ValueError, TypeError):
                            continue
                return False
        
        # For other sign types, assume match if feature ID matches
        return True
    
    def get_match_statistics(self, results: List[MatchResult]) -> Dict[str, int]:
        """
        Get statistics of matching results
        
        Args:
            results: List of match results
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_detections": len(results),
            "successful_matches": 0,
            "no_matches": 0,
            "multiple_matches": 0,
            "api_errors": 0,
            "no_mapping": 0
        }
        
        for result in results:
            if result.nvdb_status == "match":
                stats["successful_matches"] += 1
            elif result.nvdb_status == "no_match":
                stats["no_matches"] += 1
            elif result.nvdb_status == "multiple_matches":
                stats["multiple_matches"] += 1
            elif result.nvdb_status == "api_error":
                stats["api_errors"] += 1
            elif result.nvdb_status in ["no_nvdb_mapping", "no_feature_mapping"]:
                stats["no_mapping"] += 1
        
        return stats


def main():
    """Test NVDB matcher"""
    print("🔍 Testing NVDB Matcher...")
    
    # Initialize matcher
    matcher = NVDBMatcher()
    
    # Create sample detections
    sample_detections = [
        {
            "class": "speed_sign_80",
            "confidence": 0.9,
            "box": {"center_x": 640, "center_y": 360, "width": 100, "height": 100}
        },
        {
            "class": "stop_sign", 
            "confidence": 0.85,
            "box": {"center_x": 400, "center_y": 300, "width": 80, "height": 80}
        },
        {
            "class": "vegetation",
            "confidence": 0.75,
            "box": {"center_x": 200, "center_y": 400, "width": 150, "height": 200}
        }
    ]
    
    # Test matching (Oslo center)
    oslo_coords = (59.9139, 10.7522)
    results = matcher.match_detections(sample_detections, oslo_coords)
    
    # Display results
    print(f"\n📊 Matching Results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.detected_class} -> {result.nvdb_status}")
        if result.nvdb_id:
            print(f"     NVDB ID: {result.nvdb_id} (confidence: {result.confidence:.2f})")
    
    # Statistics
    stats = matcher.get_match_statistics(results)
    print(f"\n📈 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
