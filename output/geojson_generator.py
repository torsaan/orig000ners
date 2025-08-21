#!/usr/bin/env python3
"""
GeoJSON Output Generator
Generates GeoJSON format output for GIS compatibility.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ImageProcessingResult:
    """Container for complete image processing results"""
    image_path: str
    gps_coords: Tuple[float, float]
    detections: List[Dict]
    issues: List[Dict]
    nvdb_matches: List[Dict]
    processing_metadata: Dict
    original_image_b64: Optional[str] = None
    annotated_image_b64: Optional[str] = None


class GeoJSONGenerator:
    """Generates GeoJSON output from processing results"""
    
    def __init__(self):
        """Initialize GeoJSON generator"""
        print("📄 GeoJSON Generator initialized")
    
    def generate_geojson(self, results: List[ImageProcessingResult]) -> Dict:
        """
        Generate GeoJSON FeatureCollection from processing results
        
        Args:
            results: List of image processing results
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        for result in results:
            # Create feature for each detection in the image
            for i, detection in enumerate(result.detections):
                # Find corresponding issue and NVDB match
                issue_data = self._find_related_issue(detection, result.issues)
                nvdb_data = self._find_related_nvdb_match(detection, result.nvdb_matches)
                
                feature = self._create_feature(
                    result.image_path,
                    result.gps_coords,
                    detection,
                    issue_data,
                    nvdb_data,
                    result.processing_metadata,
                    detection_index=i
                )
                
                features.append(feature)
        
        # Create GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_images": len(results),
                "total_features": len(features),
                "generator": "RoadsideAssetInspection v1.0"
            },
            "features": features
        }
        
        return geojson
    
    def _create_feature(self, image_path: str, gps_coords: Tuple[float, float],
                       detection: Dict, issue_data: Optional[Dict],
                       nvdb_data: Optional[Dict], metadata: Dict,
                       detection_index: int) -> Dict:
        """
        Create a GeoJSON feature for a single detection
        
        Args:
            image_path: Path to source image
            gps_coords: GPS coordinates (lat, lon)
            detection: Detection data
            issue_data: Related issue data
            nvdb_data: Related NVDB match data
            metadata: Processing metadata
            detection_index: Index of detection in image
            
        Returns:
            GeoJSON feature
        """
        lat, lon = gps_coords
        
        # Basic properties
        properties = {
            # Image information
            "image_id": os.path.basename(image_path),
            "image_path": image_path,
            "detection_index": detection_index,
            
            # Detection information
            "detected_class": detection["class"],
            "detection_confidence": detection["confidence"],
            "bounding_box": detection["box"],
            
            # GPS information
            "gps_latitude": lat,
            "gps_longitude": lon,
            "gps_accuracy": metadata.get("gps_accuracy", "unknown"),
            
            # Processing metadata
            "processed_at": metadata.get("processed_at", datetime.now().isoformat()),
            "processing_version": metadata.get("version", "1.0")
        }
        
        # Add issue information
        if issue_data:
            properties.update({
                "issue_detected": True,
                "issue_type": issue_data.get("issue_type", "unknown"),
                "issue_severity": issue_data.get("severity", "unknown"),
                "issue_confidence": issue_data.get("confidence", 0.0),
                "issue_details": issue_data.get("details", {}),
                "recommendations": issue_data.get("recommendations", [])
            })
        else:
            properties["issue_detected"] = False
        
        # Add NVDB information
        if nvdb_data:
            properties.update({
                "nvdb_matched": True,
                "nvdb_id": nvdb_data.get("nvdb_id"),
                "nvdb_status": nvdb_data.get("nvdb_status", "unknown"),
                "nvdb_confidence": nvdb_data.get("confidence", 0.0)
            })
            
            # Add NVDB object details if available
            if nvdb_data.get("nvdb_data"):
                properties["nvdb_object_data"] = nvdb_data["nvdb_data"]
        else:
            properties["nvdb_matched"] = False
        
        # Add depth information if available
        if "depth_proxy" in detection:
            properties["depth_proxy"] = detection["depth_proxy"]
        
        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]  # GeoJSON uses [lon, lat] order
            },
            "properties": properties
        }
        
        return feature
    
    def _find_related_issue(self, detection: Dict, issues: List[Dict]) -> Optional[Dict]:
        """
        Find issue data related to a detection
        
        Args:
            detection: Detection data
            issues: List of issue data
            
        Returns:
            Related issue data or None
        """
        detected_class = detection["class"]
        
        for issue in issues:
            if issue.get("affected_object") == detected_class:
                return issue
        
        return None
    
    def _find_related_nvdb_match(self, detection: Dict, nvdb_matches: List[Dict]) -> Optional[Dict]:
        """
        Find NVDB match data related to a detection
        
        Args:
            detection: Detection data
            nvdb_matches: List of NVDB match data
            
        Returns:
            Related NVDB match data or None
        """
        detected_class = detection["class"]
        
        for match in nvdb_matches:
            if match.get("detected_class") == detected_class:
                return match
        
        return None
    
    def save_geojson(self, geojson: Dict, output_path: str) -> bool:
        """
        Save GeoJSON to file
        
        Args:
            geojson: GeoJSON data
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2, ensure_ascii=False)
            
            print(f"✅ GeoJSON saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save GeoJSON: {e}")
            return False
    
    def generate_summary_report(self, geojson: Dict) -> Dict:
        """
        Generate summary report from GeoJSON data
        
        Args:
            geojson: GeoJSON FeatureCollection
            
        Returns:
            Summary report
        """
        features = geojson.get("features", [])
        
        # Initialize counters
        class_counts = {}
        issue_counts = {}
        nvdb_match_counts = {"matched": 0, "not_matched": 0}
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        
        for feature in features:
            props = feature["properties"]
            
            # Count detected classes
            detected_class = props.get("detected_class", "unknown")
            class_counts[detected_class] = class_counts.get(detected_class, 0) + 1
            
            # Count issues
            if props.get("issue_detected", False):
                issue_type = props.get("issue_type", "unknown")
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                
                severity = props.get("issue_severity", "low")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count NVDB matches
            if props.get("nvdb_matched", False):
                nvdb_match_counts["matched"] += 1
            else:
                nvdb_match_counts["not_matched"] += 1
        
        report = {
            "total_detections": len(features),
            "detected_classes": class_counts,
            "issues": {
                "total_issues": sum(issue_counts.values()),
                "by_type": issue_counts,
                "by_severity": severity_counts
            },
            "nvdb_matching": nvdb_match_counts,
            "images_processed": geojson.get("metadata", {}).get("total_images", 0)
        }
        
        return report
    
    def create_3d_feature(self, gps_coords: Tuple[float, float], 
                         point_cloud_data: Dict, 
                         detection_data: Dict) -> Dict:
        """
        Create GeoJSON feature for 3D point cloud data (future use)
        
        Args:
            gps_coords: GPS coordinates
            point_cloud_data: 3D point cloud information
            detection_data: Detection results
            
        Returns:
            GeoJSON feature with 3D information
        """
        lat, lon = gps_coords
        
        # For 3D data, we can use MultiPoint or Polygon geometries
        # This is a placeholder for future LIDAR integration
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat, point_cloud_data.get("elevation", 0)]
            },
            "properties": {
                "data_source": "3d_pointcloud",
                "point_count": point_cloud_data.get("point_count", 0),
                "density": point_cloud_data.get("density", 0),
                "detected_class": detection_data.get("class", "unknown"),
                "3d_bbox": point_cloud_data.get("bbox_3d", {}),
                "clearance_analysis": point_cloud_data.get("clearance", {})
            }
        }
        
        return feature


def main():
    """Test GeoJSON generator"""
    print("🔍 Testing GeoJSON Generator...")
    
    # Create generator
    generator = GeoJSONGenerator()
    
    # Create sample processing results
    sample_results = [
        ImageProcessingResult(
            image_path="images/test_image_1.jpg",
            gps_coords=(59.9139, 10.7522),  # Oslo
            detections=[
                {
                    "class": "speed_sign_80",
                    "confidence": 0.9,
                    "box": {"center_x": 640, "center_y": 360, "width": 100, "height": 100}
                },
                {
                    "class": "vegetation",
                    "confidence": 0.85,
                    "box": {"center_x": 660, "center_y": 380, "width": 80, "height": 120}
                }
            ],
            issues=[
                {
                    "issue_type": "vegetation_obscuration",
                    "affected_object": "speed_sign_80",
                    "severity": "medium",
                    "confidence": 0.8,
                    "details": {"iou": 0.25},
                    "recommendations": ["Trim vegetation", "Monitor growth"]
                }
            ],
            nvdb_matches=[
                {
                    "detected_class": "speed_sign_80",
                    "nvdb_id": "12345",
                    "nvdb_status": "match",
                    "confidence": 0.9
                }
            ],
            processing_metadata={
                "processed_at": datetime.now().isoformat(),
                "version": "1.0",
                "gps_accuracy": "3-5m"
            }
        )
    ]
    
    # Generate GeoJSON
    geojson = generator.generate_geojson(sample_results)
    
    # Display results
    print(f"📊 Generated GeoJSON with {len(geojson['features'])} features")
    
    # Save to file
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_results.geojson")
    
    success = generator.save_geojson(geojson, output_path)
    if success:
        print(f"✅ Test GeoJSON saved to: {output_path}")
    
    # Generate summary report
    report = generator.generate_summary_report(geojson)
    print(f"\n📈 Summary Report:")
    print(f"  Total detections: {report['total_detections']}")
    print(f"  Issues found: {report['issues']['total_issues']}")
    print(f"  NVDB matches: {report['nvdb_matching']['matched']}")


if __name__ == "__main__":
    main()
