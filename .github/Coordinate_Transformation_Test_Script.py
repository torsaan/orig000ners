#!/usr/bin/env python3
"""
Coordinate Transformation Test Script
Tests coordinate system transformations for Norwegian GPS coordinates to NVDB-compatible formats.
Supports multiple UTM zones for future international expansion.
"""

import math
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CoordinateTestResult:
    """Container for coordinate transformation test results"""
    input_wgs84: Tuple[float, float]
    output_utm: Tuple[float, float]
    utm_zone: int
    utm_hemisphere: str
    epsg_code: int
    success: bool
    error: Optional[str] = None


class CoordinateTransformTester:
    """Test coordinate transformations for NVDB integration"""
    
    def __init__(self):
        # Test locations across Norway to determine UTM zones
        self.test_locations = [
            {"name": "Oslo", "lat": 59.9139, "lon": 10.7522, "expected_utm": 33},
            {"name": "Bergen", "lat": 60.3913, "lon": 5.3221, "expected_utm": 32},
            {"name": "Stavanger", "lat": 58.9700, "lon": 5.7331, "expected_utm": 32},
            {"name": "Trondheim", "lat": 63.4305, "lon": 10.3951, "expected_utm": 33},
            {"name": "Tromsø", "lat": 69.6489, "lon": 18.9551, "expected_utm": 33},
            {"name": "Kristiansand", "lat": 58.1467, "lon": 7.9956, "expected_utm": 32},
            {"name": "Bodø", "lat": 67.2804, "lon": 14.4059, "expected_utm": 33},
        ]
        
        # EPSG codes for Norwegian UTM zones
        self.norwegian_utm_zones = {
            32: 25832,  # EPSG:25832 - ETRS89 / UTM zone 32N
            33: 25833,  # EPSG:25833 - ETRS89 / UTM zone 33N
            35: 25835,  # EPSG:25835 - ETRS89 / UTM zone 35N (far north)
        }

    def determine_utm_zone(self, longitude: float) -> int:
        """
        Determine UTM zone from longitude
        UTM zones are 6 degrees wide, starting from -180°
        """
        return int((longitude + 180) / 6) + 1

    def wgs84_to_utm_manual(self, lat: float, lon: float) -> CoordinateTestResult:
        """
        Manual implementation of WGS84 to UTM conversion
        This is a simplified version for testing purposes
        For production, use pyproj library
        """
        try:
            # Determine UTM zone
            utm_zone = self.determine_utm_zone(lon)
            
            # WGS84 ellipsoid parameters
            a = 6378137.0  # Semi-major axis
            e2 = 0.00669437999014  # First eccentricity squared
            
            # UTM parameters
            k0 = 0.9996  # Scale factor
            false_easting = 500000.0
            false_northing = 0.0 if lat >= 0 else 10000000.0
            
            # Convert to radians
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)
            
            # Central meridian for this UTM zone
            central_meridian = math.radians((utm_zone - 1) * 6 - 180 + 3)
            
            # Calculate UTM coordinates (simplified)
            delta_lon = lon_rad - central_meridian
            
            N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
            T = math.tan(lat_rad)**2
            C = e2 * math.cos(lat_rad)**2 / (1 - e2)
            A_term = math.cos(lat_rad) * delta_lon
            
            # Simplified UTM formulas (good enough for testing)
            M = a * ((1 - e2/4 - 3*e2**2/64) * lat_rad -
                    (3*e2/8 + 3*e2**2/32) * math.sin(2*lat_rad) +
                    (15*e2**2/256) * math.sin(4*lat_rad))
            
            x = false_easting + k0 * N * (A_term + (1-T+C) * A_term**3/6)
            y = false_northing + k0 * (M + N*math.tan(lat_rad) * (A_term**2/2 + (5-T+9*C+4*C**2) * A_term**4/24))
            
            epsg_code = self.norwegian_utm_zones.get(utm_zone, 25833)  # Default to 25833
            
            return CoordinateTestResult(
                input_wgs84=(lat, lon),
                output_utm=(x, y),
                utm_zone=utm_zone,
                utm_hemisphere='N',
                epsg_code=epsg_code,
                success=True
            )
            
        except Exception as e:
            return CoordinateTestResult(
                input_wgs84=(lat, lon),
                output_utm=(0, 0),
                utm_zone=0,
                utm_hemisphere='N',
                epsg_code=0,
                success=False,
                error=str(e)
            )

    def test_pyproj_transformation(self) -> Dict[str, any]:
        """Test if pyproj library is available and working"""
        try:
            import pyproj
            
            # Test transformation from WGS84 to ETRS89/UTM33N (most common in Norway)
            transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
            
            # Test with Oslo coordinates
            lon, lat = 10.7522, 59.9139
            x, y = transformer.transform(lon, lat)
            
            return {
                "available": True,
                "version": pyproj.__version__,
                "test_transformation": {
                    "input": {"lat": lat, "lon": lon},
                    "output": {"x": x, "y": y},
                    "epsg": "25833"
                }
            }
            
        except ImportError:
            return {
                "available": False,
                "error": "pyproj not installed",
                "install_command": "pip install pyproj"
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }

    def create_bounding_box(self, center_lat: float, center_lon: float, 
                          radius_meters: int = 15) -> Dict[str, float]:
        """
        Create a bounding box around a GPS coordinate
        
        Args:
            center_lat: Center latitude in WGS84
            center_lon: Center longitude in WGS84  
            radius_meters: Radius in meters (default 15m for NVDB queries)
            
        Returns:
            Dictionary with min/max coordinates in both WGS84 and UTM
        """
        # Approximate degrees per meter (varies by latitude)
        lat_deg_per_meter = 1 / 111320
        lon_deg_per_meter = 1 / (111320 * math.cos(math.radians(center_lat)))
        
        # WGS84 bounding box
        lat_offset = radius_meters * lat_deg_per_meter
        lon_offset = radius_meters * lon_deg_per_meter
        
        wgs84_bbox = {
            "min_lat": center_lat - lat_offset,
            "max_lat": center_lat + lat_offset,
            "min_lon": center_lon - lon_offset,
            "max_lon": center_lon + lon_offset
        }
        
        # Convert to UTM for NVDB API
        center_utm = self.wgs84_to_utm_manual(center_lat, center_lon)
        if center_utm.success:
            utm_bbox = {
                "min_x": center_utm.output_utm[0] - radius_meters,
                "max_x": center_utm.output_utm[0] + radius_meters,
                "min_y": center_utm.output_utm[1] - radius_meters,
                "max_y": center_utm.output_utm[1] + radius_meters,
                "epsg": center_utm.epsg_code
            }
        else:
            utm_bbox = {"error": "UTM conversion failed"}
        
        return {
            "center": {"lat": center_lat, "lon": center_lon},
            "radius_meters": radius_meters,
            "wgs84": wgs84_bbox,
            "utm": utm_bbox
        }

    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run comprehensive coordinate transformation tests"""
        results = {
            "pyproj_test": self.test_pyproj_transformation(),
            "manual_transformations": [],
            "utm_zone_analysis": {},
            "bounding_box_tests": []
        }
        
        print("🔍 Testing coordinate transformations...")
        
        # Test each location
        utm_zones_found = set()
        for location in self.test_locations:
            lat, lon = location["lat"], location["lon"]
            
            # Manual transformation test
            manual_result = self.wgs84_to_utm_manual(lat, lon)
            results["manual_transformations"].append({
                "location": location["name"],
                "result": manual_result
            })
            
            if manual_result.success:
                utm_zones_found.add(manual_result.utm_zone)
                print(f"  {location['name']}: Zone {manual_result.utm_zone} "
                      f"(EPSG:{manual_result.epsg_code})")
            
            # Bounding box test
            bbox_test = self.create_bounding_box(lat, lon, 15)
            results["bounding_box_tests"].append({
                "location": location["name"],
                "bbox": bbox_test
            })
        
        # Analyze UTM zones
        results["utm_zone_analysis"] = {
            "zones_found": list(utm_zones_found),
            "recommended_zones": {
                "western_norway": 32,  # Bergen, Stavanger
                "eastern_norway": 33,  # Oslo, Trondheim
                "northern_norway": 33   # Tromsø, Bodø
            },
            "epsg_codes": self.norwegian_utm_zones
        }
        
        return results

    def generate_config_template(self, results: Dict[str, any]) -> str:
        """Generate configuration template based on test results"""
        config = []
        config.append("# Coordinate System Configuration")
        config.append("# Generated from coordinate transformation tests")
        config.append("")
        
        # UTM Zone mapping
        config.append("UTM_ZONES = {")
        for zone in results["utm_zone_analysis"]["zones_found"]:
            epsg = self.norwegian_utm_zones.get(zone, 25833)
            config.append(f"    {zone}: {epsg},  # EPSG:{epsg}")
        config.append("}")
        config.append("")
        
        # Default settings
        config.append("# Default settings for Norway")
        config.append("DEFAULT_UTM_ZONE = 33  # Most of Norway")
        config.append("DEFAULT_EPSG = 25833   # ETRS89 / UTM zone 33N")
        config.append("NVDB_SEARCH_RADIUS = 15  # meters")
        config.append("")
        
        # Pyproj availability
        pyproj_available = results["pyproj_test"]["available"]
        config.append(f"# PyProj library available: {pyproj_available}")
        if not pyproj_available:
            config.append(f"# Install with: {results['pyproj_test'].get('install_command', 'pip install pyproj')}")
        config.append("")
        
        # Example bounding box calculation
        if results["bounding_box_tests"]:
            example = results["bounding_box_tests"][0]["bbox"]
            config.append("# Example bounding box calculation (Oslo):")
            config.append(f"# Input: {example['center']}")
            config.append(f"# UTM: {example['utm']}")
        
        return "\n".join(config)


def main():
    """Main test execution"""
    print("=" * 60)
    print("🗺️  COORDINATE TRANSFORMATION TEST SUITE")
    print("=" * 60)
    
    tester = CoordinateTransformTester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Display results
    print("\n📊 Test Results Summary:")
    print("-" * 30)
    
    # PyProj availability
    pyproj_status = results["pyproj_test"]
    if pyproj_status["available"]:
        print(f"✅ PyProj available (v{pyproj_status['version']})")
    else:
        print(f"⚠️  PyProj not available: {pyproj_status['error']}")
    
    # UTM zones
    zones = results["utm_zone_analysis"]["zones_found"]
    print(f"🗺️  UTM zones detected: {zones}")
    
    # Manual transformations
    successful_transforms = sum(1 for t in results["manual_transformations"] 
                              if t["result"].success)
    total_transforms = len(results["manual_transformations"])
    print(f"🔄 Manual transformations: {successful_transforms}/{total_transforms} successful")
    
    # Generate configuration
    config_template = tester.generate_config_template(results)
    
    # Save results
    import json
    with open('coordinate_test_results.json', 'w') as f:
        # Convert dataclass objects to dicts for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == "manual_transformations":
                json_results[key] = []
                for item in value:
                    result = item["result"]
                    json_results[key].append({
                        "location": item["location"],
                        "result": {
                            "input_wgs84": result.input_wgs84,
                            "output_utm": result.output_utm,
                            "utm_zone": result.utm_zone,
                            "utm_hemisphere": result.utm_hemisphere,
                            "epsg_code": result.epsg_code,
                            "success": result.success,
                            "error": result.error
                        }
                    })
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    with open('coordinate_config_template.py', 'w') as f:
        f.write(config_template)
    
    print("\n📄 Files generated:")
    print("  • coordinate_test_results.json - Full test results")
    print("  • coordinate_config_template.py - Configuration template")
    
    print("\n" + "=" * 60)
    print("🎯 KEY RECOMMENDATIONS FOR AI AGENT:")
    print("=" * 60)
    
    recommendations = {
        "primary_utm_zone": 33,
        "primary_epsg": 25833,
        "pyproj_required": not pyproj_status["available"],
        "utm_zones_needed": results["utm_zone_analysis"]["zones_found"],
        "search_radius": 15,
        "coordinate_systems": results["utm_zone_analysis"]["epsg_codes"]
    }
    
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    
    return results


if __name__ == "__main__":
    main()