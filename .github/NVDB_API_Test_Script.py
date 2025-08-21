#!/usr/bin/env python3
"""
NVDB API Test Script
Tests connectivity and response formats for the Norwegian National Road Database API.
Validates coordinate transformations and feature type mappings.
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass


@dataclass
class TestResult:
    """Container for test results"""
    success: bool
    message: str
    data: Optional[Dict] = None
    error: Optional[str] = None


class NVDBAPITester:
    """Test suite for NVDB API functionality"""
    
    def __init__(self):
        # Try both V4 (current) and V3 (fallback) API endpoints
        self.api_v4_base = "https://nvdbapiles.atlas.vegvesen.no"
        self.api_v3_base = "https://nvdbapiles-v3.atlas.vegvesen.no"
        
        # Standard headers for NVDB API
        self.headers = {
            "Accept": "application/json",
            "X-Client": "demo.asset.inspection@test.com", 
            "User-Agent": "DemoAssetInspection/1.0"
        }
        

        self.expected_feature_types = {
            "speed_limit_signs": [105, 106],  # May include different speed limit types
            "stop_signs": [96],  # Stop sign type
            "warning_signs": [107, 108],  # Warning sign types
            "regulatory_signs": [109, 110]  # Other regulatory signs
        }
    
        self.test_coordinates = [
            {"name": "Oslo_center", "lat": 59.9139, "lon": 10.7522},
            {"name": "Bergen_center", "lat": 60.3913, "lon": 5.3221},
            {"name": "Trondheim_center", "lat": 63.4305, "lon": 10.3951}
        ]

    def test_api_connectivity(self) -> TestResult:
        """Test basic API connectivity and available versions"""
        print("🔍 Testing NVDB API connectivity...")
        
        # Test V4 API
        try:
            response = requests.get(f"{self.api_v4_base}/", 
                                  headers=self.headers, 
                                  timeout=10)
            if response.status_code == 200:
                return TestResult(
                    success=True,
                    message="✅ NVDB API V4 is accessible",
                    data={"version": "v4", "base_url": self.api_v4_base}
                )
        except requests.RequestException as e:
            print(f"⚠️  V4 API not accessible: {e}")
        
        # Test V3 API as fallback
        try:
            response = requests.get(f"{self.api_v3_base}/", 
                                  headers=self.headers, 
                                  timeout=10)
            if response.status_code == 200:
                return TestResult(
                    success=True,
                    message="⚠️  Using NVDB API V3 (V4 unavailable)",
                    data={"version": "v3", "base_url": self.api_v3_base}
                )
        except requests.RequestException as e:
            return TestResult(
                success=False,
                message="❌ Both V4 and V3 APIs are inaccessible",
                error=str(e)
            )

    def test_feature_types_endpoint(self, base_url: str) -> TestResult:
        """Test the feature types endpoint to discover available object types"""
        print("🔍 Testing feature types endpoint...")
        
        endpoint = f"{base_url}/vegobjekttyper"
        
        try:
            response = requests.get(endpoint, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract feature types related to signs
                sign_types = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            name = item.get('navn', '').lower()
                            if any(keyword in name for keyword in ['skilt', 'signal', 'sign', 'merke']):
                                sign_types.append({
                                    'id': item.get('id'),
                                    'navn': item.get('navn'),
                                    'beskrivelse': item.get('beskrivelse', '')
                                })
                
                return TestResult(
                    success=True,
                    message=f"✅ Found {len(sign_types)} sign-related feature types",
                    data={
                        "total_types": len(data) if isinstance(data, list) else 0,
                        "sign_types": sign_types[:10],  # First 10 for brevity
                        "sample_response": data[:3] if isinstance(data, list) else data
                    }
                )
            else:
                return TestResult(
                    success=False,
                    message=f"❌ Feature types endpoint returned {response.status_code}",
                    error=response.text[:500]
                )
                
        except requests.RequestException as e:
            return TestResult(
                success=False,
                message="❌ Failed to access feature types endpoint",
                error=str(e)
            )

    def test_geospatial_query(self, base_url: str, feature_type_id: int, 
                            lat: float, lon: float, name: str) -> TestResult:
        """Test geospatial queries for road objects"""
        print(f"🔍 Testing geospatial query for {name} (feature type {feature_type_id})...")
        
        # Create a bounding box around the coordinates (±0.01 degrees ≈ ±1km)
        bbox_size = 0.01
        bbox = f"{lon-bbox_size},{lat-bbox_size},{lon+bbox_size},{lat+bbox_size}"
        
        # Try different endpoint patterns for V3/V4 compatibility
        endpoints_to_try = [
            f"{base_url}/vegobjekter/{feature_type_id}?kartutsnitt={bbox}",
            f"{base_url}/vegobjekter/{feature_type_id}?bbox={bbox}",
            f"{base_url}/vegobjekter/{feature_type_id}?område={bbox}"
        ]
        
        for endpoint in endpoints_to_try:
            try:
                response = requests.get(endpoint, headers=self.headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract object count and sample data
                    objects = []
                    if 'objekter' in data:
                        objects = data['objekter']
                    elif isinstance(data, list):
                        objects = data
                    
                    return TestResult(
                        success=True,
                        message=f"✅ Found {len(objects)} objects near {name}",
                        data={
                            "location": name,
                            "coordinates": {"lat": lat, "lon": lon},
                            "feature_type_id": feature_type_id,
                            "object_count": len(objects),
                            "sample_objects": objects[:3] if objects else [],
                            "endpoint_used": endpoint
                        }
                    )
                elif response.status_code == 404:
                    continue  # Try next endpoint pattern
                else:
                    return TestResult(
                        success=False,
                        message=f"❌ Query failed with status {response.status_code}",
                        error=response.text[:500]
                    )
                    
            except requests.RequestException as e:
                continue  # Try next endpoint
        
        return TestResult(
            success=False,
            message=f"❌ All geospatial query patterns failed for {name}",
            error="No valid endpoint pattern found"
        )

    def discover_coordinate_system(self, base_url: str) -> TestResult:
        """Discover the coordinate system used by NVDB"""
        print("🔍 Testing coordinate system requirements...")
        
        # Test with a known Oslo location
        test_location = self.test_coordinates[0]
        
        # Try querying with different coordinate formats
        coord_tests = [
            {"name": "WGS84", "coords": f"{test_location['lon']},{test_location['lat']}"},
            {"name": "UTM33N", "coords": "598915,6642877"},  # Oslo in EPSG:25833
        ]
        
        for coord_test in coord_tests:
            try:
                # Use a common feature type for testing (try 105 first)
                endpoint = f"{base_url}/vegobjekter/105"
                params = {"kartutsnitt": coord_test["coords"] + ",1000"}  # 1000m radius
                
                response = requests.get(endpoint, headers=self.headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    object_count = len(data.get('objekter', []))
                    
                    return TestResult(
                        success=True,
                        message=f"✅ Coordinate system appears to be {coord_test['name']}",
                        data={
                            "coordinate_system": coord_test['name'],
                            "test_coords": coord_test['coords'],
                            "objects_found": object_count
                        }
                    )
                        
            except requests.RequestException:
                continue
        
        return TestResult(
            success=False,
            message="❌ Could not determine coordinate system",
            error="No coordinate format worked"
        )

    def run_comprehensive_test(self) -> Dict[str, TestResult]:
        """Run all tests and return comprehensive results"""
        results = {}
        
        print("=" * 60)
        print("🚀 Starting NVDB API Comprehensive Test Suite")
        print("=" * 60)
        
        # Test 1: API Connectivity
        connectivity_result = self.test_api_connectivity()
        results['connectivity'] = connectivity_result
        
        if not connectivity_result.success:
            print("❌ Cannot proceed with further tests - API unreachable")
            return results
        
        base_url = connectivity_result.data['base_url']
        print(f"Using API base URL: {base_url}")
        
        # Test 2: Feature Types
        feature_types_result = self.test_feature_types_endpoint(base_url)
        results['feature_types'] = feature_types_result
        
        # Test 3: Coordinate System Discovery
        coord_result = self.discover_coordinate_system(base_url)
        results['coordinate_system'] = coord_result
        
        # Test 4: Geospatial Queries
        # Try common feature types that might exist
        test_feature_types = [105, 96, 107, 108, 109, 110]
        
        for i, coord in enumerate(self.test_coordinates):
            for feature_type in test_feature_types:
                test_name = f"geospatial_{coord['name']}_type_{feature_type}"
                result = self.test_geospatial_query(
                    base_url, feature_type, 
                    coord['lat'], coord['lon'], coord['name']
                )
                results[test_name] = result
                
                # Add small delay to be respectful to API
                time.sleep(0.5)
                
                # Stop testing this location if we found objects
                if result.success and result.data['object_count'] > 0:
                    break
        
        return results

    def generate_report(self, results: Dict[str, TestResult]) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("NVDB API TEST REPORT")
        report.append("=" * 80)
        
        # Summary
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.success)
        report.append(f"Tests Run: {total_tests}")
        report.append(f"Successful: {successful_tests}")
        report.append(f"Failed: {total_tests - successful_tests}")
        report.append("")
        
        # Detailed results
        for test_name, result in results.items():
            report.append(f"TEST: {test_name}")
            report.append(f"Status: {'✅ PASS' if result.success else '❌ FAIL'}")
            report.append(f"Message: {result.message}")
            
            if result.data:
                report.append("Data:")
                report.append(json.dumps(result.data, indent=2))
            
            if result.error:
                report.append(f"Error: {result.error}")
            
            report.append("-" * 40)
        
        return "\n".join(report)


def main():
    """Main test execution"""
    tester = NVDBAPITester()
    
    print("NVDB API Test Suite")
    print("This script will test connectivity and data format for NVDB API integration")
    print()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Generate and display report
    report = tester.generate_report(results)
    print("\n" + report)
    
    # Save report to file
    with open('nvdb_api_test_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Full report saved to: nvdb_api_test_report.txt")
    
    # Extract key findings for agent instructions
    key_findings = {}
    if 'connectivity' in results and results['connectivity'].success:
        key_findings['api_version'] = results['connectivity'].data['version']
        key_findings['base_url'] = results['connectivity'].data['base_url']
    
    if 'feature_types' in results and results['feature_types'].success:
        key_findings['available_sign_types'] = results['feature_types'].data['sign_types']
    
    if 'coordinate_system' in results and results['coordinate_system'].success:
        key_findings['coordinate_system'] = results['coordinate_system'].data['coordinate_system']
    
    # Find working feature types
    working_feature_types = []
    for test_name, result in results.items():
        if test_name.startswith('geospatial_') and result.success and result.data['object_count'] > 0:
            working_feature_types.append({
                'feature_type_id': result.data['feature_type_id'],
                'object_count': result.data['object_count'],
                'location': result.data['location']
            })
    
    key_findings['working_feature_types'] = working_feature_types
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS FOR AI AGENT:")
    print("=" * 60)
    print(json.dumps(key_findings, indent=2))
    
    return key_findings


if __name__ == "__main__":
    main()