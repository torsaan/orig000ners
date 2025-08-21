#!/usr/bin/env python3
"""
EXIF GPS Extraction Test Script
Tests GPS data extraction from smartphone images and validates coordinate processing.
Creates sample images with GPS data for testing the full pipeline.
"""

import os
import json
import math
from PIL import Image, ExifTags
from PIL.ExifTags import GPSTAGS
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GPSData:
    """Container for GPS data extracted from EXIF"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    timestamp: Optional[str] = None
    source: str = "EXIF"
    accuracy: Optional[float] = None


@dataclass
class ImageTestResult:
    """Container for image processing test results"""
    filename: str
    has_exif: bool
    has_gps: bool
    gps_data: Optional[GPSData] = None
    error: Optional[str] = None
    image_info: Optional[Dict] = None


class EXIFGPSTester:
    """Test EXIF GPS extraction functionality"""
    
    def __init__(self):
        self.test_images_dir = "test_images"
        self.sample_gps_locations = [
            {"name": "Oslo_Center", "lat": 59.9139, "lon": 10.7522},
            {"name": "Bergen_Harbor", "lat": 60.3913, "lon": 5.3221},
            {"name": "Trondheim_Square", "lat": 63.4305, "lon": 10.3951},
        ]

    def decimal_to_dms(self, decimal: float) -> Tuple[int, int, float]:
        """Convert decimal degrees to degrees, minutes, seconds"""
        degrees = int(abs(decimal))
        minutes_float = (abs(decimal) - degrees) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        return degrees, minutes, seconds

    def dms_to_decimal(self, degrees: int, minutes: int, seconds: float, direction: str) -> float:
        """Convert degrees, minutes, seconds to decimal degrees"""
        decimal = degrees + minutes/60 + seconds/3600
        if direction in ['S', 'W']:
            decimal = -decimal
        return decimal

    def extract_gps_from_exif(self, image_path: str) -> ImageTestResult:
        """Extract GPS data from image EXIF"""
        try:
            # Open image and extract EXIF
            with Image.open(image_path) as image:
                exif_data = image._getexif()
                
                if not exif_data:
                    return ImageTestResult(
                        filename=os.path.basename(image_path),
                        has_exif=False,
                        has_gps=False,
                        error="No EXIF data found"
                    )
                
                # Get image info
                image_info = {
                    "size": image.size,
                    "mode": image.mode,
                    "format": image.format
                }
                
                # Look for GPS info
                gps_info = None
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        gps_info = value
                        break
                
                if not gps_info:
                    return ImageTestResult(
                        filename=os.path.basename(image_path),
                        has_exif=True,
                        has_gps=False,
                        image_info=image_info,
                        error="No GPS data in EXIF"
                    )
                
                # Parse GPS data
                gps_data = {}
                for gps_tag_id, gps_value in gps_info.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_data[gps_tag] = gps_value
                
                # Extract latitude and longitude
                lat = None
                lon = None
                altitude = None
                timestamp = None
                
                if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
                    lat_dms = gps_data['GPSLatitude']
                    lat_ref = gps_data['GPSLatitudeRef']
                    lat = self.dms_to_decimal(lat_dms[0], lat_dms[1], lat_dms[2], lat_ref)
                
                if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
                    lon_dms = gps_data['GPSLongitude']
                    lon_ref = gps_data['GPSLongitudeRef']
                    lon = self.dms_to_decimal(lon_dms[0], lon_dms[1], lon_dms[2], lon_ref)
                
                if 'GPSAltitude' in gps_data:
                    altitude = float(gps_data['GPSAltitude'])
                
                if 'GPSTimeStamp' in gps_data and 'GPSDateStamp' in gps_data:
                    time_data = gps_data['GPSTimeStamp']
                    date_data = gps_data['GPSDateStamp']
                    timestamp = f"{date_data} {time_data[0]:02d}:{time_data[1]:02d}:{time_data[2]:02d}"
                
                if lat is not None and lon is not None:
                    gps_extracted = GPSData(
                        latitude=lat,
                        longitude=lon,
                        altitude=altitude,
                        timestamp=timestamp,
                        source="EXIF"
                    )
                    
                    return ImageTestResult(
                        filename=os.path.basename(image_path),
                        has_exif=True,
                        has_gps=True,
                        gps_data=gps_extracted,
                        image_info=image_info
                    )
                else:
                    return ImageTestResult(
                        filename=os.path.basename(image_path),
                        has_exif=True,
                        has_gps=False,
                        image_info=image_info,
                        error="Incomplete GPS data in EXIF"
                    )
                    
        except Exception as e:
            return ImageTestResult(
                filename=os.path.basename(image_path),
                has_exif=False,
                has_gps=False,
                error=f"Failed to process image: {str(e)}"
            )

    def create_test_image_with_gps(self, output_path: str, gps_lat: float, gps_lon: float, 
                                  name: str) -> bool:
        """Create a test image with embedded GPS data"""
        try:
            # Create a simple test image
            img = Image.new('RGB', (640, 480), color='blue')
            
            # Note: PIL cannot easily write GPS EXIF data
            # This is a limitation - most GPS EXIF writing requires specialized libraries
            # For testing, we'll create a simple image and document the expected GPS data
            
            img.save(output_path, 'JPEG')
            
            # Create a companion JSON file with the GPS data
            gps_file = output_path.replace('.jpg', '_gps.json')
            gps_info = {
                "filename": os.path.basename(output_path),
                "location_name": name,
                "gps_data": {
                    "latitude": gps_lat,
                    "longitude": gps_lon,
                    "source": "test_generated"
                },
                "note": "Real smartphone images will have this data in EXIF"
            }
            
            with open(gps_file, 'w') as f:
                json.dump(gps_info, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create test image {output_path}: {e}")
            return False

    def test_directory_processing(self, directory: str) -> List[ImageTestResult]:
        """Test processing all images in a directory"""
        results = []
        
        if not os.path.exists(directory):
            print(f"⚠️  Directory {directory} does not exist")
            return results
        
        # Look for common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                result = self.extract_gps_from_exif(file_path)
                results.append(result)
        
        return results

    def validate_gps_accuracy(self, gps_data: GPSData, expected_lat: float, 
                            expected_lon: float, tolerance: float = 0.001) -> bool:
        """Validate GPS data accuracy against expected coordinates"""
        lat_diff = abs(gps_data.latitude - expected_lat)
        lon_diff = abs(gps_data.longitude - expected_lon)
        
        return lat_diff <= tolerance and lon_diff <= tolerance

    def calculate_gps_distance(self, lat1: float, lon1: float, 
                             lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters"""
        # Haversine formula
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance

    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run comprehensive EXIF GPS extraction tests"""
        results = {
            "test_images_created": 0,
            "test_images_processed": [],
            "user_images_processed": [],
            "summary": {},
            "recommendations": []
        }
        
        print("🔍 Starting EXIF GPS Extraction Tests...")
        
        # Create test images directory
        os.makedirs(self.test_images_dir, exist_ok=True)
        
        # Create test images (note: these won't have real EXIF GPS data)
        print(f"\n📸 Creating test images in {self.test_images_dir}/")
        for i, location in enumerate(self.sample_gps_locations):
            filename = f"test_image_{i+1}_{location['name']}.jpg"
            filepath = os.path.join(self.test_images_dir, filename)
            
            success = self.create_test_image_with_gps(
                filepath, location['lat'], location['lon'], location['name']
            )
            if success:
                results["test_images_created"] += 1
                print(f"  ✅ Created {filename}")
        
        # Test processing test images (will show no GPS in EXIF as expected)
        print(f"\n🔍 Testing image processing on test images...")
        test_results = self.test_directory_processing(self.test_images_dir)
        results["test_images_processed"] = test_results
        
        # Look for user images in common directories
        user_directories = [".", "images", "photos", "samples"]
        for dir_name in user_directories:
            if os.path.exists(dir_name) and dir_name != self.test_images_dir:
                user_results = self.test_directory_processing(dir_name)
                if user_results:
                    results["user_images_processed"].extend(user_results)
                    print(f"  📁 Found {len(user_results)} images in {dir_name}")
        
        # Generate summary
        all_processed = results["test_images_processed"] + results["user_images_processed"]
        total_images = len(all_processed)
        images_with_exif = sum(1 for r in all_processed if r.has_exif)
        images_with_gps = sum(1 for r in all_processed if r.has_gps)
        
        results["summary"] = {
            "total_images_processed": total_images,
            "images_with_exif": images_with_exif,
            "images_with_gps": images_with_gps,
            "gps_extraction_rate": images_with_gps / total_images if total_images > 0 else 0
        }
        
        # Generate recommendations
        recommendations = []
        
        if images_with_gps == 0:
            recommendations.append("⚠️  No images with GPS data found. Ensure smartphone GPS is enabled when taking photos.")
            recommendations.append("📱 Test with actual smartphone photos taken with location services enabled.")
        
        if images_with_gps < total_images and total_images > 0:
            recommendations.append(f"📊 Only {images_with_gps}/{total_images} images have GPS data.")
            recommendations.append("🔧 Implement fallback mechanism for manual GPS input.")
        
        recommendations.append("📋 For data collection: Ensure GPS accuracy is enabled in camera settings.")
        recommendations.append("🗂️  Consider batch processing workflow for multiple images.")
        
        results["recommendations"] = recommendations
        
        return results

    def generate_data_collection_guide(self) -> str:
        """Generate a guide for collecting GPS-enabled images"""
        guide = []
        guide.append("# GPS-Enabled Image Collection Guide")
        guide.append("## For Norwegian Road Asset Inspection Demo")
        guide.append("")
        
        guide.append("## Smartphone Settings")
        guide.append("1. **Enable Location Services**")
        guide.append("   - iOS: Settings > Privacy & Security > Location Services > Camera > While Using App")
        guide.append("   - Android: Settings > Apps > Camera > Permissions > Location > Allow")
        guide.append("")
        
        guide.append("2. **Camera Settings**")
        guide.append("   - Enable 'Location Tags' or 'GPS Tags' in camera settings")
        guide.append("   - Use highest resolution available (minimum 1280x720)")
        guide.append("   - Ensure good GPS signal before taking photos")
        guide.append("")
        
        guide.append("## Data Collection Protocol")
        guide.append("1. **Before Starting**")
        guide.append("   - Check GPS accuracy (use GPS app to verify ±5m accuracy)")
        guide.append("   - Plan route to include diverse sign types")
        guide.append("   - Ensure good weather and lighting conditions")
        guide.append("")
        
        guide.append("2. **Photo Taking Guidelines**")
        guide.append("   - Stand 3-10 meters from signs for optimal detection")
        guide.append("   - Take photos from slight angle to avoid reflections")
        guide.append("   - Include some vegetation in frame if present")
        guide.append("   - Take multiple angles of the same sign")
        guide.append("   - Wait for GPS lock before each photo")
        guide.append("")
        
        guide.append("3. **Target Sign Types for Norway**")
        guide.append("   - Speed limit signs (30, 50, 80 km/h)")
        guide.append("   - Stop signs (STOPP)")
        guide.append("   - Warning signs")
        guide.append("   - Regulatory signs")
        guide.append("")
        
        guide.append("4. **Documentation**")
        guide.append("   - Note actual sign content for validation")
        guide.append("   - Record any vegetation obscuration issues")
        guide.append("   - Document approximate GPS coordinates")
        guide.append("")
        
        guide.append("## Quality Control")
        guide.append("- Minimum 200-300 images for training")
        guide.append("- At least 20-30 images per sign class")
        guide.append("- Include various lighting conditions")
        guide.append("- Mix of clear and partially obscured signs")
        guide.append("")
        
        guide.append("## File Organization")
        guide.append("```")
        guide.append("collected_images/")
        guide.append("├── clear_signs/")
        guide.append("│   ├── speed_signs/")
        guide.append("│   ├── stop_signs/")
        guide.append("│   └── warning_signs/")
        guide.append("└── obscured_signs/")
        guide.append("    ├── partially_obscured/")
        guide.append("    └── heavily_obscured/")
        guide.append("```")
        
        return "\n".join(guide)


def main():
    """Main test execution"""
    print("=" * 70)
    print("📱 EXIF GPS EXTRACTION TEST SUITE")
    print("=" * 70)
    
    tester = EXIFGPSTester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Display results
    print("\n📊 Test Results Summary:")
    print("-" * 40)
    
    summary = results["summary"]
    print(f"📸 Total images processed: {summary['total_images_processed']}")
    print(f"📋 Images with EXIF data: {summary['images_with_exif']}")
    print(f"🗺️  Images with GPS data: {summary['images_with_gps']}")
    
    if summary['total_images_processed'] > 0:
        gps_rate = summary['gps_extraction_rate'] * 100
        print(f"📈 GPS extraction rate: {gps_rate:.1f}%")
    
    # Show recommendations
    print("\n💡 Recommendations:")
    for rec in results["recommendations"]:
        print(f"  {rec}")
    
    # Generate data collection guide
    guide = tester.generate_data_collection_guide()
    
    # Save results
    with open('exif_gps_test_results.json', 'w') as f:
        # Convert dataclass objects for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, list):
                json_results[key] = []
                for item in value:
                    if isinstance(item, ImageTestResult):
                        json_results[key].append({
                            "filename": item.filename,
                            "has_exif": item.has_exif,
                            "has_gps": item.has_gps,
                            "gps_data": {
                                "latitude": item.gps_data.latitude,
                                "longitude": item.gps_data.longitude,
                                "altitude": item.gps_data.altitude,
                                "timestamp": item.gps_data.timestamp,
                                "source": item.gps_data.source
                            } if item.gps_data else None,
                            "error": item.error,
                            "image_info": item.image_info
                        })
                    else:
                        json_results[key].append(item)
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    with open('data_collection_guide.md', 'w', encoding='utf-8') as f:
            f.write(guide)
            f.write(guide)
    
    print("\n📄 Files generated:")
    print("  • exif_gps_test_results.json - Test results")
    print("  • data_collection_guide.md - Photo collection guide")
    print(f"  • {tester.test_images_dir}/ - Test images with GPS reference files")
    
    print("\n" + "=" * 70)
    print("🎯 READY FOR DATA COLLECTION:")
    print("=" * 70)
    print("1. Review the data_collection_guide.md")
    print("2. Configure smartphone GPS settings")
    print("3. Collect 200-300 images with GPS tags")
    print("4. Run this script again on collected images")
    print("5. Proceed with AI agent implementation")
    
    return results


if __name__ == "__main__":
    main()