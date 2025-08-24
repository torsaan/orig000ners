# Automated Roadside Asset Inspection Demo / WORK IN PROGRESS 

- Fine-tune the model for specific detections 
- Models for specific tasks for detected objects
- Streamline integration corresponding to guidelines from the Norwegian government regarding reporting, etc
- Move from streamlit to a local application - Local GPU or Cloud, while focusing on images, batching, local should be fine -> Moving to point clouds would require a cloud solution for processing.
-  Calculating depth -~~~~~?
-  As long as the object matching works, the point clouds get redundant. Hurr durr i have lots of data thats worth alot.
- How to do Predictive maintenance work?
  * Images , training data and further analysis on cracks.
  * Images of areas continsoutly before something went wrong.




  Thoughts
  - Certain Vegetation, plants, etc, are considered harmful just by their looks. Identifying these automatically could be nice.
  - Georeferencing objects to the NVDB will be the main issue
  - Images from reports could help create a solid dataset for training for anomalies, bad sight, bad rails, and other elements that are hard to find examples of. This could be solved by solving a data problem rather than complicated technical solutions.
 
    ## 21.08.2025 Need to get model working first.
    ## 23.08.2025 Model performing poorly, good enough to move forward with testing matching objects to NVDB. :)


A proof-of-concept system that processes georeferenced smartphone images to detect traffic signs, identify vegetation obscuration issues, and verify detected assets against the Norwegian National Road Database (NVDB).

## 🎯 Project Overview

### Objective
Build a system that automates roadside asset inspection using smartphone images with GPS metadata, targeting Norwegian road infrastructure.

### Key Features
- **Object Detection**: YOLOv8-based detection of traffic signs and vegetation
- **Issue Analysis**: IoU-based vegetation obscuration detection
- **NVDB Integration**: Verification against Norwegian National Road Database
- **GeoJSON Output**: GIS-compatible results for further analysis
- **Interactive Interface**: Streamlit web app with map visualization
- **Future-Ready**: Modular architecture for LIDAR integration

## 🏗️ System Architecture

```
📁 project/
├── 🚀 main.py                    # Streamlit app entry point
├── ⚙️ config/
│   ├── config.json              # System configuration
│   └── sign_mapping.json        # NVDB feature type mappings
├── 🔍 detection/
│   ├── model.py                 # YOLOv8 wrapper
│   └── depth_estimation.py     # MiDaS depth estimation
├── 🌐 nvdb/
│   ├── api_client.py           # NVDB API client
│   └── matcher.py              # GPS-to-NVDB matching
├── 📊 analysis/
│   └── issue_detector.py       # IoU-based issue analysis
├── 📄 output/
│   └── geojson_generator.py    # GeoJSON output generation
├── 📷 images/                   # Input images directory
├── 📈 outputs/                  # Generated outputs
└── 🧪 tests/                    # Validation tests
    ├── nvdb_api_test.py
    ├── coordinate_transform_test.py
    └── exif_gps_test.py
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv env
.\env\Scripts\Activate.ps1  # Windows PowerShell
# source env/bin/activate    # Linux/Mac

# Install dependencies
pip install ultralytics pillow pyproj requests folium streamlit streamlit-folium opencv-python
```

### 2. Run Validation Tests

```bash
python run_tests.py
```

This validates:
- ✅ NVDB API connectivity
- ✅ Coordinate transformation (WGS84 ↔ UTM)
- ✅ EXIF GPS extraction

### 3. Launch the Application

```bash
streamlit run main.py
```

The web interface will open at `http://localhost:8501`

## 📱 Data Collection

### Image Requirements
- **Format**: JPEG with EXIF GPS metadata
- **Resolution**: Minimum 1280x720, preferred 1920x1080
- **GPS Accuracy**: Enable high-accuracy GPS (±3-5m)
- **Content**: Traffic signs with surrounding context

### Smartphone Setup
1. **Enable Location Services**
   - iOS: Settings > Privacy & Security > Location Services > Camera > While Using App
   - Android: Settings > Apps > Camera > Permissions > Location > Allow

2. **Camera Settings**
   - Enable 'Location Tags' or 'GPS Tags'
   - Use highest resolution available
   - Ensure GPS lock before photos

### Collection Protocol
- Stand 3-10 meters from signs
- Capture multiple angles (front, side)
- Include varied conditions (clear, partially obscured)
- Document actual sign content for validation

## 🔧 Configuration

### Main Configuration (`config/config.json`)

```json
{
  "yolo_model": "yolov8n.pt",
  "iou_threshold": 0.15,
  "confidence_threshold": 0.5,
  "nvdb_api": {
    "base_url": "https://nvdbapiles-v3.atlas.vegvesen.no",
    "client_email": "your@email.com",
    "timeout": 15
  },
  "coordinate_system": {
    "input": "EPSG:4326",
    "output": "EPSG:25833",
    "default_utm_zone": 33
  },
  "nvdb_search_radius": 15
}
```

### Sign Mapping (`config/sign_mapping.json`)

```json
{
  "stop_sign": {"nvdb_feature_id": 96},
  "speed_sign_30": {"nvdb_feature_id": 105, "speed_value": 30},
  "speed_sign_50": {"nvdb_feature_id": 105, "speed_value": 50},
  "speed_sign_80": {"nvdb_feature_id": 105, "speed_value": 80},
  "vegetation": {"nvdb_feature_id": null}
}
```

## 🧪 Testing

### Coordinate Transformation Test
```bash
python tests/coordinate_transform_test.py
```
- Validates WGS84 to UTM conversions
- Tests multiple Norwegian locations
- Confirms PyProj availability

### NVDB API Test
```bash
python tests/nvdb_api_test.py
```
- Tests API connectivity (V3/V4)
- Validates feature type mappings
- Confirms geospatial queries

### EXIF GPS Test
```bash
python tests/exif_gps_test.py
```
- Tests GPS extraction from images
- Generates data collection guidelines
- Creates test images with GPS references

## 📊 Usage

### Web Interface

1. **Upload Images**: Select GPS-tagged JPEG files
2. **Process**: Click "Process Images" to run detection pipeline
3. **View Map**: Interactive map showing results with issue indicators
4. **Export**: Download GeoJSON for GIS analysis

### Command Line Processing

```python
from detection.model import Detector
from nvdb.matcher import NVDBMatcher
from analysis.issue_detector import IssueAnalyzer

# Initialize components
detector = Detector()
matcher = NVDBMatcher()
analyzer = IssueAnalyzer()

# Process image
detections = detector.detect("image.jpg")
gps_coords = (59.9139, 10.7522)  # Oslo
matches = matcher.match_detections(detections, gps_coords)
issues = analyzer.analyze_objects(detections)
```

## 📈 Output Formats

### GeoJSON Structure
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {
      "type": "Point",
      "coordinates": [lon, lat]
    },
    "properties": {
      "image_id": "image_123.jpg",
      "detected_class": "speed_sign_80",
      "issue": "obscured",
      "iou": 0.25,
      "nvdb_id": "743981",
      "nvdb_status": "match"
    }
  }]
}
```

### Issue Detection Results
- **Vegetation Obscuration**: IoU-based overlap analysis
- **Visibility Issues**: Low detection confidence flags
- **Depth Analysis**: MiDaS-based depth ordering (optional)

## 🌍 Geographic Scope

### Primary: Norway
- **Coordinate System**: ETRS89/UTM (EPSG:25832/25833)
- **Data Source**: NVDB (Nasjonal vegdatabank)
- **Coverage**: All Norwegian public roads

### UTM Zone Support
- Zone 32N (EPSG:25832): Southern/Western Norway
- Zone 33N (EPSG:25833): Central/Eastern Norway (default)
- Zone 35N (EPSG:25835): Northern Norway

## 🔮 Future Extensions

### Phase 2: LIDAR Integration

**Data Sources**:
- Synchronized LIDAR point clouds + images
- Calibration matrices for 2D-3D projection

**Analysis Enhancements**:
- 3D proximity detection (vegetation < 0.5m from signs)
- Road clearance zone validation
- Physical obstruction modeling

**Output Extensions**:
- 3D GeoJSON geometries (MultiPoint, Polygon)
- Point cloud visualizations
- Clearance zone reports

### International Expansion
- **Coordinate Systems**: Support additional UTM zones
- **Data Sources**: Integration with other national databases
- **Sign Standards**: Configurable sign type mappings

## 🛠️ Technical Specifications

### Dependencies
- **Core**: Python 3.8+, PyTorch, OpenCV
- **Detection**: Ultralytics YOLOv8
- **Geospatial**: PyProj, Folium
- **Web**: Streamlit, PIL
- **Optional**: MiDaS (depth estimation)

### Hardware Requirements
- **Training**: GPU recommended (GTX 1650+)
- **Inference**: CPU sufficient for small batches
- **Storage**: 1-2 GB for models and outputs

### Performance Targets
- **Detection Accuracy**: mAP@50 ≥ 0.85
- **NVDB Match Rate**: ≥ 90% for known assets
- **Processing Speed**: ~30 seconds per image (CPU)

## 📋 Known Limitations

1. **Training Data**: System ships with pre-trained YOLOv8 - requires fine-tuning on Norwegian sign dataset
2. **NVDB Coverage**: Some rural areas may have incomplete NVDB data
3. **GPS Accuracy**: Smartphone GPS (±3-5m) may cause false negatives in dense areas
4. **Weather Effects**: Performance may degrade in poor weather/lighting
5. **Vegetation Seasonality**: Detection optimized for leafed vegetation

## 🆘 Troubleshooting

### Common Issues

**"No GPS data found"**
- Ensure location services enabled in camera app
- Check EXIF data with: `python tests/exif_gps_test.py`

**"NVDB API connectivity failed"**
- Verify internet connection
- Check NVDB status: https://nvdbstatus.atlas.vegvesen.no/
- Try fallback API (V3 ↔ V4)

**"Low detection accuracy"**
- Increase image resolution (minimum 1280x720)
- Ensure good lighting and contrast
- Consider model fine-tuning with local data

**"Coordinate transformation errors"**
- Verify PyProj installation: `pip install pyproj`
- Check UTM zone coverage for your area
- Validate coordinates with: `python tests/coordinate_transform_test.py`

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. **Data Collection**: Contribute GPS-tagged Norwegian sign images
2. **Model Training**: Help fine-tune YOLOv8 on Norwegian signs
3. **Testing**: Validate system in different Norwegian regions
4. **Documentation**: Improve setup guides and examples

## 📄 License

This project is for demonstration and research purposes. Please respect:
- NVDB API terms of service
- Norwegian data protection regulations
- Image privacy considerations

## 📞 Support

For technical issues:
1. Run validation tests: `python run_tests.py`
2. Check configuration files in `config/`
3. Review logs in terminal output
4. Verify NVDB API status

---

**Version**: 1.0  
**Last Updated**: August 2025  
**Tested**: Norway (Oslo, Bergen, Trondheim)  
**Status**: Demo/Proof-of-Concept
