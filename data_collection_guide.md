# GPS-Enabled Image Collection Guide
## For Norwegian Road Asset Inspection Demo

## Smartphone Settings
1. **Enable Location Services**
   - iOS: Settings > Privacy & Security > Location Services > Camera > While Using App
   - Android: Settings > Apps > Camera > Permissions > Location > Allow

2. **Camera Settings**
   - Enable 'Location Tags' or 'GPS Tags' in camera settings
   - Use highest resolution available (minimum 1280x720)
   - Ensure good GPS signal before taking photos

## Data Collection Protocol
1. **Before Starting**
   - Check GPS accuracy (use GPS app to verify ±5m accuracy)
   - Plan route to include diverse sign types
   - Ensure good weather and lighting conditions

2. **Photo Taking Guidelines**
   - Stand 3-10 meters from signs for optimal detection
   - Take photos from slight angle to avoid reflections
   - Include some vegetation in frame if present
   - Take multiple angles of the same sign
   - Wait for GPS lock before each photo

3. **Target Sign Types for Norway**
   - Speed limit signs (30, 50, 80 km/h)
   - Stop signs (STOPP)
   - Warning signs
   - Regulatory signs

4. **Documentation**
   - Note actual sign content for validation
   - Record any vegetation obscuration issues
   - Document approximate GPS coordinates

## Quality Control
- Minimum 200-300 images for training
- At least 20-30 images per sign class
- Include various lighting conditions
- Mix of clear and partially obscured signs

## File Organization
```
collected_images/
├── clear_signs/
│   ├── speed_signs/
│   ├── stop_signs/
│   └── warning_signs/
└── obscured_signs/
    ├── partially_obscured/
    └── heavily_obscured/
```# GPS-Enabled Image Collection Guide
## For Norwegian Road Asset Inspection Demo

## Smartphone Settings
1. **Enable Location Services**
   - iOS: Settings > Privacy & Security > Location Services > Camera > While Using App
   - Android: Settings > Apps > Camera > Permissions > Location > Allow

2. **Camera Settings**
   - Enable 'Location Tags' or 'GPS Tags' in camera settings
   - Use highest resolution available (minimum 1280x720)
   - Ensure good GPS signal before taking photos

## Data Collection Protocol
1. **Before Starting**
   - Check GPS accuracy (use GPS app to verify ±5m accuracy)
   - Plan route to include diverse sign types
   - Ensure good weather and lighting conditions

2. **Photo Taking Guidelines**
   - Stand 3-10 meters from signs for optimal detection
   - Take photos from slight angle to avoid reflections
   - Include some vegetation in frame if present
   - Take multiple angles of the same sign
   - Wait for GPS lock before each photo

3. **Target Sign Types for Norway**
   - Speed limit signs (30, 50, 80 km/h)
   - Stop signs (STOPP)
   - Warning signs
   - Regulatory signs

4. **Documentation**
   - Note actual sign content for validation
   - Record any vegetation obscuration issues
   - Document approximate GPS coordinates

## Quality Control
- Minimum 200-300 images for training
- At least 20-30 images per sign class
- Include various lighting conditions
- Mix of clear and partially obscured signs

## File Organization
```
collected_images/
├── clear_signs/
│   ├── speed_signs/
│   ├── stop_signs/
│   └── warning_signs/
└── obscured_signs/
    ├── partially_obscured/
    └── heavily_obscured/
```