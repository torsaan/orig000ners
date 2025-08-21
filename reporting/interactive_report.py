#!/usr/bin/env python3
"""
Interactive Report Generation Component
Provides an editable table interface for reviewing and annotating detected issues.
"""

import streamlit as st
import pandas as pd
import base64
import io
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class InteractiveReportGenerator:
    """Interactive report generation with editable fields and image display"""
    
    def __init__(self):
        """Initialize the interactive report generator"""
        self.issue_types = [
            'Pothole', 
            'Alligator Crack', 
            'Longitudinal Crack', 
            'Obscured Sign',
            'Vegetation Overgrowth',
            'Damaged Sign',
            'Missing Sign',
            'Poor Visibility',
            'Road Marking Issue',
            'Other'
        ]
        
        self.severity_levels = ['Low', 'Medium', 'High', 'Critical']
        
        # Initialize session state for detected issues
        if 'detected_issues' not in st.session_state:
            st.session_state.detected_issues = []
        
        if 'report_initialized' not in st.session_state:
            st.session_state.report_initialized = False
    
    def convert_processing_results_to_issues(self, processing_results: List[Any]) -> List[Dict]:
        """
        Convert ImageProcessingResult objects to interactive issue format
        
        Args:
            processing_results: List of ImageProcessingResult objects
            
        Returns:
            List of issue dictionaries for interactive editing
        """
        issues = []
        issue_id = 0
        
        for result in processing_results:
            # Process each detection as a separate issue
            for detection in result.detections:
                # Create cropped images for original and annotated
                original_crop_b64 = self._create_detection_crop(
                    result.original_image_b64, detection, False
                )
                annotated_crop_b64 = self._create_detection_crop(
                    result.annotated_image_b64 or result.original_image_b64, 
                    detection, True
                )
                
                # Map detection class to issue type
                issue_type = self._map_detection_to_issue_type(detection['class'])
                
                # Determine initial severity based on confidence
                initial_severity = self._determine_initial_severity(detection['confidence'])
                
                issue = {
                    "id": f"issue_{issue_id}",
                    "image_name": result.image_path,
                    "detection_class": detection['class'],
                    "confidence": detection['confidence'],
                    "original_img_b64": original_crop_b64,
                    "analyzed_img_b64": annotated_crop_b64,
                    "full_original_img_b64": result.original_image_b64,
                    "full_analyzed_img_b64": result.annotated_image_b64,
                    "issue_type": issue_type,
                    "severity": initial_severity,
                    "note": "",
                    "gps_lat": result.gps_coords[0] if result.processing_metadata.get("has_gps") else None,
                    "gps_lon": result.gps_coords[1] if result.processing_metadata.get("has_gps") else None,
                    "bbox": detection.get('bbox_xyxy', []),
                    "timestamp": result.processing_metadata.get("processed_at", datetime.now().isoformat())
                }
                issues.append(issue)
                issue_id += 1
        
        return issues
    
    def _create_detection_crop(self, image_b64: str, detection: Dict, draw_bbox: bool = False) -> str:
        """
        Create a cropped image around the detection
        
        Args:
            image_b64: Base64 encoded image
            detection: Detection dictionary with bbox information
            draw_bbox: Whether to draw bounding box on the crop
            
        Returns:
            Base64 encoded cropped image
        """
        if not image_b64:
            return ""
        
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data))
            
            # Get bbox coordinates
            bbox = detection.get('bbox_xyxy', [])
            if len(bbox) != 4:
                # If no bbox, return small version of full image
                img.thumbnail((150, 150))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                return base64.b64encode(buffer.getvalue()).decode()
            
            x1, y1, x2, y2 = bbox
            
            # Add padding around detection
            padding = 20
            x1 = max(0, int(x1 - padding))
            y1 = max(0, int(y1 - padding))
            x2 = min(img.width, int(x2 + padding))
            y2 = min(img.height, int(y2 + padding))
            
            # Crop the image
            cropped = img.crop((x1, y1, x2, y2))
            
            # Draw bounding box if requested
            if draw_bbox:
                draw = ImageDraw.Draw(cropped)
                # Adjust bbox coordinates for cropped image
                crop_x1 = padding if x1 == int(bbox[0] - padding) else 0
                crop_y1 = padding if y1 == int(bbox[1] - padding) else 0
                crop_x2 = crop_x1 + (bbox[2] - bbox[0])
                crop_y2 = crop_y1 + (bbox[3] - bbox[1])
                
                # Draw rectangle
                draw.rectangle([crop_x1, crop_y1, crop_x2, crop_y2], 
                             outline="red", width=3)
                
                # Add label
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                draw.text((crop_x1, crop_y1 - 15), label, fill="red")
            
            # Resize to thumbnail
            cropped.thumbnail((150, 150))
            
            # Convert back to base64
            buffer = io.BytesIO()
            cropped.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode()
            
        except Exception as e:
            print(f"Error creating detection crop: {e}")
            return ""
    
    def _map_detection_to_issue_type(self, detection_class: str) -> str:
        """Map YOLO detection class to issue type"""
        mapping = {
            'stop sign': 'Obscured Sign',
            'traffic light': 'Poor Visibility',
            'car': 'Other',
            'truck': 'Other',
            'bus': 'Other',
            'person': 'Other',
            'bicycle': 'Other',
            'potted plant': 'Vegetation Overgrowth',
        }
        return mapping.get(detection_class.lower(), 'Other')
    
    def _determine_initial_severity(self, confidence: float) -> str:
        """Determine initial severity based on detection confidence"""
        if confidence >= 0.9:
            return 'High'
        elif confidence >= 0.7:
            return 'Medium'
        else:
            return 'Low'
    
    def initialize_from_processing_results(self, processing_results: List[Any]):
        """Initialize interactive report from processing results"""
        # Safe access to session state with default fallback
        if not getattr(st.session_state, 'report_initialized', False):
            issues = self.convert_processing_results_to_issues(processing_results)
            st.session_state.detected_issues = issues
            st.session_state.report_initialized = True
            st.success(f"✅ Initialized interactive report with {len(issues)} issues")
    
    def render_interactive_report(self):
        """Render the interactive report interface with grouped images"""
        # Header with custom styling
        st.markdown('<div class="interactive-report-header">', unsafe_allow_html=True)
        st.markdown("# 📋 Interactive Inspection Report")
        st.markdown("Review and annotate detected issues grouped by image")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Safe access to detected issues
        detected_issues = getattr(st.session_state, 'detected_issues', [])
        if not detected_issues:
            st.info("No issues detected. Process images first to generate an interactive report.")
            return
        
        # Report controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            total_issues = len(st.session_state.detected_issues)
            unique_images = len(set(issue['image_name'] for issue in st.session_state.detected_issues))
            st.subheader(f"📊 {total_issues} Issues in {unique_images} Images")
        
        with col2:
            if st.button("🔄 Reset Report", help="Reset all changes and reload from detection results"):
                st.session_state.report_initialized = False
                st.session_state.detected_issues = []
                st.rerun()
        
        with col3:
            compact_mode = st.checkbox("� Compact View", value=True, help="Use compact issue rows")
        
        st.markdown("---")
        
        # Group issues by image
        issues_by_image = self._group_issues_by_image()
        
        # Render each image group as a card
        for image_name, image_issues in issues_by_image.items():
            self._render_image_card(image_name, image_issues, compact_mode)
        
        # Export options
        self._render_export_options()
    
    def _group_issues_by_image(self) -> Dict[str, List[Dict]]:
        """Group issues by image name"""
        issues_by_image = {}
        
        for issue in st.session_state.detected_issues:
            image_name = issue['image_name']
            if image_name not in issues_by_image:
                issues_by_image[image_name] = []
            issues_by_image[image_name].append(issue)
        
        return issues_by_image
    
    def _render_image_card(self, image_name: str, image_issues: List[Dict], compact_mode: bool):
        """Render a card for a single image with all its issues"""
        
        # Create main container with custom styling
        with st.container():
            st.markdown(f'<div class="image-report-card">', unsafe_allow_html=True)
            
            # Image header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### 📷 {image_name}")
                st.markdown(f"**{len(image_issues)} detection(s) found**")
            
            with col2:
                # Quick stats for this image
                severities = [issue['severity'] for issue in image_issues]
                critical_count = severities.count('Critical')
                high_count = severities.count('High')
                
                if critical_count > 0:
                    st.error(f"🔴 {critical_count} Critical")
                elif high_count > 0:
                    st.warning(f"🟠 {high_count} High Priority")
                else:
                    st.success("✅ Low Priority Only")
            
            st.markdown("---")
            
            # Display original and analyzed images side by side (once per image)
            if image_issues:
                first_issue = image_issues[0]  # Use first issue to get image data
                
                img_col1, img_col2 = st.columns(2)
                
                with img_col1:
                    st.markdown("**📸 Original Image**")
                    if first_issue["full_original_img_b64"]:
                        img_data = base64.b64decode(first_issue["full_original_img_b64"])
                        img = Image.open(io.BytesIO(img_data))
                        st.image(img, use_container_width=True)
                    else:
                        st.info("Original image not available")
                
                with img_col2:
                    st.markdown("**🎯 Analyzed Image**")
                    if first_issue["full_analyzed_img_b64"]:
                        img_data = base64.b64decode(first_issue["full_analyzed_img_b64"])
                        img = Image.open(io.BytesIO(img_data))
                        st.image(img, use_container_width=True)
                    else:
                        st.info("Analysis not available")
            
            st.markdown("#### 🔍 Detected Issues")
            
            # Issues table header
            if compact_mode:
                header_cols = st.columns([1.5, 2, 1.5, 1.5, 3])
                header_cols[0].markdown("**Thumbnail**")
                header_cols[1].markdown("**Detection**")
                header_cols[2].markdown("**Issue Type**")
                header_cols[3].markdown("**Severity**")
                header_cols[4].markdown("**Notes**")
            else:
                header_cols = st.columns([2, 2, 1.5, 1.5, 3])
                header_cols[0].markdown("**Crop (Original)**")
                header_cols[1].markdown("**Crop (Analyzed)**")
                header_cols[2].markdown("**Issue Type**")
                header_cols[3].markdown("**Severity**")
                header_cols[4].markdown("**Notes**")
            
            # Render each issue in this image
            for i, issue in enumerate(image_issues):
                issue_index = st.session_state.detected_issues.index(issue)
                self._render_compact_issue_row(issue_index, issue, compact_mode)
                
                # Add separator between issues
                if i < len(image_issues) - 1:
                    st.markdown('<hr style="margin: 10px 0; border-color: #4A4A4A;">', unsafe_allow_html=True)
            
            # Technical details in expandable section
            with st.expander("🔧 Show Technical Details"):
                tech_col1, tech_col2 = st.columns(2)
                
                with tech_col1:
                    st.markdown("**Image Information:**")
                    st.write(f"📁 **File:** {image_name}")
                    if first_issue['gps_lat'] and first_issue['gps_lon']:
                        st.write(f"📍 **GPS:** {first_issue['gps_lat']:.6f}, {first_issue['gps_lon']:.6f}")
                    else:
                        st.write("📍 **GPS:** Not available")
                    st.write(f"⏰ **Processed:** {first_issue.get('timestamp', 'Unknown')}")
                
                with tech_col2:
                    st.markdown("**Detection Summary:**")
                    detection_classes = [issue['detection_class'] for issue in image_issues]
                    class_counts = {}
                    for cls in detection_classes:
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    
                    for cls, count in class_counts.items():
                        avg_conf = np.mean([issue['confidence'] for issue in image_issues if issue['detection_class'] == cls])
                        st.write(f"🎯 **{cls}:** {count}x (avg conf: {avg_conf:.3f})")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    def _render_compact_issue_row(self, issue_index: int, issue: Dict, compact_mode: bool):
        """Render a compact row for a single issue within an image card"""
        
        if compact_mode:
            # Compact mode: thumbnail + detection info in one column
            row_cols = st.columns([1.5, 2, 1.5, 1.5, 3])
            
            # Thumbnail column
            with row_cols[0]:
                if issue["original_img_b64"]:
                    img_data = base64.b64decode(issue["original_img_b64"])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, width=80)
                else:
                    st.write("No thumb")
            
            # Detection info column
            with row_cols[1]:
                st.markdown(f"**{issue['detection_class']}**")
                st.caption(f"Confidence: {issue['confidence']:.3f}")
                if issue['bbox']:
                    bbox_str = f"Box: {issue['bbox'][2]-issue['bbox'][0]:.0f}×{issue['bbox'][3]-issue['bbox'][1]:.0f}"
                    st.caption(bbox_str)
            
        else:
            # Standard mode: separate columns for original and analyzed crops
            row_cols = st.columns([2, 2, 1.5, 1.5, 3])
            
            # Original crop
            with row_cols[0]:
                if issue["original_img_b64"]:
                    img_data = base64.b64decode(issue["original_img_b64"])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, width=120)
                    st.caption(f"{issue['detection_class']}")
                else:
                    st.write("No image")
            
            # Analyzed crop
            with row_cols[1]:
                if issue["analyzed_img_b64"]:
                    img_data = base64.b64decode(issue["analyzed_img_b64"])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, width=120)
                    st.caption(f"Conf: {issue['confidence']:.3f}")
                else:
                    st.write("No analysis")
        
        # Issue Type Dropdown
        type_col_index = 2 if compact_mode else 2
        with row_cols[type_col_index]:
            selected_issue = st.selectbox(
                "Type",
                options=self.issue_types,
                index=self.issue_types.index(issue["issue_type"]) if issue["issue_type"] in self.issue_types else 0,
                key=f"issue_type_{issue['id']}",
                label_visibility="collapsed"
            )
            st.session_state.detected_issues[issue_index]["issue_type"] = selected_issue
        
        # Severity Dropdown
        severity_col_index = 3 if compact_mode else 3
        with row_cols[severity_col_index]:
            selected_severity = st.selectbox(
                "Severity",
                options=self.severity_levels,
                index=self.severity_levels.index(issue["severity"]) if issue["severity"] in self.severity_levels else 0,
                key=f"severity_{issue['id']}",
                label_visibility="collapsed"
            )
            st.session_state.detected_issues[issue_index]["severity"] = selected_severity
            
            # Color-coded severity indicator
            severity_colors = {
                'Low': '🟢',
                'Medium': '🟡', 
                'High': '🟠',
                'Critical': '🔴'
            }
            st.write(f"{severity_colors.get(selected_severity, '⚪')}")
        
        # Notes Text Area
        notes_col_index = 4 if compact_mode else 4
        with row_cols[notes_col_index]:
            note_text = st.text_area(
                "Notes",
                value=issue["note"],
                key=f"note_{issue['id']}",
                height=80,
                label_visibility="collapsed",
                placeholder="Add your notes here..."
            )
            st.session_state.detected_issues[issue_index]["note"] = note_text
    
    def _render_export_options(self):
        """Render export options for the interactive report"""
        st.markdown('<div class="export-buttons-container">', unsafe_allow_html=True)
        st.subheader("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        # CSV Export
        with col1:
            if st.button("📄 Export to CSV", use_container_width=True):
                csv_data = self._generate_csv_export()
                st.download_button(
                    label="💾 Download CSV Report",
                    data=csv_data,
                    file_name=f'interactive_inspection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
        
        # JSON Export
        with col2:
            if st.button("📋 Export to JSON", use_container_width=True):
                json_data = self._generate_json_export()
                st.download_button(
                    label="💾 Download JSON Report",
                    data=json_data,
                    file_name=f'interactive_inspection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                    mime='application/json',
                    use_container_width=True
                )
        
        # Summary Stats
        with col3:
            if st.button("📊 Show Summary", use_container_width=True):
                self._show_summary_stats()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _generate_csv_export(self) -> str:
        """Generate CSV export of the interactive report"""
        export_data = []
        
        for issue in st.session_state.detected_issues:
            export_data.append({
                'ID': issue['id'],
                'Image': issue['image_name'],
                'Detection Class': issue['detection_class'],
                'Confidence': issue['confidence'],
                'Issue Type': issue['issue_type'],
                'Severity': issue['severity'],
                'Notes': issue['note'],
                'GPS Latitude': issue['gps_lat'] if issue['gps_lat'] else 'N/A',
                'GPS Longitude': issue['gps_lon'] if issue['gps_lon'] else 'N/A',
                'Timestamp': issue['timestamp']
            })
        
        df = pd.DataFrame(export_data)
        return df.to_csv(index=False)
    
    def _generate_json_export(self) -> str:
        """Generate JSON export of the interactive report"""
        export_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_issues': len(st.session_state.detected_issues),
                'version': '1.0'
            },
            'issues': []
        }
        
        for issue in st.session_state.detected_issues:
            # Remove base64 images from JSON export to reduce size
            issue_copy = issue.copy()
            issue_copy.pop('original_img_b64', None)
            issue_copy.pop('analyzed_img_b64', None)
            issue_copy.pop('full_original_img_b64', None)
            issue_copy.pop('full_analyzed_img_b64', None)
            export_data['issues'].append(issue_copy)
        
        return json.dumps(export_data, indent=2)
    
    def _show_summary_stats(self):
        """Show summary statistics of the interactive report"""
        issues = st.session_state.detected_issues
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Issues", len(issues))
        
        # Count by severity
        severity_counts = {}
        for issue in issues:
            severity = issue['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        critical_count = severity_counts.get('Critical', 0)
        col2.metric("Critical Issues", critical_count)
        
        # Count by issue type
        type_counts = {}
        for issue in issues:
            issue_type = issue['issue_type']
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        
        most_common_type = max(type_counts.keys(), key=lambda x: type_counts[x]) if type_counts else "N/A"
        col3.metric("Most Common Type", most_common_type)
        
        # Count with notes
        notes_count = len([i for i in issues if i['note'].strip()])
        col4.metric("Issues with Notes", notes_count)
        
        # Detailed breakdown
        st.subheader("📈 Detailed Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By Severity:**")
            for severity in self.severity_levels:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    st.write(f"• {severity}: {count}")
        
        with col2:
            st.write("**By Issue Type:**")
            for issue_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"• {issue_type}: {count}")


def main():
    """Test the interactive report generator"""
    st.set_page_config(page_title="Interactive Report Generator", layout="wide")
    
    generator = InteractiveReportGenerator()
    generator.render_interactive_report()


if __name__ == "__main__":
    main()
