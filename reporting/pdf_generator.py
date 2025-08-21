#!/usr/bin/env python3
"""
PDF Report Generation Module
Creates standardized PDF reports for traffic safety inspections based on approved findings.
"""

import os
import io
import base64
from datetime import datetime
from typing import List, Optional
from PIL import Image
import numpy as np

try:
    from fpdf import FPDF
except ImportError:
    print("⚠️ FPDF2 not installed. Installing...")
    os.system("pip install fpdf2")
    from fpdf import FPDF

from .database import Finding


class TrafficSafetyReportPDF(FPDF):
    """Custom PDF class for traffic safety reports"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(20, 20, 20)
    
    def header(self):
        """Add header to each page"""
        # Logo placeholder (you can add actual logo here)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 50, 100)
        self.cell(0, 10, 'TRAFFIC SAFETY INSPECTION REPORT', 0, 1, 'C')
        self.ln(5)
        
        # Add line
        self.set_draw_color(0, 50, 100)
        self.line(20, 30, 190, 30)
        self.ln(10)
    
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        # Add generation timestamp
        self.set_y(-25)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'R')


class TSReportGenerator:
    """Traffic Safety Report Generator"""
    
    def __init__(self):
        """Initialize the report generator"""
        self.severity_mappings = {
            'consequence': {
                'Lav': 1, 'Middels': 2, 'Høy': 3, 'Meget høy': 4
            },
            'probability': {
                'Svært liten': 1, 'Liten': 2, 'Middels': 3, 'Stor': 4, 'Svært stor': 5
            }
        }
        
        self.finding_types = ['Avvik', 'Feil', 'Merknad']
    
    def generate_report(self, findings: List[Finding], output_path: Optional[str] = None) -> bytes:
        """
        Generate PDF report for approved findings
        
        Args:
            findings: List of approved findings
            output_path: Optional path to save PDF file
            
        Returns:
            PDF content as bytes
        """
        if not findings:
            raise ValueError("No findings provided for report generation")
        
        # Filter only approved findings
        approved_findings = [f for f in findings if f.status == 'approved']
        if not approved_findings:
            raise ValueError("No approved findings found")
        
        pdf = TrafficSafetyReportPDF()
        
        # Add cover page
        self._add_cover_page(pdf, approved_findings)
        
        # Add finding pages
        for i, finding in enumerate(approved_findings):
            self._add_finding_page(pdf, finding, i + 1)
        
        # Add summary page
        self._add_summary_page(pdf, approved_findings)
        
        # Output PDF
        if output_path:
            pdf.output(output_path)
            print(f"✅ Report saved to: {output_path}")
        
        # Return as bytes
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            return pdf_output.encode('latin1')
        else:
            return bytes(pdf_output)
    
    def _add_cover_page(self, pdf: TrafficSafetyReportPDF, findings: List[Finding]):
        """Add cover page with report summary"""
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 20)
        pdf.set_text_color(0, 50, 100)
        pdf.cell(0, 15, 'TRAFFIC SAFETY INSPECTION', 0, 1, 'C')
        pdf.cell(0, 15, 'AUTOMATED FINDINGS REPORT', 0, 1, 'C')
        pdf.ln(20)
        
        # Report metadata
        pdf.set_font('Arial', '', 12)
        pdf.set_text_color(0, 0, 0)
        
        # Report info table
        report_date = datetime.now().strftime("%Y-%m-%d")
        pdf.cell(50, 8, 'Report Date:', 0, 0, 'L')
        pdf.cell(0, 8, report_date, 0, 1, 'L')
        
        pdf.cell(50, 8, 'Total Findings:', 0, 0, 'L')
        pdf.cell(0, 8, str(len(findings)), 0, 1, 'L')
        
        # Finding types summary
        finding_types_count = {}
        for finding in findings:
            ft = finding.expert_finding_type or 'Unspecified'
            finding_types_count[ft] = finding_types_count.get(ft, 0) + 1
        
        pdf.ln(5)
        pdf.cell(0, 8, 'Findings by Type:', 0, 1, 'L')
        for finding_type, count in finding_types_count.items():
            pdf.cell(20, 6, '', 0, 0)  # Indent
            pdf.cell(50, 6, f'{finding_type}:', 0, 0, 'L')
            pdf.cell(0, 6, str(count), 0, 1, 'L')
        
        # Road references
        road_refs = set(f.nvdb_road_reference for f in findings if f.nvdb_road_reference)
        if road_refs:
            pdf.ln(5)
            pdf.cell(0, 8, 'Road Segments Inspected:', 0, 1, 'L')
            for road_ref in sorted(road_refs):
                pdf.cell(20, 6, '', 0, 0)  # Indent
                pdf.cell(0, 6, f'• {road_ref}', 0, 1, 'L')
        
        pdf.ln(20)
        
        # Detection classes summary
        detection_classes = {}
        for finding in findings:
            cls = finding.detected_class
            detection_classes[cls] = detection_classes.get(cls, 0) + 1
        
        pdf.cell(0, 8, 'Objects Detected:', 0, 1, 'L')
        for det_class, count in sorted(detection_classes.items()):
            pdf.cell(20, 6, '', 0, 0)  # Indent
            pdf.cell(50, 6, f'{det_class}:', 0, 0, 'L')
            pdf.cell(0, 6, str(count), 0, 1, 'L')
    
    def _add_finding_page(self, pdf: TrafficSafetyReportPDF, finding: Finding, finding_number: int):
        """Add a page for individual finding"""
        pdf.add_page()
        
        # Finding header
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(0, 50, 100)
        pdf.cell(0, 10, f'FINDING #{finding_number}', 0, 1, 'C')
        pdf.ln(5)
        
        # Basic information table
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(0, 0, 0)
        
        # Two-column layout for basic info
        col_width = 85
        
        # Left column
        y_start = pdf.get_y()
        pdf.cell(col_width, 6, 'DETECTION INFORMATION', 1, 1, 'C')
        
        pdf.set_font('Arial', '', 9)
        pdf.cell(35, 5, 'Image:', 1, 0, 'L')
        pdf.cell(50, 5, os.path.basename(finding.image_path), 1, 1, 'L')
        
        pdf.cell(35, 5, 'Detected Class:', 1, 0, 'L')
        pdf.cell(50, 5, finding.detected_class, 1, 1, 'L')
        
        pdf.cell(35, 5, 'Confidence:', 1, 0, 'L')
        pdf.cell(50, 5, f'{finding.confidence_score:.3f}', 1, 1, 'L')
        
        pdf.cell(35, 5, 'Timestamp:', 1, 0, 'L')
        timestamp = finding.timestamp[:19] if len(finding.timestamp) > 19 else finding.timestamp
        pdf.cell(50, 5, timestamp, 1, 1, 'L')
        
        # Right column
        pdf.set_xy(105, y_start)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(col_width, 6, 'LOCATION INFORMATION', 1, 1, 'C')
        
        pdf.set_font('Arial', '', 9)
        pdf.cell(35, 5, 'Road Reference:', 1, 0, 'L')
        road_ref = finding.nvdb_road_reference[:15] + "..." if len(finding.nvdb_road_reference) > 15 else finding.nvdb_road_reference
        pdf.cell(50, 5, road_ref, 1, 1, 'L')
        
        pdf.cell(35, 5, 'GPS Latitude:', 1, 0, 'L')
        pdf.cell(50, 5, f'{finding.gps_lat:.6f}', 1, 1, 'L')
        
        pdf.cell(35, 5, 'GPS Longitude:', 1, 0, 'L')
        pdf.cell(50, 5, f'{finding.gps_lon:.6f}', 1, 1, 'L')
        
        pdf.ln(10)
        
        # Expert assessment section
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(0, 50, 100)
        pdf.cell(0, 8, 'EXPERT ASSESSMENT', 0, 1, 'L')
        pdf.ln(2)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(0, 0, 0)
        
        # Finding type
        pdf.cell(40, 6, 'Finding Type:', 0, 0, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, finding.expert_finding_type or 'Not specified', 0, 1, 'L')
        
        # Description
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'Description:', 0, 1, 'L')
        pdf.set_font('Arial', '', 9)
        if finding.expert_description:
            # Multi-line text
            desc_lines = finding.expert_description.split('\n')
            for line in desc_lines:
                if len(line) > 80:
                    # Word wrap
                    words = line.split(' ')
                    current_line = ""
                    for word in words:
                        if len(current_line + word) > 80:
                            pdf.cell(0, 4, current_line, 0, 1, 'L')
                            current_line = word + " "
                        else:
                            current_line += word + " "
                    if current_line:
                        pdf.cell(0, 4, current_line, 0, 1, 'L')
                else:
                    pdf.cell(0, 4, line, 0, 1, 'L')
        else:
            pdf.cell(0, 4, 'No description provided', 0, 1, 'L')
        
        pdf.ln(3)
        
        # Proposed action
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'Proposed Action:', 0, 1, 'L')
        pdf.set_font('Arial', '', 9)
        if finding.expert_action_proposal:
            action_lines = finding.expert_action_proposal.split('\n')
            for line in action_lines:
                if len(line) > 80:
                    words = line.split(' ')
                    current_line = ""
                    for word in words:
                        if len(current_line + word) > 80:
                            pdf.cell(0, 4, current_line, 0, 1, 'L')
                            current_line = word + " "
                        else:
                            current_line += word + " "
                    if current_line:
                        pdf.cell(0, 4, current_line, 0, 1, 'L')
                else:
                    pdf.cell(0, 4, line, 0, 1, 'L')
        else:
            pdf.cell(0, 4, 'No action proposed', 0, 1, 'L')
        
        pdf.ln(5)
        
        # Severity assessment table
        self._add_severity_table(pdf, finding)
        
        # Add images if available
        pdf.ln(5)
        self._add_finding_images(pdf, finding)
    
    def _add_severity_table(self, pdf: TrafficSafetyReportPDF, finding: Finding):
        """Add severity assessment table"""
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(0, 50, 100)
        pdf.cell(0, 8, 'SEVERITY ASSESSMENT', 0, 1, 'L')
        
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(0, 0, 0)
        
        # Consequence table
        pdf.cell(0, 5, 'Consequence:', 0, 1, 'L')
        pdf.set_font('Arial', '', 8)
        
        consequences = ['Lav', 'Middels', 'Høy', 'Meget høy']
        cell_width = 40
        
        # Header
        pdf.cell(30, 5, '', 1, 0, 'C')
        for cons in consequences:
            pdf.cell(cell_width, 5, cons, 1, 0, 'C')
        pdf.ln()
        
        # Data row with X mark
        pdf.cell(30, 5, 'Consequence', 1, 0, 'C')
        for cons in consequences:
            mark = 'X' if finding.expert_severity_consequence == cons else ''
            pdf.cell(cell_width, 5, mark, 1, 0, 'C')
        pdf.ln()
        
        pdf.ln(3)
        
        # Probability table
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 5, 'Probability:', 0, 1, 'L')
        pdf.set_font('Arial', '', 8)
        
        probabilities = ['Svært liten', 'Liten', 'Middels', 'Stor', 'Svært stor']
        
        # Header
        pdf.cell(30, 5, '', 1, 0, 'C')
        for prob in probabilities:
            pdf.cell(32, 5, prob, 1, 0, 'C')
        pdf.ln()
        
        # Data row with X mark
        pdf.cell(30, 5, 'Probability', 1, 0, 'C')
        for prob in probabilities:
            mark = 'X' if finding.expert_severity_probability == prob else ''
            pdf.cell(32, 5, mark, 1, 0, 'C')
        pdf.ln()
    
    def _add_finding_images(self, pdf: TrafficSafetyReportPDF, finding: Finding):
        """Add images to the finding page"""
        if not finding.original_image_b64 and not finding.annotated_image_b64:
            return
        
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(0, 50, 100)
        pdf.cell(0, 8, 'IMAGES', 0, 1, 'L')
        
        current_y = pdf.get_y()
        image_width = 80
        image_height = 60
        
        # Add original image
        if finding.original_image_b64:
            try:
                # Decode base64 image
                img_data = base64.b64decode(finding.original_image_b64)
                img = Image.open(io.BytesIO(img_data))
                
                # Save temporary image file
                temp_path = f"temp_original_{finding.id}.jpg"
                img.save(temp_path, "JPEG")
                
                # Add to PDF
                pdf.image(temp_path, x=20, y=current_y, w=image_width, h=image_height)
                pdf.set_xy(20, current_y + image_height + 2)
                pdf.set_font('Arial', '', 8)
                pdf.cell(image_width, 4, 'Original Image', 0, 0, 'C')
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                print(f"⚠️ Failed to add original image: {e}")
        
        # Add annotated image
        if finding.annotated_image_b64:
            try:
                # Decode base64 image
                img_data = base64.b64decode(finding.annotated_image_b64)
                img = Image.open(io.BytesIO(img_data))
                
                # Save temporary image file
                temp_path = f"temp_annotated_{finding.id}.jpg"
                img.save(temp_path, "JPEG")
                
                # Add to PDF
                pdf.image(temp_path, x=110, y=current_y, w=image_width, h=image_height)
                pdf.set_xy(110, current_y + image_height + 2)
                pdf.set_font('Arial', '', 8)
                pdf.cell(image_width, 4, 'With Detection', 0, 0, 'C')
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                print(f"⚠️ Failed to add annotated image: {e}")
        
        # Move cursor below images
        pdf.set_xy(20, current_y + image_height + 10)
    
    def _add_summary_page(self, pdf: TrafficSafetyReportPDF, findings: List[Finding]):
        """Add summary page with statistics"""
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.set_text_color(0, 50, 100)
        pdf.cell(0, 12, 'REPORT SUMMARY', 0, 1, 'C')
        pdf.ln(10)
        
        # Statistics
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, 'INSPECTION STATISTICS', 0, 1, 'L')
        pdf.ln(3)
        
        pdf.set_font('Arial', '', 10)
        
        # Total findings
        pdf.cell(50, 6, 'Total Findings:', 0, 0, 'L')
        pdf.cell(0, 6, str(len(findings)), 0, 1, 'L')
        
        # By finding type
        finding_types = {}
        for finding in findings:
            ft = finding.expert_finding_type or 'Unspecified'
            finding_types[ft] = finding_types.get(ft, 0) + 1
        
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'By Finding Type:', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        for ft, count in finding_types.items():
            pdf.cell(20, 5, '', 0, 0)  # Indent
            pdf.cell(40, 5, f'{ft}:', 0, 0, 'L')
            pdf.cell(0, 5, str(count), 0, 1, 'L')
        
        # By severity
        severities = {}
        for finding in findings:
            sev = f"{finding.expert_severity_consequence or 'Unknown'} / {finding.expert_severity_probability or 'Unknown'}"
            severities[sev] = severities.get(sev, 0) + 1
        
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'By Severity (Consequence/Probability):', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        for sev, count in severities.items():
            pdf.cell(20, 5, '', 0, 0)  # Indent
            pdf.cell(60, 5, f'{sev}:', 0, 0, 'L')
            pdf.cell(0, 5, str(count), 0, 1, 'L')
        
        pdf.ln(10)
        
        # Recommendations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'RECOMMENDATIONS', 0, 1, 'L')
        pdf.ln(3)
        
        pdf.set_font('Arial', '', 10)
        
        # High priority items
        high_priority = [f for f in findings if f.expert_severity_consequence in ['Høy', 'Meget høy']]
        if high_priority:
            pdf.cell(0, 6, f'• {len(high_priority)} findings require immediate attention (high consequence)', 0, 1, 'L')
        
        # Action required items
        action_items = [f for f in findings if f.expert_action_proposal and f.expert_action_proposal.strip()]
        if action_items:
            pdf.cell(0, 6, f'• {len(action_items)} findings have specific action proposals', 0, 1, 'L')
        
        # Follow-up
        pdf.ln(5)
        pdf.cell(0, 6, '• Review and prioritize findings based on severity assessment', 0, 1, 'L')
        pdf.cell(0, 6, '• Schedule remedial actions for high-priority items', 0, 1, 'L')
        pdf.cell(0, 6, '• Update asset management systems with confirmed findings', 0, 1, 'L')


def main():
    """Test PDF generation"""
    print("📄 Testing PDF Report Generation...")
    
    # Create test finding
    from .database import Finding
    
    test_finding = Finding(
        id=1,
        image_path="test_image.jpg",
        timestamp=datetime.now().isoformat(),
        gps_lat=59.9139,
        gps_lon=10.7522,
        nvdb_road_reference="Fv753 S1D1 m18",
        detected_class="stop sign",
        confidence_score=0.85,
        bounding_box="[100, 100, 200, 200]",
        status="approved",
        expert_description="Stop sign partially obscured by vegetation",
        expert_action_proposal="Trim vegetation around sign",
        expert_finding_type="Avvik",
        expert_severity_consequence="Middels",
        expert_severity_probability="Stor"
    )
    
    # Generate report
    generator = TSReportGenerator()
    pdf_content = generator.generate_report([test_finding], "test_report.pdf")
    
    print(f"✅ Generated PDF report: {len(pdf_content)} bytes")
    print("📄 Test report saved as test_report.pdf")


if __name__ == "__main__":
    main()
