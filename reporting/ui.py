#!/usr/bin/env python3
"""
Reporting UI Module for Streamlit
Provides the expert review interface for validating and enriching AI findings.
"""

import streamlit as st
import folium
from streamlit_folium import folium_static
import base64
import io
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from PIL import Image, ImageDraw
import tempfile
import os

from .database import FindingsDatabase, Finding, convert_processing_results_to_findings
from .pdf_generator import TSReportGenerator


class ExpertReviewInterface:
    """Expert review interface for findings validation"""
    
    def __init__(self):
        """Initialize the expert review interface"""
        self.db = FindingsDatabase()
        self.pdf_generator = TSReportGenerator()
        
        # Initialize session state
        if 'selected_finding_id' not in st.session_state:
            st.session_state.selected_finding_id = None
        if 'findings_filter' not in st.session_state:
            st.session_state.findings_filter = 'new'
    
    def import_processing_results(self, processing_results: List[Any]) -> int:
        """
        Import processing results into findings database
        
        Args:
            processing_results: List of ImageProcessingResult objects
            
        Returns:
            Number of findings imported
        """
        findings = convert_processing_results_to_findings(processing_results)
        imported_count = 0
        
        for finding in findings:
            try:
                self.db.add_finding(finding)
                imported_count += 1
            except Exception as e:
                st.error(f"Failed to import finding: {e}")
        
        return imported_count
    
    def render_sidebar(self) -> Optional[Finding]:
        """Render sidebar with findings list and filters"""
        with st.sidebar:
            st.header("📋 Findings Review")
            
            # Import button
            if st.button("📥 Import from Processing Results", use_container_width=True):
                if hasattr(st.session_state, 'processing_results') and st.session_state.processing_results:
                    imported = self.import_processing_results(st.session_state.processing_results)
                    if imported > 0:
                        st.success(f"✅ Imported {imported} new findings")
                        st.rerun()
                    else:
                        st.warning("No new findings to import")
                else:
                    st.warning("No processing results available. Process images first.")
            
            st.divider()
            
            # Status filter
            status_filter = st.selectbox(
                "Filter by Status",
                ['new', 'approved', 'rejected'],
                index=0,
                key='findings_filter'
            )
            
            # Get findings
            findings = self.db.get_findings_by_status(status_filter)
            
            st.subheader(f"📊 {status_filter.title()} Findings ({len(findings)})")
            
            if not findings:
                st.info(f"No {status_filter} findings available")
                return None
            
            # Class filter
            all_classes = list(set(f.detected_class for f in findings))
            class_filter = st.selectbox(
                "Filter by Class",
                ['All'] + all_classes,
                index=0
            )
            
            if class_filter != 'All':
                findings = [f for f in findings if f.detected_class == class_filter]
            
            st.divider()
            
            # Findings list
            selected_finding = None
            
            for finding in findings:
                # Create finding summary card
                with st.container():
                    # Thumbnail if available
                    if finding.original_image_b64:
                        try:
                            img_data = base64.b64decode(finding.original_image_b64)
                            img = Image.open(io.BytesIO(img_data))
                            # Create thumbnail
                            img.thumbnail((60, 60))
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.image(img, width=60)
                            with col2:
                                st.write(f"**{finding.detected_class}**")
                                st.write(f"Conf: {finding.confidence_score:.2f}")
                                road_ref = finding.nvdb_road_reference[:20] + "..." if len(finding.nvdb_road_reference) > 20 else finding.nvdb_road_reference
                                st.write(f"📍 {road_ref}" if road_ref else "📍 No road ref")
                        except:
                            st.write(f"**{finding.detected_class}** (Conf: {finding.confidence_score:.2f})")
                            road_ref = finding.nvdb_road_reference[:30] + "..." if len(finding.nvdb_road_reference) > 30 else finding.nvdb_road_reference
                            st.write(f"📍 {road_ref}" if road_ref else "📍 No road ref")
                    else:
                        st.write(f"**{finding.detected_class}** (Conf: {finding.confidence_score:.2f})")
                        road_ref = finding.nvdb_road_reference[:30] + "..." if len(finding.nvdb_road_reference) > 30 else finding.nvdb_road_reference
                        st.write(f"📍 {road_ref}" if road_ref else "📍 No road ref")
                    
                    # Select button
                    if st.button(f"🔍 Review", key=f"select_{finding.id}", use_container_width=True):
                        st.session_state.selected_finding_id = finding.id
                        selected_finding = finding
                        st.rerun()
                    
                    st.divider()
            
            # Get currently selected finding
            if st.session_state.selected_finding_id:
                selected_finding = self.db.get_finding_by_id(st.session_state.selected_finding_id)
            
            return selected_finding
    
    def render_finding_details(self, finding: Optional[Finding]):
        """Render detailed view of selected finding"""
        if not finding:
            st.info("👈 Select a finding from the sidebar to review")
            return
        
        st.header(f"🔍 Finding Review: {finding.detected_class}")
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Confidence", f"{finding.confidence_score:.3f}")
        col2.metric("Status", finding.status.title())
        col3.metric("GPS Lat", f"{finding.gps_lat:.6f}")
        col4.metric("GPS Lon", f"{finding.gps_lon:.6f}")
        
        st.divider()
        
        # Image and map section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Images")
            
            if finding.original_image_b64:
                img_data = base64.b64decode(finding.original_image_b64)
                img = Image.open(io.BytesIO(img_data))
                st.image(img, caption="Original Image", use_column_width=True)
            
            if finding.annotated_image_b64:
                img_data = base64.b64decode(finding.annotated_image_b64)
                img = Image.open(io.BytesIO(img_data))
                st.image(img, caption="With Detection", use_column_width=True)
            elif finding.original_image_b64 and finding.bounding_box:
                # Draw bounding box manually
                try:
                    bbox = json.loads(finding.bounding_box)
                    if len(bbox) == 4:
                        img_data = base64.b64decode(finding.original_image_b64)
                        img = Image.open(io.BytesIO(img_data))
                        draw = ImageDraw.Draw(img)
                        x1, y1, x2, y2 = bbox
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                        st.image(img, caption="With Bounding Box", use_column_width=True)
                except:
                    st.info("Could not draw bounding box")
        
        with col2:
            st.subheader("🗺️ Location")
            
            # Create map
            m = folium.Map(location=[finding.gps_lat, finding.gps_lon], zoom_start=16)
            folium.Marker(
                location=[finding.gps_lat, finding.gps_lon],
                popup=f"{finding.detected_class}\nConf: {finding.confidence_score:.2f}",
                icon=folium.Icon(color='red', icon='camera')
            ).add_to(m)
            
            folium_static(m, width=400, height=300)
            
            # Location details
            st.write(f"**Road Reference:** {finding.nvdb_road_reference or 'Not available'}")
            st.write(f"**Image:** {os.path.basename(finding.image_path)}")
            st.write(f"**Timestamp:** {finding.timestamp[:19]}")
        
        st.divider()
        
        # Expert assessment form
        st.subheader("📝 Expert Assessment")
        
        # Form for expert input
        with st.form(f"expert_form_{finding.id}"):
            col1, col2 = st.columns(2)
            
            with col1:
                finding_type = st.selectbox(
                    "Finding Type",
                    ["", "Avvik", "Feil", "Merknad"],
                    index=["", "Avvik", "Feil", "Merknad"].index(finding.expert_finding_type) if finding.expert_finding_type else 0
                )
                
                severity_consequence = st.selectbox(
                    "Severity - Consequence",
                    ["", "Lav", "Middels", "Høy", "Meget høy"],
                    index=["", "Lav", "Middels", "Høy", "Meget høy"].index(finding.expert_severity_consequence) if finding.expert_severity_consequence else 0
                )
            
            with col2:
                severity_probability = st.selectbox(
                    "Severity - Probability", 
                    ["", "Svært liten", "Liten", "Middels", "Stor", "Svært stor"],
                    index=["", "Svært liten", "Liten", "Middels", "Stor", "Svært stor"].index(finding.expert_severity_probability) if finding.expert_severity_probability else 0
                )
                
                # NVDB Road Reference (editable)
                nvdb_road_ref = st.text_input(
                    "NVDB Road Reference",
                    value=finding.nvdb_road_reference,
                    help="Edit or add road reference information"
                )
            
            # Description
            description = st.text_area(
                "Description",
                value=finding.expert_description,
                height=100,
                help="Detailed description of the finding"
            )
            
            # Action proposal
            action_proposal = st.text_area(
                "Proposed Action",
                value=finding.expert_action_proposal,
                height=100,
                help="Recommended action to address this finding"
            )
            
            # Form buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                save_submitted = st.form_submit_button("💾 Save Changes", type="secondary", use_container_width=True)
            
            with col2:
                approve_submitted = st.form_submit_button("✅ Approve", type="primary", use_container_width=True)
            
            with col3:
                reject_submitted = st.form_submit_button("❌ Reject", use_container_width=True)
        
        # Handle form submission
        if save_submitted or approve_submitted or reject_submitted:
            # Update finding with form data
            finding.expert_finding_type = finding_type
            finding.expert_severity_consequence = severity_consequence
            finding.expert_severity_probability = severity_probability
            finding.nvdb_road_reference = nvdb_road_ref
            finding.expert_description = description
            finding.expert_action_proposal = action_proposal
            
            if approve_submitted:
                finding.status = "approved"
                st.success("✅ Finding approved!")
            elif reject_submitted:
                finding.status = "rejected"
                st.warning("❌ Finding rejected")
            else:
                st.success("💾 Changes saved!")
            
            # Update in database
            self.db.update_finding(finding)
            
            # Refresh the page
            st.rerun()
    
    def render_report_generation(self):
        """Render PDF report generation section"""
        st.header("📄 Report Generation")
        
        # Get approved findings
        approved_findings = self.db.get_findings_by_status("approved")
        
        if not approved_findings:
            st.warning("No approved findings available for reporting.")
            return
        
        st.success(f"✅ {len(approved_findings)} approved findings ready for reporting")
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        
        # By finding type
        finding_types = {}
        for finding in approved_findings:
            ft = finding.expert_finding_type or 'Unspecified'
            finding_types[ft] = finding_types.get(ft, 0) + 1
        
        col1.subheader("By Type")
        for ft, count in finding_types.items():
            col1.write(f"• **{ft}**: {count}")
        
        # By detection class
        detection_classes = {}
        for finding in approved_findings:
            cls = finding.detected_class
            detection_classes[cls] = detection_classes.get(cls, 0) + 1
        
        col2.subheader("By Object Class")
        for cls, count in detection_classes.items():
            col2.write(f"• **{cls}**: {count}")
        
        # By severity
        severities = {}
        for finding in approved_findings:
            sev = f"{finding.expert_severity_consequence or 'Unknown'}"
            severities[sev] = severities.get(sev, 0) + 1
        
        col3.subheader("By Severity")
        for sev, count in severities.items():
            col3.write(f"• **{sev}**: {count}")
        
        st.divider()
        
        # Report generation
        if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating PDF report..."):
                    # Generate PDF
                    pdf_bytes = self.pdf_generator.generate_report(approved_findings)
                    
                    # Create download button
                    filename = f"traffic_safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    
                    st.download_button(
                        label="💾 Download PDF Report",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                    
                    st.success("✅ PDF report generated successfully!")
                    
            except Exception as e:
                st.error(f"❌ Failed to generate report: {e}")
        
        # Show approved findings table
        if st.checkbox("📋 Show Approved Findings Details"):
            st.subheader("Approved Findings")
            
            findings_data = []
            for finding in approved_findings:
                findings_data.append({
                    'ID': finding.id,
                    'Image': os.path.basename(finding.image_path),
                    'Class': finding.detected_class,
                    'Confidence': f"{finding.confidence_score:.3f}",
                    'Type': finding.expert_finding_type or 'Not specified',
                    'Severity': f"{finding.expert_severity_consequence or 'Unknown'} / {finding.expert_severity_probability or 'Unknown'}",
                    'GPS': f"{finding.gps_lat:.6f}, {finding.gps_lon:.6f}",
                    'Road Ref': finding.nvdb_road_reference[:30] + "..." if len(finding.nvdb_road_reference) > 30 else finding.nvdb_road_reference
                })
            
            st.dataframe(findings_data, use_container_width=True, hide_index=True)
    
    def render_statistics(self):
        """Render findings statistics"""
        st.header("📊 Findings Statistics")
        
        # Get summary
        summary = self.db.get_findings_summary()
        
        if not summary:
            st.info("No findings in database")
            return
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Findings", sum(summary.values()))
        col2.metric("New", summary.get('new', 0))
        col3.metric("Approved", summary.get('approved', 0))
        col4.metric("Rejected", summary.get('rejected', 0))
        
        st.divider()
        
        # Get all findings for detailed analysis
        all_findings = []
        for status in ['new', 'approved', 'rejected']:
            all_findings.extend(self.db.get_findings_by_status(status))
        
        if all_findings:
            # Detection class distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Detection Classes")
                class_counts = {}
                for finding in all_findings:
                    cls = finding.detected_class
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• **{cls}**: {count}")
            
            with col2:
                st.subheader("⚖️ Status Distribution")
                for status, count in summary.items():
                    percentage = (count / sum(summary.values())) * 100
                    st.write(f"• **{status.title()}**: {count} ({percentage:.1f}%)")
        
        # Database management
        st.divider()
        st.subheader("🗄️ Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clear Rejected Findings", help="Remove all rejected findings from database"):
                rejected_findings = self.db.get_findings_by_status("rejected")
                for finding in rejected_findings:
                    if finding.id is not None:
                        self.db.delete_finding(finding.id)
                st.success(f"✅ Removed {len(rejected_findings)} rejected findings")
                st.rerun()
        
        with col2:
            st.write(f"**Database file**: {self.db.db_path}")
            if os.path.exists(self.db.db_path):
                file_size = os.path.getsize(self.db.db_path) / 1024  # KB
                st.write(f"**File size**: {file_size:.1f} KB")
    
    def run(self):
        """Run the expert review interface"""
        st.set_page_config(
            page_title="Expert Review - RoadSight AI",
            page_icon="👨‍💼",
            layout="wide"
        )
        
        st.title("👨‍💼 Expert Review Interface")
        st.markdown("**Review, validate, and enrich AI findings for official reporting**")
        st.markdown("---")
        
        # Render sidebar and get selected finding
        selected_finding = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Finding Review", "📄 Report Generation", "📊 Statistics"])
        
        with tab1:
            self.render_finding_details(selected_finding)
        
        with tab2:
            self.render_report_generation()
        
        with tab3:
            self.render_statistics()


def main():
    """Main entry point for expert review interface"""
    interface = ExpertReviewInterface()
    interface.run()


if __name__ == "__main__":
    main()
