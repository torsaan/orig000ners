#!/usr/bin/env python3
"""
Database module for storing and managing findings from AI model detection.
Handles the lifecycle of findings from detection to expert review to reporting.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class Finding:
    """Data model for a finding detected by the AI model"""
    id: Optional[int] = None
    image_path: str = ""
    timestamp: str = ""
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    nvdb_road_reference: str = ""
    detected_class: str = ""
    confidence_score: float = 0.0
    bounding_box: str = ""  # JSON string of bbox coordinates
    segmentation_mask: str = ""  # JSON string if available
    status: str = "new"  # 'new', 'approved', 'rejected'
    expert_description: str = ""
    expert_action_proposal: str = ""
    expert_finding_type: str = ""  # 'Avvik', 'Feil', 'Merknad'
    expert_severity_consequence: str = ""  # Severity - Consequence
    expert_severity_probability: str = ""  # Severity - Probability
    created_at: str = ""
    updated_at: str = ""
    original_image_b64: str = ""  # Base64 encoded original image
    annotated_image_b64: str = ""  # Base64 encoded annotated image


class FindingsDatabase:
    """Database manager for findings"""
    
    def __init__(self, db_path: str = "findings.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    gps_lat REAL,
                    gps_lon REAL,
                    nvdb_road_reference TEXT,
                    detected_class TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    bounding_box TEXT,
                    segmentation_mask TEXT,
                    status TEXT DEFAULT 'new',
                    expert_description TEXT,
                    expert_action_proposal TEXT,
                    expert_finding_type TEXT,
                    expert_severity_consequence TEXT,
                    expert_severity_probability TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    original_image_b64 TEXT,
                    annotated_image_b64 TEXT
                )
            """)
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic closing"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def add_finding(self, finding: Finding) -> int:
        """Add a new finding to the database"""
        now = datetime.now().isoformat()
        finding.created_at = now
        finding.updated_at = now
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO findings (
                    image_path, timestamp, gps_lat, gps_lon, nvdb_road_reference,
                    detected_class, confidence_score, bounding_box, segmentation_mask,
                    status, expert_description, expert_action_proposal, expert_finding_type,
                    expert_severity_consequence, expert_severity_probability,
                    created_at, updated_at, original_image_b64, annotated_image_b64
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                finding.image_path, finding.timestamp, finding.gps_lat, finding.gps_lon,
                finding.nvdb_road_reference, finding.detected_class, finding.confidence_score,
                finding.bounding_box, finding.segmentation_mask, finding.status,
                finding.expert_description, finding.expert_action_proposal, finding.expert_finding_type,
                finding.expert_severity_consequence, finding.expert_severity_probability,
                finding.created_at, finding.updated_at, finding.original_image_b64,
                finding.annotated_image_b64
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_findings_by_status(self, status: str = "new") -> List[Finding]:
        """Get all findings with specified status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM findings WHERE status = ? ORDER BY created_at DESC", (status,))
            rows = cursor.fetchall()
            return [self._row_to_finding(row) for row in rows]
    
    def get_finding_by_id(self, finding_id: int) -> Optional[Finding]:
        """Get a specific finding by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM findings WHERE id = ?", (finding_id,))
            row = cursor.fetchone()
            return self._row_to_finding(row) if row else None
    
    def update_finding(self, finding: Finding):
        """Update an existing finding"""
        finding.updated_at = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE findings SET
                    image_path = ?, timestamp = ?, gps_lat = ?, gps_lon = ?,
                    nvdb_road_reference = ?, detected_class = ?, confidence_score = ?,
                    bounding_box = ?, segmentation_mask = ?, status = ?,
                    expert_description = ?, expert_action_proposal = ?, expert_finding_type = ?,
                    expert_severity_consequence = ?, expert_severity_probability = ?,
                    updated_at = ?, original_image_b64 = ?, annotated_image_b64 = ?
                WHERE id = ?
            """, (
                finding.image_path, finding.timestamp, finding.gps_lat, finding.gps_lon,
                finding.nvdb_road_reference, finding.detected_class, finding.confidence_score,
                finding.bounding_box, finding.segmentation_mask, finding.status,
                finding.expert_description, finding.expert_action_proposal, finding.expert_finding_type,
                finding.expert_severity_consequence, finding.expert_severity_probability,
                finding.updated_at, finding.original_image_b64, finding.annotated_image_b64,
                finding.id
            ))
            conn.commit()
    
    def delete_finding(self, finding_id: int):
        """Delete a finding by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM findings WHERE id = ?", (finding_id,))
            conn.commit()
    
    def get_findings_summary(self) -> Dict[str, int]:
        """Get summary statistics of findings"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM findings 
                GROUP BY status
            """)
            rows = cursor.fetchall()
            return {row['status']: row['count'] for row in rows}
    
    def get_findings_by_class(self, detected_class: str) -> List[Finding]:
        """Get findings filtered by detected class"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM findings WHERE detected_class = ? ORDER BY created_at DESC", 
                          (detected_class,))
            rows = cursor.fetchall()
            return [self._row_to_finding(row) for row in rows]
    
    def _row_to_finding(self, row) -> Finding:
        """Convert database row to Finding object"""
        return Finding(
            id=row['id'],
            image_path=row['image_path'],
            timestamp=row['timestamp'],
            gps_lat=row['gps_lat'],
            gps_lon=row['gps_lon'],
            nvdb_road_reference=row['nvdb_road_reference'] or "",
            detected_class=row['detected_class'],
            confidence_score=row['confidence_score'],
            bounding_box=row['bounding_box'] or "",
            segmentation_mask=row['segmentation_mask'] or "",
            status=row['status'],
            expert_description=row['expert_description'] or "",
            expert_action_proposal=row['expert_action_proposal'] or "",
            expert_finding_type=row['expert_finding_type'] or "",
            expert_severity_consequence=row['expert_severity_consequence'] or "",
            expert_severity_probability=row['expert_severity_probability'] or "",
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            original_image_b64=row['original_image_b64'] or "",
            annotated_image_b64=row['annotated_image_b64'] or ""
        )


def convert_processing_results_to_findings(processing_results: List[Any], 
                                          nvdb_road_refs: Dict[str, str] = None) -> List[Finding]:
    """
    Convert ImageProcessingResult objects to Finding objects
    
    Args:
        processing_results: List of ImageProcessingResult objects from main processing
        nvdb_road_refs: Optional mapping of image paths to road references
        
    Returns:
        List of Finding objects
    """
    findings = []
    nvdb_road_refs = nvdb_road_refs or {}
    
    for result in processing_results:
        # Only create findings for images with GPS coordinates
        if not result.processing_metadata.get("has_gps", False):
            continue
            
        # Create a finding for each detection
        for detection in result.detections:
            finding = Finding(
                image_path=result.image_path,
                timestamp=result.processing_metadata.get("processed_at", datetime.now().isoformat()),
                gps_lat=result.gps_coords[0],
                gps_lon=result.gps_coords[1],
                nvdb_road_reference=nvdb_road_refs.get(result.image_path, ""),
                detected_class=detection["class"],
                confidence_score=detection["confidence"],
                bounding_box=json.dumps(detection.get("bbox_xyxy", [])),
                segmentation_mask="",
                status="new",
                original_image_b64=getattr(result, 'original_image_b64', ''),
                annotated_image_b64=getattr(result, 'annotated_image_b64', '')
            )
            findings.append(finding)
    
    return findings


def main():
    """Test the database functionality"""
    print("🗄️ Testing Findings Database...")
    
    # Initialize database
    db = FindingsDatabase("test_findings.db")
    
    # Create test finding
    test_finding = Finding(
        image_path="test_image.jpg",
        timestamp=datetime.now().isoformat(),
        gps_lat=59.9139,
        gps_lon=10.7522,
        nvdb_road_reference="Fv753 S1D1 m18",
        detected_class="stop sign",
        confidence_score=0.85,
        bounding_box="[100, 100, 200, 200]",
        status="new"
    )
    
    # Add to database
    finding_id = db.add_finding(test_finding)
    print(f"✅ Added finding with ID: {finding_id}")
    
    # Retrieve findings
    new_findings = db.get_findings_by_status("new")
    print(f"📊 Found {len(new_findings)} new findings")
    
    # Get summary
    summary = db.get_findings_summary()
    print(f"📈 Summary: {summary}")
    
    # Clean up test database
    os.remove("test_findings.db")
    print("🧹 Cleaned up test database")


if __name__ == "__main__":
    main()
