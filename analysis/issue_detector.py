#!/usr/bin/env python3
"""
Issue Detection and Analysis
Detects vegetation obscuration and other issues using IoU analysis.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Issue:
    """Detected issue"""
    issue_type: str
    affected_object: str
    severity: str  # low, medium, high
    confidence: float
    details: Dict
    recommendations: List[str]


class IssueAnalyzer:
    """Analyzes detection results for various issues"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize issue analyzer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.iou_threshold = self.config.get("iou_threshold", 0.15)
        
        print(f"🔍 Issue Analyzer initialized (IoU threshold: {self.iou_threshold})")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return {"iou_threshold": 0.15}
    
    def analyze_objects(self, detections: List[Dict], 
                       depth_map: Optional[np.ndarray] = None,
                       source_type: str = "2d_image") -> List[Issue]:
        """
        Analyze detected objects for issues
        
        Args:
            detections: List of detected objects
            depth_map: Optional depth map for depth analysis
            source_type: Type of data source (2d_image, 3d_pointcloud)
            
        Returns:
            List of detected issues
        """
        issues = []
        
        if source_type == "2d_image":
            issues.extend(self._analyze_2d_obscuration(detections))
            
            if depth_map is not None:
                issues.extend(self._analyze_depth_issues(detections, depth_map))
        
        elif source_type == "3d_pointcloud":
            # Future: 3D point cloud analysis
            issues.extend(self._analyze_3d_proximity(detections))
        
        # Add general visibility issues
        issues.extend(self._analyze_visibility_issues(detections))
        
        return issues
    
    def _analyze_2d_obscuration(self, detections: List[Dict]) -> List[Issue]:
        """
        Analyze 2D bounding box overlaps for obscuration
        
        Args:
            detections: List of detected objects
            
        Returns:
            List of obscuration issues
        """
        issues = []
        
        # Separate signs and vegetation
        signs = [d for d in detections if "sign" in d["class"]]
        vegetation = [d for d in detections if d["class"] == "vegetation"]
        
        print(f"🔍 Analyzing {len(signs)} signs vs {len(vegetation)} vegetation objects")
        
        for sign in signs:
            for veg in vegetation:
                iou = self._calculate_iou(sign["box"], veg["box"])
                
                if iou > self.iou_threshold:
                    severity = self._determine_severity(iou)
                    
                    issue = Issue(
                        issue_type="vegetation_obscuration",
                        affected_object=sign["class"],
                        severity=severity,
                        confidence=min(iou / self.iou_threshold, 1.0),
                        details={
                            "iou": iou,
                            "sign_confidence": sign["confidence"],
                            "vegetation_confidence": veg["confidence"],
                            "sign_bbox": sign["box"],
                            "vegetation_bbox": veg["box"]
                        },
                        recommendations=self._get_obscuration_recommendations(severity, iou)
                    )
                    
                    issues.append(issue)
        
        return issues
    
    def _analyze_depth_issues(self, detections: List[Dict], 
                             depth_map: np.ndarray) -> List[Issue]:
        """
        Analyze depth-based issues
        
        Args:
            detections: List of detected objects
            depth_map: Depth map
            
        Returns:
            List of depth-related issues
        """
        issues = []
        
        # Import depth estimator for object depth calculation
        try:
            from detection.depth_estimation import DepthEstimator
            depth_estimator = DepthEstimator()
            
            signs = [d for d in detections if "sign" in d["class"]]
            vegetation = [d for d in detections if d["class"] == "vegetation"]
            
            for sign in signs:
                sign_depth = depth_estimator.get_object_depth(depth_map, sign["box"])
                
                for veg in vegetation:
                    veg_depth = depth_estimator.get_object_depth(depth_map, veg["box"])
                    
                    if sign_depth and veg_depth:
                        # Check if vegetation is significantly in front of sign
                        if veg_depth > sign_depth + 0.1:  # Threshold for "in front"
                            
                            issue = Issue(
                                issue_type="depth_obscuration",
                                affected_object=sign["class"],
                                severity="medium",
                                confidence=0.8,
                                details={
                                    "sign_depth": sign_depth,
                                    "vegetation_depth": veg_depth,
                                    "depth_difference": veg_depth - sign_depth
                                },
                                recommendations=[
                                    "Vegetation appears to be in front of sign",
                                    "Consider trimming vegetation for better visibility"
                                ]
                            )
                            
                            issues.append(issue)
        
        except ImportError:
            print("⚠️  Depth estimation not available")
        
        return issues
    
    def _analyze_3d_proximity(self, detections: List[Dict]) -> List[Issue]:
        """
        Analyze 3D point cloud proximity (future implementation)
        
        Args:
            detections: List of detected objects with 3D information
            
        Returns:
            List of proximity issues
        """
        # Future: Implement 3D proximity analysis
        # This would analyze point cloud clusters for:
        # - Vegetation growing too close to signs
        # - Physical clearance violations
        # - Road encroachment
        
        return []
    
    def _analyze_visibility_issues(self, detections: List[Dict]) -> List[Issue]:
        """
        Analyze general visibility issues
        
        Args:
            detections: List of detected objects
            
        Returns:
            List of visibility issues
        """
        issues = []
        
        for detection in detections:
            if "sign" in detection["class"]:
                # Check if sign detection confidence is low
                if detection["confidence"] < 0.7:
                    issue = Issue(
                        issue_type="low_visibility",
                        affected_object=detection["class"],
                        severity="medium",
                        confidence=1.0 - detection["confidence"],
                        details={
                            "detection_confidence": detection["confidence"],
                            "reason": "Low detection confidence may indicate visibility issues"
                        },
                        recommendations=[
                            "Sign may be partially obscured or damaged",
                            "Consider manual inspection",
                            "Check for weather effects or lighting conditions"
                        ]
                    )
                    issues.append(issue)
        
        return issues
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes
        
        Args:
            box1: First bounding box (center_x, center_y, width, height)
            box2: Second bounding box
            
        Returns:
            IoU value between 0 and 1
        """
        try:
            # Convert center format to corner format
            x1_1 = box1["center_x"] - box1["width"] / 2
            y1_1 = box1["center_y"] - box1["height"] / 2
            x2_1 = box1["center_x"] + box1["width"] / 2
            y2_1 = box1["center_y"] + box1["height"] / 2
            
            x1_2 = box2["center_x"] - box2["width"] / 2
            y1_2 = box2["center_y"] - box2["height"] / 2
            x2_2 = box2["center_x"] + box2["width"] / 2
            y2_2 = box2["center_y"] + box2["height"] / 2
            
            # Calculate intersection
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate areas
            area1 = box1["width"] * box1["height"]
            area2 = box2["width"] * box2["height"]
            
            # Calculate union
            union = area1 + area2 - intersection
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception as e:
            print(f"❌ IoU calculation failed: {e}")
            return 0.0
    
    def _determine_severity(self, iou: float) -> str:
        """
        Determine issue severity based on IoU
        
        Args:
            iou: Intersection over Union value
            
        Returns:
            Severity level
        """
        if iou > 0.5:
            return "high"
        elif iou > 0.3:
            return "medium"
        else:
            return "low"
    
    def _get_obscuration_recommendations(self, severity: str, iou: float) -> List[str]:
        """
        Get recommendations based on obscuration severity
        
        Args:
            severity: Issue severity
            iou: IoU value
            
        Returns:
            List of recommendations
        """
        base_recommendations = [
            f"Vegetation obscuration detected (IoU: {iou:.3f})",
            "Sign visibility may be compromised"
        ]
        
        if severity == "high":
            base_recommendations.extend([
                "🚨 URGENT: Immediate vegetation trimming required",
                "Sign is significantly obscured and may not be visible to drivers",
                "Safety hazard - prioritize for maintenance"
            ])
        elif severity == "medium":
            base_recommendations.extend([
                "⚠️  Moderate obscuration - schedule vegetation maintenance",
                "Monitor for continued growth",
                "Consider seasonal trimming schedule"
            ])
        else:
            base_recommendations.extend([
                "💡 Minor obscuration detected",
                "Include in routine maintenance schedule",
                "Monitor during vegetation growth seasons"
            ])
        
        return base_recommendations
    
    def generate_issue_report(self, issues: List[Issue]) -> Dict:
        """
        Generate comprehensive issue report
        
        Args:
            issues: List of detected issues
            
        Returns:
            Report dictionary
        """
        report = {
            "summary": {
                "total_issues": len(issues),
                "high_severity": len([i for i in issues if i.severity == "high"]),
                "medium_severity": len([i for i in issues if i.severity == "medium"]),
                "low_severity": len([i for i in issues if i.severity == "low"])
            },
            "issue_types": {},
            "affected_objects": {},
            "recommendations": [],
            "details": []
        }
        
        # Analyze issue types
        for issue in issues:
            issue_type = issue.issue_type
            report["issue_types"][issue_type] = report["issue_types"].get(issue_type, 0) + 1
            
            affected_obj = issue.affected_object
            report["affected_objects"][affected_obj] = report["affected_objects"].get(affected_obj, 0) + 1
            
            # Add to details
            report["details"].append({
                "type": issue.issue_type,
                "object": issue.affected_object,
                "severity": issue.severity,
                "confidence": issue.confidence,
                "details": issue.details,
                "recommendations": issue.recommendations
            })
        
        # Generate priority recommendations
        high_issues = [i for i in issues if i.severity == "high"]
        if high_issues:
            report["recommendations"].append("🚨 URGENT: Address high-severity issues immediately")
        
        medium_issues = [i for i in issues if i.severity == "medium"]
        if medium_issues:
            report["recommendations"].append("⚠️  Schedule maintenance for medium-severity issues")
        
        return report


def main():
    """Test issue analyzer"""
    print("🔍 Testing Issue Analyzer...")
    
    # Initialize analyzer
    analyzer = IssueAnalyzer()
    
    # Create sample detections with overlapping boxes
    sample_detections = [
        {
            "class": "speed_sign_80",
            "confidence": 0.9,
            "box": {"center_x": 640, "center_y": 360, "width": 100, "height": 100}
        },
        {
            "class": "vegetation",
            "confidence": 0.85,
            "box": {"center_x": 660, "center_y": 380, "width": 80, "height": 120}  # Overlapping
        },
        {
            "class": "stop_sign",
            "confidence": 0.6,  # Low confidence
            "box": {"center_x": 400, "center_y": 300, "width": 80, "height": 80}
        }
    ]
    
    # Analyze issues
    issues = analyzer.analyze_objects(sample_detections)
    
    # Display results
    print(f"\n📊 Analysis Results:")
    print(f"Found {len(issues)} issues")
    
    for i, issue in enumerate(issues):
        print(f"\n{i+1}. {issue.issue_type.upper()}")
        print(f"   Object: {issue.affected_object}")
        print(f"   Severity: {issue.severity}")
        print(f"   Confidence: {issue.confidence:.2f}")
        print(f"   Recommendations:")
        for rec in issue.recommendations[:2]:  # Show first 2 recommendations
            print(f"     • {rec}")
    
    # Generate report
    report = analyzer.generate_issue_report(issues)
    print(f"\n📈 Report Summary:")
    print(f"  Total issues: {report['summary']['total_issues']}")
    print(f"  High severity: {report['summary']['high_severity']}")
    print(f"  Medium severity: {report['summary']['medium_severity']}")
    print(f"  Low severity: {report['summary']['low_severity']}")


if __name__ == "__main__":
    main()
