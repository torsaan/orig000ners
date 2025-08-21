#!/usr/bin/env python3
"""
Expert Review Standalone Application
Dedicated interface for expert review and PDF report generation.
"""

from reporting.ui import ExpertReviewInterface


def main():
    """Main entry point for standalone expert review application"""
    interface = ExpertReviewInterface()
    interface.run()


if __name__ == "__main__":
    main()
