"""Helper to ensure old src modules can be imported from compatibility bridges."""

import sys
import os


def ensure_src_path():
    """Add project root to Python path so we can import from src.* modules."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)