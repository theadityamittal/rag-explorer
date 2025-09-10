"""Utils module compatibility bridge - bridges old utils functions to new or existing implementations."""

from ._path_helper import ensure_src_path

def init_clean_cli():
    """Initialize clean CLI environment (warnings suppression)."""
    try:
        ensure_src_path()
        from src.utils.warnings_suppressor import init_clean_cli as old_init_clean_cli
        return old_init_clean_cli()
    except ImportError:
        # Skip if not available
        pass


class Meter:
    """Meter class for tracking performance metrics."""
    
    def __init__(self):
        try:
            ensure_src_path()
            from src.utils.metrics import Meter as OldMeter
            self._meter = OldMeter()
        except ImportError:
            # Simple fallback implementation
            self._observations = []
    
    def observe(self, value: float):
        """Record an observation."""
        try:
            self._meter.observe(value)
        except AttributeError:
            # Fallback implementation
            self._observations.append(value)
    
    def summary(self):
        """Get summary statistics."""
        try:
            return self._meter.summary()
        except AttributeError:
            # Fallback implementation
            if not self._observations:
                return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
            return {
                "count": len(self._observations),
                "avg": sum(self._observations) / len(self._observations),
                "min": min(self._observations),
                "max": max(self._observations),
            }