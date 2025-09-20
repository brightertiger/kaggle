__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

from .core.config import Config
from .pipeline.pipeline import SaltSegmentationPipeline

__all__ = ["Config", "SaltSegmentationPipeline"]
