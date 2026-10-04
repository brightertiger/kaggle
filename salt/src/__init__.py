__version__ = "1.0.0"
__author__ = "Ujjwal Singh Rao"

from .core.config import Config
from .pipeline.pipeline import SaltSegmentationPipeline

__all__ = ["Config", "SaltSegmentationPipeline"]
