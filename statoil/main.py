import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import IcebergPipeline

if __name__ == "__main__":
    pipeline = IcebergPipeline()
    pipeline.run_full_pipeline()
