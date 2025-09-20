from src.pipeline import IcebergPipeline

if __name__ == "__main__":
    pipeline = IcebergPipeline()
    pipeline.run_full_pipeline()
