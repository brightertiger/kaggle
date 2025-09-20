import torch
from src.pipeline import JigsawPipeline

def example_data_preparation():
    pipeline = JigsawPipeline()
    
    print("Preparing raw data...")
    pipeline.prepare_data('../../data')
    
    print("Generating embeddings...")
    pipeline.generate_embeddings('../../data')
    
    print("Generating adversarial data...")
    pipeline.generate_adversarial_data('../../data')

def example_training():
    pipeline = JigsawPipeline()
    
    print("Training Version 1 - Fold 0")
    pipeline.train_version1(subset=0, load_pretrained=False)
    
    print("Training Version 2 - Fold 0 (with Version 1 weights)")
    pipeline.train_version2(subset=0, load_from_version1=True)

def example_scoring():
    pipeline = JigsawPipeline()
    
    print("Scoring all models...")
    pipeline.scoring_pipeline.score_all_models('../../data/process/foreign/test_english.csv')

def example_full_pipeline():
    pipeline = JigsawPipeline()
    
    print("Running complete pipeline...")
    final_predictions = pipeline.run_full_pipeline(
        data_dir='../../data',
        test_path='../../data/process/foreign/test_english.csv'
    )
    
    print(f"Final predictions shape: {final_predictions.shape}")
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    example_data_preparation()
