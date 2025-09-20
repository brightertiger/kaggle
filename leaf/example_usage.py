import numpy as np
from src.pipeline import CassavaPipeline

def main():
    pipeline = CassavaPipeline()
    
    print("=== Cassava Leaf Disease Classification Pipeline ===\n")
    
    print("1. Preparing data...")
    pipeline.prepare_data('../../data')
    
    print("\n2. Training individual models...")
    for version in ['version0', 'version1', 'version2']:
        for fold in range(2):  # Train only 2 folds for demo
            print(f"Training {version} fold {fold}")
            pipeline.train_model(version, fold)
    
    print("\n3. Scoring models...")
    for version in ['version0', 'version1', 'version2']:
        pipeline.score_model(version, '../../data/test.csv')
    
    print("\n4. Creating ensemble...")
    ensemble_results = pipeline.create_ensemble()
    
    print("\n=== Pipeline completed! ===")
    print(f"Final ensemble accuracy: {np.mean([r['model_acc'] for r in ensemble_results]):.4f}")

if __name__ == '__main__':
    main()
