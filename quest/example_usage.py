#!/usr/bin/env python3

"""
Example usage of the Question Understanding Pipeline

This script demonstrates how to use the refactored pipeline for training
and inference on the Google AI Question Understanding competition.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.pipeline import QuestionUnderstandingPipeline


def main():
    """Main example function"""
    
    # Initialize configuration
    config = Config()
    
    # You can customize the config here
    config.batch_size = 2
    config.learning_rate = 1e-5
    config.num_epochs = 6
    config.n_folds = 5
    config.data_dir = "data/"
    config.model_dir = "models/"
    config.output_dir = "outputs/"
    
    # Initialize pipeline
    pipeline = QuestionUnderstandingPipeline(config)
    
    # Example 1: Train all folds
    print("Example 1: Training all folds")
    print("-" * 40)
    
    # Uncomment to run training
    # fold_results = pipeline.train_all_folds(
    #     data_path=config.data_dir,
    #     model_path=config.model_dir
    # )
    # print(f"Training completed. Results: {fold_results}")
    
    # Example 2: Train single fold
    print("\nExample 2: Training single fold")
    print("-" * 40)
    
    # Uncomment to run single fold training
    # val_loss = pipeline.train_fold(
    #     fold=1,
    #     data_path=config.data_dir,
    #     model_path=config.model_dir
    # )
    # print(f"Fold 1 training completed. Validation loss: {val_loss:.4f}")
    
    # Example 3: Evaluate cross-validation performance
    print("\nExample 3: Cross-validation evaluation")
    print("-" * 40)
    
    # Uncomment to run evaluation
    # cv_results = pipeline.evaluate_cv_performance(
    #     data_path=config.data_dir,
    #     model_path=config.model_dir
    # )
    # print(f"CV evaluation completed. Results: {cv_results}")
    
    # Example 4: Generate predictions on test data
    print("\nExample 4: Test inference")
    print("-" * 40)
    
    # Uncomment to run inference
    # predictions = pipeline.inference(
    #     data_path=config.data_dir,
    #     model_path=config.model_dir,
    #     output_path=config.output_dir
    # )
    # print(f"Inference completed. Predictions shape: {predictions.shape}")
    
    # Example 5: Hyperparameter search
    print("\nExample 5: Hyperparameter search")
    print("-" * 40)
    
    param_grid = {
        'learning_rate': [5e-6, 1e-5, 2e-5],
        'batch_size': [1, 2, 4],
        'num_epochs': [4, 6, 8]
    }
    
    # Uncomment to run hyperparameter search
    # search_results = pipeline.hyperparameter_search(
    #     data_path=config.data_dir,
    #     model_path=config.model_dir,
    #     param_grid=param_grid
    # )
    # print(f"Hyperparameter search completed.")
    # print(f"Best params: {search_results['best_params']}")
    # print(f"Best score: {search_results['best_score']:.4f}")
    
    # Example 6: Model architecture comparison
    print("\nExample 6: Model architecture comparison")
    print("-" * 40)
    
    model_types = ["question_understanding", "dual_bert"]
    
    # Uncomment to run model comparison
    # comparison_results = pipeline.create_model_comparison(
    #     data_path=config.data_dir,
    #     model_path=config.model_dir,
    #     model_types=model_types
    # )
    # print(f"Model comparison completed.")
    
    print("\nAll examples completed!")
    print("Uncomment the desired sections to run the actual training/evaluation.")


if __name__ == "__main__":
    main()
