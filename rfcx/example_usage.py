import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.pipeline import run_full_pipeline, create_folds, resample_audio, ensemble_predictions
from src.trainer import train_model
from src.predictor import generate_predictions

def example_basic_training():
    config = Config()
    
    print("Example: Basic Training Pipeline")
    print("=" * 50)
    
    train_model(config, model_type="resnet")

def example_prediction():
    config = Config()
    
    print("Example: Generate Predictions")
    print("=" * 50)
    
    generate_predictions(config, model_type="resnet", apply_tta=False)

def example_tta_prediction():
    config = Config()
    
    print("Example: TTA Predictions")
    print("=" * 50)
    
    generate_predictions(config, model_type="resnet", apply_tta=True, 
                        output_name="resnet_tta_predictions")

def example_ensemble():
    print("Example: Ensemble Predictions")
    print("=" * 50)
    
    prediction_files = [
        "predictions/resnet_predictions.csv",
        "predictions/resnet_tta_predictions.csv"
    ]
    
    ensemble_predictions(prediction_files, "predictions/ensemble_predictions.csv")

def example_full_pipeline():
    config = Config()
    
    print("Example: Full Pipeline")
    print("=" * 50)
    
    run_full_pipeline(config, model_type="resnet", apply_tta=True, create_ensemble=True)

def example_data_preprocessing():
    print("Example: Data Preprocessing")
    print("=" * 50)
    
    create_folds("data/train_tp.csv", "data/positive.csv", n_folds=5)
    resample_audio("data/train/", "data/resample/train/")
    resample_audio("data/test/", "data/resample/test/")

if __name__ == "__main__":
    import argparse

    examples = {
        "preprocess": example_data_preprocessing,
        "train": example_basic_training,
        "predict": example_prediction,
        "tta": example_tta_prediction,
        "ensemble": example_ensemble,
        "full": example_full_pipeline,
    }
    parser = argparse.ArgumentParser(description="Run one RFCX Python API example")
    parser.add_argument("example", choices=examples)
    args = parser.parse_args()
    examples[args.example]()
