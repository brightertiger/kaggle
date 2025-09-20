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
    config = Config()
    
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
    resample_audio("data/raw/train/", "data/resample/train/")
    resample_audio("data/raw/test/", "data/resample/test/")

if __name__ == "__main__":
    print("RFCX Species Audio Detection - Example Usage")
    print("=" * 60)
    
    print("\n1. Data Preprocessing:")
    example_data_preprocessing()
    
    print("\n2. Basic Training:")
    example_basic_training()
    
    print("\n3. Generate Predictions:")
    example_prediction()
    
    print("\n4. TTA Predictions:")
    example_tta_prediction()
    
    print("\n5. Ensemble Predictions:")
    example_ensemble()
    
    print("\n6. Full Pipeline:")
    example_full_pipeline()
