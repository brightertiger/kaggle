import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .config import Config
from .trainer import train_model
from .predictor import generate_predictions

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def create_folds(data_path: str, output_path: str, n_folds: int = 5, random_state: int = 2017) -> None:
    positive_data = pd.read_csv(data_path)
    positive_data['species_id'] = positive_data['species_id'].astype(str).str.zfill(2)
    positive_data['songtype_id'] = positive_data['songtype_id'].astype(str)
    
    positive_data['fold'] = -1
    splits = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    
    for fold_idx, (_, val_idx) in enumerate(splits.split(positive_data.index, positive_data.species_id)):
        positive_data.loc[val_idx, 'fold'] = fold_idx + 1
    
    positive_data.to_csv(output_path, index=False)
    print(f"Created {n_folds} folds. Data saved to {output_path}")

def resample_audio(input_path: str, output_path: str, target_sr: int = 32000) -> None:
    import soundfile as sf
    import librosa as lb
    
    os.makedirs(output_path, exist_ok=True)
    
    for filename in os.listdir(input_path):
        if filename.endswith('.flac'):
            name = filename.split('.')[0]
            sound, orig_sr = sf.read(os.path.join(input_path, filename))
            sound = lb.resample(sound, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best")
            np.save(os.path.join(output_path, f'{name}.npy'), sound)
    
    print(f"Resampled audio files saved to {output_path}")

def ensemble_predictions(prediction_files: List[str], output_path: str) -> None:
    predictions = []
    
    for file_path in prediction_files:
        df = pd.read_csv(file_path)
        prediction_cols = [col for col in df.columns if col.startswith('s')]
        df[prediction_cols] = df[prediction_cols].rank(axis=1)
        predictions.append(df)
    
    ensemble_df = pd.concat(predictions, ignore_index=True)
    ensemble_df = ensemble_df.groupby('recording_id').mean().reset_index()
    
    ensemble_df.to_csv(output_path, index=False)
    print(f"Ensemble predictions saved to {output_path}")

def run_full_pipeline(config: Config, model_type: str = "resnet", 
                     apply_tta: bool = False, create_ensemble: bool = False) -> None:
    set_seed(config.seed)
    
    print("Starting RFCX Species Audio Detection Pipeline...")
    print(f"Model type: {model_type}")
    print(f"Apply TTA: {apply_tta}")
    
    train_model(config, model_type)
    
    generate_predictions(config, model_type, apply_tta)
    
    if apply_tta:
        generate_predictions(config, model_type, apply_tta=True, 
                           output_name=f"{model_type}_predictions_tta")
    
    if create_ensemble:
        prediction_files = [
            f"{config.data.predictions_path}/{model_type}_predictions.csv"
        ]
        if apply_tta:
            prediction_files.append(f"{config.data.predictions_path}/{model_type}_predictions_tta.csv")
        
        ensemble_predictions(prediction_files, 
                           f"{config.data.predictions_path}/ensemble_predictions.csv")
    
    print("Pipeline completed successfully!")
