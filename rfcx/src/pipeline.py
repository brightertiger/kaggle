import os
import random
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from sklearn.model_selection import StratifiedGroupKFold
from .config import Config
from .trainer import train_model
from .predictor import generate_predictions

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_folds(data_path: str, output_path: str, n_folds: int = 5, random_state: int = 2017) -> None:
    positive_data = pd.read_csv(data_path)
    if positive_data['recording_id'].nunique() < n_folds:
        raise ValueError("Need at least one distinct recording per fold")
    
    positive_data['fold'] = -1
    splits = StratifiedGroupKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    
    for fold_idx, (_, val_idx) in enumerate(splits.split(
            positive_data.index, positive_data.species_id, groups=positive_data.recording_id)):
        positive_data.loc[val_idx, 'fold'] = fold_idx + 1
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    positive_data.to_csv(output_path, index=False)
    print(f"Created {n_folds} folds. Data saved to {output_path}")

def resample_audio(input_path: str, output_path: str, target_sr: int = 32000) -> None:
    import soundfile as sf
    import librosa as lb
    
    os.makedirs(output_path, exist_ok=True)
    
    files = sorted(Path(input_path).glob('*.flac'))
    if not files:
        raise ValueError(f"No FLAC files found in {input_path}")
    for file_path in files:
        name = file_path.stem
        sound, orig_sr = sf.read(file_path, dtype='float32')
        if sound.ndim == 2:
            sound = sound.mean(axis=1)
        sound = lb.resample(y=sound, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best")
        np.save(os.path.join(output_path, f'{name}.npy'), sound)
    
    print(f"Resampled audio files saved to {output_path}")

def ensemble_predictions(prediction_files: List[str], output_path: str) -> None:
    if not prediction_files:
        raise ValueError("Provide at least one prediction file")
    predictions = []
    recording_ids = None
    prediction_cols = None
    
    for file_path in prediction_files:
        df = pd.read_csv(file_path)
        columns = [col for col in df.columns if col.startswith('s') and col[1:].isdigit()]
        if df['recording_id'].duplicated().any() or not columns:
            raise ValueError(f"Invalid prediction schema: {file_path}")
        if recording_ids is None:
            recording_ids = df['recording_id'].tolist()
            prediction_cols = columns
        if set(df['recording_id']) != set(recording_ids) or columns != prediction_cols:
            raise ValueError("Prediction files must contain the same recordings and species columns")
        df = df.set_index('recording_id').loc[recording_ids, prediction_cols]
        if not np.isfinite(df.to_numpy()).all():
            raise ValueError(f"Nonfinite predictions: {file_path}")
        predictions.append(df.rank(axis=1, pct=True))
    
    ensemble_df = predictions[0].copy()
    ensemble_df[:] = np.mean([df.to_numpy() for df in predictions], axis=0)
    ensemble_df = ensemble_df.reset_index()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(output_path, index=False)
    print(f"Ensemble predictions saved to {output_path}")

def run_full_pipeline(config: Config, model_type: str = "resnet", 
                     apply_tta: bool = False, create_ensemble: bool = False) -> None:
    set_seed(config.seed)
    
    print("Starting RFCX Species Audio Detection Pipeline...")
    print(f"Model type: {model_type}")
    print(f"Apply TTA: {apply_tta}")
    
    train_model(config, model_type)
    
    generate_predictions(config, model_type, apply_tta=False)
    
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
