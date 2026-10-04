import pandas as pd
from sklearn.model_selection import KFold
from typing import Tuple, List
import os


class DataSplitter:
    def __init__(self, config):
        self.config = config
        self.kfold = KFold(
            n_splits=config.avito.N_FOLDS,
            shuffle=True,
            random_state=config.avito.RANDOM_STATE
        )
    
    def create_cv_folds(self) -> None:
        self.config.avito.create_directories()
        train_data = pd.read_csv(self.config.avito.TRAIN_DATA_PATH)
        train_data = train_data.drop(['image'], axis=1, errors='ignore')
        
        if 'activation_date' in train_data.columns:
            train_data['activation_date'] = pd.to_datetime(train_data['activation_date'])
        
        print(f'Training data shape: {train_data.shape}')
        
        os.makedirs(self.config.avito.FOLDS_DIR, exist_ok=True)
        
        fold = 1
        for train_idx, valid_idx in self.kfold.split(train_data):
            train_fold = train_data.iloc[train_idx][[self.config.avito.ID_COLUMN]]
            valid_fold = train_data.iloc[valid_idx][[self.config.avito.ID_COLUMN]]
            
            print(f'Fold {fold} - Train: {train_fold.shape}, Valid: {valid_fold.shape}')
            
            train_fold.to_csv(f'{self.config.avito.FOLDS_DIR}/train_{fold}.csv', index=False)
            valid_fold.to_csv(f'{self.config.avito.FOLDS_DIR}/valid_{fold}.csv', index=False)
            fold += 1
        
        test_data = pd.read_csv(self.config.avito.TEST_DATA_PATH)[[self.config.avito.ID_COLUMN]]
        test_data.to_csv(f'{self.config.avito.FOLDS_DIR}/score.csv', index=False)
        print(f'Test data shape: {test_data.shape}')


class DataLoader:
    def __init__(self, config):
        self.config = config
    
    def load_train_data(self, columns: List[str] = None) -> pd.DataFrame:
        if columns is None:
            return pd.read_csv(self.config.avito.TRAIN_DATA_PATH)
        return pd.read_csv(self.config.avito.TRAIN_DATA_PATH, usecols=columns)
    
    def load_test_data(self, columns: List[str] = None) -> pd.DataFrame:
        if columns is None:
            return pd.read_csv(self.config.avito.TEST_DATA_PATH)
        return pd.read_csv(self.config.avito.TEST_DATA_PATH, usecols=columns)
    
    def load_train_active(self, columns: List[str] = None) -> pd.DataFrame:
        if columns is None:
            return pd.read_csv(self.config.avito.TRAIN_ACTIVE_PATH)
        return pd.read_csv(self.config.avito.TRAIN_ACTIVE_PATH, usecols=columns)
    
    def load_test_active(self, columns: List[str] = None) -> pd.DataFrame:
        if columns is None:
            return pd.read_csv(self.config.avito.TEST_ACTIVE_PATH)
        return pd.read_csv(self.config.avito.TEST_ACTIVE_PATH, usecols=columns)
    
    def load_full_dataset(self, columns: List[str] = None) -> pd.DataFrame:
        train_data = self.load_train_data(columns)
        test_data = self.load_test_data(columns)
        return pd.concat([train_data, test_data], ignore_index=True)
    
    def load_active_dataset(self, columns: List[str] = None) -> pd.DataFrame:
        train_active = self.load_train_active(columns)
        test_active = self.load_test_active(columns)
        return pd.concat([train_active, test_active], ignore_index=True)
    
    def load_fold_data(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_idx = pd.read_csv(f'{self.config.avito.FOLDS_DIR}/train_{fold}.csv')
        valid_idx = pd.read_csv(f'{self.config.avito.FOLDS_DIR}/valid_{fold}.csv')
        return train_idx, valid_idx


class FeatureValidator:
    def __init__(self, config):
        self.config = config
    
    def validate_feature_file(self, file_path: str, expected_columns: List[str] = None) -> bool:
        if not os.path.exists(file_path):
            print(f"Feature file not found: {file_path}")
            return False
        
        try:
            df = pd.read_csv(file_path)
            if expected_columns and not all(col in df.columns for col in expected_columns):
                missing_cols = [col for col in expected_columns if col not in df.columns]
                print(f"Missing columns in {file_path}: {missing_cols}")
                return False
            
            if self.config.avito.ID_COLUMN not in df.columns:
                print(f"Missing ID column in {file_path}")
                return False
            
            print(f"Feature file validated: {file_path} - Shape: {df.shape}")
            return True
            
        except Exception as e:
            print(f"Error validating {file_path}: {e}")
            return False
    
    def check_for_duplicates(self, file_path: str) -> bool:
        df = pd.read_csv(file_path)
        duplicates = df[self.config.avito.ID_COLUMN].duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate IDs in {file_path}")
            return False
        return True


def align_rows(driver: pd.DataFrame, values: pd.DataFrame, key: str) -> pd.DataFrame:
    """Join by ID without silently dropping, duplicating, or reordering rows."""
    result = driver.merge(values, on=key, how='left', sort=False,
                          validate='one_to_one', indicator=True)
    if (result['_merge'] != 'both').any():
        raise ValueError(f'Missing rows when joining on {key}')
    return result.drop(columns='_merge')
