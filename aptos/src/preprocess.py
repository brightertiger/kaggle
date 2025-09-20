import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from config import Config

def create_folds(data_path: str, output_path: str, n_splits: int, random_state: int = 2017):
    data = pd.read_csv(data_path)
    data.columns = ['id_code', 'diagnosis']
    data['fold'] = 0
    
    print(f"Dataset shape: {data.shape}")
    print(f"Diagnosis distribution:\n{data.diagnosis.value_counts()}")
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    for fold_idx, (_, valid_idx) in enumerate(skf.split(data.index, data.diagnosis), 1):
        data.loc[valid_idx, 'fold'] = fold_idx
    
    data.to_csv(output_path, index=False)
    print(f"Folds saved to: {output_path}")
    return data

def main():
    config = Config()
    
    print("Creating cross-validation folds for APTOS dataset")
    print("=" * 50)
    
    # 2015 Pretrain data
    print("\nProcessing 2015 Pretrain Data:")
    pretrain_train_path = f"{config.PRETRAIN_DATA_PATH}/{config.TRAIN_LABELS_2015}"
    pretrain_train_output = f"{config.PRETRAIN_DATA_PATH}/{config.TRAIN_FOLDS_FILE}"
    create_folds(pretrain_train_path, pretrain_train_output, config.PRETRAIN_FOLDS)
    
    # 2015 Test data
    print("\nProcessing 2015 Test Data:")
    pretrain_test_path = f"{config.PRETRAIN_DATA_PATH}/{config.TEST_LABELS_2015}"
    pretrain_test_output = f"{config.PRETRAIN_DATA_PATH}/{config.TEST_FOLDS_FILE}"
    create_folds(pretrain_test_path, pretrain_test_output, config.PRETRAIN_FOLDS)
    
    # 2019 Train data
    print("\nProcessing 2019 Train Data:")
    train_path = f"{config.TRAIN_DATA_PATH}/{config.TRAIN_LABELS_2019}"
    train_output = f"{config.TRAIN_DATA_PATH}/{config.TRAIN_FOLDS_FILE}"
    create_folds(train_path, train_output, config.TRAIN_FOLDS)
    
    print("\nAll folds created successfully!")

if __name__ == "__main__":
    main()
