"""Normalize native/legacy CSV schemas and create stratified folds."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_folds(data_path: str, output_path: str, n_splits: int, random_state: int = 2017):
    data = pd.read_csv(data_path, dtype={'id_code': str, 'image': str})
    data = data.rename(columns={'image': 'id_code', 'level': 'diagnosis'})
    required = {'id_code', 'diagnosis'}
    if not required.issubset(data.columns):
        raise ValueError(f'{data_path} needs id_code/diagnosis or image/level columns')
    data = data[['id_code', 'diagnosis']].copy()
    if data.isna().any().any() or data['id_code'].duplicated().any():
        raise ValueError('Labels must have unique image IDs and no missing values')
    if not data['diagnosis'].isin(range(5)).all():
        raise ValueError('diagnosis must be an integer severity from 0 to 4')
    if n_splits < 2 or data['diagnosis'].value_counts().min() < n_splits:
        raise ValueError(f'Each present class needs at least {n_splits} rows for stratified folds')
    data['diagnosis'] = data['diagnosis'].astype(int)
    data['fold'] = 0
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for fold_idx, (_, valid_idx) in enumerate(skf.split(data.index, data.diagnosis), 1):
        data.loc[valid_idx, 'fold'] = fold_idx
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f'Folds saved to: {output_path} ({len(data)} images)')
    return data


def main():
    from .pipeline import APTOSPipeline
    APTOSPipeline().preprocess_data()


if __name__ == '__main__':
    main()
