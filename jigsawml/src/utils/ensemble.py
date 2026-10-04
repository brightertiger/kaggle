from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config


class ModelEnsemble:
    def __init__(self, model_dir, config=None):
        self.model_dir = Path(model_dir)
        self.config = config or Config(model_dir=model_dir)

    @staticmethod
    def average(frames, weights=None):
        # Duplicate IDs may represent translated views; average those first.
        frames = [frame.groupby('id', sort=False, as_index=False)['toxic'].mean() for frame in frames]
        ids = frames[0]['id']
        if any(set(frame['id']) != set(ids) for frame in frames):
            raise ValueError('Prediction files have different sets of IDs')
        values = np.stack([frame.set_index('id').loc[ids, 'toxic'].to_numpy() for frame in frames])
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError('Predictions must be finite probabilities')
        return pd.DataFrame({'id': ids, 'toxic': np.average(values, axis=0, weights=weights)})

    def post_process_version(self, version):
        frames = [pd.read_csv(self.model_dir / f'version{version}/score_{fold}.csv', dtype={'id': str})
                  for fold in range(self.config.N_FOLDS)]
        result = self.average(frames)
        result.to_csv(self.model_dir / f'version_{version}.csv', index=False)
        return result

    def create_final_ensemble(self):
        result = self.average([self.post_process_version(1), self.post_process_version(2)], weights=[1, 3])
        result.to_csv(self.model_dir / 'combined.csv', index=False)
        result.to_csv(self.model_dir / 'submission.csv', index=False)
        return result
