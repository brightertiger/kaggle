"""Blend aligned out-of-fold probabilities, then apply the fit to test images."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from ..data.data_preprocessing import validate_image_ids


class ModelEnsemble:
    def __init__(self, output_dir='./output/blend', num_classes=5):
        self.models = {}
        self.blend_model = None
        self.output_dir = Path(output_dir)
        self.num_classes = num_classes
        self.prob_columns = [f'prob_{i}' for i in range(num_classes)]

    def add_model_predictions(self, model_name, predictions_path):
        predictions = pd.read_csv(predictions_path)
        validate_image_ids(predictions)
        if not set(self.prob_columns).issubset(predictions.columns):
            raise ValueError(f'{model_name} requires probability columns {self.prob_columns}')
        values = predictions[self.prob_columns].to_numpy()
        if (not np.isfinite(values).all() or (values < 0).any() or
                not np.allclose(values.sum(axis=1), 1, atol=1e-5)):
            raise ValueError(f'{model_name} has invalid probabilities')
        self.models[model_name] = predictions

    def create_blend_features(self, data):
        validate_image_ids(data)
        blend_data = data.copy()
        features = []
        for name, predictions in self.models.items():
            if set(data['image_id']) != set(predictions['image_id']):
                raise ValueError(f'{name}: image IDs must match exactly; use OOF predictions for training')
            mapping = {column: f'{name}_{column}' for column in self.prob_columns}
            # Join only by ID: a predicted label must never be a training merge key.
            frame = predictions[['image_id'] + self.prob_columns].rename(columns=mapping)
            blend_data = blend_data.merge(frame, on='image_id', how='left', sort=False, validate='one_to_one')
            features.extend(mapping.values())
        if not features:
            raise ValueError('No model predictions supplied')
        return blend_data, features

    @staticmethod
    def _new_model():
        # Explicit OVR preserves the original multi_class='ovr' behavior in modern sklearn.
        return OneVsRestClassifier(LogisticRegression(max_iter=500, C=0.2))

    def train_blend_model(self, fold, features, baseline_features=None):
        data, _ = self.create_blend_features(self.data)
        train, valid = data[data['fold'] != fold], data[data['fold'] == fold]
        if train.empty or valid.empty:
            raise ValueError(f'Empty blend split for fold {fold}')
        model = self._new_model()
        model.fit(train[features], train['label'])
        predictions = model.predict(valid[features])
        baseline_acc = None
        if baseline_features:
            baseline_acc = float((valid[baseline_features].to_numpy().argmax(axis=1) == valid['label']).mean())
        metric = float((predictions == valid['label']).mean())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / f'blend_{fold}.pkl').open('wb') as handle:
            pickle.dump(model, handle)
        print(f'Blend fold {fold}: accuracy={metric:.4f}')
        return dict(fold=int(fold), baseline_acc=baseline_acc, model_acc=metric, predictions=predictions)

    def create_ensemble(self, data_path, model_predictions, test_predictions=None, test_data=None):
        self.data = pd.read_csv(data_path)
        self.models = {}
        for name, path in model_predictions.items():
            self.add_model_predictions(name, path)
        data, features = self.create_blend_features(self.data)
        baseline_name = 'version7' if 'version7' in self.models else next(iter(self.models))
        baseline_features = [f'{baseline_name}_{column}' for column in self.prob_columns]
        results = [self.train_blend_model(fold, features, baseline_features)
                   for fold in sorted(data['fold'].unique())]
        self.blend_model = self._new_model()
        self.blend_model.fit(data[features], data['label'])
        with (self.output_dir / 'blend.pkl').open('wb') as handle:
            pickle.dump({'model': self.blend_model, 'features': features}, handle)
        if test_predictions is not None:
            if set(test_predictions) != set(model_predictions) or test_data is None:
                raise ValueError('Test predictions must match the OOF model names and include test_data')
            test_ensemble = ModelEnsemble(self.output_dir, self.num_classes)
            for name in model_predictions:
                test_ensemble.add_model_predictions(name, test_predictions[name])
            test_frame, test_features = test_ensemble.create_blend_features(test_data[['image_id']])
            if test_features != features:
                raise ValueError('Training/test feature order differs')
            self.submission = test_frame[['image_id']].copy()
            self.submission['label'] = self.blend_model.predict(test_frame[features]).astype(int)
            self.submission.to_csv(self.output_dir.parent / 'submission.csv', index=False)
        return results
