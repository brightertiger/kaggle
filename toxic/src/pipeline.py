"""Cross-validated training with separate out-of-fold and test predictions."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data_utils import DataProcessor, read_csv
from .models import NeuralNetworkModel, NaiveBayesSVM, LogisticRegressionModel
from .ensemble import EnsembleModel, ModelEvaluator, align_predictions


class ToxicCommentPipeline:
    """Preprocess, train each fold, evaluate OOF scores, and write submissions."""

    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.ensemble_model = EnsembleModel(config)
        self.evaluator = ModelEvaluator(config)
        self.config.create_directories()

    def prepare_data(self):
        self.data_processor.process_all_data()

    def _load_fold(self, method: str, fold: int):
        source = Path(self.config.data.output_dir, method)
        return (
            read_csv(source / 'train' / f'train_data_{fold}.csv'),
            read_csv(source / 'train' / f'train_labels_{fold}.csv'),
            read_csv(source / 'train' / f'test_data_{fold}.csv'),
            read_csv(source / 'train' / f'test_labels_{fold}.csv'),
            read_csv(source / 'score' / 'score_data.csv'),
        )

    def train_neural_network_models(self, embedding_type: str = 'glove') -> Dict[str, List[Tuple[pd.DataFrame, pd.DataFrame]]]:
        if embedding_type not in {'glove', 'fasttext'}:
            raise ValueError('embedding_type must be glove or fasttext')
        neural_model = NeuralNetworkModel(self.config)
        if self.config.model.random_embeddings:
            embeddings_index = {}
        else:
            path = getattr(self.config.model, f'{embedding_type}_path')
            embeddings_index = neural_model.load_embeddings(path)
        return {
            method: self._train_nn_for_preprocessing(neural_model, embeddings_index, method, embedding_type)
            for method in self.config.data.preprocessing_methods[:2]
        }

    def _train_nn_for_preprocessing(self, neural_model: NeuralNetworkModel,
                                    embeddings_index: Dict[str, np.ndarray],
                                    preprocessing_method: str, embedding_type: str
                                    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        import tensorflow as tf
        results = []
        targets = self.config.evaluation.target_columns
        for fold in range(1, self.config.data.n_folds + 1):
            print(f'Training neural {embedding_type}/{preprocessing_method} fold {fold}')
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(self.config.data.random_state + fold)
            train_text, train_label, valid_text, valid_label, test_text = self._load_fold(preprocessing_method, fold)
            train_label = align_predictions(train_label, train_text['id'], targets)
            valid_label = align_predictions(valid_label, valid_text['id'], targets)
            train_seq, valid_seq, _ = neural_model.prepare_data(train_text, valid_text)
            embedding_matrix = neural_model.create_embedding_matrix(neural_model.tokenizer, embeddings_index)
            model = neural_model.build_model(embedding_matrix)
            # Bound tf.data's worker pool on laptops as well as competition machines.
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            batch_size = self.config.model.batch_size
            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_seq, train_label[targets].to_numpy(dtype='float32'))
            ).shuffle(len(train_seq), seed=self.config.data.random_state + fold).batch(batch_size).with_options(options)
            valid_ds = tf.data.Dataset.from_tensor_slices(
                (valid_seq, valid_label[targets].to_numpy(dtype='float32'))
            ).batch(batch_size).with_options(options)
            model.fit(
                train_ds, validation_data=valid_ds, epochs=self.config.model.epochs, shuffle=False,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=self.config.model.patience, restore_best_weights=True,
                )], verbose=0,
            )
            fold_results = []
            for frame, sequences in ((valid_text, valid_seq), (test_text, neural_model.transform(test_text))):
                dataset = tf.data.Dataset.from_tensor_slices(sequences).batch(batch_size).with_options(options)
                probabilities = model.predict(dataset, verbose=0)
                result = frame[['id']].reset_index(drop=True)
                result[targets] = probabilities
                fold_results.append(result)
            results.append(tuple(fold_results))
        return results

    def _combine_folds(self, results):
        # Validation folds contain different IDs: concatenate, never average them.
        validation = pd.concat([pair[0] for pair in results], ignore_index=True)
        test = self.ensemble_model.simple_averaging(
            [pair[1] for pair in results], self.config.evaluation.target_columns,
        )
        validation = align_predictions(validation, read_csv(self.config.data.train_path)['id'],
                                       self.config.evaluation.target_columns)
        test = align_predictions(test, read_csv(self.config.data.test_path)['id'],
                                 self.config.evaluation.target_columns)
        return validation, test

    def train_traditional_models(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        return {
            'nbsvm': self._train_traditional_model(NaiveBayesSVM(self.config), 'nbsvm'),
            'logistic_regression': self._train_traditional_model(LogisticRegressionModel(self.config), 'logistic_regression'),
        }

    def _train_traditional_model(self, model, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        methods = self.config.data.preprocessing_methods
        method = 'preprocessed' if 'preprocessed' in methods else methods[0]
        results = []
        for fold in range(1, self.config.data.n_folds + 1):
            print(f'Training {model_name}/{method} fold {fold}')
            train_text, train_label, valid_text, _, test_text = self._load_fold(method, fold)
            results.append(model.train_model(train_label, train_text, valid_text, test_text))
        return self._combine_folds(results)

    def create_ensemble_predictions(self, model_predictions: Dict[str, pd.DataFrame],
                                     weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        return self.ensemble_model.blend_predictions(
            model_predictions, weights or {}, self.config.evaluation.target_columns,
        )

    def evaluate_models(self, predictions: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        labels = read_csv(self.config.data.train_path)
        return self.evaluator.evaluate_ensemble(predictions, labels, self.config.evaluation.target_columns)

    def save_predictions(self, validation_predictions: Dict[str, pd.DataFrame],
                         test_predictions: Dict[str, pd.DataFrame]):
        targets = self.config.evaluation.target_columns
        for predictions, directory, suffix, path in (
            (validation_predictions, self.config.model.model_dir, 'validation', self.config.data.train_path),
            (test_predictions, self.config.model.submission_dir, 'submission', self.config.data.test_path),
        ):
            ids = read_csv(path)['id']
            for name, frame in predictions.items():
                aligned = align_predictions(frame, ids, targets)
                aligned.to_csv(Path(directory, f'{name}_{suffix}.csv'), index=False)

    def run_full_pipeline(self, train_neural: bool = True, train_traditional: bool = True):
        if not train_neural and not train_traditional:
            raise ValueError('Enable at least one model family')
        self.prepare_data()
        validation_predictions, test_predictions = {}, {}
        if train_neural:
            neural_results = self.train_neural_network_models(self.config.model.embedding_type)
            for method, folds in neural_results.items():
                validation_predictions[f'neural_{method}'], test_predictions[f'neural_{method}'] = self._combine_folds(folds)
        if train_traditional:
            for name, (validation, test) in self.train_traditional_models().items():
                validation_predictions[name], test_predictions[name] = validation, test
        validation_predictions['final_ensemble'] = self.create_ensemble_predictions(validation_predictions)
        test_predictions['final_ensemble'] = self.create_ensemble_predictions(test_predictions)
        evaluation = self.evaluate_models(validation_predictions)
        self.evaluator.print_evaluation_results(evaluation)
        self.save_predictions(validation_predictions, test_predictions)
        pd.DataFrame.from_dict(evaluation, orient='index').to_csv(Path(self.config.model.log_dir, 'evaluation.csv'))
        # Preserve the original return convention: test frames carry a _test suffix.
        predictions = {**validation_predictions,
                       **{f'{name}_test': frame for name, frame in test_predictions.items()}}
        print(f'Pipeline complete: {self.config.model.submission_dir}/final_ensemble_submission.csv')
        return predictions, evaluation
