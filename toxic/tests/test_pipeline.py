"""Regression checks for fold aggregation, label alignment, and current APIs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config
from src.data_utils import TextPreprocessor
from src.ensemble import EnsembleModel, ModelEvaluator
from src.models import NaiveBayesSVM, LogisticRegressionModel
from src.pipeline import ToxicCommentPipeline


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.config = get_config()
        self.targets = self.config.evaluation.target_columns

    def frame(self, ids, values):
        result = pd.DataFrame({'id': ids})
        result[self.targets] = np.repeat(np.array(values)[:, None], len(self.targets), axis=1)
        return result

    def test_blending_and_auc_align_shuffled_ids(self):
        labels = self.frame(['a', 'b', 'c'], [0, 1, 0])
        pred = self.frame(['c', 'a', 'b'], [0.1, 0.2, 0.9])
        blended = EnsembleModel(self.config).simple_averaging([pred, pred.iloc[::-1]], self.targets)
        np.testing.assert_allclose(blended[self.targets], pred[self.targets])
        result = ModelEvaluator(self.config).evaluate_single_model(blended, labels, self.targets)
        self.assertEqual(result['overall'], 1.0)
        with self.assertRaisesRegex(ValueError, 'IDs do not match'):
            EnsembleModel(self.config).simple_averaging([pred, pred.iloc[:2]], self.targets)
        with self.assertRaises(ValueError):
            EnsembleModel(self.config).weighted_averaging([pred], [0], self.targets)

    def test_uneven_oof_folds_concatenate_and_test_folds_average(self):
        (Path(__file__).resolve().parents[1] / 'dry_run_output').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'dry_run_output') as temp:
            base = Path(temp)
            self.config.data.train_path = str(base / 'train.csv')
            self.config.data.test_path = str(base / 'test.csv')
            self.frame(['0001', '0002', '0003'], [0, 1, 0]).to_csv(self.config.data.train_path, index=False)
            self.frame(['0099', '0088'], [0, 0]).to_csv(self.config.data.test_path, index=False)
            self.config.data.output_dir = str(base / 'processed')
            self.config.model.model_dir = str(base / 'validation')
            self.config.model.log_dir = str(base / 'logs')
            self.config.model.submission_dir = str(base / 'submissions')
            pipeline = ToxicCommentPipeline(self.config)
            folds = [
                (self.frame(['0002', '0003'], [0.8, 0.1]), self.frame(['0099', '0088'], [0.2, 0.6])),
                (self.frame(['0001'], [0.2]), self.frame(['0088', '0099'], [0.8, 0.4])),
            ]
            valid, test = pipeline._combine_folds(folds)
            self.assertEqual(valid.id.tolist(), ['0001', '0002', '0003'])
            np.testing.assert_allclose(valid.toxic, [0.2, 0.8, 0.1])
            np.testing.assert_allclose(test.toxic, [0.3, 0.7])
            pipeline.save_predictions({'model': valid}, {'model': test})
            self.assertEqual(len(pd.read_csv(base / 'submissions/model_submission.csv')), 2)

    def test_sparse_models_support_constant_labels_and_nondefault_indices(self):
        train = pd.DataFrame({'id': ['a', 'b', 'c', 'd'],
                              'comment_text': ['helpful kind edit', 'rude harsh comment',
                                               'friendly kind work', 'rude nasty words']}, index=[8, 6, 4, 2])
        labels = self.frame(['d', 'c', 'b', 'a'], [1, 0, 1, 0])
        labels['threat'] = 0
        self.config.model.nb_min_df = 1
        for model_type in (NaiveBayesSVM, LogisticRegressionModel):
            valid, test = model_type(self.config).train_model(labels, train, train.iloc[:2], train.iloc[2:])
            self.assertEqual(valid.id.tolist(), ['a', 'b'])
            self.assertEqual(test.id.tolist(), ['c', 'd'])
            self.assertTrue(np.isfinite(valid[self.targets]).all().all())
            self.assertTrue((test.threat == 0).all())
        evaluation = ModelEvaluator(self.config).evaluate_single_model(labels, labels, self.targets)
        self.assertTrue(np.isnan(evaluation['threat']))
        self.assertTrue(np.isnan(evaluation['overall']))

    def test_stacking_aligns_labels_and_handles_constant_target(self):
        labels = self.frame(['a', 'b', 'c', 'd'], [0, 1, 0, 1])
        labels['threat'] = 0
        pred = self.frame(['c', 'a', 'd', 'b'], [0.2, 0.1, 0.8, 0.9])
        valid, test = EnsembleModel(self.config).stacking(
            [pred, pred.iloc[::-1]], [pred, pred.iloc[::-1]], labels, self.targets,
        )
        self.assertEqual(valid.id.tolist(), pred.id.tolist())
        self.assertTrue((test.threat == 0).all())
        self.assertGreater(valid.loc[valid.id == 'b', 'toxic'].iloc[0],
                           valid.loc[valid.id == 'a', 'toxic'].iloc[0])

    def test_preprocessing_preserves_markers_and_handles_missing_text(self):
        processed = TextPreprocessor.tokenized('WOW!!! https://example.org #TalkPage :-)')
        for token in ['<url>', '<allcaps>', '<repeat>', '<hashtag>', '<smile>']:
            self.assertIn(token, processed)
        self.assertEqual(TextPreprocessor.basic_clean(None), 'nan')
        self.assertNotIn('127.0.0.1', TextPreprocessor.basic_clean('edit 127.0.0.1'))


if __name__ == '__main__':
    (Path(__file__).resolve().parents[1] / 'dry_run_output').mkdir(exist_ok=True)
    unittest.main()
