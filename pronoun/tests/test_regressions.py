"""Regression checks for failures introduced during migration; no downloads."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Doc
import torch
from transformers import BertConfig, BertTokenizer

from src.config import Config, DataConfig
from src.data_utils import PronounDataset, collate_fn, read_gap
from src.feature_engineering import FeatureExtractor
from src.models import PronounResolutionModel
from src.pipeline import PronounResolutionPipeline


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        cls.root = Path(cls.temp.name)
        words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'alice', 'beth',
                 'she', 'met', 'said', '.', 'padding', 'smith', 'thanked']
        (cls.root / 'vocab.txt').write_text('\n'.join(words) + '\n')
        (cls.root / 'tokenizer_config.json').write_text(json.dumps({'tokenizer_class': 'BertTokenizer'}))
        cls.tokenizer = BertTokenizer.from_pretrained(cls.root, local_files_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def dataset(self, tagged, max_length=500):
        features = pd.DataFrame([[1., 0., 0., 0., 0., 0., 0.]])
        return PronounDataset(pd.DataFrame({'Text': [tagged]}), (features, features),
                              pd.DataFrame({'Label': [0]}), self.tokenizer, max_length)

    def test_exact_offsets_preserve_repeated_and_multiword_mentions(self):
        text = 'Alice Smith met Beth. Alice Smith said she thanked Beth.'
        data = pd.DataFrame([{'Text': text, 'A-offset': text.rindex('Alice'),
                              'B-offset': text.rindex('Beth'), 'Pronoun-offset': text.index('she')}])
        pipeline = PronounResolutionPipeline.__new__(PronounResolutionPipeline)
        tagged = pipeline._add_tags_to_text(data).iloc[0]['Text']
        self.assertEqual(tagged.replace(' [A] ', '').replace(' [B] ', '').replace(' [P] ', ''), text)
        tokens, offsets = self.dataset(tagged)._tokenize_text(tagged)
        words = self.tokenizer.convert_ids_to_tokens(tokens)
        self.assertEqual([words[i] for i in offsets], ['alice', 'beth', 'she'])
        self.assertGreater(offsets[0], words.index('alice'))
        self.assertGreater(offsets[1], words.index('beth'))

    def test_crop_retains_late_mentions_and_rejects_impossible_span(self):
        text = 'padding ' * 40 + '[A] Alice met [B] Beth. [P] she said.'
        tokens, offsets = self.dataset(text, 12)._tokenize_text(text)
        self.assertLessEqual(len(tokens), 12)
        self.assertEqual([self.tokenizer.convert_ids_to_tokens(tokens)[i] for i in offsets],
                         ['alice', 'beth', 'she'])
        with self.assertRaisesRegex(ValueError, 'Mention span'):
            self.dataset('', 8)._tokenize_text('[A] Alice ' + 'padding ' * 30 + '[B] Beth [P] she')

    def test_singleton_training_and_exact_layer_freezing(self):
        config = BertConfig(vocab_size=len(self.tokenizer), hidden_size=8,
                            intermediate_size=16, num_attention_heads=2,
                            num_hidden_layers=24, max_position_embeddings=32)
        model = PronounResolutionModel('unused', 8, bert_config=config, freeze_layers=12)
        for index, layer in enumerate(model.bert_encoder.bert.encoder.layer):
            self.assertTrue(all(p.requires_grad == (index >= 12) for p in layer.parameters()))
        batch = collate_fn([self.dataset('[A] Alice met [B] Beth. [P] she said.')[0]])
        logits = model(*batch[:4])
        self.assertEqual(tuple(logits.shape), (1, 3))
        torch.nn.functional.cross_entropy(logits, batch[4]).backward()
        self.assertTrue(torch.isfinite(model.classifier[-1].weight.grad).all())

    def test_syntactic_distances_use_the_original_document(self):
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        extractor._setup_spacy_extensions()
        doc = Doc(spacy.blank('en').vocab, words=['Alice', 'said', 'she', 'left'],
                  heads=[1, 1, 3, 1], deps=['nsubj', 'ROOT', 'nsubj', 'ccomp'])
        extractor.nlp = spacy.blank('en')
        self.assertEqual(extractor._word_index('Beth met Mary-Jane.', 'Mary-Jane', 9), 2)
        scores = extractor._compute_syntactic_features(doc[2], {'Alice': [doc[0]]}, 'Alice', None, doc.text)
        self.assertEqual(scores['a_par'], 1)
        self.assertEqual(extractor._compute_syntactic_distance(doc[0], doc[2], doc), 3)

    def test_fold_one_is_not_discarded_and_labels_are_parsed(self):
        rows = []
        for i in range(7):
            rows.append({'ID': f'row-{i}', 'Text': 'Alice met Beth. she left.',
                         'A': 'Alice', 'A-offset': 0, 'B': 'Beth', 'B-offset': 10,
                         'Pronoun': 'she', 'Pronoun-offset': 16, 'URL': 'https://example.com/Alice',
                         'A-coref': 'FALSE', 'B-coref': 'TRUE'})
        data = pd.DataFrame(rows)
        train, val = self.root / 'train.tsv', self.root / 'val.tsv'
        data.iloc[:5].to_csv(train, sep='\t', index=False)
        data.iloc[5:].to_csv(val, sep='\t', index=False)
        pipeline = PronounResolutionPipeline.__new__(PronounResolutionPipeline)
        pipeline.config = Config(data=DataConfig(n_folds=3))
        pipeline._features = lambda table: table
        combined = pipeline.prepare_data(train, val)
        self.assertEqual(len(combined), len(data))
        for fold in range(1, 4):
            self.assertGreater(len(combined[combined['fold'] == fold]), 0)
            self.assertGreater(len(combined[combined['fold'] != fold]), 0)
        self.assertEqual(pipeline._create_labels(combined)['Label'].tolist(), [1] * 7)
        data.loc[0, 'A-offset'] = 1
        data.to_csv(train, sep='\t', index=False)
        with self.assertRaisesRegex(ValueError, 'invalid A character offset'):
            read_gap(train, labeled=True)


if __name__ == '__main__':
    unittest.main()
