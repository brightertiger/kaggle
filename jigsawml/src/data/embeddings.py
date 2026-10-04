from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..utils.config import Config


class UniversalSentenceEncoder:
    def __init__(self, model_url=None):
        # Optional and lazy: importing the training pipeline never loads TensorFlow.
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            import tensorflow_text  # noqa: F401; registers multilingual text ops
        except ImportError as exc:
            raise ImportError('USE requires requirements-use.txt; the offline dry run does not.') from exc
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        self.model = hub.load(model_url or Config.USE_MODEL)

    def get_embeddings(self, texts):
        rows = [self.model([str(text)]).numpy()[0] for text in tqdm(texts, desc='USE embeddings')]
        return np.asarray(rows, dtype=np.float32).reshape(-1, len(Config.USE_FEATURES)).round(5)

    def process_dataset(self, input_path, output_path):
        return EmbeddingProcessor(encoder=self).process_dataset(input_path, output_path)


class EmbeddingProcessor:
    def __init__(self, config=None, encoder=None):
        self.config = config or Config()
        self.use_model = encoder

    def process_dataset(self, input_path, output_path):
        if self.use_model is None:
            self.use_model = UniversalSentenceEncoder(self.config.USE_MODEL)
        data = pd.read_csv(input_path)
        embeddings = self.use_model.get_embeddings(data['comment_text'].fillna(''))
        if embeddings.shape != (len(data), len(self.config.USE_FEATURES)):
            raise ValueError('Sentence embeddings have the wrong shape')
        result = data.join(pd.DataFrame(embeddings, columns=self.config.USE_FEATURES))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return result

    def process_all_datasets(self, data_dir):
        for name in ['english/train_english', 'english/valid_english', 'english/test_english',
                     'foreign/valid_foreign', 'foreign/test_foreign',
                     'foreign/train_foreign', 'subtitle/subtitle']:
            path = Path(data_dir) / f'{name}.csv'
            if path.exists():
                self.process_dataset(path, path.with_name(f'{path.stem}_embed.csv'))
