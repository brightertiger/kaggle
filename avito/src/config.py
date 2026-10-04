"""Configuration with paths resolved from the competition folder by default."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AvitoConfig:
    RANDOM_STATE: int = 2017
    N_FOLDS: int = 5
    DATA_ROOT: str = str(ROOT / 'data')
    FEATURES_DIR: str = str(ROOT / 'output' / 'features')
    MODEL_DIR: str = str(ROOT / 'output' / 'models')
    OUTPUT_DIR: str = str(ROOT / 'output')
    TARGET_COLUMN: str = 'deal_probability'
    ID_COLUMN: str = 'item_id'
    TEXT_COLUMNS: list[str] = field(default_factory=lambda: ['title', 'description'])
    CATEGORICAL_COLUMNS: list[str] = field(default_factory=lambda: [
        'parent_category_name', 'category_name', 'user_type', 'region', 'city',
        'user_id', 'param_1', 'param_2', 'param_3'])
    NUMERICAL_COLUMNS: list[str] = field(default_factory=lambda: [
        'price', 'image_top_1', 'item_seq_number'])

    @property
    def TRAIN_DATA_PATH(self):
        return str(Path(self.DATA_ROOT) / 'train.csv')

    @property
    def TEST_DATA_PATH(self):
        return str(Path(self.DATA_ROOT) / 'test.csv')

    @property
    def TRAIN_ACTIVE_PATH(self):
        return str(Path(self.DATA_ROOT) / 'train_active.csv')

    @property
    def TEST_ACTIVE_PATH(self):
        return str(Path(self.DATA_ROOT) / 'test_active.csv')

    @property
    def FOLDS_DIR(self):
        return str(Path(self.MODEL_DIR) / 'folds')

    @property
    def INSAMPLE_DIR(self):
        return str(Path(self.MODEL_DIR) / 'insample')

    @property
    def OUTSAMPLE_DIR(self):
        return str(Path(self.MODEL_DIR) / 'outsample')

    def create_directories(self):
        for directory in [self.MODEL_DIR, self.OUTPUT_DIR, self.FOLDS_DIR,
                          self.INSAMPLE_DIR, self.OUTSAMPLE_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        for subdir in ['count', 'date', 'text_title', 'user']:
            (Path(self.FEATURES_DIR) / subdir).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    RIDGE_ALPHA: float = 20.0
    RIDGE_MAX_ITER: int | None = None
    RIDGE_TOL: float = 0.001
    RIDGE_SOLVER: str = 'auto'
    TFIDF_MAX_FEATURES: int = 50000
    TFIDF_NGRAM_RANGE: tuple = (1, 2)
    TFIDF_SUBLINEAR_TF: bool = True
    TFIDF_NORM: str = 'l2'
    TFIDF_SMOOTH_IDF: bool = False
    COUNT_VECTORIZER_NGRAM_RANGE: tuple = (1, 2)


class Config:
    def __init__(self, avito=None, model=None):
        self.avito = avito or AvitoConfig()
        self.model = model or ModelConfig()
