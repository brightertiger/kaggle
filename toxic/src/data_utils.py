import pandas as pd
from pathlib import Path
import re
import string
import nltk
from typing import Tuple
from sklearn.model_selection import KFold
from .config import Config


class TextPreprocessor:
    """Text preprocessing utilities for toxic comment classification"""

    @staticmethod
    def basic_clean(text: str) -> str:
        """Basic text cleaning - remove URLs, IP addresses, normalize whitespace"""
        if pd.isna(text):
            return 'nan'

        text = str(text)
        ipaddress = re.findall(r'[0-9]+(?:\.[0-9]+){3}', text)
        for ip in ipaddress:
            text = text.replace(ip, ' ')

        text = text.replace('\n', ' ')
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def basic_clean_lower(text: str) -> str:
        """Basic cleaning with lowercase conversion"""
        return TextPreprocessor.basic_clean(text).lower()

    @staticmethod
    def tokenized(text: str) -> str:
        """Advanced tokenization with emoji and special character handling"""
        if pd.isna(text):
            return 'nan'

        text = str(text)
        FLAGS = re.MULTILINE | re.DOTALL

        def hashtag(text_match):
            text = text_match.group()
            hashtag_body = text[1:]
            if hashtag_body.isupper():
                result = "<hashtag> {} <allcaps>".format(hashtag_body)
            else:
                result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
            return result

        def allcaps(text_match):
            text = text_match.group()
            return text.lower() + " <allcaps> "

        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=FLAGS)

        eyes = r"[8:=;]"
        nose = r"['`\-]?"

        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = re_sub(r"/", " / ")
        text = re_sub(r"@\w+", "<user>")
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
        text = re_sub(r"<3", "<heart>")
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = re_sub(r"#\S+", hashtag)
        text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
        text = re_sub(r"([A-Z]){2,}", allcaps)

        punct = re.compile('[%s]' % re.escape(string.punctuation.replace('<', '').replace('>', '')))
        text = punct.sub(' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower()

    @staticmethod
    def nltk_tokenized(text: str) -> str:
        """NLTK-based tokenization with basic cleaning"""
        if pd.isna(text):
            return 'nan'

        text = str(text)
        ipaddress = re.findall(r'[0-9]+(?:\.[0-9]+){3}', text)
        for ip in ipaddress:
            text = text.replace(ip, ' ')

        text = text.replace('\n', ' ')
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        try:
            tokens = nltk.word_tokenize(text)
            return ' '.join([x.strip() for x in tokens])
        except LookupError:
            # Treebank tokenization needs no downloaded Punkt resources.
            return ' '.join(nltk.tokenize.TreebankWordTokenizer().tokenize(text))

    @staticmethod
    def preprocessed(text: str) -> str:
        """Use preprocessed text as-is"""
        return str(text) if not pd.isna(text) else 'nan'

    @classmethod
    def apply_preprocessing(cls, text_series: pd.Series, method: str) -> pd.Series:
        """Apply specified preprocessing method to text series"""
        method_map = {
            'basic_clean': cls.basic_clean,
            'basic_clean_lower': cls.basic_clean_lower,
            'tokenized': cls.tokenized,
            'nltk_tokenized': cls.nltk_tokenized,
            'preprocessed': cls.preprocessed
        }

        if method not in method_map:
            raise ValueError(f"Unknown preprocessing method: {method}")

        return text_series.map(method_map[method])


class DataProcessor:
    """Data processing and cross-validation utilities"""

    def __init__(self, config: Config):
        self.config = config

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data"""
        train_data = read_csv(self.config.data.train_path)
        test_data = read_csv(self.config.data.test_path)
        targets = self.config.evaluation.target_columns
        for name, frame, required in (
            ('train', train_data, ['id', 'comment_text', *targets]),
            ('test', test_data, ['id', 'comment_text']),
        ):
            missing = set(required) - set(frame.columns)
            if missing:
                raise ValueError(f"{name} data is missing columns: {sorted(missing)}")
            if frame.empty or frame['id'].isna().any() or frame['id'].eq('').any() or frame['id'].duplicated().any():
                raise ValueError(f"{name} data must have nonempty, unique IDs")
            frame['comment_text'] = frame['comment_text'].fillna('nan').astype(str)
        if not train_data[targets].isin([0, 1]).all().all():
            raise ValueError('Training labels must be binary, without missing values')
        if not 2 <= self.config.data.n_folds <= len(train_data):
            raise ValueError('n_folds must be between 2 and the training row count')
        if not self.config.data.preprocessing_methods:
            raise ValueError('At least one preprocessing method is required')
        return train_data, test_data

    def create_cross_validation_splits(self, train_data: pd.DataFrame,
                                     preprocessing_method: str) -> None:
        """Create cross-validation splits for a specific preprocessing method"""
        train_data_copy = train_data.copy()
        train_data_copy['comment_text'] = TextPreprocessor.apply_preprocessing(
            train_data_copy['comment_text'], preprocessing_method
        )

        comments = train_data_copy['id'].unique()
        folds = KFold(n_splits=self.config.data.n_folds,
                     shuffle=True,
                     random_state=self.config.data.random_state)

        labels = ['id', *self.config.evaluation.target_columns]
        text_cols = ['id', 'comment_text']

        fold = 1
        source_dir = f"{self.config.data.output_dir}/{preprocessing_method}"
        Path(source_dir, 'train').mkdir(parents=True, exist_ok=True)

        for train_idx, test_idx in folds.split(comments):
            train_ids = comments[train_idx]
            test_ids = comments[test_idx]

            X_train = train_data_copy[train_data_copy['id'].isin(train_ids)][text_cols]
            y_train = train_data_copy[train_data_copy['id'].isin(train_ids)][labels]
            X_test = train_data_copy[train_data_copy['id'].isin(test_ids)][text_cols]
            y_test = train_data_copy[train_data_copy['id'].isin(test_ids)][labels]

            X_train.to_csv(f"{source_dir}/train/train_data_{fold}.csv", index=False)
            y_train.to_csv(f"{source_dir}/train/train_labels_{fold}.csv", index=False)
            X_test.to_csv(f"{source_dir}/train/test_data_{fold}.csv", index=False)
            y_test.to_csv(f"{source_dir}/train/test_labels_{fold}.csv", index=False)

            fold += 1

    def prepare_test_data(self, test_data: pd.DataFrame,
                         preprocessing_method: str) -> None:
        """Prepare test data for scoring"""
        test_data_copy = test_data.copy()
        test_data_copy['comment_text'] = TextPreprocessor.apply_preprocessing(
            test_data_copy['comment_text'], preprocessing_method
        )

        source_dir = f"{self.config.data.output_dir}/{preprocessing_method}"
        Path(source_dir, 'score').mkdir(parents=True, exist_ok=True)
        test_data_copy.to_csv(f"{source_dir}/score/score_data.csv", index=False)

    def process_all_data(self) -> None:
        """Process all data with different preprocessing methods"""
        train_data, test_data = self.load_data()

        for method in self.config.data.preprocessing_methods:
            print(f"Processing data with method: {method}")
            self.create_cross_validation_splits(train_data, method)
            self.prepare_test_data(test_data, method)


def read_csv(path) -> pd.DataFrame:
    """Keep hexadecimal/numeric-looking competition IDs intact, including zeros."""
    return pd.read_csv(path, dtype={'id': str, 'comment_text': str}, keep_default_na=False)
