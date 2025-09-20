import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from typing import List, Dict, Any
from .config import Config

class TextFeatureEngineer:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.porter = PorterStemmer()
        
    def clean_string(self, text: str) -> str:
        table = str.maketrans('', '', string.punctuation)
        return text.lower().translate(table)
    
    def count_word(self, text: str) -> int:
        return len(self.clean_string(text).split())
    
    def count_word_unique(self, text: str) -> int:
        return len(set(self.clean_string(text).split()))
    
    def word_length(self, text: str) -> float:
        words = self.clean_string(text).split()
        return np.mean([len(word) for word in words]) if words else 0
    
    def count_punct(self, text: str) -> int:
        return sum(1 for char in text if char in string.punctuation)
    
    def count_upper(self, text: str) -> int:
        return sum(1 for word in text.split() if word.istitle())
    
    def count_stopword(self, text: str) -> int:
        return sum(1 for word in self.clean_string(text).split() if word in self.stopwords)
    
    def count_stemwords(self, text: str) -> int:
        words = self.clean_string(text).split()
        stemmed_words = [self.porter.stem(word) for word in words]
        return sum(1 for i in range(len(words)) if words[i] != stemmed_words[i])
    
    def count_pos_tag(self, text: str, pos_tag: str) -> int:
        tokenized_text = nltk.word_tokenize(self.clean_string(text))
        pos_tagged = nltk.pos_tag(tokenized_text)
        return sum(1 for word, tag in pos_tagged if pos_tag in tag)
    
    def count_noun(self, text: str) -> int:
        return self.count_pos_tag(text, 'NN')
    
    def count_adj(self, text: str) -> int:
        return self.count_pos_tag(text, 'JJ')
    
    def count_det(self, text: str) -> int:
        return self.count_pos_tag(text, 'DT')
    
    def count_verb(self, text: str) -> int:
        return self.count_pos_tag(text, 'VB')
    
    def count_pronoun(self, text: str) -> int:
        return self.count_pos_tag(text, 'PRP')
    
    def count_chars(self, text: str) -> int:
        return len(text)
    
    def extract_all_features(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        result_df = df.copy()
        
        feature_functions = {
            'count_word': self.count_word,
            'count_word_unique': self.count_word_unique,
            'word_length': self.word_length,
            'count_punct': self.count_punct,
            'count_upper': self.count_upper,
            'count_stemwords': self.count_stemwords,
            'count_stopword': self.count_stopword,
            'count_noun': self.count_noun,
            'count_pronoun': self.count_pronoun,
            'count_det': self.count_det,
            'count_adj': self.count_adj,
            'count_verb': self.count_verb,
            'count_chars': self.count_chars
        }
        
        for feature_name, feature_func in feature_functions.items():
            result_df[feature_name] = result_df[text_column].apply(feature_func)
        
        self._add_ratio_features(result_df)
        
        return result_df
    
    def _add_ratio_features(self, df: pd.DataFrame) -> None:
        df['ratio_punct'] = df['count_punct'] / df['count_chars']
        df['ratio_upper'] = df['count_upper'] / df['count_word']
        df['ratio_stemwords'] = df['count_stemwords'] / df['count_word']
        df['ratio_stopword'] = df['count_stopword'] / df['count_word']
        df['ratio_noun'] = df['count_noun'] / df['count_word']
        df['ratio_pronoun'] = df['count_pronoun'] / df['count_word']
        df['ratio_det'] = df['count_det'] / df['count_word']
        df['ratio_adj'] = df['count_adj'] / df['count_word']
        df['ratio_verb'] = df['ratio_verb'] / df['count_word']

class NaiveBayesFeatureEngineer:
    def __init__(self):
        self.word_vectorizer = None
        self.char_cnt_vectorizer = None
        self.char_tf_vectorizer = None
        self.word_svd = None
        self.char_svd = None
        
    def fit_transform_word_features(self, train_texts: List[str], test_texts: List[str]) -> tuple:
        from sklearn.feature_extraction.text import CountVectorizer
        
        self.word_vectorizer = CountVectorizer(
            stop_words='english', 
            ngram_range=Config.NGRAM_RANGE_WORD
        )
        
        full_texts = train_texts + test_texts
        self.word_vectorizer.fit(full_texts)
        
        train_features = self.word_vectorizer.transform(train_texts)
        test_features = self.word_vectorizer.transform(test_texts)
        
        return train_features, test_features
    
    def fit_transform_char_count_features(self, train_texts: List[str], test_texts: List[str]) -> tuple:
        from sklearn.feature_extraction.text import CountVectorizer
        
        self.char_cnt_vectorizer = CountVectorizer(
            ngram_range=Config.NGRAM_RANGE_CHAR_CNT, 
            analyzer='char'
        )
        
        full_texts = train_texts + test_texts
        self.char_cnt_vectorizer.fit(full_texts)
        
        train_features = self.char_cnt_vectorizer.transform(train_texts)
        test_features = self.char_cnt_vectorizer.transform(test_texts)
        
        return train_features, test_features
    
    def fit_transform_char_tfidf_features(self, train_texts: List[str], test_texts: List[str]) -> tuple:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.char_tf_vectorizer = TfidfVectorizer(
            ngram_range=Config.NGRAM_RANGE_CHAR, 
            analyzer='char'
        )
        
        full_texts = train_texts + test_texts
        self.char_tf_vectorizer.fit(full_texts)
        
        train_features = self.char_tf_vectorizer.transform(train_texts)
        test_features = self.char_tf_vectorizer.transform(test_texts)
        
        return train_features, test_features
    
    def fit_transform_svd_features(self, train_texts: List[str], test_texts: List[str]) -> tuple:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        char_vectorizer = TfidfVectorizer(
            ngram_range=Config.NGRAM_RANGE_CHAR, 
            analyzer='char'
        )
        word_vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=Config.NGRAM_RANGE_WORD
        )
        
        full_texts = train_texts + test_texts
        
        char_features = char_vectorizer.fit_transform(full_texts)
        word_features = word_vectorizer.fit_transform(full_texts)
        
        char_svd = TruncatedSVD(n_components=Config.SVD_COMPONENTS, algorithm='arpack')
        word_svd = TruncatedSVD(n_components=Config.SVD_COMPONENTS, algorithm='arpack')
        
        char_svd.fit(char_features)
        word_svd.fit(word_features)
        
        train_char_svd = pd.DataFrame(
            char_svd.transform(char_vectorizer.transform(train_texts)),
            columns=[f'svd_char_{i}' for i in range(Config.SVD_COMPONENTS)]
        )
        test_char_svd = pd.DataFrame(
            char_svd.transform(char_vectorizer.transform(test_texts)),
            columns=[f'svd_char_{i}' for i in range(Config.SVD_COMPONENTS)]
        )
        
        train_word_svd = pd.DataFrame(
            word_svd.transform(word_vectorizer.transform(train_texts)),
            columns=[f'svd_wrd_{i}' for i in range(Config.SVD_COMPONENTS)]
        )
        test_word_svd = pd.DataFrame(
            word_svd.transform(word_vectorizer.transform(test_texts)),
            columns=[f'svd_wrd_{i}' for i in range(Config.SVD_COMPONENTS)]
        )
        
        train_svd_features = pd.concat([train_word_svd, train_char_svd], axis=1)
        test_svd_features = pd.concat([test_word_svd, test_char_svd], axis=1)
        
        return train_svd_features, test_svd_features
