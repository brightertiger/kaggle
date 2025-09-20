import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import GroupKFold
from typing import List, Tuple, Dict, Any


class QuestionAnswerDataset(Dataset):
    """Dataset class for question-answer pairs with multi-label classification"""
    
    def __init__(self, text_data: pd.DataFrame, meta_data: pd.DataFrame, 
                 labels: pd.DataFrame, tokenizer: BertTokenizer, 
                 max_length: int = 512):
        self.text_data = text_data
        self.meta_data = meta_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qa_ids = text_data['qa_id'].tolist()
        
        # Build vocabulary for text cleaning
        self.vocab = set(tokenizer.get_vocab().keys())
    
    def __len__(self):
        return len(self.text_data)
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing out-of-vocabulary words"""
        words = text.split()
        clean_words = []
        
        for word in words:
            if word in self.vocab:
                clean_words.append(word)
            elif re.sub(r'[^\w\s]', '', word) in self.vocab:
                clean_words.append(re.sub(r'[^\w\s]', '', word))
            else:
                # Split by punctuation and check sub-words
                clean_word = re.sub(r'[^\w\s]', ' ', word)
                sub_words = clean_word.split()
                for sub_word in sub_words:
                    if sub_word in self.vocab:
                        clean_words.append(sub_word)
        
        return ' '.join(clean_words)
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text with fallback to cleaned text if too long"""
        text = text.replace('\n', ' ').lower()
        original_text = text
        
        # First attempt: tokenize as-is
        tokens = self.tokenizer.tokenize(text)
        
        # If too long, try with cleaned text
        if len(tokens) >= 400:
            text = self.clean_text(original_text)
            tokens = self.tokenizer.tokenize(text)
        
        # Add special tokens and truncate
        tokens = ['[CLS]'] + tokens[:self.max_length-2] + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return token_ids
    
    def get_question_tokens(self, qa_id: int) -> np.ndarray:
        """Get tokenized question (title + body)"""
        row = self.text_data[self.text_data['qa_id'] == qa_id].iloc[0]
        title = row['question_title']
        body = row['question_body']
        
        question_text = f"{title} [SEP] {body}"
        tokens = self.tokenize_text(question_text)
        
        # Pad to max_length
        tokens = tokens + [0] * (self.max_length - len(tokens))
        return np.array(tokens, dtype=np.int64)
    
    def get_answer_tokens(self, qa_id: int) -> np.ndarray:
        """Get tokenized answer"""
        row = self.text_data[self.text_data['qa_id'] == qa_id].iloc[0]
        answer_text = row['answer']
        
        tokens = self.tokenize_text(answer_text)
        
        # Pad to max_length
        tokens = tokens + [0] * (self.max_length - len(tokens))
        return np.array(tokens, dtype=np.int64)
    
    def get_labels(self, qa_id: int) -> np.ndarray:
        """Get labels for the given qa_id"""
        row = self.labels[self.labels['qa_id'] == qa_id].iloc[0]
        return row.iloc[1:].values.astype(np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        qa_id = self.qa_ids[idx]
        
        question_tokens = self.get_question_tokens(qa_id)
        answer_tokens = self.get_answer_tokens(qa_id)
        labels = self.get_labels(qa_id)
        
        return {
            'question': torch.tensor(question_tokens, dtype=torch.long),
            'answer': torch.tensor(answer_tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


class DataProcessor:
    """Data processing utilities for question understanding"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
    
    def create_folds(self, data: pd.DataFrame, n_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified folds based on question body content"""
        group_kfold = GroupKFold(n_splits=n_folds)
        folds = list(group_kfold.split(X=data['qa_id'], groups=data['question_body']))
        return folds
    
    def split_data(self, data: pd.DataFrame, train_idx: np.ndarray, 
                   valid_idx: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and validation sets"""
        train_data = data.iloc[train_idx].reset_index(drop=True)
        valid_data = data.iloc[valid_idx].reset_index(drop=True)
        return train_data, valid_data
    
    def save_split_data(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                       fold: int, data_dir: str):
        """Save train/validation splits to separate files"""
        os.makedirs(f"{data_dir}/split", exist_ok=True)
        
        # Save text data
        train_data[['qa_id'] + self.config.text_columns].to_csv(
            f"{data_dir}/split/text_train_{fold}.csv", index=False)
        valid_data[['qa_id'] + self.config.text_columns].to_csv(
            f"{data_dir}/split/text_valid_{fold}.csv", index=False)
        
        # Save metadata
        meta_columns = self.config.user_columns + self.config.url_columns + self.config.categorical_columns
        train_data[['qa_id'] + meta_columns].to_csv(
            f"{data_dir}/split/data_train_{fold}.csv", index=False)
        valid_data[['qa_id'] + meta_columns].to_csv(
            f"{data_dir}/split/data_valid_{fold}.csv", index=False)
        
        # Save labels
        train_data[['qa_id'] + self.config.label_columns].to_csv(
            f"{data_dir}/split/label_train_{fold}.csv", index=False)
        valid_data[['qa_id'] + self.config.label_columns].to_csv(
            f"{data_dir}/split/label_valid_{fold}.csv", index=False)
    
    def create_data_loaders(self, fold: int, data_dir: str, 
                          batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training and validation"""
        # Load data
        train_text = pd.read_csv(f"{data_dir}/split/text_train_{fold}.csv")
        train_meta = pd.read_csv(f"{data_dir}/split/data_train_{fold}.csv")
        train_labels = pd.read_csv(f"{data_dir}/split/label_train_{fold}.csv")
        
        valid_text = pd.read_csv(f"{data_dir}/split/text_valid_{fold}.csv")
        valid_meta = pd.read_csv(f"{data_dir}/split/data_valid_{fold}.csv")
        valid_labels = pd.read_csv(f"{data_dir}/split/label_valid_{fold}.csv")
        
        # Create datasets
        train_dataset = QuestionAnswerDataset(
            train_text, train_meta, train_labels, 
            self.tokenizer, self.config.max_length
        )
        valid_dataset = QuestionAnswerDataset(
            valid_text, valid_meta, valid_labels,
            self.tokenizer, self.config.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        
        return train_loader, valid_loader
