import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from ..utils.config import Config

class UniversalSentenceEncoder:
    def __init__(self):
        tf.config.experimental.set_memory_growth(
            tf.config.experimental.list_physical_devices('GPU')[0], True
        )
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    
    def get_embeddings(self, texts):
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            embedding = self.model([text]).numpy()
            embeddings.append(embedding[0])
        
        embeddings = np.vstack(embeddings).round(5).astype(np.float32)
        return embeddings
    
    def process_dataset(self, input_path, output_path):
        data = pd.read_csv(input_path)
        embeddings = self.get_embeddings(data['comment_text'])
        
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.columns = [f'use_{i}' for i in range(512)]
        
        result = data.join(embedding_df)
        result.to_csv(output_path, index=False)
        
        print(f"Processed {len(data)} samples and saved to {output_path}")
        return result

class EmbeddingProcessor:
    def __init__(self):
        self.use_model = UniversalSentenceEncoder()
    
    def process_all_datasets(self, data_dir):
        datasets = [
            ('english/train_english.csv', 'english/train_english_embed.csv'),
            ('english/valid_english.csv', 'english/valid_english_embed.csv'),
            ('english/test_english.csv', 'english/test_english_embed.csv'),
            ('foreign/train_foreign.csv', 'foreign/train_foreign_embed.csv'),
            ('foreign/valid_foreign.csv', 'foreign/valid_foreign_embed.csv'),
            ('foreign/test_foreign.csv', 'foreign/test_foreign_embed.csv'),
            ('subtitle/subtitle.csv', 'subtitle/subtitle_embed.csv')
        ]
        
        for input_file, output_file in datasets:
            input_path = f"{data_dir}/{input_file}"
            output_path = f"{data_dir}/{output_file}"
            
            print(f"Processing {input_file}...")
            self.use_model.process_dataset(input_path, output_path)
        
        print("All datasets processed successfully!")
