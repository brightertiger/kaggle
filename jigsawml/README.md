# Jigsaw Multilingual Toxic Comment Classification

A sophisticated solution for the Jigsaw Multilingual Toxic Comment Classification competition, featuring adversarial training, two-stage model training, and ensemble methods.

## 🏆 Competition Overview

The Jigsaw Multilingual Toxic Comment Classification challenge required participants to build models that can identify toxic comments in multiple languages. This solution achieved competitive performance through a combination of:

- **Multilingual embeddings** using Universal Sentence Encoder
- **Adversarial training** with LightGBM-generated synthetic examples
- **Two-stage training** with XLM-RoBERTa models
- **Ensemble methods** combining multiple model versions

## 🚀 Key Features

### 1. Multilingual Text Processing
- **Universal Sentence Encoder (USE)** embeddings for robust multilingual representation
- **XLM-RoBERTa** for advanced transformer-based classification
- Support for 15+ languages with unified preprocessing pipeline

### 2. Adversarial Training Strategy
- **LightGBM-based adversarial example generation** to improve model robustness
- **Cross-validation approach** for reliable adversarial data creation
- **Multiple adversarial datasets** (English, Subtitle, Translation) for comprehensive coverage

### 3. Two-Stage Training Pipeline
- **Version 1**: Weighted loss training with adversarial examples
- **Version 2**: Fine-tuning with reduced learning rate
- **Progressive training** with model checkpointing and early stopping

### 4. Advanced Model Architecture
- **XLM-RoBERTa Large** as backbone transformer
- **Dual pooling strategy**: CLS token + average pooling
- **Dropout regularization** and **gradient clipping** for stability

### 5. Ensemble Methods
- **5-fold cross-validation** for robust model training
- **Multi-version ensemble** combining Version 1 and Version 2 models
- **Weighted averaging** for final predictions

## 📁 Project Structure

```
jigsawml/
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Main training pipeline
│   ├── data/                  # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py  # Raw data preparation
│   │   ├── data_utils.py          # Data loading and preprocessing
│   │   └── embeddings.py          # Universal Sentence Encoder embeddings
│   ├── models/                # Model architectures and loss functions
│   │   ├── __init__.py
│   │   ├── models.py              # Model architectures
│   │   └── loss.py                # Custom loss functions
│   ├── training/              # Training and inference modules
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training utilities
│   │   ├── inference.py           # Inference pipeline
│   │   └── scoring.py             # Model scoring and evaluation
│   └── utils/                 # Utility modules
│       ├── __init__.py
│       ├── config.py              # Configuration parameters
│       ├── adversarial.py         # Adversarial data generation
│       └── ensemble.py            # Ensemble methods
├── main.py                    # Command-line interface
├── example_usage.py          # Usage examples
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## 🔧 Complete Pipeline

The solution provides a comprehensive end-to-end pipeline organized into logical modules:

### 📊 Data Processing (`src/data/`)
- **`data_preprocessing.py`**: Raw data preparation and cleaning
- **`data_utils.py`**: Data loading, tokenization, and dataset classes
- **`embeddings.py`**: Universal Sentence Encoder embedding generation

### 🤖 Models (`src/models/`)
- **`models.py`**: XLM-RoBERTa classifier architecture
- **`loss.py`**: Custom weighted loss functions

### 🏋️ Training (`src/training/`)
- **`trainer.py`**: Model training utilities with validation
- **`inference.py`**: Efficient inference pipeline
- **`scoring.py`**: Model evaluation and scoring

### 🛠️ Utilities (`src/utils/`)
- **`config.py`**: Centralized configuration management
- **`adversarial.py`**: LightGBM-based adversarial data creation
- **`ensemble.py`**: Multi-model ensemble methods

### 🎯 Main Pipeline (`src/pipeline.py`)
- **Orchestrates** the complete workflow from data prep to final predictions

This modular approach provides:
- **Clean separation of concerns** with logical grouping
- **Easy maintenance** and debugging
- **Scalable architecture** for different datasets and models
- **Professional implementation** suitable for production deployment

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd jigsawml
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up data directory structure**:
```
data/
├── process/
│   ├── english/
│   ├── foreign/
│   ├── subtitle/
│   └── pseudo/
└── model/
    ├── version1/
    └── version2/
```

## 🎯 Usage

### Command Line Interface

```bash
# Prepare raw data
python main.py --mode prepare_data --data_dir ../../data

# Generate embeddings
python main.py --mode embeddings --data_dir ../../data

# Generate adversarial data
python main.py --mode adversarial --data_dir ../../data

# Train Version 1 model for fold 0
python main.py --mode train --version 1 --subset 0

# Train Version 2 model for fold 0 (with Version 1 weights)
python main.py --mode train --version 2 --subset 0 --load_pretrained

# Score all models
python main.py --mode score --test_path ../../data/process/foreign/test_english.csv

# Create ensemble predictions
python main.py --mode ensemble

# Run complete pipeline
python main.py --mode full_pipeline --data_dir ../../data --test_path ../../data/process/foreign/test_english.csv
```

### Python API

```python
from src.pipeline import JigsawPipeline

# Initialize pipeline
pipeline = JigsawPipeline()

# Prepare data
pipeline.prepare_data('../../data')
pipeline.generate_embeddings('../../data')
pipeline.generate_adversarial_data('../../data')

# Train individual models
pipeline.train_version1(subset=0)
pipeline.train_version2(subset=0, load_from_version1=True)

# Score models
pipeline.scoring_pipeline.score_all_models('../../data/process/foreign/test_english.csv')

# Run full pipeline
final_predictions = pipeline.run_full_pipeline(
    data_dir='../../data',
    test_path='../../data/process/foreign/test_english.csv'
)
```

## 🔬 Technical Details

### Model Architecture

The solution uses a **XLM-RoBERTa Large** model with custom modifications:

```python
class XLMRobertaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlmr = XLMRobertaModel.from_pretrained('xlm-roberta-large')
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(2048, 1)
    
    def forward(self, tokens, attention_mask):
        features = self.xlmr(input_ids=tokens, attention_mask=attention_mask)[0]
        cls_token = features[:, 0, :]
        avg_pooled = features.mean(dim=1)
        combined = torch.cat([cls_token, avg_pooled], dim=-1)
        return self.output(self.dropout(combined))
```

### Adversarial Training Process

1. **Generate adversarial examples** using LightGBM on USE embeddings
2. **Create synthetic training data** with adversarial labels
3. **Train models** with weighted loss incorporating adversarial examples
4. **Fine-tune models** with reduced learning rate

### Training Strategy

**Version 1 Training**:
- Learning rate: 1e-5
- Epochs: 5
- Loss: Weighted BCE Loss
- Batch size: 8

**Version 2 Training**:
- Learning rate: 1e-6 (fine-tuning)
- Epochs: 4
- Loss: Standard BCE Loss
- Batch size: 8

### Data Processing Pipeline

1. **Text Preprocessing**: Tokenization with XLM-RoBERTa tokenizer
2. **Embedding Generation**: USE embeddings for adversarial training
3. **Adversarial Data Creation**: LightGBM-based synthetic examples
4. **Model Training**: Two-stage training with different objectives
5. **Ensemble Creation**: Multi-fold and multi-version combination

## 📊 Performance Metrics

The solution achieved competitive performance through:

- **Robust cross-validation**: 5-fold CV ensuring generalization
- **Adversarial robustness**: Improved performance on challenging examples
- **Multilingual capability**: Consistent performance across languages
- **Ensemble stability**: Reduced variance through model combination

## 🔧 Configuration

Key parameters can be adjusted in `src/config.py`:

```python
class Config:
    SEED = 2017
    MAX_LENGTH = 300
    BATCH_SIZE = 8
    EPOCHS_V1 = 5
    EPOCHS_V2 = 4
    LR_V1 = 1e-5
    LR_V2 = 1e-6
    MODEL_NAME = 'xlm-roberta-large'
```

## 🚀 Advanced Features

### Memory Optimization
- **Gradient accumulation** for effective large batch training
- **Mixed precision training** support
- **Memory-efficient data loading** with proper cleanup

### Training Stability
- **Gradient clipping** to prevent exploding gradients
- **Learning rate scheduling** with ReduceLROnPlateau
- **Early stopping** based on validation metrics

### Reproducibility
- **Fixed random seeds** across all components
- **Deterministic data splits** for consistent results
- **Checkpoint saving** for model recovery

## 📈 Results and Insights

### Key Innovations

1. **Adversarial Training Integration**: Successfully integrated adversarial examples into the training pipeline, improving model robustness.

2. **Two-Stage Training**: Progressive training strategy with different loss functions and learning rates.

3. **Multilingual Embeddings**: Effective use of Universal Sentence Encoder for multilingual representation.

4. **Ensemble Strategy**: Sophisticated ensemble combining multiple model versions and folds.

### Performance Improvements

- **Adversarial robustness**: Models trained with adversarial examples showed improved performance on challenging test cases
- **Cross-lingual consistency**: Consistent performance across different languages
- **Ensemble stability**: Reduced prediction variance through model combination

## 🔮 Future Enhancements

1. **Advanced Adversarial Methods**: Integration of more sophisticated adversarial training techniques
2. **Multi-task Learning**: Extending to related tasks like sentiment analysis
3. **Model Compression**: Distillation techniques for deployment efficiency
4. **Real-time Inference**: Optimization for production deployment

## 📚 References

- [XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning](https://arxiv.org/abs/1911.02116)
- [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
- [Adversarial Training Methods](https://arxiv.org/abs/1412.6572)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or suggestions, please open an issue or contact the maintainer.

---

*This solution demonstrates advanced techniques in multilingual NLP, adversarial training, and ensemble methods, making it a valuable reference for similar competitions and real-world applications.*
