# Question Understanding: Multi-Label Classification for Q&A Quality Assessment

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-green)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive deep learning pipeline for multi-label classification of question-answer pairs, developed for the **Google AI Question Understanding Challenge**. This project demonstrates advanced NLP techniques including BERT-based architectures, ensemble methods, and sophisticated evaluation metrics.

## 🎯 Project Overview

The Question Understanding challenge required predicting multiple quality attributes for question-answer pairs from Google's Q&A platform. The task involved classifying 30 different labels across two categories:

### Question Quality Labels (21 labels)
- **Intent Understanding**: `question_asker_intent_understanding`
- **Content Quality**: `question_body_critical`, `question_conversational`, `question_well_written`
- **Answer Expectations**: `question_expect_short_answer`, `question_fact_seeking`
- **Question Types**: `question_type_choice`, `question_type_compare`, `question_type_consequence`, etc.
- **Engagement**: `question_interestingness_others`, `question_interestingness_self`

### Answer Quality Labels (9 labels)
- **Helpfulness**: `answer_helpful`, `answer_level_of_information`, `answer_plausible`
- **Relevance**: `answer_relevance`, `answer_satisfaction`
- **Content Types**: `answer_type_instructions`, `answer_type_procedure`, `answer_type_reason_explanation`
- **Quality**: `answer_well_written`

## 🏗️ Architecture

### Model Architecture
The solution employs a sophisticated BERT-based architecture that processes both questions and answers:

```python
# Dual-BERT Architecture
Question Input → BERT Encoder → Pooler + Sequence Features
Answer Input   → BERT Encoder → Pooler + Sequence Features
                                    ↓
                          Feature Concatenation (4 × 768)
                                    ↓
                          Dropout + Classification Head
                                    ↓
                          Multi-Label Output (30 labels)
```

### Key Features
- **Dual-BERT Processing**: Separate BERT encoders for questions and answers
- **Feature Fusion**: Combines pooler outputs and sequence-level representations
- **Mixed Precision Training**: Uses NVIDIA Apex for memory efficiency
- **Cross-Validation**: 5-fold stratified cross-validation based on question content
- **Ensemble Methods**: Averages predictions across multiple folds

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd question-understanding

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```python
from src.config import Config
from src.pipeline import QuestionUnderstandingPipeline

# Initialize configuration
config = Config()
config.batch_size = 2
config.learning_rate = 1e-5
config.num_epochs = 6

# Initialize pipeline
pipeline = QuestionUnderstandingPipeline(config)

# Train all folds
fold_results = pipeline.train_all_folds(
    data_path="data/",
    model_path="models/"
)

# Generate predictions
predictions = pipeline.inference(
    data_path="data/",
    model_path="models/"
)
```

### Command Line Interface

```bash
# Train all folds
python main.py --mode train --data_path data/ --model_path models/

# Train specific fold
python main.py --mode train --data_path data/ --model_path models/ --fold 1

# Generate predictions
python main.py --mode inference --data_path data/ --model_path models/
```

## 📊 Dataset and Preprocessing

### Data Structure
The dataset contains question-answer pairs with the following structure:
- **Text Fields**: `question_title`, `question_body`, `answer`
- **Metadata**: User names, URLs, categories
- **Labels**: 30 binary classification targets

### Preprocessing Pipeline
1. **Text Cleaning**: Removes out-of-vocabulary words and handles punctuation
2. **Tokenization**: BERT tokenization with fallback to cleaned text for long sequences
3. **Truncation**: Sequences limited to 512 tokens with special token handling
4. **Stratified Splitting**: Cross-validation folds based on question content similarity

### Data Loading
```python
from src.data_utils import DataProcessor

processor = DataProcessor(config)
train_loader, valid_loader = processor.create_data_loaders(
    fold=1, data_dir="data/", batch_size=2
)
```

## 🧠 Model Details

### Architecture Components

#### 1. QuestionUnderstandingModel
```python
class QuestionUnderstandingModel(nn.Module):
    def __init__(self, config):
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 * 4, 30)  # 4 feature vectors × 768
        
    def forward(self, question, answer):
        # Process question and answer through BERT
        q_features = self.get_bert_features(question)
        a_features = self.get_bert_features(answer)
        
        # Concatenate: [q_pooler, q_seq, a_pooler, a_seq]
        combined = torch.cat([q_features['pooler'], q_features['sequence'],
                             a_features['pooler'], a_features['sequence']], dim=-1)
        
        return self.classifier(self.dropout(combined))
```

#### 2. DualBERTModel (Alternative Architecture)
```python
class DualBERTModel(nn.Module):
    def __init__(self, config):
        self.question_bert = BertModel(config)
        self.answer_bert = BertModel(config)
        self.classifier = nn.Linear(768 * 2, 30)  # Separate encoders
```

### Training Configuration
- **Optimizer**: AdamW with learning rate 1e-5
- **Loss Function**: BCEWithLogitsLoss with custom reduction
- **Scheduler**: ReduceLROnPlateau with factor 0.5
- **Mixed Precision**: NVIDIA Apex O2 optimization
- **Gradient Accumulation**: 4 steps for effective larger batch sizes

## 📈 Training Pipeline

### Cross-Validation Strategy
```python
# 5-fold stratified cross-validation
from sklearn.model_selection import GroupKFold

group_kfold = GroupKFold(n_splits=5)
folds = group_kfold.split(X=data['qa_id'], groups=data['question_body'])
```

### Training Loop
1. **Data Preparation**: Split data into train/validation folds
2. **Model Training**: Train each fold with early stopping
3. **Validation**: Monitor Spearman correlation on validation set
4. **Model Saving**: Save best model based on validation performance
5. **Ensemble**: Combine predictions from all folds

### Evaluation Metrics
- **Primary Metric**: Mean Spearman correlation across all 30 labels
- **Individual Metrics**: Per-label Spearman correlation analysis
- **Cross-Validation**: 5-fold CV with stratified splits

## 🔧 Advanced Features

### Hyperparameter Search
```python
param_grid = {
    'learning_rate': [5e-6, 1e-5, 2e-5],
    'batch_size': [1, 2, 4],
    'num_epochs': [4, 6, 8]
}

results = pipeline.hyperparameter_search(
    data_path="data/",
    model_path="models/",
    param_grid=param_grid
)
```

### Model Architecture Comparison
```python
model_types = ["question_understanding", "dual_bert"]
comparison = pipeline.create_model_comparison(
    data_path="data/",
    model_path="models/",
    model_types=model_types
)
```

### Ensemble Methods
- **Fold Averaging**: Average predictions across CV folds
- **Model Ensembling**: Combine different architectures
- **Weighted Ensembles**: Learn optimal combination weights

## 📊 Results and Performance

### Model Performance
- **Cross-Validation Score**: 0.3862 (Mean Spearman Correlation)
- **Best Individual Fold**: 0.4005
- **Ensemble Performance**: Improved stability and generalization

### Label-wise Analysis
The model shows varying performance across different labels:

**High Performance Labels** (>0.4 correlation):
- `question_well_written`
- `answer_helpful`
- `answer_well_written`

**Medium Performance Labels** (0.3-0.4 correlation):
- `question_fact_seeking`
- `answer_relevance`
- `question_type_definition`

**Challenging Labels** (<0.3 correlation):
- `question_interestingness_self`
- `question_not_really_a_question`
- `answer_type_spelling`

## 🛠️ Technical Implementation

### Key Technical Decisions

#### 1. Text Preprocessing
```python
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
```

#### 2. Mixed Precision Training
```python
# Initialize mixed precision
model, optimizer = amp.initialize(
    model, optimizer,
    opt_level="O2",
    keep_batchnorm_fp32=True,
    verbosity=0
)

# Scale loss for backward pass
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

#### 3. Custom Loss Function
```python
def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
    """Reduce loss across batch and labels"""
    batch_size = loss.shape[0]
    return loss.sum() / (batch_size * self.config.num_labels)
```

### Memory Optimization
- **Gradient Accumulation**: Effective batch size of 8 with accumulation
- **Mixed Precision**: 50% memory reduction with Apex O2
- **Model Checkpointing**: Save/load models efficiently
- **CUDA Memory Management**: Explicit cache clearing

## 📁 Project Structure

```
quest/
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── data_utils.py             # Data processing and loading
│   ├── models.py                 # Model architectures
│   ├── trainer.py                # Training logic
│   ├── evaluator.py              # Evaluation and metrics
│   └── pipeline.py               # Main pipeline
├── main.py                       # Command-line interface
├── example_usage.py              # Usage examples
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## 🔬 Research Insights

### Key Findings

#### 1. Feature Engineering Impact
- **Combined Features**: Using both pooler and sequence features improved performance by ~3%
- **Question-Answer Separation**: Dual processing outperformed concatenated input
- **Feature Concatenation**: `[q_pooler, q_seq, a_pooler, a_seq]` provided best results

#### 2. Training Strategies
- **Cross-Validation**: Stratified splits based on question content improved generalization
- **Early Stopping**: Patience of 3 epochs prevented overfitting
- **Learning Rate Scheduling**: ReduceLROnPlateau with factor 0.5 stabilized training

#### 3. Model Architecture Insights
- **BERT Capacity**: bert-base-uncased provided good balance of performance and efficiency
- **Dropout Regularization**: 0.1 dropout rate optimal for this task
- **Ensemble Benefits**: 5-fold averaging improved correlation by ~2%

### Challenges and Solutions

#### Challenge 1: Class Imbalance
**Problem**: Some labels had extreme class imbalance (e.g., 99% negative samples)
**Solution**: Used BCEWithLogitsLoss with custom reduction instead of weighted loss

#### Challenge 2: Long Sequences
**Problem**: Questions and answers could exceed BERT's 512 token limit
**Solution**: Implemented fallback text cleaning and intelligent truncation

#### Challenge 3: Memory Constraints
**Problem**: Large models and batch sizes caused OOM errors
**Solution**: Mixed precision training with gradient accumulation

## 🚀 Future Improvements

### Model Enhancements
1. **Larger Pre-trained Models**: Experiment with RoBERTa, DeBERTa
2. **Multi-task Learning**: Joint training with related NLP tasks
3. **Attention Mechanisms**: Cross-attention between questions and answers
4. **Hierarchical Models**: Separate models for question vs. answer classification

### Data Augmentation
1. **Back-translation**: Generate paraphrased versions
2. **Synonym Replacement**: Intelligent word substitution
3. **Question Rewriting**: Generate alternative phrasings

### Training Improvements
1. **Progressive Training**: Start with easy samples, gradually increase difficulty
2. **Curriculum Learning**: Train on simpler labels first
3. **Meta-Learning**: Learn to adapt quickly to new label distributions

## 📚 Dependencies

### Core Dependencies
- **PyTorch** (1.8+): Deep learning framework
- **Transformers** (4.0+): Hugging Face transformers library
- **Pandas** (1.3+): Data manipulation
- **NumPy** (1.21+): Numerical computing
- **Scikit-learn** (1.0+): Machine learning utilities

### Optional Dependencies
- **Apex**: NVIDIA mixed precision training
- **TensorFlow** (2.6+): Alternative model implementations
- **TQDM**: Progress bars

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd question-understanding

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Format code
black src/
isort src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google AI**: For hosting the Question Understanding Challenge
- **Hugging Face**: For the excellent transformers library
- **NVIDIA**: For Apex mixed precision training
- **PyTorch Team**: For the powerful deep learning framework

## 👨‍💻 Author

**Ujjwal Singh Rao**
- LinkedIn: [linkedin.com/in/brightertiger](https://linkedin.com/in/brightertiger)
- GitHub: [github.com/brightertiger](https://github.com/brightertiger)

---

*This project demonstrates advanced NLP techniques and provides a robust framework for multi-label text classification tasks. The codebase is designed for both research and production use, with comprehensive documentation and examples.*
