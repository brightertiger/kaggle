# Avito Deal Probability Prediction

A comprehensive machine learning solution for the [Avito Demand Prediction Challenge](https://www.kaggle.com/c/avito-demand-prediction), which aims to predict the probability that an advertisement will result in a deal based on various features including text, categorical, and user behavior data.

## 🏆 Competition Overview

**Challenge**: Predict the probability that an advertisement on Avito will result in a deal
- **Target**: `deal_probability` (continuous value between 0 and 1)
- **Evaluation Metric**: Root Mean Squared Error (RMSE)
- **Dataset**: ~1.5M advertisements with text, categorical, and user features
- **Domain**: Russian online classified advertisements marketplace

**Business Impact**: Helping Avito optimize their marketplace by predicting advertisement success rates, enabling better pricing strategies and user experience improvements.

## 🚀 Key Features

- **Multi-Level Ensemble Architecture**: Level-1 feature-specific models + Level-2 ensemble blending
- **Advanced Text Processing**: TF-IDF vectorization with Russian language optimization
- **Comprehensive Feature Engineering**: 60+ features from text, categorical, and user behavior data
- **Cross-Validation Strategy**: 5-fold stratified cross-validation for robust evaluation
- **Modular Pipeline Design**: Clean, maintainable codebase following software engineering best practices

## 📁 Project Structure

```
avito/
├── main.py              # Main entry point with CLI interface
├── example_usage.py     # Usage demonstrations and examples
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── src/                # Source code package
│   ├── __init__.py     # Package initialization
│   ├── config.py       # Centralized configuration management
│   ├── data_utils.py   # Data loading and preprocessing utilities
│   ├── feature_engineering.py # Feature generation and text processing
│   ├── models.py       # Model architectures and training logic
│   └── pipeline.py     # Main training pipeline orchestration
├── features/           # Original feature engineering scripts (preserved)
├── model/              # Original model scripts (preserved)
└── data/               # Original data processing scripts (preserved)
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd avito
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare data**:
   - Download competition data from Kaggle
   - Place files in `../../data/download/` directory:
     - `train.csv`
     - `test.csv`
     - `train_active.csv`
     - `test_active.csv`

## 📊 Data Preparation

### Dataset Structure
```
data/
├── download/
│   ├── train.csv           # Training data with deal_probability
│   ├── test.csv            # Test data for predictions
│   ├── train_active.csv    # Active user data
│   └── test_active.csv     # Active user data for test
└── data/
    ├── files/              # Cross-validation folds
    └── features/           # Generated feature files
```

### Create Cross-Validation Folds
```bash
python main.py --step preprocess
```

This creates 5-fold cross-validation splits with consistent random state for reproducibility.

## 🏗️ Feature Engineering

### Text Features (High Impact)
- **TF-IDF Vectorization**: Title and description text with Russian stop words
- **Text Statistics**: Word count, character count, punctuation ratio
- **Language Detection**: Russian vs English character ratios
- **Sentiment Analysis**: Positive/negative word counting based on target correlation
- **Parameter Concatenation**: Combined categorical parameters as text features

### User Behavior Features (Medium Impact)
- **Activity Patterns**: Unique categories, titles, and parameters per user
- **Engagement Metrics**: Number of listings and diversity metrics
- **Data Quality**: Missing value patterns per user

### Categorical Features (Medium Impact)
- **Count Aggregations**: Category-city, category, user, parameter combinations
- **Frequency Features**: Image top-1 and parameter frequency counts
- **Hierarchical Aggregations**: Multi-level categorical combinations

### Temporal Features (Low Impact)
- **Day of Week**: Activation date weekday encoding

### Feature Engineering Pipeline
```python
from src.feature_engineering import FeaturePipeline
from src.config import Config

config = Config()
pipeline = FeaturePipeline(config)
pipeline.generate_all_features()
```

## 🎯 Model Architecture

### Level-1 Models (Feature-Specific)
1. **Text Model**: Ridge Regression on TF-IDF features
   - Title + Description + Parameters TF-IDF
   - 50,000 features for description, unlimited for title
   - Alpha: 20.0 for regularization

2. **User Model**: Ridge Regression on user behavior features
   - 4 normalized user features
   - Alpha: 0.00000001 (minimal regularization)

### Level-2 Ensemble (Meta-Learning)
- **Linear Regression**: Blends Level-1 predictions
- **Cross-validation**: 5-fold ensemble training
- **Prediction Clipping**: Ensures outputs in [0,1] range

### Model Configuration
```python
# Text Model Parameters
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
RIDGE_ALPHA = 20.0

# User Model Parameters  
USER_FEATURES = 4
USER_RIDGE_ALPHA = 0.00000001
```

## 🎯 Training Pipeline

### Phase 1: Feature Engineering
- Text preprocessing and TF-IDF vectorization
- User behavior feature generation
- Categorical aggregation features
- Cross-validation fold creation

### Phase 2: Level-1 Training
- Text model training on each fold
- User model training on each fold
- Out-of-fold predictions generation

### Phase 3: Level-2 Ensemble
- Meta-feature creation from Level-1 predictions
- Ensemble model training with cross-validation
- Final prediction blending

### Training Configuration
```python
# Cross-Validation
N_FOLDS = 5
RANDOM_STATE = 2017

# Text Processing
STOP_WORDS = 'russian'
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 50000
```

## 🔧 Text Processing

### Russian Language Optimization
The solution includes specialized processing for Russian text:

1. **Stop Words Removal**: Russian language stop words filtering
2. **Character Analysis**: Russian vowel detection and English character counting
3. **Sentiment Analysis**: Positive/negative word identification based on target correlation
4. **Text Normalization**: Consistent preprocessing pipeline

### Text Feature Pipeline
```python
class TextPreprocessor:
    def clean_text(self, text: str) -> str:
        # Lowercase conversion
        # Digit separation
        # Punctuation removal
        # Whitespace normalization
    
    def has_russian_vowels(self, text: str) -> int:
        # Russian vowel detection
    
    def count_english_chars(self, text: str) -> int:
        # English character counting
```

## 📈 Results

### Validation Performance
- **Cross-Validation RMSE**: ~0.215-0.220 (5-fold CV)
- **Model Architecture**: Multi-level ensemble significantly improved performance
- **Key Insights**:
  - Text features provide the strongest predictive signal
  - User behavior features add valuable complementary information
  - Ensemble approach reduces overfitting and improves generalization

### Feature Importance Analysis
1. **Text Features** (High): TF-IDF features from title and description
2. **User Features** (Medium): User activity and behavior patterns
3. **Count Features** (Medium): Categorical aggregation patterns
4. **Date Features** (Low): Temporal patterns have limited impact

### Model Performance Breakdown
- **Text Model**: Primary predictive power from content analysis
- **User Model**: Secondary signal from user behavior patterns
- **Ensemble**: Combines strengths while reducing individual model weaknesses

## 🚀 Usage

### Quick Start
```bash
# 1. Run complete pipeline
python main.py --step all

# 2. Or run individual steps
python main.py --step preprocess    # Create cross-validation folds
python main.py --step features      # Generate features
python main.py --step train         # Train models
python main.py --step evaluate      # Evaluate performance
python main.py --step submission    # Generate submission
```

### Custom Configuration
```bash
# Custom configuration via command line
python main.py --step all \
    --n-folds 10 \
    --random-state 42 \
    --output-dir ./custom_output
```

### Programmatic Usage
```python
from src.config import Config
from src.pipeline import AvitoPipeline

# Custom configuration
config = Config()
config.avito.N_FOLDS = 10
config.avito.RANDOM_STATE = 42

# Initialize pipeline
pipeline = AvitoPipeline(config)

# Run specific steps
pipeline.preprocess_data()
pipeline.generate_features()
pipeline.train_models()
submission = pipeline.generate_submission()
```

### Model Inference
```python
from src.models import TextModel, UserModel
from src.config import Config

# Load trained models (after training)
config = Config()
text_model = TextModel(config)
user_model = UserModel(config)

# Make predictions on new data
text_predictions = text_model.predict(new_text_data)
user_predictions = user_model.predict(new_user_data)
```

## 🔬 Technical Details

### Cross-Validation Strategy
- **Stratified Splits**: Maintains target distribution across folds
- **Random State**: 2017 for reproducible results
- **Validation**: Hold-out validation within each fold
- **Out-of-fold Predictions**: Prevents data leakage

### Text Processing Pipeline
- **TF-IDF Vectorization**: Sublinear TF scaling, L2 normalization
- **N-gram Range**: (1,2) for unigrams and bigrams
- **Russian Stop Words**: Language-specific preprocessing
- **Feature Selection**: Top 50,000 features by frequency

### Optimization Strategy
- **Ridge Regression**: L2 regularization for text features
- **Feature Normalization**: Z-score normalization for user features
- **Ensemble Blending**: Linear regression for meta-learning
- **Prediction Clipping**: Ensures valid probability range

### Hardware Requirements
- **RAM**: 8GB+ system memory (for TF-IDF processing)
- **Storage**: 10GB+ for dataset and feature files
- **CPU**: Multi-core recommended for parallel processing

## 📚 Key Learnings

1. **Text Features Dominate**: TF-IDF features from title and description provide the strongest predictive signal
2. **Multi-Level Ensembles**: Hierarchical ensemble architecture improves performance over single models
3. **Language-Specific Processing**: Russian language optimization significantly improved text feature quality
4. **Feature Engineering**: Careful feature creation and validation prevents data leakage
5. **Cross-Validation**: Robust CV strategy essential for reliable performance estimation

## 🎯 Business Impact

### Marketplace Optimization
- **Pricing Strategy**: Deal probability helps optimize advertisement pricing
- **User Experience**: Better ad placement and visibility for high-probability deals
- **Revenue Optimization**: Focus resources on advertisements with higher success rates

### Technical Innovation
- **Multi-Modal Features**: Combining text, categorical, and behavioral data
- **Ensemble Architecture**: Hierarchical model combination for improved performance
- **Scalable Pipeline**: Modular design enables easy feature addition and model updates

## 🎯 Future Improvements

- **Deep Learning**: Neural networks for text processing (BERT, RoBERTa)
- **Image Features**: Computer vision models for advertisement images
- **Temporal Modeling**: Time series features for user behavior patterns
- **Advanced Ensembles**: Stacking with non-linear meta-learners
- **Real-Time Inference**: Model optimization for production deployment

## 📖 References

- [Avito Demand Prediction Challenge](https://www.kaggle.com/c/avito-demand-prediction)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Cross-Validation Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)

## 📄 License

This project is for educational and research purposes. Please ensure compliance with competition rules and data usage policies.

---

**Note**: This solution achieved competitive performance in the Avito Demand Prediction Challenge through comprehensive feature engineering, multi-level ensemble architecture, and Russian language-specific text processing. The codebase has been refactored for clarity, maintainability, and reproducibility, making it suitable for portfolio demonstration and further research.