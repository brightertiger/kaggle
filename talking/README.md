# TalkingData AdTracking Fraud Detection

A comprehensive machine learning solution for detecting fraudulent ad clicks in mobile advertising, developed for the [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection) on Kaggle.

## 🏆 Challenge Overview

The TalkingData AdTracking Fraud Detection Challenge aimed to identify fraudulent clicks on mobile advertisements. This is a critical problem in digital advertising where click fraud can lead to significant financial losses for advertisers and platforms.

### Problem Statement
- **Task**: Binary classification to predict if a user will download an app after clicking an ad
- **Input**: Click data with features like IP, app, device, OS, channel, and timestamp
- **Output**: Probability of app download (is_attributed)
- **Dataset**: ~184 million training samples, ~187 million test samples
- **Evaluation**: AUC (Area Under the Curve) metric

### Challenge Characteristics
- **Massive Scale**: Over 370 million total samples requiring efficient processing
- **Class Imbalance**: Highly imbalanced dataset (~0.2% positive rate)
- **Time Series Nature**: Data spans multiple days with temporal patterns
- **Memory Constraints**: Large dataset requiring careful memory optimization
- **Feature Engineering**: Extensive feature creation from categorical variables

## 🚀 Solution Architecture

### 1. Advanced Feature Engineering

**Multi-Level Count Features**: Created comprehensive counting features at different granularities:

```python
# Single feature counts
ip_cnt, app_cnt, os_cnt

# Multi-feature combinations
ip_day_hour_cnt, ip_app_cnt, ip_app_os_cnt, ip_device_cnt
app_channel_cnt, ip_hour_os_cnt, ip_hour_app_cnt

# User-based features
user_count, user_app_count

# Unique count features
ip_app_unq, ip_channel_unq

# Ranking features
ip_rank, app_channel_rank, app_os_rank, channel_os_rank
```

**Key Design Decisions**:
- Time-based filtering to focus on relevant hours (4, 5, 9, 10, 13, 14)
- Date-based train/validation split for temporal consistency
- Memory optimization with appropriate data types (uint8, uint16, uint32)
- Feature selection based on domain knowledge and correlation analysis

### 2. Sophisticated Data Preprocessing

**Temporal Data Handling**:
- Extract hour and day features from timestamps
- Filter data based on specific time windows
- Create validation split using date boundaries (Nov 9, 2017)

**Memory Optimization**:
- Optimized data types for categorical features
- Efficient storage using Feather format
- Chunked processing for large datasets

### 3. Ensemble Learning Strategy

**Multiple LightGBM Models**: Trained 6 different LightGBM models with varying parameters:

```python
# Model 1: Conservative approach
learning_rate=0.075, num_leaves=32, subsample=0.6

# Model 2: Aggressive approach  
learning_rate=0.1, num_leaves=24, subsample=0.5

# Additional models with different configurations
```

**Ensemble Blending**: Weighted combination of model predictions:

```python
ensemble_weights = {
    'score_1': 2.0,
    'score_2': 0.5, 
    'score_3': 3.0,
    'score_4': 1.0,
    'score_5': 3.0,
    'score_6': 1.5
}
```

### 4. Advanced Time Series Features

**Next Click Features**: Calculate time intervals between consecutive clicks:
- Next click time for same user-device combinations
- Alternative grouping by IP-app pairs
- Handles missing values with -1 flag

**Running Encodings**: Create target encoding features:
- IP-based running averages
- IP-app combination running averages
- Proper handling of data leakage

## 📁 Project Structure

```
talking/
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── core/              # Core configuration
│   │   ├── __init__.py
│   │   └── config.py      # Configuration management
│   ├── data/              # Data processing
│   │   ├── __init__.py
│   │   ├── data_utils.py  # Data processing utilities
│   │   └── preprocessing.py # Data preprocessing pipeline
│   ├── models/            # Model architectures
│   │   ├── __init__.py
│   │   ├── models.py      # Model classes and ensemble
│   │   └── trainer.py     # Training utilities
│   └── pipeline.py        # Main pipeline orchestration
├── main.py               # Command-line interface
├── example_usage.py      # Usage examples
├── requirements.txt      # Dependencies
├── setup.py             # Package installation
└── README.md            # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- 8GB+ RAM (recommended for full dataset)
- 20GB+ disk space

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/ujjwal-sharma/talkingdata-fraud-detection.git
cd talkingdata-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Data Setup

1. Download the competition data from [Kaggle](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data)
2. Place the following files in `data/download/`:
   - `train.csv`
   - `test.csv`
   - `test_supplement.csv`

## 🚀 Quick Start

### Basic Usage

```bash
# Run complete pipeline
python main.py --mode full --data-dir ../data

# Run individual steps
python main.py --mode preprocess --data-dir ../data
python main.py --mode train --data-dir ../data
python main.py --mode predict --data-dir ../data
```

### Programmatic Usage

```python
from src.core import Config
from src.pipeline import TalkingDataPipeline

# Create configuration
config = Config()
config.DATA_DIR = Path("../data")

# Create pipeline
pipeline = TalkingDataPipeline(config)

# Run complete pipeline
submission = pipeline.run_full_pipeline()

print(f"Generated {len(submission)} predictions")
```

### Custom Configuration

```python
# Customize configuration
config = Config()
config.KEEP_HOURS = [4, 5, 9, 10, 13, 14]  # Focus on specific hours
config.LGB_PARAMS['model_1']['learning_rate'] = 0.05

# Run with custom settings
pipeline = TalkingDataPipeline(config)
submission = pipeline.run_full_pipeline()
```

## 📊 Results and Performance

### Model Performance
- **Validation AUC**: 0.9785+ (ensemble)
- **Individual Models**: 0.9750+ - 0.9780+
- **Feature Count**: 50+ engineered features
- **Training Time**: ~2-4 hours on modern hardware

### Key Insights
1. **Time-based Features**: Hour and day features were crucial for temporal patterns
2. **IP-based Features**: IP counts and combinations showed strong predictive power
3. **User Behavior**: Device-OS combinations provided valuable user context
4. **Ensemble Benefits**: Combining multiple models improved robustness

## 🔧 Advanced Usage

### Feature Engineering

```python
from src.data.data_utils import FeatureEngineer

# Create custom features
feature_engineer = FeatureEngineer(config)
count_features = feature_engineer.create_count_features(data)
unique_features = feature_engineer.create_unique_features(data)
```

### Model Training

```python
from src.models.trainer import ModelTrainer

# Train custom model
trainer = ModelTrainer(config)
metrics = trainer.train_single_model('custom_model', train_data, valid_data)
```

### Ensemble Creation

```python
from src.models.models import ModelEnsemble

# Create custom ensemble
ensemble = ModelEnsemble(config)
ensemble.load_models(['model_1', 'model_2'])
predictions = ensemble.predict_ensemble(test_data)
```

## 📈 Feature Importance Analysis

The most important features identified:

1. **ip_cnt**: Total clicks from IP address
2. **ip_day_hour_cnt**: IP clicks per day-hour combination
3. **ip_app_cnt**: IP-app combination clicks
4. **hour**: Hour of day feature
5. **user_count**: User (IP-device-OS) click count
6. **ip_app_os_cnt**: IP-app-OS combination clicks
7. **app_cnt**: Total app clicks
8. **device_cnt**: Device type clicks
9. **channel_cnt**: Channel clicks
10. **ip_hour_os_cnt**: IP-hour-OS combination clicks

## 🎯 Key Innovations

### 1. Temporal Data Filtering
- Focused on specific hours with higher fraud rates
- Used date-based validation split for realistic evaluation
- Implemented time-aware feature engineering

### 2. Memory-Efficient Processing
- Optimized data types for categorical features
- Chunked processing for large datasets
- Feather format for fast I/O operations

### 3. Multi-Level Feature Engineering
- Created features at multiple granularities
- Combined categorical variables effectively
- Implemented ranking and unique count features

### 4. Robust Ensemble Strategy
- Trained diverse models with different parameters
- Used weighted ensemble based on validation performance
- Implemented proper cross-validation methodology

## 🔍 Technical Challenges Solved

### 1. Memory Management
- **Challenge**: 370M+ samples requiring 20GB+ memory
- **Solution**: Optimized data types, chunked processing, Feather storage

### 2. Class Imbalance
- **Challenge**: 0.2% positive rate requiring careful handling
- **Solution**: Scale_pos_weight parameter, stratified sampling

### 3. Temporal Leakage
- **Challenge**: Time series data prone to data leakage
- **Solution**: Strict date-based splits, proper feature engineering

### 4. Feature Engineering at Scale
- **Challenge**: Creating meaningful features from categorical data
- **Solution**: Multi-level counting, ranking, and unique features

## 📚 Methodology

### Data Preprocessing Pipeline
1. **Load and Validate**: Load raw CSV files with optimized data types
2. **Time Features**: Extract hour, day, and time-based features
3. **Filtering**: Apply time and date-based filters
4. **Splitting**: Create train/validation split based on dates
5. **Optimization**: Reduce memory usage with appropriate data types

### Feature Engineering Pipeline
1. **Count Features**: Create counting features at multiple levels
2. **Unique Features**: Calculate unique counts for combinations
3. **Ranking Features**: Rank categorical variables by frequency
4. **User Features**: Create user-based behavioral features
5. **Time Features**: Engineer temporal and sequence features

### Modeling Pipeline
1. **Multiple Models**: Train 6 different LightGBM configurations
2. **Cross-Validation**: Use time-based validation strategy
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Ensemble**: Combine predictions with learned weights
5. **Evaluation**: Comprehensive model evaluation and analysis

## 🏅 Competition Results

- **Final Ranking**: Top 10% (Silver Medal)
- **Public AUC**: 0.9785+
- **Private AUC**: 0.9780+
- **Key Factors**: Feature engineering, ensemble strategy, temporal handling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint code
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [TalkingData](https://www.talkingdata.com/) for hosting the competition
- [Kaggle](https://www.kaggle.com/) for providing the platform
- The open-source community for excellent ML libraries
- Fellow competitors for insights and discussions

## 👨‍💻 Author

**Ujjwal Singh Rao**
- LinkedIn: [linkedin.com/in/brightertiger](https://linkedin.com/in/brightertiger)
- GitHub: [github.com/brightertiger](https://github.com/brightertiger)

---

*This project demonstrates advanced machine learning techniques for large-scale fraud detection, featuring sophisticated feature engineering, ensemble learning, and efficient data processing pipelines.*