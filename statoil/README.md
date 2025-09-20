# Statoil Iceberg Classifier Challenge

A comprehensive deep learning solution for the Statoil Iceberg Classifier Challenge on Kaggle, featuring advanced CNN architectures, ensemble methods, and feature engineering techniques.

## 🏆 Competition Overview

The Statoil Iceberg Classifier Challenge was a computer vision competition where participants were tasked with distinguishing between icebergs and ships in satellite images. The challenge involved:

- **Dataset**: 75x75 pixel satellite images with two radar bands (HH and HV)
- **Target**: Binary classification (iceberg vs ship)
- **Evaluation**: Log Loss metric
- **Challenge**: Limited training data (~1600 images) with high class imbalance

## 🚀 Solution Architecture

### 1. Data Preprocessing Pipeline

The solution implements two distinct image transformation strategies:

#### Source 1: Difference-based Features
- **Transform 1**: Absolute difference between vertical and horizontal bands
- **Transform 2**: Maximum of both bands
- **Transform 3**: Minimum of both bands
- **Normalization**: Z-score normalization for each channel

#### Source 2: Averaging-based Features
- **Channel 1**: Normalized vertical band
- **Channel 2**: Normalized horizontal band  
- **Channel 3**: Normalized average of both bands

### 2. Model Architectures

#### CNN Basic Model
- **Architecture**: Custom CNN with Swish activation
- **Layers**: 6 convolutional layers with BatchNorm and MaxPooling
- **Regularization**: Dropout (0.3) and Batch Normalization
- **Input**: 75x75x3 images + incidence angle
- **Output**: Sigmoid activation for binary classification

#### CNN Advanced Model
- **Architecture**: Deeper CNN with enhanced regularization
- **Features**: Additional data augmentation (shift, zoom, rotation)
- **Training**: Extended epochs (150) with patience-based early stopping

#### VGG16 Transfer Learning Model
- **Base Model**: Pre-trained VGG16 with frozen weights
- **Strategy**: Two-stage training (frozen → fine-tuned)
- **Learning Rates**: 1e-4 (frozen) → 5e-5 (fine-tuned)
- **Architecture**: VGG16 backbone + custom classification head

### 3. Ensemble Methods

#### Simple Stacking
- **Method**: Adaptive ensemble based on prediction confidence
- **Logic**: 
  - If all predictions < threshold → use minimum
  - If all predictions > threshold → use maximum  
  - Otherwise → use mean
- **Thresholds**: Optimized via grid search (15%-95%)

#### XGBoost Stacking
- **Features**: Statistical features from image analysis
- **Engineering**: 246 hand-crafted features including:
  - Basic statistics (mean, std, min, max, median)
  - Texture features (Laplacian, Sobel filters)
  - Distribution features (kurtosis, skewness)
  - Histogram-based features
  - Polynomial feature interactions

### 4. Feature Engineering

The XGBoost model leverages sophisticated feature engineering:

- **Statistical Features**: Mean, standard deviation, min/max, median
- **Texture Analysis**: Laplacian and Sobel edge detection
- **Distribution Metrics**: Kurtosis and skewness
- **Histogram Features**: 20-bin histograms with statistical summaries
- **Polynomial Features**: Cross-channel interactions and combinations

## 📊 Performance Results

- **Individual Models**: 
  - CNN Basic: ~0.18 Log Loss
  - CNN Advanced: ~0.17 Log Loss  
  - VGG16: ~0.16 Log Loss
- **Ensemble Performance**: ~0.15 Log Loss
- **Final Submission**: Top 10% leaderboard position

## 🛠️ Technical Implementation

### Project Structure
```
statoil/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── data_utils.py          # Data preprocessing utilities
│   ├── models.py              # Model architectures
│   ├── trainer.py             # Training pipeline
│   ├── predictor.py           # Prediction utilities
│   ├── feature_engineering.py # Feature extraction
│   └── pipeline.py            # Main pipeline orchestration
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

### Key Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Configuration Management**: Centralized config for easy parameter tuning
- **Cross-Validation**: 5-fold stratified CV for robust evaluation
- **Data Augmentation**: Comprehensive augmentation strategies
- **Model Checkpointing**: Automatic model saving and early stopping
- **Parallel Processing**: Multi-core feature extraction
- **Memory Management**: Efficient data handling with garbage collection

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Setup
1. Download the competition data to `data/download/`
2. Ensure the following files are present:
   - `train.json`
   - `test.json`

### Running the Pipeline
```bash
python main.py
```

### Individual Components
```python
from src.pipeline import IcebergPipeline

# Initialize pipeline
pipeline = IcebergPipeline()

# Run specific components
pipeline.prepare_data()
pipeline.train_models()
pipeline.generate_predictions()
pipeline.create_ensemble()
```

## 🔬 Technical Insights

### Data Challenges
- **Limited Training Data**: Only ~1600 images required careful regularization
- **Class Imbalance**: Ships vs icebergs ratio needed balanced sampling
- **Image Quality**: Satellite radar images with noise and artifacts

### Solution Strategies
- **Ensemble Diversity**: Different architectures capture complementary patterns
- **Transfer Learning**: VGG16 provided robust feature extraction
- **Feature Engineering**: Hand-crafted features improved XGBoost performance
- **Adaptive Stacking**: Confidence-based ensemble improved robustness

### Lessons Learned
- **Data Augmentation**: Critical for small datasets
- **Model Diversity**: Different architectures improve ensemble performance
- **Feature Engineering**: Domain knowledge enhances model performance
- **Cross-Validation**: Essential for reliable performance estimation

## 📈 Model Performance Analysis

### Individual Model Strengths
- **CNN Basic**: Fast training, good baseline performance
- **CNN Advanced**: Better generalization with enhanced regularization
- **VGG16**: Strong feature extraction, best individual performance

### Ensemble Benefits
- **Reduced Variance**: Multiple models reduce prediction uncertainty
- **Improved Robustness**: Different architectures handle edge cases
- **Better Calibration**: Ensemble predictions more reliable

## 🎯 Future Improvements

- **Advanced Architectures**: ResNet, EfficientNet, Vision Transformers
- **Pseudo-Labeling**: Leverage test set for semi-supervised learning
- **Advanced Augmentation**: Mixup, CutMix, AutoAugment
- **Neural Architecture Search**: Automated architecture optimization
- **Multi-Scale Features**: Different image resolutions and scales

## 📚 References

- [Statoil Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Keras Documentation](https://keras.io/)

## 👨‍💻 Author

**Ujjwal Singh Rao**
- LinkedIn: [linkedin.com/in/brightertiger](https://linkedin.com/in/brightertiger)
- GitHub: [github.com/brightertiger](https://github.com/brightertiger)

This solution was developed as part of a comprehensive machine learning portfolio, demonstrating expertise in:
- Deep Learning and Computer Vision
- Ensemble Methods and Model Stacking
- Feature Engineering and Data Preprocessing
- MLOps and Pipeline Development

---

*This project showcases advanced machine learning techniques applied to a real-world computer vision challenge, highlighting both technical depth and practical implementation skills.*