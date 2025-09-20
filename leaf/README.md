# Cassava Leaf Disease Classification

A comprehensive deep learning solution for the Cassava Leaf Disease Classification competition, featuring advanced computer vision techniques, ensemble methods, and robust training strategies.

## 🏆 Competition Overview

The Cassava Leaf Disease Classification challenge required participants to build models that can identify diseases in cassava plant leaves from images. This solution achieved competitive performance through a combination of:

- **EfficientNet-B4** architecture with transfer learning
- **Advanced data augmentation** with Albumentations
- **Stochastic Weight Averaging (SWA)** for improved generalization
- **Multi-model ensemble** with logistic regression blending
- **5-fold cross-validation** for robust model evaluation

## 🚀 Key Features

### 1. Advanced Computer Vision Pipeline
- **EfficientNet-B4** backbone with ImageNet pretrained weights
- **Comprehensive data augmentation** including geometric and color transformations
- **Smart preprocessing** with padding, resizing, and normalization
- **Test Time Augmentation (TTA)** support for improved predictions

### 2. Robust Training Strategy
- **5-fold stratified cross-validation** ensuring balanced splits
- **Cosine annealing learning rate scheduling** with warm restarts
- **Stochastic Weight Averaging** for better generalization
- **Early stopping** and model checkpointing
- **Gradient accumulation** for effective large batch training

### 3. Ensemble Methods
- **Multiple model versions** with different configurations
- **Logistic regression blending** for optimal model combination
- **Cross-validation based ensemble** training
- **Weighted averaging** for final predictions

### 4. Professional Code Architecture
- **Modular design** with clear separation of concerns
- **Configurable parameters** for easy experimentation
- **Comprehensive logging** and progress tracking
- **Memory-efficient** data loading and training

## 📁 Project Structure

```
leaf/
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Main training pipeline
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py    # Data preparation and analysis
│   │   └── data_utils.py           # Dataset classes and data loaders
│   ├── models/                  # Model architectures and loss functions
│   │   ├── __init__.py
│   │   ├── models.py               # EfficientNet model architecture
│   │   └── loss.py                 # Custom loss functions
│   ├── training/                # Training and inference modules
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training utilities with SWA
│   │   ├── inference.py            # Prediction pipeline
│   │   └── scoring.py              # Model evaluation and metrics
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── config.py               # Configuration parameters
│       └── ensemble.py             # Ensemble methods
├── main.py                     # Command-line interface
├── example_usage.py           # Usage examples
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## 🔧 Complete Pipeline

The solution provides a comprehensive end-to-end pipeline organized into logical modules:

### 📊 Data Processing (`src/data/`)
- **`data_preprocessing.py`**: Data preparation, fold creation, and analysis
- **`data_utils.py`**: Dataset classes, data loaders, and augmentation pipelines

### 🤖 Models (`src/models/`)
- **`models.py`**: EfficientNet-B4 classifier architecture
- **`loss.py`**: Custom loss functions including Focal Loss and Label Smoothing

### 🏋️ Training (`src/training/`)
- **`trainer.py`**: Advanced training utilities with SWA and validation
- **`inference.py`**: Efficient inference pipeline with TTA support
- **`scoring.py`**: Model evaluation, metrics, and visualization

### 🛠️ Utilities (`src/utils/`)
- **`config.py`**: Centralized configuration management
- **`ensemble.py`**: Multi-model ensemble and blending methods

### 🎯 Main Pipeline (`src/pipeline.py`)
- **Orchestrates** the complete workflow from data prep to final predictions

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd leaf
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up data directory structure**:
```
data/
├── raw/
│   ├── train.csv
│   ├── data.csv
│   └── train_images/
├── merged/
│   ├── data.csv
│   └── train_images/
├── test_images/
└── test.csv
```

## 🎯 Usage

### Command Line Interface

```bash
# Prepare data and create folds
python main.py --mode prepare_data --data_dir ../../data

# Train a specific model version and fold
python main.py --mode train --version version7 --fold 0

# Score a model on test data
python main.py --mode score --version version7 --test_path ../../data/test.csv

# Create ensemble predictions
python main.py --mode ensemble

# Run complete pipeline
python main.py --mode full_pipeline --data_dir ../../data --test_path ../../data/test.csv
```

### Python API

```python
from src.pipeline import CassavaPipeline

# Initialize pipeline
pipeline = CassavaPipeline()

# Prepare data
pipeline.prepare_data('../../data')

# Train individual models
for version in ['version0', 'version1', 'version2']:
    for fold in range(5):
        pipeline.train_model(version, fold)

# Score models
for version in ['version0', 'version1', 'version2']:
    pipeline.score_model(version, '../../data/test.csv')

# Create ensemble
ensemble_results = pipeline.create_ensemble()

# Run full pipeline
final_predictions = pipeline.run_full_pipeline('../../data', '../../data/test.csv')
```

## 🔬 Technical Details

### Model Architecture

The solution uses **EfficientNet-B4** with custom modifications:

```python
class EfficientNetModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4_ns', num_classes=5, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
```

### Data Augmentation Strategy

**Training Augmentations**:
- Padding to 600x800, then resize
- Random resized crop to 512x512
- Geometric transformations (transpose, flip, rotate)
- Color augmentations (brightness, contrast, hue, saturation)
- Advanced augmentations (coarse dropout, cutout)
- ImageNet normalization

**Validation Augmentations**:
- Padding to 600x800, then resize
- Center crop to 512x512
- ImageNet normalization

### Training Strategy

**Key Parameters**:
- Learning rate: 1e-4 with cosine annealing
- Batch size: 6 with gradient accumulation
- Epochs: 20 with early stopping
- Optimizer: AdamW with weight decay 1e-6
- Loss: Cross Entropy Loss

**Advanced Features**:
- **Stochastic Weight Averaging**: Applied after epoch 7
- **Gradient Accumulation**: Every 4 steps
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Early Stopping**: Based on validation accuracy

### Ensemble Strategy

1. **Multiple Model Versions**: 8 different configurations (version0-version7)
2. **Cross-Validation**: 5-fold CV for each version
3. **Logistic Regression Blending**: Combines predictions optimally
4. **Polynomial Features**: Enhanced feature space for blending

## 📊 Performance Metrics

The solution achieved competitive performance through:

- **Robust cross-validation**: 5-fold CV ensuring generalization
- **Advanced augmentation**: Improved model robustness to variations
- **Ensemble stability**: Reduced variance through model combination
- **SWA optimization**: Better generalization through weight averaging

### Class Distribution Analysis

The dataset contains 5 classes of cassava diseases:
- **Cassava Bacterial Blight (CBB)**: ~5%
- **Cassava Brown Streak Disease (CBSD)**: ~10%
- **Cassava Green Mottle (CGM)**: ~5%
- **Cassava Mosaic Disease (CMD)**: ~60% (majority class)
- **Healthy**: ~20%

## 🔧 Configuration

Key parameters can be adjusted in `src/utils/config.py`:

```python
class Config:
    SEED = 42
    NUM_CLASSES = 5
    IMAGE_SIZE = 512
    BATCH_SIZE = 6
    NUM_WORKERS = 2
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    DEVICE = 'cuda:0'
    MODEL_NAME = 'tf_efficientnet_b4_ns'
```

## 🚀 Advanced Features

### Memory Optimization
- **Gradient accumulation** for effective large batch training
- **Memory-efficient data loading** with proper cleanup
- **Mixed precision training** support
- **CUDA memory management** with explicit cleanup

### Training Stability
- **Gradient clipping** to prevent exploding gradients
- **Learning rate scheduling** with cosine annealing
- **Early stopping** based on validation metrics
- **Model checkpointing** for recovery

### Reproducibility
- **Fixed random seeds** across all components
- **Deterministic data splits** for consistent results
- **Checkpoint saving** for model recovery
- **Comprehensive logging** for debugging

## 📈 Results and Insights

### Key Innovations

1. **Advanced Data Augmentation**: Comprehensive augmentation pipeline significantly improved model robustness and generalization.

2. **Stochastic Weight Averaging**: SWA implementation provided better generalization compared to standard training.

3. **Ensemble Strategy**: Sophisticated ensemble combining multiple model versions with logistic regression blending.

4. **Cross-Validation Approach**: 5-fold stratified CV ensured robust evaluation and prevented overfitting.

### Performance Improvements

- **Augmentation robustness**: Models trained with comprehensive augmentations showed improved performance on test variations
- **Ensemble stability**: Reduced prediction variance through model combination
- **SWA benefits**: Better generalization through weight averaging
- **Cross-validation reliability**: Consistent performance across different data splits

## 🔮 Future Enhancements

1. **Advanced Architectures**: Integration of Vision Transformers (ViT) or ConvNeXt
2. **Self-Supervised Learning**: Pretraining on unlabeled cassava images
3. **Multi-Scale Training**: Training with multiple image resolutions
4. **Active Learning**: Intelligent sample selection for annotation
5. **Model Compression**: Knowledge distillation for deployment efficiency

## 📚 References

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407)
- [Albumentations: fast and flexible image augmentations](https://arxiv.org/abs/1809.06839)
- [Cassava Leaf Disease Classification Competition](https://www.kaggle.com/c/cassava-leaf-disease-classification)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or suggestions, please open an issue or contact the maintainer.

---

*This solution demonstrates advanced techniques in computer vision, deep learning, and ensemble methods, making it a valuable reference for similar competitions and real-world applications in agricultural AI.*
