# APTOS 2019 Diabetic Retinopathy Detection

A deep learning solution for the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) competition, which aims to automatically detect diabetic retinopathy severity levels from retinal fundus photographs.

## 🏆 Competition Overview

**Challenge**: Classify diabetic retinopathy severity levels (0-4) from retinal fundus images
- **0**: No DR
- **1**: Mild DR  
- **2**: Moderate DR
- **3**: Severe DR
- **4**: Proliferative DR

**Evaluation Metric**: Quadratic Weighted Kappa Score
**Dataset**: ~36,000 retinal fundus images from multiple sources

## 🚀 Key Features

- **Multi-Stage Training Pipeline**: Pretraining → Fine-tuning → Combined training
- **Advanced Data Augmentation**: Custom transforms preserving retinal image characteristics
- **Ensemble Learning**: Multiple EfficientNet models with different configurations
- **Noise Augmentation**: Label smoothing and variance regularization for robustness
- **Cross-Validation**: Stratified k-fold validation ensuring balanced splits

## 📁 Project Structure

```
aptos/
├── main.py              # Main entry point with CLI interface
├── example_usage.py     # Usage demonstrations and examples
├── train.py             # Legacy training script (deprecated)
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── src/                # Source code package
│   ├── __init__.py     # Package initialization
│   ├── config.py       # Centralized configuration management
│   ├── data_utils.py   # Data loading and preprocessing utilities
│   ├── model.py        # EfficientNet-based model architecture
│   ├── loss.py         # Custom loss functions (MSE + BCE + Variance)
│   ├── trainer.py      # Training loop and validation logic
│   ├── optimizer.py    # RAdam optimizer implementation
│   ├── pipeline.py     # Main training pipeline orchestration
│   └── preprocess.py   # Data preprocessing and fold creation
└── notebook/           # Original Jupyter notebooks (preserved)
    ├── 00-split.ipynb  # Data splitting
    ├── 01-augment.ipynb# Data augmentation visualization
    ├── 02-pretrain.ipynb# Pretraining phase
    ├── 03-train.ipynb  # Fine-tuning phase
    └── 04-combine.ipynb# Combined training phase
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd aptos
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install NVIDIA Apex** (for mixed precision training):
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

## 📊 Data Preparation

### Dataset Structure
```
data/
├── pretrain/
│   ├── train/           # 2015 training images
│   ├── test/           # 2015 test images
│   ├── trainLabels15.csv
│   └── testLabels15.csv
└── train/
    ├── train/          # 2019 training images
    └── trainLabels19.csv
```

### Create Cross-Validation Folds
```bash
python preprocess.py
```

This script creates stratified k-fold splits:
- **2015 Data**: 10-fold cross-validation
- **2019 Data**: 5-fold cross-validation

## 🏗️ Model Architecture

### EfficientNet Backbone
- **Model**: EfficientNet-B5 (pre-trained on ImageNet)
- **Input Size**: 256×256 (pretrain/train) → 330×330 (combined)
- **Features**: 2048-dimensional feature vector

### Classification Head
```python
ClassificationHead:
├── Regression Branch: 2048 → 1 (continuous severity score)
└── Classification Branch: 2048 → 5 (discrete severity levels)
```

### Custom Loss Function
The model uses a combination of three loss components:

1. **MSE Loss**: Regression loss for continuous severity prediction
2. **BCE Loss**: Classification loss with label smoothing (0.9 + 0.02)
3. **Variance Loss**: Consistency regularization for noise-augmented training

```python
Total Loss = α × MSE + (1-α) × BCE + β × Variance
```

Where:
- α = 0.75 (MSE weight)
- β = 0.2 (Variance weight, only in combined training)

## 🎯 Training Pipeline

### Phase 1: Pretraining
- **Data**: 2015 APTOS dataset (train + test)
- **Epochs**: 10
- **Image Size**: 256×256
- **Purpose**: Learn general retinal image features

### Phase 2: Fine-tuning
- **Data**: 2019 APTOS dataset
- **Epochs**: 12
- **Image Size**: 256×256
- **Purpose**: Adapt to competition-specific data distribution

### Phase 3: Combined Training
- **Data**: All datasets with weighted sampling
- **Epochs**: 10
- **Image Size**: 330×330
- **Purpose**: Final model optimization with larger input resolution

### Training Configuration
```python
# Optimizer: RAdam
learning_rate = 1e-4
weight_decay = 1e-5

# Scheduler: StepLR
step_size = 2-5 epochs
gamma = 0.1-0.5

# Mixed Precision: NVIDIA Apex O2
batch_size = 20
num_workers = 6
```

## 🔧 Data Augmentation

### Image-Specific Transforms
The augmentation strategy considers retinal image characteristics:

1. **Aspect Ratio Preservation**: Different transforms for square vs. rectangular images
2. **Geometric Augmentations**:
   - Random horizontal/vertical flips (50% probability)
   - Random crop with padding
   - Random affine transformations
   - Scale range: 1.0-1.25

3. **Color Augmentations**:
   - Color jitter (brightness: 0.5, contrast: 0.3, saturation: 0.3)

### Label Augmentation
- **Label Noise**: Gaussian noise (σ=0.05) during training
- **Label Smoothing**: Soft targets (0.9 + 0.02) for classification

## 📈 Results

### Validation Performance
- **Quadratic Kappa**: ~0.92-0.93 (5-fold CV)
- **Training Strategy**: Multi-stage approach significantly improved performance
- **Key Insights**:
  - Pretraining on 2015 data provided strong initialization
  - Combined training with larger images improved fine-grained classification
  - Noise augmentation enhanced model robustness

### Model Ensemble
The final solution combines multiple models:
- Different EfficientNet variants (B3, B5)
- Various input resolutions (256, 330)
- Multiple training phases (pretrain, train, combine)

## 🚀 Usage

### Quick Start
```bash
# 1. Run complete training pipeline
python main.py --step all

# 2. Or run individual steps
python main.py --step preprocess    # Create cross-validation folds
python main.py --step pretrain      # Pretrain on 2015 data
python main.py --step train         # Train on 2019 data
python main.py --step combine       # Combined training
```

### Custom Training
```bash
# Custom configuration via command line
python main.py --step combine \
    --image-size 512 \
    --batch-size 16 \
    --epochs-combine 15 \
    --learning-rate 5e-5

# Train specific fold
python main.py --step combine --fold 1
```

### Programmatic Usage
```python
from src.config import Config
from src.pipeline import APTOSPipeline

# Custom configuration
config = Config()
config.IMAGE_SIZE = 512
config.BATCH_SIZE = 16
config.NUM_EPOCHS_COMBINE = 15

# Initialize pipeline
pipeline = APTOSPipeline(config)

# Train specific fold
pipeline._combine_fold(fold=1)
```

### Model Inference
```python
import torch
from src.model import DiabeticRetinopathyModel
from src.config import Config

# Load model
config = Config()
model = DiabeticRetinopathyModel(config.MODEL_NAME, config)
checkpoint = torch.load('model/combine/model_1.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
model.eval()
with torch.no_grad():
    regression_output, classification_output = model(image_tensor)
    predicted_severity = torch.round(regression_output.clamp(0, 4))
```

## 🔬 Technical Details

### Cross-Validation Strategy
- **Stratified Splits**: Maintains class distribution across folds
- **Random State**: 2017 (reproducible results)
- **Validation**: Hold-out validation within each fold

### Optimization
- **RAdam Optimizer**: Rectified Adam with adaptive learning rates
- **Mixed Precision**: NVIDIA Apex for faster training and reduced memory usage
- **Gradient Accumulation**: Effective batch size management

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3080/4080 or better)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for dataset and model checkpoints

## 📚 Key Learnings

1. **Multi-Stage Training**: Progressive training from general to specific features
2. **Data Augmentation**: Domain-specific augmentations preserve medical image characteristics
3. **Loss Function Design**: Combining regression and classification losses improves performance
4. **Model Architecture**: EfficientNet provides excellent feature extraction for medical images
5. **Ensemble Methods**: Multiple models with different configurations enhance robustness

## 🎯 Future Improvements

- **Advanced Architectures**: Vision Transformers (ViT) for retinal image analysis
- **Self-Supervised Learning**: Contrastive learning on unlabeled retinal images
- **Multi-Modal Fusion**: Combining fundus images with clinical metadata
- **Uncertainty Quantification**: Bayesian neural networks for confidence estimation
- **Real-Time Inference**: Model optimization for deployment in clinical settings

## 📖 References

- [APTOS 2019 Competition](https://www.kaggle.com/c/aptos2019-blindness-detection)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [RAdam: On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)
- [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)

## 📄 License

This project is for educational and research purposes. Please ensure compliance with competition rules and data usage policies.

---

## 👨‍💻 Author

**Ujjwal Singh Rao**
- LinkedIn: [linkedin.com/in/brightertiger](https://linkedin.com/in/brightertiger)
- GitHub: [github.com/brightertiger](https://github.com/brightertiger)

---

**Note**: This solution achieved competitive performance in the APTOS 2019 competition through careful data preprocessing, advanced augmentation strategies, and multi-stage training approaches. The codebase has been refactored for clarity, maintainability, and reproducibility.
