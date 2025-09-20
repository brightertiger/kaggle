# RFCX Species Audio Detection

A deep learning solution for classifying rainforest audio recordings into 24 different species using convolutional neural networks on mel-spectrograms.

## 🎯 Project Overview

This project tackles the **Rainforest Connection Species Audio Detection** challenge, where the goal is to identify different species from audio recordings collected in rainforest environments. The solution uses advanced deep learning techniques including:

- **Audio Signal Processing**: Converting raw audio to mel-spectrograms
- **Computer Vision**: Applying CNN architectures to spectrogram images
- **Data Augmentation**: Mixup, audio augmentation, and image augmentation
- **Ensemble Methods**: Multiple model architectures and test-time augmentation
- **Cross-Validation**: Stratified k-fold validation for robust evaluation

## 🏗️ Architecture

### Data Pipeline
1. **Audio Preprocessing**: Resample audio to 32kHz and extract 5-second segments
2. **Feature Extraction**: Convert audio to mel-spectrograms (300 mel bins)
3. **Image Conversion**: Transform spectrograms to RGB images for CNN processing
4. **Augmentation**: Apply mixup, audio noise, and image transformations

### Model Architecture
- **Backbone**: ResNet50 variants (Res2Net50, ResNeSt50)
- **Input**: 300x300 RGB spectrogram images
- **Output**: 24-class species classification
- **Loss Function**: Binary Cross-Entropy with Logits

### Training Strategy
- **Cross-Validation**: 5-fold stratified validation
- **Optimization**: Adam optimizer with ReduceLROnPlateau scheduler
- **Regularization**: Dropout, data augmentation, early stopping
- **Ensemble**: Multiple model predictions with ranking-based fusion

## 📁 Project Structure

```
rfcx/
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── data_utils.py      # Data loading and preprocessing
│   ├── models.py          # Model architectures
│   ├── trainer.py         # Training logic
│   ├── predictor.py       # Inference and prediction
│   └── pipeline.py        # End-to-end pipeline
├── main.py                # Command-line interface
├── example_usage.py       # Usage examples
├── requirements.txt       # Dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rfcx
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare data structure**:
```
data/
├── raw/
│   ├── train/           # Original FLAC files
│   └── test/            # Test FLAC files
├── resample/
│   ├── train/           # Resampled NPY files
│   └── test/            # Resampled NPY files
├── train_tp.csv         # Training labels
├── train_fp.csv         # False positive labels
└── sample_submission.csv # Submission format
```

### Basic Usage

**Train a model**:
```bash
python main.py --mode train --model resnet --epochs 15
```

**Generate predictions**:
```bash
python main.py --mode predict --model resnet --tta
```

**Run full pipeline**:
```bash
python main.py --mode full --model resnet --tta --ensemble
```

### Python API

```python
from src.config import Config
from src.pipeline import run_full_pipeline

# Configure the pipeline
config = Config()
config.training.epochs = 15
config.training.batch_size = 8

# Run complete pipeline
run_full_pipeline(config, model_type="resnet", apply_tta=True, create_ensemble=True)
```

## 🔧 Configuration

The `Config` class allows easy customization of all parameters:

```python
from src.config import Config

config = Config()

# Audio processing
config.audio.sample_rate = 32000
config.audio.n_mels = 300
config.audio.segment_length = 5

# Model settings
config.model.num_classes = 24
config.model.pretrained = True

# Training parameters
config.training.batch_size = 8
config.training.learning_rate = 1e-4
config.training.epochs = 15
config.training.num_folds = 5
```

## 🎵 Audio Processing Pipeline

### 1. Audio Resampling
- Convert audio files to 32kHz sampling rate
- Save as NumPy arrays for faster loading

### 2. Spectrogram Generation
- Extract mel-spectrograms with 300 mel bins
- Use librosa for high-quality audio processing
- Convert to dB scale for better dynamic range

### 3. Image Conversion
- Transform spectrograms to RGB images
- Normalize and scale to 0-255 range
- Apply color mapping for CNN compatibility

### 4. Data Augmentation
- **Audio**: Gaussian noise injection
- **Image**: Random crops, brightness/contrast, dropout
- **Mixup**: Linear combination of samples and labels

## 🧠 Model Architectures

### ResNet50 Variants
- **Res2Net50**: Improved residual connections
- **ResNeSt50**: Split-attention mechanism
- **Transfer Learning**: Pre-trained ImageNet weights

### Training Features
- **Mixed Precision**: Efficient GPU utilization
- **Gradient Clipping**: Stable training
- **Learning Rate Scheduling**: Adaptive optimization
- **Early Stopping**: Prevent overfitting

## 📊 Evaluation Metrics

- **Accuracy**: Top-1 classification accuracy
- **Cross-Validation**: Stratified 5-fold validation
- **Ensemble Performance**: Multiple model fusion
- **Test-Time Augmentation**: Robust inference

## 🔄 Advanced Techniques

### Test-Time Augmentation (TTA)
- Multiple augmented versions of test samples
- Average predictions for improved robustness
- Audio noise and image transformations

### Ensemble Methods
- Multiple model architectures
- Ranking-based prediction fusion
- Cross-validation model averaging

### Data Augmentation Strategy
- **Mixup**: Interpolation between samples
- **Audio Augmentation**: Noise injection
- **Image Augmentation**: Geometric and photometric transforms

## 📈 Results

The solution achieved competitive performance through:

- **Robust Preprocessing**: High-quality audio feature extraction
- **Advanced Augmentation**: Mixup and multi-modal augmentation
- **Ensemble Methods**: Multiple architectures and TTA
- **Cross-Validation**: Reliable performance estimation

## 🛠️ Technical Highlights

### Code Quality
- **Modular Design**: Clean separation of concerns
- **Configuration Management**: Centralized parameter control
- **Type Hints**: Improved code readability
- **Error Handling**: Robust pipeline execution

### Performance Optimizations
- **Efficient Data Loading**: Multi-process data loading
- **Memory Management**: Proper GPU memory handling
- **Batch Processing**: Optimized inference pipeline
- **Caching**: Preprocessed data storage

### Reproducibility
- **Seed Management**: Deterministic results
- **Version Control**: Tracked model versions
- **Logging**: Comprehensive training logs
- **Checkpointing**: Model state preservation

## 🔮 Future Improvements

1. **Advanced Architectures**: Vision Transformers, EfficientNet
2. **Audio-Specific Models**: Wav2Vec2, AudioCLIP
3. **Multi-Modal Learning**: Combine audio and metadata
4. **Active Learning**: Intelligent sample selection
5. **Model Compression**: Quantization and pruning

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **timm**: Pre-trained model library
- **librosa**: Audio processing
- **albumentations**: Image augmentation
- **audiomentations**: Audio augmentation
- **scikit-learn**: Machine learning utilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Rainforest Connection**: For organizing the competition
- **Kaggle Community**: For insights and discussions
- **Open Source Libraries**: PyTorch, librosa, timm, and others
- **Research Papers**: Mixup, ResNet, ResNeSt, and related work

## 👨‍💻 Author

**Ujjwal Singh Rao**
- LinkedIn: [linkedin.com/in/brightertiger](https://linkedin.com/in/brightertiger)
- GitHub: [github.com/brightertiger](https://github.com/brightertiger)

---

*This project demonstrates advanced deep learning techniques for audio classification, showcasing expertise in signal processing, computer vision, and ensemble methods.*
