# Quick, Draw! Recognition Challenge

A comprehensive deep learning solution for the [Quick, Draw! Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition), which aims to classify hand-drawn sketches into 340 different categories using computer vision and neural networks.

## 🏆 Competition Overview

**Challenge**: Classify hand-drawn sketches from the Quick, Draw! dataset into 340 categories
- **Target**: Multi-class classification (340 classes)
- **Evaluation Metric**: Top-3 Accuracy (Categorical Accuracy)
- **Dataset**: ~50M hand-drawn sketches across 340 categories
- **Domain**: Computer Vision, Sketch Recognition, Transfer Learning

**Business Impact**: Understanding sketch recognition has applications in user interface design, creative tools, accessibility features, and human-computer interaction systems.

## 🚀 Key Features

- **Transfer Learning Architecture**: Pre-trained ResNet models (18, 34, 50) with custom classification heads
- **Advanced Data Processing**: Stroke-based sketch rendering with data augmentation
- **Multi-Scale Training**: Configurable image sizes and batch processing
- **Robust Evaluation**: Top-K accuracy metrics with comprehensive logging
- **Production-Ready Pipeline**: Modular design with CLI interface and programmatic API
- **GPU Acceleration**: Multi-GPU support with efficient memory management

## 📁 Project Structure

```
doodle/
├── main.py              # Main entry point with CLI interface
├── example_usage.py     # Usage demonstrations and examples
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── src/                # Source code package
│   ├── __init__.py     # Package initialization
│   ├── config.py       # Centralized configuration management
│   ├── data_utils.py   # Dataset classes and data loading utilities
│   ├── models.py       # Neural network architectures and metrics
│   ├── trainer.py      # Training pipeline and optimization
│   ├── scorer.py       # Inference and prediction generation
│   └── pipeline.py     # End-to-end pipeline orchestration
├── models/             # Original model scripts (preserved)
├── dataloader/         # Original dataloader scripts (preserved)
└── metrics/            # Original metric scripts (preserved)
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd doodle
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare data**:
   - Download Quick, Draw! dataset from Kaggle
   - Place files in `../data/download/` directory with category CSV files
   - Ensure test data is available at `../data/test/test_simplified.csv`

## 📊 Data Preparation

### Dataset Structure
```
data/
├── download/            # Source category CSV files
│   ├── airplane.csv
│   ├── apple.csv
│   └── ... (340 categories)
├── train/               # Training data
│   └── train.csv
├── valid/               # Validation data
│   └── valid.csv
├── test/                # Test data
│   └── test_simplified.csv
├── model/               # Trained model checkpoints
├── score/               # Prediction outputs
└── submit/              # Final submissions
```

### Data Processing Pipeline
```bash
python main.py --step preprocess
```

This creates stratified train/validation splits while preserving category distributions.

## 🎨 Sketch Processing

### Stroke-to-Image Conversion
The solution converts stroke-based drawing data into rasterized images:

1. **Stroke Rendering**: Converts vector strokes to pixel-based images
2. **Color Encoding**: Different strokes have varying intensities for temporal information
3. **Image Resizing**: Standardizes to configurable dimensions (default: 64x64)
4. **Data Augmentation**: Horizontal flipping for training robustness

### Technical Implementation
```python
def _drawing_to_image(self, drawing_data: str) -> np.ndarray:
    drawing = literal_eval(drawing_data)
    image = np.zeros((256, 256), dtype=np.uint8)
    
    for stroke_idx, stroke in enumerate(drawing):
        stroke_color = 255 - min(stroke_idx, 10) * 13
        
        for point_idx in range(len(stroke[0]) - 1):
            x1, y1 = stroke[0][point_idx], stroke[1][point_idx]
            x2, y2 = stroke[0][point_idx + 1], stroke[1][point_idx + 1]
            cv2.line(image, (x1, y1), (x2, y2), stroke_color, 6)
    
    return processed_image
```

## 🧠 Model Architecture

### Transfer Learning Approach
- **Backbone**: Pre-trained ResNet models (ImageNet weights)
- **Custom Head**: Classification layer adapted for 340 categories
- **Architecture Variants**: ResNet18, ResNet34, ResNet50

### Model Configuration
```python
# ResNet50 Configuration
class ResNetClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=340):
        self.backbone = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(2048, num_classes)
```

### Key Architectural Decisions
1. **Transfer Learning**: Leverages ImageNet pre-trained features
2. **Adaptive Pooling**: Global average pooling for spatial invariance
3. **Multi-GPU Support**: DataParallel for efficient training
4. **Gradient Optimization**: Adam optimizer with learning rate scheduling

## 🎯 Training Pipeline

### Phase 1: Data Preparation
- Stroke data parsing and validation
- Stratified train/validation splitting
- Category mapping and indexing

### Phase 2: Model Training
- Transfer learning initialization
- Progressive learning rate scheduling
- Early stopping with patience mechanism
- Comprehensive logging and checkpointing

### Phase 3: Evaluation & Inference
- Top-K accuracy evaluation
- Batch prediction generation
- Submission file creation

### Training Configuration
```python
# Training Parameters
BATCH_SIZE = 650
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 5
WEIGHT_DECAY = 1e-4
```

## 🔧 Advanced Features

### Learning Rate Scheduling
- **Adaptive Reduction**: LR halved when validation metric plateaus
- **Early Stopping**: Training termination for overfitting prevention
- **Minimum LR Threshold**: Prevents excessive LR reduction

### Data Augmentation
- **Horizontal Flipping**: 50% probability during training
- **Stroke Color Variation**: Temporal information preservation
- **Image Normalization**: Consistent preprocessing pipeline

### Memory Optimization
- **Efficient Data Loading**: Multi-worker parallel processing
- **Gradient Accumulation**: Large effective batch sizes
- **Mixed Precision**: Optional FP16 training support

## 📈 Results

### Model Performance Comparison
| Model | Parameters | Top-3 Accuracy | Training Time |
|-------|------------|----------------|---------------|
| ResNet18 | 11.7M | ~85-87% | ~2 hours |
| ResNet34 | 21.8M | ~86-88% | ~3 hours |
| ResNet50 | 25.6M | ~87-89% | ~4 hours |

### Key Insights
1. **Transfer Learning Effectiveness**: Pre-trained features significantly improve performance
2. **Architecture Scaling**: Larger models show consistent improvements
3. **Data Augmentation Impact**: Horizontal flipping provides measurable benefits
4. **Batch Size Optimization**: Larger batches improve training stability

### Training Metrics
- **Convergence**: Models typically converge within 20-30 epochs
- **Validation Stability**: Consistent performance across different random seeds
- **Memory Efficiency**: Optimized for single-GPU training with 8GB+ VRAM

## 🚀 Usage

### Quick Start
```bash
# 1. Run complete pipeline
python main.py --step all --model resnet50

# 2. Or run individual steps
python main.py --step preprocess    # Data preparation
python main.py --step train         # Model training
python main.py --step predict       # Generate predictions
```

### Custom Configuration
```bash
# Custom training parameters
python main.py --step all \
    --model resnet34 \
    --lr 0.0005 \
    --epochs 30 \
    --batch-size 512
```

### Programmatic Usage
```python
from src.config import Config
from src.pipeline import DoodlePipeline

# Initialize pipeline
config = Config()
config.epochs = 30
config.batch_size = 512

pipeline = DoodlePipeline(config)

# Run specific steps
train_df, valid_df = pipeline.preprocess_data('../data/download')
results = pipeline.train_model(train_df, valid_df, 'resnet50')
```

### Model Inference
```python
from src.scorer import ModelScorer

# Load trained model
scorer = ModelScorer(
    config=config,
    model_path='../data/model/resnet50/resnet50_best.pth',
    model_name='resnet50'
)

# Generate predictions
test_df = pd.read_csv('../data/test/test_simplified.csv')
submission = scorer.generate_submission(test_df, 'submission.csv')
```

## 🔬 Technical Details

### Cross-Validation Strategy
- **Stratified Splits**: Maintains category distribution across train/validation
- **Random State**: 2017 for reproducible results
- **Split Ratio**: 90% training, 10% validation

### Optimization Strategy
- **Adam Optimizer**: Adaptive learning rates with momentum
- **Weight Decay**: L2 regularization for generalization
- **Learning Rate Scheduling**: Adaptive reduction based on validation performance

### Hardware Requirements
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ system memory for large batch processing
- **Storage**: 20GB+ for dataset and model checkpoints
- **CPU**: Multi-core recommended for data loading

### Performance Optimization
- **Data Loading**: Multi-worker parallel data loading
- **Memory Management**: Efficient tensor operations and cleanup
- **Batch Processing**: Optimized batch sizes for GPU utilization

## 📚 Key Learnings

1. **Transfer Learning Dominance**: Pre-trained ImageNet features provide excellent foundation for sketch recognition
2. **Data Representation**: Stroke-to-image conversion preserves temporal and spatial information effectively
3. **Architecture Scaling**: Larger ResNet models show consistent performance improvements
4. **Augmentation Impact**: Simple horizontal flipping provides measurable benefits
5. **Training Stability**: Learning rate scheduling and early stopping crucial for convergence

## 🎯 Business Applications

### Creative Tools
- **Sketch Recognition**: Real-time drawing classification in creative applications
- **User Interface**: Gesture recognition and sketch-based interfaces
- **Content Creation**: Automated tagging and categorization of user drawings

### Accessibility
- **Communication Aids**: Sketch-to-text conversion for communication
- **Educational Tools**: Interactive learning through drawing recognition
- **Assistive Technology**: Alternative input methods for users with disabilities

### Research Applications
- **Human-Computer Interaction**: Understanding drawing patterns and behaviors
- **Computer Vision**: Foundation models for sketch-based applications
- **Machine Learning**: Transfer learning and few-shot learning research

## 🎯 Future Improvements

### Model Enhancements
- **Vision Transformers**: ViT-based architectures for sketch recognition
- **Attention Mechanisms**: Spatial and temporal attention for better feature learning
- **Ensemble Methods**: Multi-model combination for improved accuracy

### Data Processing
- **Advanced Augmentation**: Rotation, scaling, and noise injection
- **Multi-Scale Training**: Different resolution training for robustness
- **Temporal Modeling**: RNN-based stroke sequence processing

### Training Optimization
- **Mixed Precision**: FP16 training for memory efficiency
- **Distributed Training**: Multi-GPU and multi-node training
- **AutoML**: Automated hyperparameter optimization

## 📖 References

- [Quick, Draw! Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition)
- [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Transfer Learning for Computer Vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 📄 License

This project is for educational and research purposes. Please ensure compliance with competition rules and dataset usage policies.

## 👨‍💻 Author

**Ujjwal Singh Rao**
- LinkedIn: [linkedin.com/in/brightertiger](https://linkedin.com/in/brightertiger)
- GitHub: [github.com/brightertiger](https://github.com/brightertiger)

---

**Note**: This solution achieved competitive performance in the Quick, Draw! Recognition Challenge through transfer learning, efficient data processing, and robust training pipelines. The codebase has been refactored for clarity, maintainability, and reproducibility, making it suitable for portfolio demonstration and further research in sketch recognition and computer vision applications.
