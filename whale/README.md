# Whale Identification Challenge

A comprehensive deep learning solution for identifying individual whales from images, developed for the [Happywhale - Whale and Dolphin Identification](https://www.kaggle.com/c/happywhale-cetacean-identification) challenge on Kaggle.

## 🏆 Challenge Overview

The Happywhale Whale and Dolphin Identification Challenge aimed to identify individual whales and dolphins from images taken in the wild. This is a critical problem in marine biology research where individual identification helps track migration patterns, population dynamics, and conservation efforts.

### Problem Statement
- **Task**: Multi-class classification to identify individual whales/dolphins from images
- **Input**: Images of whales and dolphins in their natural habitat
- **Output**: Top-5 predictions with confidence scores for each individual
- **Dataset**: ~25,000 training images, ~7,960 test images
- **Classes**: ~5,004 unique individual whales/dolphins
- **Evaluation**: Mean Average Precision @ 5 (MAP@5)

### Challenge Characteristics
- **Fine-grained Classification**: Distinguishing between very similar individuals
- **Class Imbalance**: Highly imbalanced dataset with many singleton classes
- **Domain Adaptation**: Images taken in various lighting and weather conditions
- **Limited Data**: Small number of images per individual whale
- **Similarity Learning**: Need to learn robust embeddings for individual identification

## 🚀 Solution Architecture

### 1. Multi-Stage Training Pipeline

**Progressive Training Strategy**: Implemented a sophisticated multi-stage approach:

```python
# Stage 1: Basic Classification
pipeline.train_classification_model(
    train_csv_path="data/train.csv",
    image_dir="data/train",
    model_name="classification"
)

# Stage 2: Pseudo Labeling
pipeline.train_with_pseudo_labels(
    train_csv_path="data/train.csv",
    pseudo_csv_path="data/pseudo_labels.csv",
    image_dir="data/train",
    model_name="pseudo_label"
)

# Stage 3: Center Loss Training
pipeline.train_classification_model(
    train_csv_path="data/train.csv",
    image_dir="data/train",
    use_center_loss=True,
    model_name="center_loss"
)

# Stage 4: Siamese Network
pipeline.train_siamese_model(
    train_csv_path="data/train.csv",
    image_dir="data/train",
    backbone_path="models/center_loss/model.pth",
    model_name="siamese"
)
```

### 2. Advanced Model Architecture

**ResNet50 with Custom Head**: Enhanced ResNet50 backbone with specialized components:

```python
class WhaleResNet(nn.Module):
    def __init__(self, num_classes=5004, freeze_layers=None):
        super().__init__()
        
        # Pretrained ResNet50 backbone
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Custom head with adaptive pooling
        head = [
            AdaptiveConcatPool2d(1),  # Global average + max pooling
            Flatten(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.25),
            nn.Linear(4096, 2048, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.33)
        ]
        self.head = nn.Sequential(*head)
        
        # Dual output: classification + embedding
        self.classifier = nn.Linear(2048, num_classes, bias=True)
        self.embedding = nn.Linear(2048, 256, bias=False)
```

**Key Design Decisions**:
- **Adaptive Concat Pooling**: Combines global average and max pooling for richer features
- **Dual Output**: Classification head for training, embedding head for similarity
- **Progressive Unfreezing**: Gradually unfreeze layers during training
- **L2 Normalization**: Normalized embeddings for better similarity computation

### 3. Center Loss for Metric Learning

**Center Loss Implementation**: Added center loss to learn discriminative embeddings:

```python
class CenterLoss(nn.Module):
    def __init__(self, num_classes=5004, feat_dim=256, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, x, labels):
        # Compute distances to class centers
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        
        # Extract distances for true classes
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = [distmat[i][mask[i]] for i in range(batch_size)]
        
        return torch.cat(dist).mean()
```

**Benefits**:
- **Discriminative Embeddings**: Forces similar individuals to cluster together
- **Better Generalization**: Improves performance on unseen individuals
- **Metric Learning**: Learns meaningful distance relationships

### 4. Siamese Network for Similarity Learning

**Pair-based Learning**: Implemented siamese network for direct similarity learning:

```python
class SiameseNetwork(nn.Module):
    def forward(self, image1, image2):
        # Get embeddings from both images
        _, embed1 = self.backbone(image1)
        _, embed2 = self.backbone(image2)
        
        # Create feature combinations
        add_feat = embed1 + embed2
        mul_feat = embed1 * embed2
        diff_feat = torch.abs(embed1 - embed2)
        
        # Concatenate all features
        combined_feats = torch.cat([embed1, embed2, add_feat, mul_feat, diff_feat], dim=1)
        
        # Apply normalization and classification
        combined_feats = self.norm(combined_feats)
        output = self.sigmoid(self.head(combined_feats))
        
        return output
```

**Feature Engineering**:
- **Element-wise Addition**: Captures shared features
- **Element-wise Multiplication**: Captures interaction features
- **Absolute Difference**: Captures distinguishing features
- **Concatenation**: Preserves all information for final decision

### 5. Advanced Data Augmentation

**Albumentations Pipeline**: Comprehensive augmentation strategy:

```python
def _get_train_transforms(self):
    return A.Compose([
        A.HorizontalFlip(p=0.5),           # Mirror images
        A.ShiftScaleRotate(rotate_limit=15, p=0.5),  # Geometric transforms
        A.RandomBrightnessContrast(p=0.3), # Lighting variations
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
```

**Augmentation Strategy**:
- **Geometric Transforms**: Rotation, scaling, translation
- **Photometric Transforms**: Brightness, contrast adjustments
- **Preserve Identity**: Augmentations that don't change whale identity
- **Robust Training**: Improves generalization to various conditions

### 6. Sophisticated Training Strategy

**Multi-Stage Learning**:

1. **Pretraining**: Train on external whale data (pseudo labels)
2. **Fine-tuning**: Adapt to competition-specific data
3. **Progressive Unfreezing**: Gradually unfreeze backbone layers
4. **Center Loss**: Add metric learning objective
5. **Siamese Training**: Learn pairwise similarities

**Training Configuration**:
- **Optimizer**: AdamW with decoupled weight decay
- **Scheduler**: Cosine annealing with warm restarts
- **Loss Function**: Cross-entropy + Center loss
- **Batch Size**: 64 for classification, 32 for siamese
- **Learning Rates**: 1e-3 for classification, 1e-4 for siamese

## 📊 Results and Performance

### Model Performance
- **Best Single Model**: Center Loss ResNet50 achieved strong performance
- **Ensemble Strategy**: Combined multiple model predictions
- **Final Score**: Competitive performance on private leaderboard

### Key Insights
1. **Center Loss Impact**: Significant improvement in embedding quality
2. **Pseudo Labeling**: External data helped with generalization
3. **Siamese Networks**: Effective for learning fine-grained similarities
4. **Progressive Training**: Multi-stage approach crucial for convergence

## 🛠️ Technical Implementation

### Code Structure
```
whale/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── data_utils.py      # Data loading and preprocessing
│   ├── models.py          # Model architectures
│   ├── trainer.py         # Training loops and utilities
│   ├── pipeline.py        # End-to-end pipeline
│   ├── optimizer.py       # Custom AdamW optimizer
│   └── scheduler.py       # Cosine annealing scheduler
├── main.py               # Command-line interface
├── example_usage.py      # Usage examples
├── requirements.txt      # Dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

### Key Features
- **Modular Design**: Clean separation of concerns
- **Configurable**: Easy to modify hyperparameters
- **Extensible**: Simple to add new models or training strategies
- **Reproducible**: Deterministic training with proper seeding
- **Scalable**: Efficient data loading and training

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/ujjwalrao/kaggle-whale-identification.git
cd kaggle-whale-identification

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Data Preparation

```bash
# Organize your data as follows:
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── test/
│   ├── test1.jpg
│   ├── test2.jpg
│   └── ...
├── train.csv
└── pseudo_labels.csv  # Optional external data
```

### Training Models

```bash
# Basic classification training
python main.py --mode train_classification \
               --data_dir data \
               --train_csv data/train.csv \
               --epochs 20 \
               --batch_size 64

# Training with center loss
python main.py --mode train_center_loss \
               --data_dir data \
               --train_csv data/train.csv \
               --epochs 20

# Full pipeline training
python main.py --mode full_pipeline \
               --data_dir data \
               --train_csv data/train.csv \
               --epochs 20
```

### Generating Predictions

```bash
# Generate predictions
python main.py --mode predict \
               --data_dir data \
               --model_path models/center_loss/model.pth \
               --output_file submission.csv
```

### Using the Python API

```python
from src.config import Config
from src.pipeline import WhaleIdentificationPipeline

# Create configuration
config = Config(
    data_dir="data",
    image_size=448,
    batch_size=64,
    learning_rate=1e-3,
    num_epochs=20
)

# Initialize pipeline
pipeline = WhaleIdentificationPipeline(config)

# Train model
history = pipeline.train_classification_model(
    train_csv_path="data/train.csv",
    image_dir="data/train",
    use_center_loss=True,
    model_name="center_loss"
)

# Generate predictions
submission = pipeline.predict(
    test_image_dir="data/test",
    model_path="models/center_loss/model.pth",
    model_type="classification"
)
```

## 🔬 Advanced Usage

### Custom Model Architecture

```python
from src.models import WhaleResNet

# Create custom model
model = WhaleResNet(
    num_classes=5004,
    freeze_layers=3  # Freeze first 3 layers
)

# Modify architecture
model.classifier = nn.Linear(2048, 1000)  # Reduce classes
model.embedding = nn.Linear(2048, 512)    # Increase embedding dim
```

### Custom Training Loop

```python
from src.trainer import Trainer, ModelCheckpoint

# Initialize trainer
trainer = Trainer(model, device='cuda')

# Custom checkpointing
checkpoint = ModelCheckpoint(
    save_dir="models/custom",
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

# Train with custom parameters
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric_fn=metric_fn,
    num_epochs=30,
    checkpoint=checkpoint
)
```

### Hyperparameter Tuning

```python
# Grid search example
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]
image_sizes = [224, 448]

for lr in learning_rates:
    for batch_size in batch_sizes:
        for image_size in image_sizes:
            config = Config(
                learning_rate=lr,
                batch_size=batch_size,
                image_size=image_size
            )
            
            pipeline = WhaleIdentificationPipeline(config)
            # Train and evaluate...
```

## 📈 Performance Optimization

### Memory Optimization
- **Gradient Accumulation**: Train with larger effective batch sizes
- **Mixed Precision**: Use FP16 for faster training
- **Data Loading**: Optimized data loaders with multiple workers

### Training Speed
- **Progressive Resizing**: Start with smaller images, increase gradually
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Early Stopping**: Prevent overfitting with validation monitoring

### Model Efficiency
- **Model Pruning**: Remove unnecessary parameters
- **Knowledge Distillation**: Train smaller models from larger ones
- **Quantization**: Reduce model size for deployment

## 🧪 Experimental Features

### Advanced Augmentation
- **Mixup**: Interpolate between training examples
- **CutMix**: Combine parts of different images
- **AutoAugment**: Learn optimal augmentation policies

### Ensemble Methods
- **Model Averaging**: Combine multiple model predictions
- **Stacking**: Train meta-model on base model predictions
- **Bagging**: Train models on different data subsets

### Loss Functions
- **Focal Loss**: Handle class imbalance better
- **Label Smoothing**: Improve generalization
- **Triplet Loss**: Alternative metric learning approach

## 📚 References and Acknowledgments

### Key Papers
1. **Center Loss**: "A Discriminative Feature Learning Approach for Deep Face Recognition" (Wen et al., 2016)
2. **Siamese Networks**: "Siamese Neural Networks for One-shot Image Recognition" (Koch et al., 2015)
3. **ResNet**: "Deep Residual Learning for Image Recognition" (He et al., 2016)
4. **AdamW**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)

### Datasets
- **Happywhale Challenge**: Kaggle competition dataset
- **External Whale Data**: Additional whale images for pseudo labeling

### Libraries and Tools
- **PyTorch**: Deep learning framework
- **Albumentations**: Image augmentation library
- **scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ujjwal Singh Rao**
- LinkedIn: [linkedin.com/in/brightertiger](https://linkedin.com/in/brightertiger)
- GitHub: [github.com/brightertiger](https://github.com/brightertiger)

---

*This project demonstrates advanced deep learning techniques for fine-grained image classification and metric learning. The multi-stage training pipeline, center loss implementation, and siamese network architecture showcase sophisticated approaches to solving challenging computer vision problems.*
