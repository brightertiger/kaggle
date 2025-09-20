import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from .config import Config

class MelanomaClassifier(nn.Module):
    def __init__(self, model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, metadata_dim=Config.METADATA_DIM):
        super(MelanomaClassifier, self).__init__()
        
        self.backbone = EfficientNet.from_pretrained(model_name, advprop=True)
        
        # Replace the classifier
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, 512, bias=True)
        
        # Metadata processing
        self.metadata_processor = nn.Linear(metadata_dim, 256, bias=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, metadata):
        image_features = self.backbone(image)
        metadata_features = self.metadata_processor(metadata)
        
        combined_features = torch.cat([image_features, metadata_features], dim=-1)
        output = self.classifier(combined_features)
        
        return output

class MelanomaClassifierV2(nn.Module):
    def __init__(self, model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, metadata_dim=Config.METADATA_DIM):
        super(MelanomaClassifierV2, self).__init__()
        
        self.backbone = EfficientNet.from_pretrained(model_name, advprop=True)
        
        # Replace the classifier
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, 512, bias=True)
        
        # Metadata processing with attention
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image, metadata):
        image_features = self.backbone(image)
        metadata_features = self.metadata_processor(metadata)
        
        # Apply attention
        image_features_attended, _ = self.attention(
            image_features.unsqueeze(1), 
            image_features.unsqueeze(1), 
            image_features.unsqueeze(1)
        )
        image_features_attended = image_features_attended.squeeze(1)
        
        combined_features = torch.cat([image_features_attended, metadata_features], dim=-1)
        output = self.classifier(combined_features)
        
        return output
