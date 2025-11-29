"""
Hybrid CNN model for Speech Emotion Recognition
Combines mel-spectrogram CNN with prosodic features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class MelSpectrogramCNN(nn.Module):
    """CNN branch for processing mel-spectrograms"""
    
    def __init__(self):
        super(MelSpectrogramCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 256)
        
        return x


class ProsodicBranch(nn.Module):
    """Branch for processing prosodic features"""
    
    def __init__(self, input_dim=13):
        super(ProsodicBranch, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch, 13)
        
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        
        return x


class HybridEmotionModel(nn.Module):
    """
    Hybrid model combining CNN (mel-spectrogram) + Prosodic features
    """
    
    def __init__(self, num_classes=NUM_CLASSES):
        super(HybridEmotionModel, self).__init__()
        
        # Two branches
        self.mel_cnn = MelSpectrogramCNN()
        self.prosodic_branch = ProsodicBranch()
        
        # Fusion layers
        # 256 (from CNN) + 128 (from prosodic) = 384
        self.fc1 = nn.Linear(384, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, mel_spec, prosodic):
        """
        Args:
            mel_spec: (batch, 1, n_mels, time)
            prosodic: (batch, 13)
        """
        # Process through both branches
        mel_features = self.mel_cnn(mel_spec)
        prosodic_features = self.prosodic_branch(prosodic)
        
        # Concatenate features
        combined = torch.cat([mel_features, prosodic_features], dim=1)
        
        # Fusion layers
        x = self.dropout1(F.relu(self.bn1(self.fc1(combined))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x
    
    def get_num_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    model = HybridEmotionModel()
    print(f"Total parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    mel_spec = torch.randn(batch_size, 1, N_MELS, 87)  # 87 time frames for 2 sec
    prosodic = torch.randn(batch_size, 13)
    
    output = model(mel_spec, prosodic)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, num_classes)