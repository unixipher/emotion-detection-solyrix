"""
Enhanced Hybrid CNN model with attention mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class AttentionBlock(nn.Module):
    """Channel and Spatial Attention"""
    def __init__(self, channels):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels)
        )
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_att = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class ResidualBlock(nn.Module):
    """Residual block with batch norm"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EnhancedMelCNN(nn.Module):
    """Deeper CNN with residual connections and attention"""
    def __init__(self):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Attention
        self.attention = AttentionBlock(512)
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.attention(x)
        
        # Combine avg and max pooling
        avg_pool = self.gap(x).view(x.size(0), -1)
        max_pool = self.gmp(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)  # 1024 features
        
        return x


class EnhancedProsodicBranch(nn.Module):
    """Enhanced prosodic feature processing"""
    def __init__(self, input_dim=13):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        return x


class EnhancedHybridModel(nn.Module):
    """Enhanced hybrid model with better fusion"""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        self.mel_cnn = EnhancedMelCNN()
        self.prosodic_branch = EnhancedProsodicBranch()
        
        # Cross-attention fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, batch_first=True
        )
        
        # Fusion layers (1024 from CNN + 256 from prosodic = 1280)
        self.fc1 = nn.Linear(1280, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, mel_spec, prosodic):
        # Extract features
        mel_features = self.mel_cnn(mel_spec)  # (batch, 1024)
        prosodic_features = self.prosodic_branch(prosodic)  # (batch, 256)
        
        # Concatenate
        combined = torch.cat([mel_features, prosodic_features], dim=1)
        
        # Fusion
        x = self.dropout1(F.relu(self.bn1(self.fc1(combined))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.fc_out(x)
        
        return x
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test
if __name__ == "__main__":
    model = EnhancedHybridModel()
    print(f"Parameters: {model.get_num_parameters():,}")
    
    mel = torch.randn(4, 1, 128, 130)
    pros = torch.randn(4, 13)
    out = model(mel, pros)
    print(f"Output: {out.shape}")