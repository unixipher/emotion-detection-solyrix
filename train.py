"""
Advanced training techniques for higher accuracy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance better"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing to prevent overconfidence"""
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class MixUpAugmentation:
    """MixUp data augmentation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class AdvancedTrainer:
    """Enhanced trainer with modern techniques"""
    
    def __init__(self, model, train_loader, val_loader, device, config=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Default config
        self.config = config or {
            'lr': 0.001,
            'weight_decay': 1e-4,
            'epochs': 150,
            'use_focal_loss': True,
            'use_label_smoothing': True,
            'use_mixup': True,
            'gradient_clip': 1.0,
            'use_swa': True,  # Stochastic Weight Averaging
        }
        
        # Loss function
        if self.config['use_focal_loss']:
            # Class weights for focal loss
            alpha = torch.FloatTensor([1.0, 2.0, 4.0, 6.0, 2.4, 6.0]).to(device)
            self.criterion = FocalLoss(alpha=alpha, gamma=2)
        elif self.config['use_label_smoothing']:
            self.criterion = LabelSmoothingLoss(classes=6, smoothing=0.1)
        else:
            weights = torch.FloatTensor([1.0, 2.0, 4.0, 6.0, 2.4, 6.0]).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler (OneCycle for better convergence)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['lr'] * 10,
            epochs=self.config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Alternative: Cosine Annealing with Warm Restarts
        # self.scheduler = CosineAnnealingWarmRestarts(
        #     self.optimizer, T_0=10, T_mult=2
        # )
        
        # MixUp
        self.mixup = MixUpAugmentation(alpha=0.2) if self.config['use_mixup'] else None
        
        # SWA (Stochastic Weight Averaging)
        if self.config['use_swa']:
            self.swa_model = optim.swa_utils.AveragedModel(model)
            self.swa_scheduler = optim.swa_utils.SWALR(
                self.optimizer, swa_lr=self.config['lr'] * 0.1
            )
            self.swa_start = self.config['epochs'] * 0.75  # Start SWA at 75% training
        else:
            self.swa_model = None
        
        self.best_val_acc = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, epoch):
        """Train one epoch with advanced techniques"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (mel_spec, prosodic, labels) in enumerate(self.train_loader):
            mel_spec = mel_spec.to(self.device)
            prosodic = prosodic.to(self.device)
            labels = labels.to(self.device)
            
            # MixUp augmentation
            if self.mixup and np.random.random() > 0.5:
                mel_spec, labels_a, labels_b, lam = self.mixup(mel_spec, labels)
                
                self.optimizer.zero_grad()
                outputs = self.model(mel_spec, prosodic)
                
                loss = lam * self.criterion(outputs, labels_a) + \
                       (1 - lam) * self.criterion(outputs, labels_b)
            else:
                self.optimizer.zero_grad()
                outputs = self.model(mel_spec, prosodic)
                loss = self.criterion(outputs, labels)
            
            # Backward pass with gradient clipping
            loss.backward()
            
            if self.config['gradient_clip']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validation with TTA (Test Time Augmentation)"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for mel_spec, prosodic, labels in self.val_loader:
                mel_spec = mel_spec.to(self.device)
                prosodic = prosodic.to(self.device)
                labels = labels.to(self.device)
                
                # Regular prediction
                outputs = self.model(mel_spec, prosodic)
                loss = self.criterion(outputs, labels)
                
                # Optional: Test Time Augmentation (average multiple predictions)
                # outputs_tta = []
                # for _ in range(3):
                #     # Add slight noise
                #     mel_noisy = mel_spec + torch.randn_like(mel_spec) * 0.01
                #     outputs_tta.append(self.model(mel_noisy, prosodic))
                # outputs = torch.stack(outputs_tta).mean(dim=0)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Full training loop"""
        print(f"Training Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Update SWA model
            if self.swa_model and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'models/best_model.pth')
                print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
        
        # Finalize SWA
        if self.swa_model:
            print("\nFinalizing SWA model...")
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, self.device)
            
            # Validate SWA model
            swa_val_loss, swa_val_acc = self.validate_swa()
            print(f"SWA Val Acc: {swa_val_acc:.2f}%")
            
            if swa_val_acc > self.best_val_acc:
                torch.save({
                    'model_state_dict': self.swa_model.state_dict(),
                    'val_acc': swa_val_acc,
                }, 'models/best_swa_model.pth')
                print("✓ SWA model is better! Saved.")
    
    def validate_swa(self):
        """Validate SWA model"""
        self.swa_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for mel_spec, prosodic, labels in self.val_loader:
                mel_spec = mel_spec.to(self.device)
                prosodic = prosodic.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.swa_model(mel_spec, prosodic)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return running_loss / len(self.val_loader), 100 * correct / total