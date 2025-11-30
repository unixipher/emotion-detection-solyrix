"""
Working Training script for Speech Emotion Recognition
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

print("=" * 60)
print("SPEECH EMOTION RECOGNITION - TRAINING")
print("=" * 60)

print("\n1. Loading configuration...", flush=True)
from config import *

print(f"✓ Config loaded: {NUM_CLASSES} emotion classes", flush=True)
print(f"✓ Device: {DEVICE}", flush=True)
print(f"✓ Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}", flush=True)

print("\n2. Loading model...", flush=True)
from model import EnhancedHybridModel

print("\n3. Loading dataset...", flush=True)
from dataset import create_data_loaders


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
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


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, resume_path=None):
        print("\n4. Initializing trainer...", flush=True)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.start_epoch = 0
        
        # Class weights based on imbalance
        # Order: happiness, sadness, fear, anger, surprise, disgust
        class_weights = [1.0, 2.0, 4.0, 6.0, 2.4, 6.0]
        weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        print(f"   Class weights: {class_weights}", flush=True)
        
        # Use Focal Loss if enabled
        if USE_FOCAL_LOSS:
            self.criterion = FocalLoss(alpha=weights_tensor, gamma=2)
            print("   Using: Focal Loss", flush=True)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            print("   Using: Weighted CrossEntropy", flush=True)
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.early_stopping_counter = 0

        # Resume if checkpoint exists
        if resume_path and os.path.exists(resume_path):
            print(f"   Resuming from: {resume_path}", flush=True)
            checkpoint = torch.load(resume_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accs = checkpoint.get('train_accs', [])
            self.val_accs = checkpoint.get('val_accs', [])
            
            self.start_epoch = len(self.train_losses)
            if self.val_losses:
                self.best_val_loss = min(self.val_losses)
            if self.val_accs:
                self.best_val_acc = max(self.val_accs)
            
            print(f"   Resumed at epoch {self.start_epoch + 1}", flush=True)
        
        # Create directories
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        print("✓ Trainer initialized", flush=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', ncols=100)
        for mel_spec, prosodic, labels in pbar:
            mel_spec = mel_spec.to(self.device)
            prosodic = prosodic.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(mel_spec, prosodic)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for mel_spec, prosodic, labels in tqdm(self.val_loader, desc='Validation', ncols=100):
                mel_spec = mel_spec.to(self.device)
                prosodic = prosodic.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(mel_spec, prosodic)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs=EPOCHS):
        """Full training loop"""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Model parameters: {self.model.get_num_parameters():,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Starting from epoch: {self.start_epoch + 1}")
        print("=" * 60 + "\n")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, preds, labels = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.save_model('best_model.pth')
                print(f"   ✅ NEW BEST! Saved model (Val Acc: {val_acc:.2f}%)")
            else:
                self.early_stopping_counter += 1
                print(f"   No improvement ({self.early_stopping_counter}/{EARLY_STOPPING_PATIENCE})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
                print(f"   💾 Checkpoint saved")
            
            # Early stopping
            if self.early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"   Best Val Acc: {self.best_val_acc:.2f}%")
                break
        
        # Save final results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        
        self.save_training_plots()
        self.save_confusion_matrix(preds, labels)
        self.save_classification_report(preds, labels)
        
        print("\n✅ All results saved!")
        print(f"   📊 Plots: {RESULTS_DIR}/training_history.png")
        print(f"   🔲 Confusion Matrix: {RESULTS_DIR}/confusion_matrix.png")
        print(f"   📄 Report: {RESULTS_DIR}/classification_report.txt")
        print(f"   🎯 Best Model: {MODEL_DIR}/best_model.pth")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(MODEL_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
        }, filepath)
    
    def save_training_plots(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=150)
        plt.close()
    
    def save_confusion_matrix(self, preds, labels):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
        plt.close()
    
    def save_classification_report(self, preds, labels):
        """Save classification report"""
        report = classification_report(labels, preds, target_names=EMOTIONS, digits=4)
        
        with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.2f}%\n")
            f.write(f"Best Validation Loss: {self.best_val_loss:.4f}\n")
        
        print("\n" + "=" * 60)
        print("FINAL CLASSIFICATION REPORT")
        print("=" * 60)
        print(report)


def main():
    try:
        # Set device
        device = torch.device(DEVICE)
        print(f"\n✓ Using device: {device}", flush=True)
        
        # Create data loaders
        print(f"\nLoading data from: {DATA_DIR}", flush=True)
        train_loader, val_loader = create_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
        
        # Create model
        print("\nCreating model...", flush=True)
        model = EnhancedHybridModel()
        print(f"✓ Model created: {model.get_num_parameters():,} parameters", flush=True)
        
        # Check if resuming from checkpoint
        resume_path = os.path.join(MODEL_DIR, 'best_model.pth')
        if not os.path.exists(resume_path):
            resume_path = None
        
        # Create trainer and train
        trainer = Trainer(model, train_loader, val_loader, device, resume_path=resume_path)
        trainer.train()
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()