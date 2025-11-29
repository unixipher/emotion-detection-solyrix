"""
Training script for Speech Emotion Recognition
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

from config import *
from model import HybridEmotionModel
from dataset import create_data_loaders


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, resume_path=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.start_epoch = 0  # Default start
        
        # --- NEW: CLASS WEIGHTING LOGIC ---
        # Based on your dataset distribution:
        # Happiness: ~12k (Base)
        # Sadness: ~6k (2x rarer)
        # Fear: ~3k (4x rarer)
        # Anger: ~2k (6x rarer)
        # Surprise: ~5k (2.4x rarer)
        # Disgust: ~2k (6x rarer)
        
        # Order must match config.py: ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
        class_weights = [1.0, 2.0, 4.0, 6.0, 2.4, 6.0]
        
        # Convert to tensor and move to device
        weights_tensor = torch.FloatTensor(class_weights).to(device)
        print(f"⚖️  Using Class Weights: {class_weights}")
        
        # Apply weights to Loss function
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        # ----------------------------------

        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

        # Resume Logic
        if resume_path and os.path.exists(resume_path):
            print(f"Loading checkpoint from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accs = checkpoint.get('train_accs', [])
            self.val_accs = checkpoint.get('val_accs', [])
            
            # Calculate start epoch based on history
            self.start_epoch = len(self.train_losses)
            if self.val_losses:
                self.best_val_loss = min(self.val_losses)
            
            print(f"✓ Resuming from epoch {self.start_epoch + 1}")
        
        # Create directories
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
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
            for mel_spec, prosodic, labels in tqdm(self.val_loader, desc='Validation'):
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
        print(f"\nStarting training on {self.device}")
        print(f"Model parameters: {self.model.get_num_parameters():,}")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
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
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.save_model('best_model.pth')
                print("✓ Best model saved!")
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save final results
        self.save_training_plots()
        self.save_confusion_matrix(preds, labels)
        self.save_classification_report(preds, labels)
        
        print("\n✓ Training completed!")
    
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
        }, filepath)
    
    def save_training_plots(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'))
        print(f"✓ Training plots saved to {RESULTS_DIR}/training_history.png")
    
    def save_confusion_matrix(self, preds, labels):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=EMOTIONS, yticklabels=EMOTIONS)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
        print(f"✓ Confusion matrix saved to {RESULTS_DIR}/confusion_matrix.png")
    
    def save_classification_report(self, preds, labels):
        """Save classification report"""
        report = classification_report(labels, preds, target_names=EMOTIONS)
        
        with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        print(f"✓ Classification report saved to {RESULTS_DIR}/classification_report.txt")
        print("\nClassification Report:")
        print(report)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_data_loaders(DATA_DIR)
    
    # Create model
    print("\nInitializing model...")
    model = HybridEmotionModel()
    
    # Check if a model exists to resume from
    resume_path = os.path.join(MODEL_DIR, 'best_model.pth')
    if not os.path.exists(resume_path):
        resume_path = None
    
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, device, resume_path=resume_path)
    trainer.train()


if __name__ == "__main__":
    main()