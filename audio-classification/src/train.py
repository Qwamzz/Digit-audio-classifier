import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import os

from models.model import AudioCNN
from dataset import SpokenDigitDataset

def train(model, train_loader, num_epochs=10, device='cuda'):
    """Train the model"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = load_dataset("Robolab/FSDD-AIFF")
    train_dataset = SpokenDigitDataset(dataset['train'])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = AudioCNN().to(device)
    
    # Train model
    model = train(model, train_loader, device=device)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/audio_cnn.pt')
    print("Model saved to models/audio_cnn.pt")

if __name__ == "__main__":
    main()
