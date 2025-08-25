import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from fusion.fusion_model import EEGfMRIFusionNet
import matplotlib.pyplot as plt

def main():
    # ========== STEP 1: Load Preprocessed Features ==========
    # Replace these with np.load('preprocessed/eeg.npy') etc.
    eeg_features = np.random.rand(100, 128).astype(np.float32)   # 100 samples, 128 EEG features
    fmri_features = np.random.rand(100, 256).astype(np.float32)  # 100 samples, 256 fMRI features
    labels = np.random.randint(0, 2, size=(100,)).astype(np.int64)  # Binary classification

    # Convert to PyTorch tensors
    eeg_tensor = torch.tensor(eeg_features)
    fmri_tensor = torch.tensor(fmri_features)
    labels_tensor = torch.tensor(labels)

    # Dataset & Loader
    dataset = TensorDataset(eeg_tensor, fmri_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # ========== STEP 2: Model Setup ==========
    model = EEGfMRIFusionNet(eeg_dim=eeg_features.shape[1], fmri_dim=fmri_features.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ========== STEP 3: Training Loop ==========
    epochs = 10
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for eeg_batch, fmri_batch, label_batch in loader:
            optimizer.zero_grad()
            outputs = model(eeg_batch, fmri_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # ========== STEP 4: Save Model ==========
    torch.save(model.state_dict(), "models/fusion_model.pth")
    print("Model saved to models/fusion_model.pth")

    # ========== STEP 5: Plot Training Loss ==========
    import os
    os.makedirs("results", exist_ok=True)
    plt.plot(range(1, epochs+1), losses, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("results/training_loss.png")
    print("Saved training curve to results/training_loss.png")

if __name__ == "__main__":
    main()
