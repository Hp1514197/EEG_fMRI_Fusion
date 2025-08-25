import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGEncoder(nn.Module):
    """Encodes flattened EEG features into an embedding."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class fMRIEncoder(nn.Module):
    """Encodes flattened fMRI features into an embedding."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class EEGfMRIFusionNet(nn.Module):
    """Simple late-fusion MLP for EEG + fMRI."""
    def __init__(self, eeg_dim: int = 128, fmri_dim: int = 256, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.eeg_encoder = EEGEncoder(eeg_dim, hidden_dim)
        self.fmri_encoder = fMRIEncoder(fmri_dim, hidden_dim)
        self.fc_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, eeg, fmri):
        eeg_feat = self.eeg_encoder(eeg)
        fmri_feat = self.fmri_encoder(fmri)
        fused = torch.cat((eeg_feat, fmri_feat), dim=1)
        x = F.relu(self.fc_fusion(fused))
        out = self.fc_out(x)
        return out
