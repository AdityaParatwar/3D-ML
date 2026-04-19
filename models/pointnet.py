import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()

        # Feature extraction
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)

        # Classification
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x shape: (B, N, 3)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Max pooling (IMPORTANT)
        x = torch.max(x, dim=1)[0]   # (B, 256)

        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)

        return x