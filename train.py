import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.dataset import PointCloudDataset
from models.pointnet import PointNet

train_dataset = PointCloudDataset("data/pointclouds", split="train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = PointNet(num_classes=len(train_dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for points, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/pointnet.pth")
print("Model saved!")