from dataset import PointCloudDataset
from torch.utils.data import DataLoader

dataset = PointCloudDataset("data/pointclouds", split="train")

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for points, labels in dataloader:
    print("Batch points:", points.shape)  # (16, 1024, 3)
    print("Batch labels:", labels.shape)
    break