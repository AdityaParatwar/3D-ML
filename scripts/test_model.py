import torch
from models.pointnet import PointNet

model = PointNet(num_classes=10)

dummy_input = torch.rand(2, 1024, 3) 
output = model(dummy_input)

print("Output shape:", output.shape)