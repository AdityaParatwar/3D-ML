import torch
from models.pointnet import PointNet
from scripts.dataset import PointCloudDataset

# Load dataset
dataset = PointCloudDataset("data/pointclouds", split="test")
classes = dataset.classes

# Load model
model = PointNet(num_classes=len(classes))
model.load_state_dict(torch.load("models/pointnet.pth"))
model.eval()

print("✅ Model Loaded Successfully!\n")

# Initialize variables
num_samples = 5
correct = 0

# Testing loop
for i in range(num_samples):
    points, label = dataset[i]
    points = points.unsqueeze(0)

    with torch.no_grad():
        output = model(points)
        pred = torch.argmax(output, dim=1).item()

    # Accuracy count
    if pred == label:
        correct += 1

    # Print results
    print(f"Sample {i+1}")
    print(f"Actual: {classes[label]}")
    print(f"Predicted: {classes[pred]}")
    print("-" * 30)

# Final accuracy
print(f"\nAccuracy: {correct}/{num_samples}")