import os
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points = points / max_dist
    return points

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.files = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            class_path = os.path.join(root_dir, cls, split)
            
            if not os.path.exists(class_path):
                continue
            
            for file in os.listdir(class_path):
                if file.endswith(".npy"):
                    self.files.append(os.path.join(class_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        points = np.load(self.files[idx])
        points = normalize(points)

        return torch.tensor(points, dtype=torch.float32), self.labels[idx]