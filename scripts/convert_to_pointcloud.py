import os
import trimesh
import numpy as np

DATASET_PATH = "data/ModelNet10"
OUTPUT_PATH = "data/pointclouds"
NUM_POINTS = 1024

os.makedirs(OUTPUT_PATH, exist_ok=True)

def sample_points_from_mesh(file_path):
    mesh = trimesh.load(file_path)
    points = mesh.sample(NUM_POINTS)
    return points

for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)
    
    if not os.path.isdir(category_path):
        continue

    for split in ["train", "test"]:
        split_path = os.path.join(category_path, split)
        
        if not os.path.exists(split_path):
            continue

        save_folder = os.path.join(OUTPUT_PATH, category, split)
        os.makedirs(save_folder, exist_ok=True)

        for file in os.listdir(split_path):
            if file.endswith(".off"):
                file_path = os.path.join(split_path, file)
                
                points = sample_points_from_mesh(file_path)

                save_path = os.path.join(save_folder, file.replace(".off", ".npy"))
                np.save(save_path, points)

print("Conversion Done!")