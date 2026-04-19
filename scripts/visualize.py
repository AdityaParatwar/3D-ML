import numpy as np
import open3d as o3d

points = np.load("data/pointclouds/chair/train/chair_0001.npy")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])