import open3d as o3d
import torch
import os
import pytorch_volumetric as pv
from pytorch_volumetric import sample_mesh_points
from pytorch_volumetric import voxel
from pytorch_volumetric import sdf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time
from sklearn.decomposition import PCA

OBJS_DIR = "objs/"
PCD_DIR = "pcd/"


# find centroid of point_cloud 
def pc_centroid(point_cloud):
    return np.mean(point_cloud, axis=0)

# find principal axis of point_cloud 
def pc_principal_axis(point_cloud):
    pc_np = np.asarray(point_cloud.points)
    pca_pc = PCA(n_components=3)
    pca_pc.fit(pc_np)
    eigen_vals_pc = pca_pc.explained_variance_
    axes_pc = pca_pc.components_
    return axes_pc[0] * 3 * np.sqrt(eigen_vals_pc[0]) 

# find rotation matrix to rotate vector a onto vector b
def rot_mat(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    cos_theta = np.dot(a, b)
    sin_theta = np.linalg.norm(v)
    if sin_theta == 0:
        return np.eye(3)
    v = v / sin_theta
    v_cross = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    R = np.eye(3) + sin_theta * v_cross + (1 - cos_theta) * np.dot(v_cross, v_cross)
    return R


# get point clouds (ground truth and transformed)
GT_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "drill.pcd"))
transformed_pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, "transformed_drill.pcd")) 
GT_np = np.asarray(GT_pcd.points)
transformed_np = np.asarray(transformed_pcd.points)

# find principal axes 
GT_pa = pc_principal_axis(GT_pcd)
transformed_pa = pc_principal_axis(transformed_pcd)

# find rotation matrix to align principal axes 
R = rot_mat(transformed_pa, GT_pa)

# find translation matrix to overlay centroids 
GT_centroid = pc_centroid(GT_np)
transformed_centroid = pc_centroid(transformed_np)
t = transformed_centroid - GT_centroid 

print("R ", R) # similar to GT t 
print("t ", t) # very close to GT t 
# take these as initial R and t, the continue to update with minimization function

# Visualize rotation 
# fig_gt = plt.figure(1) 
# ax_gt = fig_gt.add_subplot(111, projection='3d')
# ax_gt.scatter(GT_np[:, 0], GT_np[:, 1], GT_np[:, 2], alpha=0.2)

# fig_gt = plt.figure(2) 
# ax_gt = fig_gt.add_subplot(111, projection='3d')
# ax_gt.scatter(transformed_np[:, 0], transformed_np[:, 1], transformed_np[:, 2], alpha=0.2)

# fig_gt = plt.figure(3)
# ax_gt = fig_gt.add_subplot(111, projection='3d')
# aligned_np = transformed_np @ R
# ax_gt.scatter(aligned_np[:, 0], aligned_np[:, 1], aligned_np[:, 2], alpha=0.2)

# plt.show()

# Visuaize full transformation 

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud Localization")
GT_pcd.paint_uniform_color([1, 0, 0]) # Ground truth: red 
transformed_pcd.paint_uniform_color([0, 0, 1])  # Transformed: blue
recovered_points = (transformed_np - t) @ R.T
recovered_pcd = o3d.geometry.PointCloud()
recovered_pcd.points = o3d.utility.Vector3dVector(recovered_points)
recovered_pcd.paint_uniform_color([0, 1, 0])  # Recovered: green

vis.add_geometry(GT_pcd)
vis.add_geometry(transformed_pcd)
vis.add_geometry(recovered_pcd)

vis.run()